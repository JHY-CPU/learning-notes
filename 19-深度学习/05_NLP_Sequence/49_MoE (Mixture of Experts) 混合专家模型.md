# 49_MoE (Mixture of Experts) 混合专家模型

## 核心概念

- **MoE (Mixture of Experts)**：一种条件计算（Conditional Computation）技术。将神经网络的 FFN 层替换为多个并行的"专家"(Experts)子网络，每个输入 token 由路由(Router)选择激活少数专家。
- **路由机制 (Router)**：一个轻量级分类器，为每个 token 计算选择每个专家的概率。通常使用 softmax 选择 top-k 专家。路由是 MoE 的核心——它决定了哪些专家处理哪些 token。
- **Top-k 稀疏门控**：不是所有专家都对每个 token 生效。传统上使用 top-2 门控（激活得分最高的 2 个专家），稀疏激活极大减少了计算量。
- **负载均衡损失 (Load Balancing Loss)**：防止路由将所有 token 分配给少数专家。通过辅助损失惩罚专家负载的不均衡分布，鼓励均匀使用所有专家。
- **专家容量 (Expert Capacity)**：每个专家在一个 batch 中能处理的最大 token 数。超过容量的 token 被丢弃（dropped）或跳过当前层。容量是计算效率和模型质量之间的权衡。
- **Switch Transformer**：Google 提出的 MoE 变体，使用 top-1 门控（每个 token 只激活 1 个专家），更极致的稀疏性。
- **Mixtral 8x7B**：Mistral AI 的 MoE 模型，8 个专家（每次激活 2 个），总参数量 46.7B，但每次只激活 12.9B 参数。性能超越 LLaMA-2-70B 和 GPT-3.5。
- **训练与推理的差异**：训练时使用完整的 batch 数据训练所有专家。推理时每个 token 只路由到 top-k 专家，总 FLOPs 远少于同等稠密模型（但内存占用依然大，因为所有专家的参数都需要加载）。

## 数学推导

**MoE 层的前向传播**：
$$
y = \sum_{i=1}^{N} G(x)_i \cdot E_i(x)
$$

其中 $N$ 是专家数量，$G(x)$ 是路由权重，$E_i(x)$ 是第 $i$ 个专家的输出。

**Top-k 门控**：
$$
G(x) = \text{softmax}(\text{TopK}(x \cdot W_g, k))
$$

$$
\text{TopK}(v, k)_i = \begin{cases}
v_i & \text{if } v_i \text{ is in top k} \\
-\infty & \text{otherwise}
\end{cases}
$$

**负载均衡损失**（辅助损失）：
$$
\mathcal{L}_{\text{balance}} = \alpha \cdot N \sum_{i=1}^{N} f_i \cdot P_i
$$

其中 $f_i$ 是分配给专家 $i$ 的 token 比例，$P_i$ 是路由给专家 $i$ 的平均概率。$\alpha$ 是损失权重（通常 0.01）。

当所有专家被均匀使用时，$f_i = P_i = 1/N$，损失最小。

## 直观理解

- **MoE 像"专家会诊"**：一个病人（token）不需要见所有医生（全部专家），只需由分诊台（Router）判断问题类型，然后分配给 2 个相关专家——心脏问题和高血压问题各一位专家。虽然医院有 100 位专家，但你每次只看其中 2 位，效率非常高。
- **稀疏激活的含义**：总参数量 46.7B 的 Mixtral，每次推理只激活 12.9B 参数。类比：图书馆有 10000 本书，但你每次只借 2 本（top-2）。你可以访问所有书的知识，但实际上处理的只是一小部分。
- **负载均衡损失像"不让某位医生太累"**：如果没有均衡机制，也许"内科专家"专家总是被选中（因为它什么都能看），其他专家闲置。负载均衡确保每个专家被选中的频率大致相等——分摊工作。
- **Top-1 vs Top-2**：Switch Transformer 用"只看一个专家"（top-1），更高效（计算更少），但对路由准确性要求更高。Mistral 8x7B 用"看两个专家"（top-2），更稳健一些。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class MoELayer(nn.Module):
    """混合专家层（简化实现）"""
    def __init__(self, dim, num_experts=8, top_k=2, capacity_factor=1.25):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.capacity_factor = capacity_factor

        # 路由器
        self.router = nn.Linear(dim, num_experts, bias=False)

        # 多个专家（每个专家是 2 层 FFN）
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim)
            ) for _ in range(num_experts)
        ])

    def forward(self, x):
        # x: (batch, seq, dim)
        batch, seq, dim = x.shape
        x_flat = x.view(-1, dim)                      # (batch*seq, dim)

        # 路由：计算每个 token 选择各专家的概率
        router_logits = self.router(x_flat)            # (batch*seq, n_experts)
        router_weights = F.softmax(router_logits, dim=-1)

        # Top-k 选择
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)

        # 计算专家容量
        capacity = int((batch * seq * self.top_k / self.num_experts) * self.capacity_factor)

        # 逐专家处理
        output = torch.zeros_like(x_flat)
        for expert_idx in range(self.num_experts):
            # 找到被分配给当前专家的 token
            mask = (top_k_indices == expert_idx)
            if not mask.any():
                continue

            # 限制容量
            selected = mask.nonzero(as_tuple=True)
            if selected[0].size(0) > capacity:
                # 超出容量时丢弃
                selected = (selected[0][:capacity], selected[1][:capacity])

            expert_input = x_flat[selected[0]]
            expert_output = self.experts[expert_idx](expert_input)

            # 累积输出（乘以对应权重）
            weights = top_k_weights[selected[0], selected[1]]
            output.index_add_(0, selected[0], expert_output * weights.unsqueeze(-1))

        # 负载均衡损失计算
        # f_i: 被分配给各专家的 token 比例
        f_i = torch.zeros(self.num_experts, device=x.device)
        for i in range(self.num_experts):
            f_i[i] = (top_k_indices == i).float().mean()

        P_i = router_weights.mean(dim=0)  # 平均 router 概率
        balance_loss = self.num_experts * (f_i * P_i).sum()

        return output.view(batch, seq, dim), balance_loss

# 演示 MoE 的计算特性
dim = 768
batch, seq = 4, 128
x = torch.randn(batch, seq, dim)

moe = MoELayer(dim, num_experts=8, top_k=2)
output, bal_loss = moe(x)

total_params = 8 * (2 * 768 * 3072)  # 8 个专家
activated_params = 2 * (2 * 768 * 3072)  # top-2
print(f"总参数量: {total_params / 1e6:.1f}M")
print(f"激活参数量: {activated_params / 1e6:.1f}M")
print(f"稀疏度: {activated_params / total_params * 100:.1f}%")
print(f"负载均衡损失: {bal_loss.item():.4f}")
```

## 深度学习关联

- **万亿参数模型的范式**：MoE 是实现超大模型（如 GPT-4、PaLM-2、Mixtral）的核心技术。稀疏激活使得模型可以用千亿参数的总容量包含更多知识，但每次推理的计算成本只增加 20-40%。
- **训练稳定性挑战**：MoE 的训练比稠密模型更不稳定，路由坍塌（router collapse——所有 token 都选同一专家）和负载不均衡是常见问题。近年来的 Z-loss 辅助损失、RotoGrad 等方法被提出以稳定训练。
- **分布式训练的挑战**：MoE 需要额外的"all-to-all"通信操作来将 token 分发给正确的专家所在的 GPU。高效的分布式 MoE 训练是系统研究的热点（如 GShard、Switch Transformer 的分片策略）。
