# 41_LoRA：低秩自适应微调原理

## 核心概念

- **LoRA (Low-Rank Adaptation)**：由 Hu et al. (2021) 提出，冻结预训练模型的权重，在模型层中插入可训练的低秩分解矩阵来微调。大幅降低显存和存储需求。
- **核心假设**：预训练模型具有较低的"内在维度"（intrinsic dimension），即模型适应新任务所需要的权重变化 $\Delta W$ 具有低秩特性。
- **低秩分解**：将权重更新矩阵 $\Delta W \in \mathbb{R}^{d \times k}$ 分解为两个小矩阵 $B \in \mathbb{R}^{d \times r}$ 和 $A \in \mathbb{R}^{r \times k}$ 的乘积，其中 $r \ll \min(d, k)$。
- **前向传播变化**：$h = W_0 x + \Delta W x = W_0 x + BAx$，原始的 $W_0$ 保持冻结，只有 $A$ 和 $B$ 在训练中更新。
- **推理零开销**：训练完成后，可以将 $B \cdot A$ 合并入原始权重 $W' = W_0 + BA$，推理时与原模型完全相同的计算量。
- **缩放因子**：$\Delta W$ 乘以 $\alpha/r$ 控制更新幅度，其中 $\alpha$ 是缩放常数（通常设为 $r$ 的倍数）。
- **应用范围**：通常应用于注意力层的 $W_Q, W_K, W_V, W_O$ 权重矩阵，也可用于 FFN 层。每个模块可以独立选择是否应用 LoRA。
- **秩的选择**：$r$ 通常取 4-64 之间。$r$ 越大，表达能力越强但参数量也越大。经验表明 $r=8$ 或 $r=16$ 在多数任务上已足够。

## 数学推导

对权重矩阵 $W_0 \in \mathbb{R}^{d \times k}$ 的更新：
$$
W = W_0 + \Delta W = W_0 + BA, \quad B \in \mathbb{R}^{d \times r}, A \in \mathbb{R}^{r \times k}
$$

前向传播：
$$
h = W_0 x + BAx
$$

**参数量对比**：
- 全量微调：$d \times k$
- LoRA 微调：$d \times r + r \times k = r \times (d + k)$

当 $d = k = 4096$, $r = 8$ 时：
- 全量微调：$4096^2 \approx 16.8M$ 参数
- LoRA：$8 \times (4096 + 4096) \approx 65K$ 参数（约 0.4%）

**初始化**：$A$ 使用随机高斯初始化，$B$ 初始化为 0，确保初始时 $\Delta W = 0$，训练从原始模型开始。

## 直观理解

- **LoRA 像给模型装"小配件"**：原始模型是一台功能强大的通用机器。LoRA 不是改造机器本身，而是在关键位置加装可调节的小旋钮。对于不同的任务，只需要调节这些小旋钮（$A, B$ 矩阵），而不是重新造机器。
- **低秩的含义**：权重更新 $\Delta W$ 是 $d \times k$ 的大矩阵，但实际上 $\Delta W$ 的有效信息可以用更少的维度表示。就像一张高清照片可以用"压缩"的方式存储——LoRA 就是权重更新的"压缩形式"。
- **推理零开销的含义**：训练完 LoRA 后，$BA$ 可以合并进 $W_0$，得到 $W' = W_0 + BA$。这意味着部署时加载 $W'$ 就相当于原始模型 + 微调效果，推理速度完全不变。

## 代码示例

```python
import torch
import torch.nn as nn

class LoRALayer(nn.Module):
    """LoRA 层实现"""
    def __init__(self, in_dim, out_dim, rank=8, alpha=16):
        super().__init__()
        self.alpha = alpha
        self.scaling = alpha / rank
        # 低秩矩阵 A 和 B
        self.A = nn.Parameter(torch.randn(in_dim, rank) * 0.01)
        self.B = nn.Parameter(torch.zeros(rank, out_dim))

    def forward(self, x):
        # x @ A @ B * scaling
        return (x @ self.A @ self.B) * self.scaling

class LoRALinear(nn.Module):
    """将 LoRA 应用于 Linear 层"""
    def __init__(self, linear, rank=8, alpha=16):
        super().__init__()
        self.linear = linear  # 冻结的原始线性层
        self.lora = LoRALayer(linear.in_features, linear.out_features, rank, alpha)

    def forward(self, x):
        return self.linear(x) + self.lora(x)

# 在 Transformer 注意力层上应用 LoRA
class MultiHeadAttentionWithLoRA(nn.Module):
    def __init__(self, d_model, n_heads, lora_rank=8):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        # 原始权重冻结
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
        # LoRA 适配器
        self.lora_q = LoRALayer(d_model, d_model, lora_rank)
        self.lora_v = LoRALayer(d_model, d_model, lora_rank)

    def forward(self, x):
        Q = self.W_Q(x) + self.lora_q(x)  # W_Q + LoRA
        K = self.W_K(x)
        V = self.W_V(x) + self.lora_v(x)  # W_V + LoRA
        # ... 标准注意力计算
        return self.W_O(torch.matmul(F.softmax(
            torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5), dim=-1), V))

# 参数量对比
d_model = 768
lora_r = 8
full_params = d_model * d_model * 4  # Q,K,V,O
lora_params = lora_r * d_model * 2   # 只在 Q,V 用 LoRA
print(f"全量微调参数量: {full_params:,}")
print(f"LoRA 微调参数量: {lora_params:,}")
print(f"参数减少比例: {(1 - lora_params / full_params) * 100:.2f}%")
```

## 深度学习关联

- **参数高效微调 (PEFT) 的标杆**：LoRA 是参数高效微调（Parameter-Efficient Fine-Tuning）的代表方法。与 Adapter、Prefix Tuning、Prompt Tuning 等方法一起构成了 PEFT 技术体系。
- **大模型微调的标配**：对于 GPT-3 175B、LLaMA-65B 等大模型，全量微调成本极高。LoRA 使得在消费级 GPU（如 RTX 3090 24GB）上微调 7B-13B 模型成为可能。
- **QLoRA 的扩展**：QLoRA (Dettmers et al., 2023) 将 LoRA 与 4-bit 量化结合，进一步降低了显存需求，使得在单张 24GB GPU 上微调 65B 模型成为可能。
