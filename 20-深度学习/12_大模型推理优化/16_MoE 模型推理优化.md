# 16_MoE 模型推理优化

## 1. MoE 模型回顾

Mixture of Experts (MoE) 模型通过**条件计算**实现了"大参数、低计算量"的效果。

```
MoE 前向传播:

输入 x → Router → 选择 Top-K 个专家
                    ↓
              Expert₁  Expert₂  Expert₃  Expert₄
              (激活)   (激活)   (休眠)   (休眠)
                ↓        ↓
              加权组合 → 输出

Mixtral 8x7B:
  - 总参数: 46.7B (8 个 7B 专家)
  - 每 token 激活参数: ~12.9B (选择 2 个专家)
  - 计算量: 等同于 12.9B 密集模型
  - 优势: 46.7B 参数的容量，12.9B 的计算成本
```

## 2. MoE 推理挑战

```
MoE 推理的主要挑战:

1. 内存占用
   所有专家参数都要加载到 GPU
   Mixtral 8x7B FP16: ~94 GB → 单卡放不下

2. 专家负载不均衡
   Router 倾向于选择少数"热门"专家
   → 部分 GPU 过载，部分空闲

3. All-to-All 通信
   分布式推理时 token 需要路由到对应专家所在 GPU
   → 通信开销大

4. 内存带宽浪费
   加载所有专家但只计算 2 个
   → 大量参数读取无用
```

## 3. 专家卸载 (Expert Offloading)

```python
class ExpertOffloadingEngine:
    """专家卸载：将不活跃专家放在 CPU/磁盘"""

    def __init__(self, model, gpu_memory_gb: float):
        self.model = model
        self.gpu_budget = gpu_memory_gb * 1024**3

        # 分析专家使用频率
        self.expert_usage = {}  # expert_id -> 使用次数

        # 决定哪些专家放 GPU，哪些放 CPU
        self.gpu_experts = set()
        self.cpu_experts = set()

    def plan_expert_placement(self, calibration_data):
        """规划专家放置策略"""
        # 1. 收集专家使用统计
        for data in calibration_data:
            routing_decision = self.model.route(data)
            for expert_id in routing_decision:
                self.expert_usage[expert_id] = self.expert_usage.get(expert_id, 0) + 1

        # 2. 按使用频率排序
        sorted_experts = sorted(
            self.expert_usage.items(),
            key=lambda x: x[1],
            reverse=True
        )

        # 3. 贪心放置：高频专家放 GPU
        current_memory = 0
        expert_size = self.model.get_expert_size()

        for expert_id, usage in sorted_experts:
            if current_memory + expert_size < self.gpu_budget:
                self.gpu_experts.add(expert_id)
                current_memory += expert_size
            else:
                self.cpu_experts.add(expert_id)

        print(f"GPU 专家: {len(self.gpu_experts)}, CPU 专家: {len(self.cpu_experts)}")

    def forward_with_offloading(self, x, top_k=2):
        """带卸载的前向传播"""
        # Router 决策
        expert_ids, weights = self.model.router(x, top_k)

        outputs = []
        for i, eid in enumerate(expert_ids):
            if eid in self.gpu_experts:
                # 直接从 GPU 计算
                out = self.model.experts[eid](x)
            else:
                # 从 CPU 加载并计算
                expert_weights = self.load_expert_from_cpu(eid)
                out = self.compute_expert(x, expert_weights)
            outputs.append(out * weights[i])

        return sum(outputs)

    def load_expert_from_cpu(self, expert_id):
        """从 CPU 加载专家到 GPU"""
        expert_data = self.cpu_expert_storage[expert_id]
        return expert_data.to("cuda")
```

## 4. 动态专家路由

```python
class DynamicExpertRouter:
    """动态路由优化"""

    def __init__(self, num_experts, top_k=2):
        self.num_experts = num_experts
        self.top_k = top_k
        self.load_balancer = ExpertLoadBalancer(num_experts)

    def route(self, x) -> tuple:
        """
        路由函数 + 负载均衡
        """
        # 原始路由分数
        router_logits = self.router_linear(x)  # [batch, num_experts]
        router_probs = F.softmax(router_logits, dim=-1)

        # Top-K 选择
        top_k_probs, top_k_indices = torch.topk(router_probs, self.top_k)

        # 负载均衡损失
        # 鼓励均匀使用所有专家
        load_balance_loss = self.load_balancer.penalty(router_probs)

        # 添加辅助损失到训练中
        return top_k_indices, top_k_probs, load_balance_loss


class ExpertLoadBalancer:
    """专家负载均衡"""

    def __init__(self, num_experts):
        self.num_experts = num_experts
        self.expert_loads = [0] * num_experts

    def penalty(self, router_probs):
        """负载均衡损失"""
        # f_i: 分配给专家 i 的 token 比例
        # P_i: router 选择专家 i 的平均概率
        # 损失 = α × num_experts × Σ(f_i × P_i)
        # 鼓励 f 和 P 都均匀分布
        f = (router_probs > 0).float().mean(dim=0)  # 实际负载
        P = router_probs.mean(dim=0)                 # 期望负载

        return self.num_experts * (f * P).sum()
```

## 5. 专家合并 (Expert Merging)

```python
"""
专家合并: 将多个专家合并为一个，减少参数量

方法 1: 加权平均
  W_merged = Σ(w_i × Expert_i)  (按使用频率加权)

方法 2: 奇异值分解合并
  对每层的权重矩阵做 SVD，保留主要分量

方法 3: 知识蒸馏
  用 8 个专家的输出训练 2 个合并专家
"""

class ExpertMerger:
    def merge_by_weighted_average(self, experts, usage_freqs):
        """加权平均合并"""
        total_weight = sum(usage_freqs)
        merged = {}

        for key in experts[0].keys():
            merged[key] = sum(
                experts[i][key] * usage_freqs[i] / total_weight
                for i in range(len(experts))
            )

        return merged

    def merge_by_svd(self, experts, rank_ratio=0.5):
        """SVD 合并"""
        # 合并所有专家的权重
        W_stack = torch.stack([e.weight for e in experts])  # [num_experts, out, in]

        # 重塑为 2D 矩阵
        num_experts, out, inp = W_stack.shape
        W_2d = W_stack.reshape(num_experts * out, inp)

        # SVD
        U, S, V = torch.svd(W_2d)

        # 保留主要分量
        rank = int(inp * rank_ratio)
        W_reconstructed = U[:, :rank] @ torch.diag(S[:rank]) @ V[:, :rank].T

        # 重塑回专家维度
        return W_reconstructed.reshape(num_experts, out, inp)
```

## 6. 分布式 MoE 推理

```python
class DistributedMoEInference:
    """分布式 MoE 推理"""

    def __init__(self, model, num_gpu):
        self.model = model
        self.num_gpu = num_gpu

        # 专家分配到不同 GPU
        experts_per_gpu = model.num_experts // num_gpu
        self.expert_placement = {}
        for i in range(model.num_experts):
            gpu_id = i // experts_per_gpu
            self.expert_placement[i] = min(gpu_id, num_gpu - 1)

    def forward(self, x):
        # 1. Router 决策 (在 GPU 0 上)
        expert_ids, weights = self.model.router(x)

        # 2. All-to-All 通信: 将 token 发送到对应专家所在 GPU
        # token → expert mapping → GPU mapping
        dispatched = self.dispatch_tokens(x, expert_ids)

        # 3. 各 GPU 并行计算本地专家
        local_outputs = {}
        for gpu_id in range(self.num_gpu):
            local_experts = [e for e, g in self.expert_placement.items() if g == gpu_id]
            for eid in local_experts:
                if eid in dispatched[gpu_id]:
                    local_outputs[eid] = self.compute_expert(gpu_id, eid, dispatched[gpu_id][eid])

        # 4. All-to-All: 收集结果
        gathered = self.gather_outputs(local_outputs, expert_ids)

        # 5. 加权组合
        return self.combine(gathered, weights)

    def dispatch_tokens(self, x, expert_ids):
        """将 token 路由到对应 GPU"""
        # 构建 send/recv 计划
        # 使用 NCCL All-to-All 通信
        pass
```

## 7. 性能对比

```
MoE 推理优化效果 (Mixtral 8x7B, A100 80GB):

方案                    │ 内存占用 │ 延迟    │ 吞吐量
────────────────────────┼──────────┼─────────┼────────
FP16 全部加载           │ 94 GB   │ -       │ -
INT4 全部加载 (AWQ)    │ 24 GB   │ 30ms/tok│ 800 tok/s
专家卸载 (8 GPU→2 GPU) │ 24 GB   │ 45ms/tok│ 500 tok/s
专家合并 (8→2 experts) │ 24 GB   │ 20ms/tok│ 1000 tok/s*
* 合并后质量有轻微下降
```

## 总结

MoE 模型推理的核心挑战是**内存**和**通信**。专家卸载和量化是最实用的方案 -- AWQ 4-bit 可以将 Mixtral 8x7B 放入单张 24GB GPU。分布式推理通过专家分片解决多卡场景，但 All-to-All 通信是瓶颈。
