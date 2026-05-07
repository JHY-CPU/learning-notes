# 8_Tensor Parallelism 张量并行

## 1. 张量并行概述

张量并行 (Tensor Parallelism, TP) 将**单个层的权重矩阵切分到多个 GPU**上，每个 GPU 负责部分计算，通过通信合并结果。

```
模型并行策略:

张量并行 (TP):     流水线并行 (PP):
  列切分/行切分       层切分
  GPU间高频通信       GPU间低频通信
  需要高速互联        可用普通互联
  单层跨多GPU         不同层在不同GPU
```

## 2. 线性层的张量并行

### 2.1 列切分 (Column Parallel)

```
Y = X @ W

W [d_out, d_in] 切分为 [d_out/N, d_in] 分到 N 个 GPU:

GPU 0: Y₀ = X @ W₀  [batch, d_out/N]
GPU 1: Y₁ = X @ W₁  [batch, d_out/N]
...
GPU N: Yₙ = X @ Wₙ  [batch, d_out/N]

拼接: Y = concat(Y₀, Y₁, ..., Yₙ)  [batch, d_out]

每个 GPU:
  - 存储: W/N 的权重
  - 计算: 1/N 的矩阵乘法
  - 通信: 无（结果直接拼接为完整输出）
```

### 2.2 行切分 (Row Parallel)

```
Y = X @ W

W [d_out, d_in] 切分为 [d_out, d_in/N]:

GPU 0: Y₀ = X @ W₀  [batch, d_out]
GPU 1: Y₁ = X @ W₁  [batch, d_out]
...
GPU N: Yₙ = X @ Wₙ  [batch, d_out]

求和: Y = Y₀ + Y₁ + ... + Yₙ  (All-Reduce)

每个 GPU:
  - 存储: W/N 的权重
  - 计算: 1/N 的矩阵乘法
  - 通信: All-Reduce 求和
```

## 3. Transformer 层的张量并行

```python
import torch
import torch.distributed as dist

class TensorParallelLinear(nn.Module):
    """张量并行线性层"""

    def __init__(self, in_features, out_features, parallel_mode="column"):
        super().__init__()
        self.parallel_mode = parallel_mode
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()

        if parallel_mode == "column":
            # 列切分: 每个 GPU 持有 out_features/world_size 行
            assert out_features % self.world_size == 0
            local_out = out_features // self.world_size
            self.weight = nn.Parameter(torch.randn(local_out, in_features))

        elif parallel_mode == "row":
            # 行切分: 每个 GPU 持有 in_features/world_size 列
            assert in_features % self.world_size == 0
            local_in = in_features // self.world_size
            self.weight = nn.Parameter(torch.randn(out_features, local_in))

    def forward(self, x):
        if self.parallel_mode == "column":
            # 本地计算，无需通信
            return x @ self.weight.T

        elif self.parallel_mode == "row":
            # 本地计算后 All-Reduce
            local_out = x @ self.weight.T
            dist.all_reduce(local_out)
            return local_out

class TensorParallelAttention(nn.Module):
    """张量并行多头注意力"""

    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads

        # Q, K, V 投影: 列切分 (每个 GPU 处理部分头)
        self.q_proj = TensorParallelLinear(d_model, d_model, "column")
        self.k_proj = TensorParallelLinear(d_model, d_model, "column")
        self.v_proj = TensorParallelLinear(d_model, d_model, "column")

        # 输出投影: 行切分
        self.o_proj = TensorParallelLinear(d_model, d_model, "row")

    def forward(self, x, kv_cache=None):
        B, S, _ = x.shape

        # 每个 GPU 计算部分头的 Q, K, V
        Q = self.q_proj(x).view(B, S, -1, self.head_dim)
        K = self.k_proj(x).view(B, S, -1, self.head_dim)
        V = self.v_proj(x).view(B, S, -1, self.head_dim)

        # 注意力计算 (本地)
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attn = F.softmax(scores, dim=-1)
        out = torch.matmul(attn, V)

        # 输出投影 (需要 All-Reduce)
        out = out.reshape(B, S, -1)
        return self.o_proj(out)
```

## 4. 完整 Transformer 的 TP 划分

```
Transformer 层的 TP 划分:

┌──────────────────────────────────────────┐
│              Input X                      │
├──────────────────────────────────────────┤
│  Attention Q/K/V (Column Parallel)        │
│  ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ GPU 0  │ │ GPU 1  │ │ GPU 2  │       │
│  │ Q₀K₀V₀ │ │ Q₁K₁V₁ │ │ Q₂K₂V₂ │       │
│  └────┬───┘ └────┬───┘ └────┬───┘       │
│       └─────┬────┘──────────┘            │
│             ↓ 拼接                        │
│  Attention Output (Row Parallel)          │
│  ┌────────┐ ┌────────┐ ┌────────┐       │
│  │ GPU 0  │ │ GPU 1  │ │ GPU 2  │       │
│  │   O₀   │ │   O₁   │ │   O₂   │       │
│  └────┬───┘ └────┬───┘ └────┬───┘       │
│       └─────┬────┘──────────┘            │
│             ↓ All-Reduce                  │
├──────────────────────────────────────────┤
│  FFN (类似划分)                           │
│  Linear1 (Column) → Activation → Linear2 (Row) │
│  最后 All-Reduce                          │
└──────────────────────────────────────────┘
```

## 5. 通信分析

```python
"""
通信量分析 (N 个 GPU):

Column Parallel:
  输出 = [batch, d_out]
  无通信 (直接拼接)

Row Parallel:
  需要 All-Reduce
  通信量 = 2 × (N-1)/N × batch × d_out × dtype_size

Attention 层总通信:
  Q/K/V: 无
  Output: All-Reduce [batch, seq, d_model]
  总计: ~2 × (N-1)/N × batch × seq × d_model

FFN 层总通信:
  Linear1: 无
  Linear2: All-Reduce [batch, seq, d_ffn]
  总计: ~2 × (N-1)/N × batch × seq × d_ffn

每层通信量 ≈ O(batch × seq × d_model)

使用 NVLink (600 GB/s):
  batch=32, seq=2048, d=4096, FP16
  通信时间 ≈ 32 × 2048 × 4096 × 2 / (600 × 10⁹) ≈ 0.9ms
  相比计算时间很小 → TP 在 NVLink 上效果好
"""
```

## 6. vLLM 中的张量并行

```python
# vLLM 使用 Ray 进行分布式张量并行

# 启动方式
# 单机多卡:
# python -m vllm.entrypoints.openai.api_server \
#     --model meta-llama/Llama-2-70b-hf \
#     --tensor-parallel-size 4 \  # 4 GPU
#     --gpu-memory-utilization 0.9

# 代码中使用:
from vllm import LLM

llm = LLM(
    model="meta-llama/Llama-2-70b-hf",
    tensor_parallel_size=4,  # 4 GPU 张量并行
    dtype="float16",
)

outputs = llm.generate(["你的提示词"])
```

## 7. TP 度选择指南

```
TP 度选择:

模型大小    推荐 TP 度    原因
────────────────────────────────────────
7B          1 (不需要)    单卡可放
13B         1-2           视 GPU 内存
70B         4-8           需要多卡
175B+       8+            必须多卡

注意事项:
1. TP 度必须能整除模型维度 (d_model, num_heads)
2. TP 度越大，通信开销越大
3. 推荐使用 NVLink/NVSwitch 互联
4. PCIe 互联不建议 TP > 2
5. TP 和 PP 可以组合使用 (TP×PP = 总 GPU 数)
```

## 总结

张量并行将单层计算分布到多个 GPU，通过 All-Reduce 通信合并结果。关键优势是**每个 GPU 只需存储 1/N 的权重**，可以运行更大模型。通信开销是主要瓶颈，需要高速互联（NVLink）支持。
