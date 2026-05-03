# 08_FSDP (Fully Sharded Data Parallel) 原理

## 核心概念

- **完全分片数据并行 (FSDP)**：在 DDP 基础上将模型参数、梯度和优化器状态也分布到各 GPU，而非在每个 GPU 上保留完整副本。它相当于 ZeRO-Stage 3 在 PyTorch 中的原生实现，可以训练 DDP 无法容纳的大模型。
- **参数分片 (Parameter Sharding)**：将模型参数沿第 0 维切分为 world_size 块，每个 GPU 只持有 1/world_size 的参数。在前向和反向传播时，按需从其他 GPU 收集完整参数，计算完后释放非本地的参数分片。
- **前向预取 (Forward Prefetch)**：FSDP 在执行当前模块的 forward 时，异步预取下一个模块的完整参数，使通信与计算重叠。同样地，在 backward 时预取上一个模块的参数梯度。
- **参数卸载 (CPU Offload)**：支持将优化器状态和参数卸载到 CPU RAM，进一步释放 GPU 显存。训练时仅将当前需要的参数分片放在 GPU 上，其余在 CPU 内存中。这使得单 GPU 可训练的模型规模从数亿参数扩展到数十亿参数。
- **混合分片策略 (Sharding Strategy)**：FSDP 提供了三种策略：`FULL_SHARD`（ZeRO-3，完整分片）、`SHARD_GRAD_OP`（ZeRO-2，不切分参数，只切分梯度和优化器状态）、`NO_SHARD`（等价于 DDP）。根据模型大小灵活选择。
- **通信-计算重叠优化**：FSDP 在 forward/backward 的每个模块边界处插入通信操作（all-gather 收集参数，reduce-scatter 归约梯度），并通过 CUDA 流调度使通信与模块计算并行执行。

## 数学推导

FSDP 的内存节省分析。设模型参数量为 $\Phi$，优化器状态占用为 $K\Phi$（Adam 需 2 个动量项，$K=2$；SGD 动量 $K=1$），梯度占用 $\Phi$，参数占用 $\Phi$。

**未分片（DDP）单 GPU 内存需求**：
$$
M_{\text{DDP}} = \underbrace{\Phi}_{\text{参数}} + \underbrace{\Phi}_{\text{梯度}} + \underbrace{K\Phi}_{\text{优化器状态}} = (2+K)\Phi
$$

**FSDP 完全分片后单 GPU 内存需求**：
$$
M_{\text{FSDP}} = \frac{\Phi}{N} + \frac{\Phi}{N} + \frac{K\Phi}{N} + \underbrace{\alpha\Phi}_{\text{单个 FSDP 单元完整参数暂存}} = \frac{(2+K)\Phi}{N} + \alpha\Phi
$$

其中 $N$ 是 GPU 数量，$\alpha$ 是同时暂存的 FSDP 单元比例（通常为 1-2 个 Transformer Block），远小于 1。

以 7B 模型 (Φ ≈ 7×10⁹) 在 8 卡上为例：
- DDP: $M \approx 7B × (2+2) = 28B$ 参数 ≈ 112 GB（以 FP32 计）
- FSDP: $M \approx 28B/8 + 7B × 0.1 ≈ 4.2B$ 参数 ≈ 16.8 GB

这使得 7B 模型可以在 8×A100-80GB 上训练，而 DDP 需要 4 倍以上的显存。

## 直观理解

- **FSDP = 分布式共享书库**：DDP 是每人拿一本完整的书（完整模型），很占地方；FSDP 是每人拿书的几页（参数分片），要看某页时从持有者那里借来，看完还回去。显存占用量从"书的厚度"变成了"几页纸"。
- **最佳实践**：对于 >1B 参数的模型，使用 FSDP 的 `FULL_SHARD` 策略并开启 CPU Offload；对于 100M-1B 的模型，`SHARD_GRAD_OP` 策略通常足够且通信开销更小。
- **常见陷阱**：FSDP 的通信量是 DDP 的 O(N) 倍（每个模块都需要 all-gather + reduce-scatter），因此在小模型或低速网络（如 EFA 带宽不足）上反而比 DDP 慢。
- **经验法则**：`forward_prefetch=True` 对 Transformer 类模型效果明显；`limit_all_gathers=True` 可防止 all-gather 操作过度堆积导致 OOM。

## 代码示例

```python
import torch
import torch.nn as nn
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    CPUOffload,
    MixedPrecision,
    BackwardPrefetch,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import (
    transformer_auto_wrap_policy,
    size_based_auto_wrap_policy,
)
import torch.distributed as dist
from functools import partial

# ========== 1. 基础 FSDP 使用 ==========

# 1.1 定义模型（需要明确可包装的子模块边界）
class TransformerBlock(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.attn = nn.MultiheadAttention(dim, 8, batch_first=True)
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Linear(dim * 4, dim),
        )
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x):
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]
        x = x + self.mlp(self.norm2(x))
        return x

class LargeTransformer(nn.Module):
    def __init__(self, num_layers=12, dim=768):
        super().__init__()
        self.embed = nn.Embedding(32000, dim)
        self.blocks = nn.ModuleList([
            TransformerBlock(dim) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, 32000)

    def forward(self, x):
        x = self.embed(x)
        for block in self.blocks:
            x = block(x)
        return self.head(self.norm(x))

# 1.2 配置 FSDP
def get_fsdp_model(rank, world_size):
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

    model = LargeTransformer(num_layers=12, dim=768).cuda()

    # 自动包装策略：基于 transformer block 边界
    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={TransformerBlock},
    )

    fsdp_model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        sharding_strategy=ShardingStrategy.FULL_SHARD,  # ZeRO-3
        cpu_offload=CPUOffload(offload_params=False),
        mixed_precision=MixedPrecision(
            param_dtype=torch.bfloat16,
            reduce_dtype=torch.bfloat16,
            buffer_dtype=torch.bfloat16,
        ),
        backward_prefetch=BackwardPrefetch.BACKWARD_PRE,
        forward_prefetch=True,
        limit_all_gathers=True,
    )

    return fsdp_model

# ========== 2. FSDP 训练循环 ==========
# def train_fsdp(fsdp_model, dataloader, rank):
#     optimizer = torch.optim.AdamW(fsdp_model.parameters(), lr=1e-4)
#     for epoch in range(3):
#         for batch in dataloader:
#             inputs = batch["input_ids"].cuda()
#             labels = batch["labels"].cuda()
#             outputs = fsdp_model(inputs)
#             loss = nn.CrossEntropyLoss()(outputs.view(-1, 32000), labels.view(-1))
#             loss.backward()
#             optimizer.step()
#             optimizer.zero_grad()
#         # 只在 rank 0 保存
#         if rank == 0:
#             torch.save(fsdp_model.state_dict(), f"fsdp_epoch{epoch}.pt")

# ========== 3. 保存与加载 FSDP 模型 ==========
def save_fsdp_model(fsdp_model, rank, path):
    """FSDP 模型的正确保存方式"""
    if rank == 0:
        # 需要先收集完整权重
        with FSDP.summon_full_params(fsdp_model, writeback=False, recurse=True):
            torch.save(fsdp_model.state_dict(), path)

def load_fsdp_model(fsdp_model, rank, path):
    """加载 FSDP 模型"""
    # FSDP 会自动重新分片
    state_dict = torch.load(path, map_location="cpu")
    missing, unexpected = fsdp_model.load_state_dict(state_dict, strict=False)
    if rank == 0 and (missing or unexpected):
        print(f"Missing: {missing}, Unexpected: {unexpected}")

# ========== 4. 分片策略选择 ==========
# FULL_SHARD:    参数 + 梯度 + 优化器状态全分片 (ZeRO-3)，最大显存节省
# SHARD_GRAD_OP: 只分片梯度和优化器状态 (ZeRO-2)
# NO_SHARD:      等价于 DDP，但可以使用 FSDP 的 CPU Offload 功能
# HYBRID_SHARD:  节点内 FULL_SHARD，节点间 DDP，适合多机场景
```

## 深度学习关联

- **大语言模型 (LLM) 训练标准方案**：FSDP 是当前训练 7B~70B 参数规模 LLM 的主流方案。Hugging Face Transformers 已原生支持 FSDP，只需在 `TrainingArguments` 中设置 `fsdp=True` 即可。在 MLflow 中记录 FSDP 的分片策略和 GPU 数量，有助于分析训练的扩展效率 (scaling efficiency)。
- **Kubernetes 上的弹性训练**：FSDP 支持 `world_size` 的动态变化（需配合 `torch.distributed.elastic`），允许在 Spot Instance 回收、节点故障时自动调整 GPU 数量并继续训练，而无需从头开始。
- **与 DeepSpeed ZeRO-3 的对比**：FSDP 作为 PyTorch 原生实现，与 torch.compile 的兼容性更好且无需额外安装；DeepSpeed ZeRO-3 则在 offload 策略（NVMe offload）和通信优化上更成熟。选择 FSDP 还是 DeepSpeed 取决于具体场景的显存需求和生态兼容性。
