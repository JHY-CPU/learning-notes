# FSDP与ZeRO - 大模型训练工程

*完全分片数据并行：ZeRO 三个阶段与 PyTorch FSDP 实战*


## FSDP 与 ZeRO


完全分片数据并行：ZeRO 三个阶段与 PyTorch FSDP 实战

[88-大模型训练工程](../index.html)
>
[01-分布式训练](./)
>
            02_FSDP与ZeRO

### 目录


1. [ZeRO 概述与动机](#zero-overview)
2. [ZeRO Stage 1：优化器状态分片](#zero-stage1)
3. [ZeRO Stage 2：梯度分片](#zero-stage2)
4. [ZeRO Stage 3：参数分片](#zero-stage3)
5. [ZeRO 各阶段对比](#zero-comparison)
6. [FSDP 实战](#fsdp)
7. [Activation Checkpointing](#activation-checkpointing)
8. [最佳实践](#best-practice)


## 1. ZeRO 概述与动机


ZeRO（Zero Redundancy Optimizer）是微软 DeepSpeed 提出的一种内存优化技术，旨在消除数据并行中各 GPU 之间的内存冗余。标准数据并行（DDP）要求每个 GPU 都持有完整的模型副本，这限制了可训练模型的大小。


### 1.1 内存消耗分析


假设模型参数量为 Ψ（以字节计），在混合精度训练下，每个参数需要：


$$
总内存 = 参数(2Ψ) + 梯度(2Ψ) + 优化器状态(12Ψ) = 16Ψ
$$


其中：


- **参数 (Parameters)：**
   2Ψ — FP16/BF16 模型参数
- **梯度 (Gradients)：**
   2Ψ — FP16/BF16 梯度
- **优化器状态 (Optimizer States)：**
   12Ψ — Adam 优化器需要 FP32 参数副本(4Ψ) + 动量 m(4Ψ) + 方差 v(4Ψ)


> **Note:** **显存估算示例：**
> 一个 7B 参数的模型，混合精度 + Adam 优化器：
>
> - 参数：7B × 2 bytes = 14 GB
> - 梯度：7B × 2 bytes = 14 GB
> - 优化器状态：7B × 12 bytes = 84 GB
> - **总计：112 GB**
>    （不含激活值和临时缓冲区）
>
> 单卡 A100 (80GB) 无法训练，但通过 ZeRO 可以在 8 卡 A100 上训练。


### 1.2 ZeRO 的核心思想


ZeRO 将模型状态（参数、梯度、优化器状态）在数据并行的各 GPU 之间进行**分片**（partition/shard），每个 GPU 只维护模型状态的 1/world_size 部分，在需要时通过通信收集完整状态。


## 2. ZeRO Stage 1：优化器状态分片


ZeRO Stage 1 只对优化器状态进行分片，这是最保守但开销最小的方案。


### 2.1 内存节省


$$
每 GPU 内存 = 2Ψ + 2Ψ + 12Ψ/N = 4Ψ + 12Ψ/N
$$


其中 N 为 GPU 数量。当 N 很大时，优化器状态的内存占用接近 0。


### 2.2 工作流程


```
ZeRO Stage 1 工作流程 (N=4 GPU):

每个 GPU 持有:
┌──────────┬──────────┬──────────┬──────────┐
│  GPU 0   │  GPU 1   │  GPU 2   │  GPU 3   │
├──────────┼──────────┼──────────┼──────────┤
│ 完整参数 │ 完整参数 │ 完整参数 │ 完整参数 │
│ 完整梯度 │ 完整梯度 │ 完整梯度 │ 完整梯度 │
│ 优化器   │ 优化器   │ 优化器   │ 优化器   │
│ 状态 1/4 │ 状态 2/4 │ 状态 3/4 │ 状态 4/4 │
└──────────┴──────────┴──────────┴──────────┘

优化步骤:
1. 每个 GPU 更新自己负责的参数切片
2. All-Gather 收集所有更新后的参数
3. 每个 GPU 获得完整的最新参数
```


### 2.3 通信开销


Stage 1 的额外通信开销为每步一次 All-Gather，与标准 DDP 相同（DDP 本身就需要一次 All-Reduce，可被等价替换）。


## 3. ZeRO Stage 2：梯度分片


在 Stage 1 的基础上，Stage 2 进一步对梯度进行分片，每个 GPU 只保存自己负责的那部分梯度。


### 3.1 内存节省


$$
每 GPU 内存 = 2Ψ + 2Ψ/N + 12Ψ/N = 2Ψ + 14Ψ/N
$$


### 3.2 工作流程


```
ZeRO Stage 2 工作流程 (N=4 GPU):

反向传播阶段:
1. 每个 GPU 计算完整梯度 ∇L_i
2. Reduce-Scatter: 将梯度分片归约
   GPU 0 收到 Σ∇L 的 1/4 (参数 0~Ψ/4)
   GPU 1 收到 Σ∇L 的 1/4 (参数 Ψ/4~Ψ/2)
   GPU 2 收到 Σ∇L 的 1/4 (参数 Ψ/2~3Ψ/4)
   GPU 3 收到 Σ∇L 的 1/4 (参数 3Ψ/4~Ψ)
3. 每个 GPU 只保留自己的梯度分片, 释放其余

优化步骤:
4. 各 GPU 用分片梯度更新分片参数
5. All-Gather 收集更新后的完整参数
```


> **Note:** **通信对比：**
> 标准 DDP 使用 All-Reduce 同步梯度，而 ZeRO Stage 2 使用 Reduce-Scatter。两者在 Ring 算法下的理论通信量相同（都是 2K），但 Reduce-Scatter 可以让 GPU 只保留部分结果。


### 3.3 梯度桶的组织


Stage 2 中，梯度按参数分配关系组织成桶。每个 GPU 负责管理某些参数的梯度和优化器状态，当反向传播产生梯度时，先在本地缓冲，桶满后触发 Reduce-Scatter 将梯度发送到正确的 GPU。


## 4. ZeRO Stage 3：参数分片


Stage 3 是最激进的方案，将参数、梯度、优化器状态全部分片，每个 GPU 只持有模型状态的 1/N。


### 4.1 内存节省


$$
每 GPU 内存 = (2Ψ + 2Ψ + 12Ψ) / N = 16Ψ / N
$$


内存随 GPU 数量线性扩展！


### 4.2 工作流程


```
ZeRO Stage 3 工作流程 (N=4 GPU):

前向传播:
  ┌─────────┐  All-Gather  ┌─────────┐
  │ 各 GPU  │ ──────────► │ 完整参数 │
  │ 参数1/4 │   收集参数    │ (临时)  │
  └─────────┘              └────┬────┘
                                │ 计算前向
                                ▼
                         释放未使用的参数 (按层)

反向传播:
                         ┌─────────┐
                         │ 需要参数 │ (逐层)
                         └────┬────┘
                              │ All-Gather 收集
                              ▼
                         ┌─────────┐
                         │ 计算梯度 │
                         └────┬────┘
                              │ Reduce-Scatter
                              ▼
                    ┌─────────────────┐
                    │ 各 GPU 保留梯度1/4│
                    └─────────────────┘

优化步骤:
  各 GPU 用分片梯度更新分片参数
  All-Gather 收集完整参数用于下一轮
```


### 4.3 前向和反向的参数预取


Stage 3 最大的挑战是通信延迟。为减少等待参数的时间，FSDP 实现了预取（prefetching）机制：


- **前向预取：**
   在当前层计算时，预取下一层的参数
- **反向预取：**
   在当前层反向计算时，预取上一层的参数
- **后向重叠：**
   将 All-Gather 与计算流重叠


## 5. ZeRO 各阶段对比


| 阶段 | 分片内容 | 每 GPU 内存 | 额外通信 | 适用场景 |
| --- | --- | --- | --- | --- |
| Stage 0 (DDP) | 无分片 | 16Ψ | All-Reduce | 小模型 |
| Stage 1 | 优化器状态 | 4Ψ + 12Ψ/N | All-Gather | 中等模型 |
| Stage 2 | 优化器 + 梯度 | 2Ψ + 14Ψ/N | Reduce-Scatter | 大模型 |
| Stage 3 | 优化器 + 梯度 + 参数 | 16Ψ/N | All-Gather × 2 | 超大模型 |


### 5.1 数值对比示例 (7B 模型)


| 配置 | 单 GPU 内存 | 4 GPU | 8 GPU | 16 GPU |
| --- | --- | --- | --- | --- |
| DDP | 112 GB | 112 GB | 112 GB | 112 GB |
| ZeRO-1 | 112 GB | 49 GB | 38.5 GB | 33.25 GB |
| ZeRO-2 | 112 GB | 38.5 GB | 33.25 GB | 30.625 GB |
| ZeRO-3 | 112 GB | 28 GB | 14 GB | 7 GB |


## 6. FSDP 实战


PyTorch 的 `FullyShardedDataParallel`（FSDP）是 ZeRO Stage 3 的官方实现，原生集成在 PyTorch 中。


### 6.1 基础用法


```
python
import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import (
    CPUOffload, MixedPrecision, ShardingStrategy
)
from torch.distributed.fsdp.wrap import (
    size_based_auto_wrap_policy,
    transformer_auto_wrap_policy,
    always_wrap_policy
)

# 混合精度策略
mp_policy = MixedPrecision(
    param_dtype=torch.bfloat16,      # 参数精度
    reduce_dtype=torch.float32,      # 梯度归约精度 (推荐 FP32)
    buffer_dtype=torch.bfloat16      # 缓冲区精度
)

# 自动包装策略 —— 基于 Transformer 层
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
auto_wrap_policy = transformer_auto_wrap_policy(
    transformer_layer_cls={LlamaDecoderLayer}
)

# 创建 FSDP 模型
model = FSDP(
    MyModel(),
    auto_wrap_policy=auto_wrap_policy,
    mixed_precision=mp_policy,
    sharding_strategy=ShardingStrategy.FULL_SHARD,  # 等价 ZeRO-3
    device_id=torch.cuda.current_device(),
    limit_all_gathers=True,           # 限制同时进行的 All-Gather
    use_orig_params=True,            # 兼容某些 optimizer
)
```


### 6.2 Wrapping 策略详解


Wrapping 策略决定了哪些模块被整体分片（作为一个 FSDP unit）。合适的 wrapping 策略对性能至关重要。


```
三种 Wrapping 策略对比:

1. always_wrap_policy —— 每个模块都包装
   ├── FSDP(Embedding)
   ├── FSDP(Layer0)
   ├── FSDP(Layer1)
   ├── ...
   └── FSDP(Head)
   → 粒度最细, 通信最多, 内存最省

2. transformer_auto_wrap_policy —— 按 Transformer 层包装 (推荐)
   ├── FSDP(Embedding)
   ├── FSDP(Layer0)  ← 整个 DecoderLayer 作为单位
   ├── FSDP(Layer1)
   ├── ...
   └── FSDP(Head)
   → 平衡通信与内存

3. size_based_auto_wrap_policy —— 按参数量阈值包装
   ├── FSDP(module_100M_params)
   ├── FSDP(module_50M_params)
   └── ...
   → 灵活但不直观
```


### 6.3 Sharding 策略


| 策略 | 枚举值 | 描述 | 内存节省 |
| --- | --- | --- | --- |
| FULL_SHARD | ShardingStrategy.FULL_SHARD | 参数+梯度+优化器全分片 | 最大 |
| SHARD_GRAD_OP | ShardingStrategy.SHARD_GRAD_OP | 梯度+优化器分片 (ZeRO-2) | 中等 |
| NO_SHARD | ShardingStrategy.NO_SHARD | 不分片 (等价 DDP) | 无 |
| HYBRID_SHARD | ShardingStrategy.HYBRID_SHARD | 节点内分片 + 节点间复制 | 高, 且通信优化 |


### 6.4 CPU Offloading


```
python
# CPU Offloading — 将参数和梯度卸载到 CPU
model = FSDP(
    model,
    cpu_offload=CPUOffload(offload_params=True),  # 参数卸载到 CPU
    ...
)
# 注意: 开启 CPU Offloading 后, 训练速度会显著下降
# 适用于 GPU 显存严重不足的场景
```


## 7. Activation Checkpointing


Activation Checkpointing（梯度检查点）通过牺牲部分计算来换取内存节省。不保存中间激活值，反向传播时重新计算。


### 7.1 原理


```
标准前向 vs Activation Checkpointing:

标准模式:
  Layer1 ──► Layer2 ──► Layer3 ──► Layer4
    │          │          │          │
    ▼          ▼          ▼          ▼
  保存A1     保存A2     保存A3     保存A4
  (占用内存)

Checkpoint 模式:
  Layer1 ──► Layer2 ──► Layer3 ──► Layer4
    │                                     │
    ▼                                     ▼
  保存A1                               保存A4
  (仅保存关键点)

反向传播时:
  Layer4 ◄── Layer3 ◄── Layer2 ◄── Layer1
    │          │          │          │
  已有A4    重新计算A3  重新计算A2  已有A1
            (从A2重新前向)
```


### 7.2 PyTorch FSDP 中的实现


```
python
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
    apply_activation_checkpointing,
    CheckpointImpl
)

# 方法一: 手动包装
model.encoder.layer[0] = checkpoint_wrapper(
    model.encoder.layer[0],
    checkpoint_impl=CheckpointImpl.NO_REENTRANT
)

# 方法二: 自动应用到所有 Transformer 层
apply_activation_checkpointing(
    model,
    checkpoint_wrapper_fn=checkpoint_wrapper,
    check_fn=lambda submodule: isinstance(submodule, LlamaDecoderLayer)
)
```


> **Warning:** **注意事项：**
>
> - Activation Checkpointing 会增加约 33% 的计算时间（需要重新前向）
> - 选择 checkpoint 的粒度很关键：太细则内存节省少，太粗则重计算多
> - 通常选择每个 Transformer 层作为一个 checkpoint unit


## 8. 最佳实践


### 8.1 选择建议


> **Tip:** 1. **模型能在单卡放下：**
>    使用 DDP，无需分片
> 2. **需要更多 GPU 来降低内存：**
>    使用 ZeRO-2 或 FSDP SHARD_GRAD_OP
> 3. **模型本身很大：**
>    使用 ZeRO-3 / FSDP FULL_SHARD
> 4. **显存极度紧张：**
>    FSDP FULL_SHARD + CPU Offload + Activation Checkpointing


### 8.2 FSDP 配置推荐


```
python
# 推荐的 FSDP 配置 (LLM 训练)
model = FSDP(
    model,
    # Wrapping: 按 Transformer 层包装
    auto_wrap_policy=transformer_auto_wrap_policy,
    # 分片策略: 完全分片
    sharding_strategy=ShardingStrategy.FULL_SHARD,
    # 混合精度: 参数用 BF16, 梯度归约用 FP32
    mixed_precision=MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    ),
    # 设备指定
    device_id=torch.cuda.current_device(),
    # 同步梯度的时机
    sync_module_states=True,       # 跨 rank 同步初始参数
    limit_all_gathers=True,        # 控制内存使用
    use_orig_params=True,          # 兼容 param_groups
    forward_prefetch=True,         # 预取下一层参数
)
```


### 8.3 常见问题


- **OOM on checkpoint save：**
   使用
   `torch.distributed.checkpoint`
   保存分片 checkpoint
- **训练速度慢：**
   检查 wrapping 策略，避免过细粒度
- **梯度不一致：**
   确保
   `reduce_dtype=torch.float32`
- **Checkpoint 兼容性：**
   保存时记录 wrapping 策略，加载时需相同配置

大模型训练工程 - FSDP与ZeRO | 最后更新: 2025年


<!-- Converted from: 02_FSDP与ZeRO.html -->
