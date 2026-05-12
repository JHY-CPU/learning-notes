# 数据并行DDP - 大模型训练工程

*PyTorch DistributedDataParallel 深度解析*


## 数据并行 DDP


PyTorch DistributedDataParallel 深度解析

[88-大模型训练工程](../index.html)
>
[01-分布式训练](./)
>
            01_数据并行DDP

### 目录


1. [数据并行概述](#overview)
2. [DDP 架构原理](#ddp-arch)
3. [All-Reduce 梯度同步](#allreduce)
4. [Bucket 通信机制](#bucket)
5. [梯度累积策略](#grad-accum)
6. [DDP vs DP 对比](#ddp-vs-dp)
7. [最佳实践](#best-practice)


## 1. 数据并行概述


数据并行（Data Parallelism）是大模型训练中最基础也是最常用的并行策略。其核心思想是将训练数据分割到多个 GPU 上，每个 GPU 持有完整的模型副本，各自计算梯度后进行同步。


### 1.1 基本原理


```
┌─────────────────────────────────────────────────┐
│                  数据并行流程                    │
├─────────────────────────────────────────────────┤
│                                                 │
│   输入数据 (N samples)                          │
│       │                                         │
│       ▼                                         │
│   ┌───────┬───────┬───────┬───────┐             │
│   │GPU 0  │GPU 1  │GPU 2  │GPU 3  │             │
│   │N/4    │N/4    │N/4    │N/4    │  分发数据   │
│   │       │       │       │       │             │
│   │前向   │前向   │前向   │前向   │  并行计算   │
│   │↓      │↓      │↓      │↓      │             │
│   │反向   │反向   │反向   │反向   │             │
│   │↓      │↓      │↓      │↓      │             │
│   │∇L_0   │∇L_1   │∇L_2   │∇L_3   │             │
│   └───┬───┴───┬───┴───┬───┴───┬───┘             │
│       │       │       │       │                 │
│       ▼       ▼       ▼       ▼                 │
│   ┌─────────────────────────────┐               │
│   │     All-Reduce 同步梯度      │               │
│   │  ∇L = (∇L_0+∇L_1+∇L_2+∇L_3)/4 │            │
│       ▼       ▼       ▼       ▼                 │
│   │    各 GPU 更新模型参数 (相同) │               │
│   └─────────────────────────────┘               │
└─────────────────────────────────────────────────┘
```


### 1.2 数据并行的分类


| 类型 | 同步方式 | 通信模式 | 适用场景 |
| --- | --- | --- | --- |
| 同步数据并行 | All-Reduce 同步梯度 | 集中式 | PyTorch DDP |
| 异步数据并行 | 参数服务器异步更新 | 参数服务器 | 大规模异构集群 |
| 完全分片数据并行 | 分片梯度/参数/优化器 | All-Reduce/Gather | FSDP / ZeRO |


> **Note:** **核心公式：**
> 设模型参数为 W，损失函数为 L，第 i 个 GPU 上的数据为 D_i，则数据并行的梯度更新为：
>
> $$
> W_{t+1} = W_t - lr · (1/N) · Σ_{i=0}^{N-1} ∇L(W_t, D_i)
> $$
>
> 其中 N 为 GPU 数量，lr 为学习率。


## 2. DDP 架构原理


PyTorch 的 `DistributedDataParallel`（DDP）通过多进程并行和高效的梯度同步机制，实现了比 `DataParallel` 更优越的性能。


### 2.1 进程模型


- **多进程架构：**
   每个 GPU 对应一个独立进程（rank），通过
   `torch.multiprocessing`
   或 SLURM 启动
- **进程间通信：**
   使用 NCCL 后端进行 GPU-to-GPU 通信
- **主进程协调：**
   DDP 不需要主进程，所有进程对等（peer-to-peer）


```
python
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler

def setup(rank, world_size):
    # 初始化进程组
    dist.init_process_group(
        backend="nccl",           # GPU 通信使用 NCCL
        init_method="env://",      # 通过环境变量初始化
        world_size=world_size,
        rank=rank
    )
    torch.cuda.set_device(rank)

def cleanup():
    dist.destroy_process_group()

def train(rank, world_size):
    setup(rank, world_size)

    # 创建模型并包装为 DDP
    model = MyModel().to(rank)
    ddp_model = DDP(model, device_ids=[rank])

    # 使用 DistributedSampler 分发数据
    dataset = MyDataset()
    sampler = DistributedSampler(dataset, num_replicas=world_size, rank=rank)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, sampler=sampler
    )

    for epoch in range(num_epochs):
        sampler.set_epoch(epoch)  # 确保每个 epoch 的 shuffle 不同
for batch in dataloader:
            output = ddp_model(batch)
            loss = criterion(output)
            loss.backward()   # DDP 自动同步梯度
            optimizer.step()
            optimizer.zero_grad()

    cleanup()
```


### 2.2 Reducer 与参数注册


DDP 在初始化时会创建一个 `Reducer` 对象，负责：


1. **参数注册：**
   遍历模型的所有参数，按梯度的 dtype 和 device 进行分组
2. **桶分配：**
   将参数梯度组织成固定大小的桶（bucket），桶大小由
   `bucket_cap_mb`
   控制（默认 25MB）
3. **钩子注册：**
   为每个参数注册梯度累积钩子（
   `AccumulateGrad`
   hook）
4. **梯度就绪触发：**
   当一个桶内所有参数的梯度都就绪时，触发 All-Reduce


### 2.3 前向传播与反向传播


```
┌──────────── 前向传播 ────────────┐
│                                  │
│  Rank 0: input_0 ──► model ──► out_0
│  Rank 1: input_1 ──► model ──► out_1
│  ...                             │
│  (各自独立计算, 无需通信)          │
└──────────────────────────────────┘
                │
                ▼
┌──────────── 反向传播 ────────────┐
│                                  │
│  backward() 触发梯度计算          │
│       │                          │
│       ▼                          │
│  梯度就绪 (按桶)                  │
│       │                          │
│       ▼                          │
│  Reducer 检测桶就绪               │
│       │                          │
│       ▼                          │
│  All-Reduce 同步该桶的梯度        │
│       │                          │
│       ▼                          │
│  梯度除以 world_size (平均)       │
└──────────────────────────────────┘
```


## 3. All-Reduce 梯度同步


All-Reduce 是 DDP 中最核心的通信原语，它将所有进程的数据进行归约操作（如求和/求均值），并将结果广播到所有进程。


### 3.1 Ring All-Reduce


Ring All-Reduce 将 N 个 GPU 排列成环状，通过 2(N-1) 步完成同步：


> **Note:** **算法复杂度分析：**
>
> - 假设数据总量为 K 字节，进程数为 N
> - 每步传输量：K/N 字节
> - 总传输量：2 × (N-1) × K/N ≈ 2K 字节
> - **通信量与 GPU 数量无关（渐近最优）**


```
Ring All-Reduce (4 GPU, 4 数据块)

Phase 1: Scatter-Reduce (3 步)
┌─────┐   ┌─────┐   ┌─────┐   ┌─────┐
│GPU 0│   │GPU 1│   │GPU 2│   │GPU 3│
│ a   │   │ b   │   │ c   │   │ d   │
└──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
   │a→d      │b→a      │c→b      │d→c
┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
│a+d  │   │b+a  │   │c+b  │   │d+c  │
└──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
   │→c      │→d      │→a      │→b
┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
│a+d+c │   │b+a+d │   │c+b+a │   │d+c+b│
└──┬──┘   └──┬──┘   └──┬──┘   └──┬──┘
   │→b      │→c      │→d      │→a
┌──▼──┐   ┌──▼──┐   ┌──▼──┐   ┌──▼──┐
│a+b+c+d│  │b+a+d+c│  │c+b+a+d│  │d+c+b+a│

Phase 2: All-Gather (3 步)
→ 所有 GPU 最终获得完整结果 Σ
```


### 3.2 Hierarchical All-Reduce


当 GPU 数量较多或跨节点通信时，层级式 All-Reduce 更高效：


1. **节点内 Reduce：**
   每个节点内将数据归约到一个 leader GPU
2. **节点间 Reduce：**
   各节点 leader GPU 执行 Ring All-Reduce
3. **节点内 Broadcast：**
   将结果广播回节点内所有 GPU


$$
通信复杂度: T = T_intra + T_inter = (K/n) + (K·n/N) = K·(1/n + n/N)
$$


其中 N 为总 GPU 数，n 为每节点 GPU 数，K 为数据量。最优节点数 n = √N。


### 3.3 梯度同步的时序


> **Warning:** **关键优化：**
> DDP 将梯度计算与 All-Reduce 通信重叠执行。当一个桶内所有参数的梯度就绪后，立即触发该桶的 All-Reduce，而无需等待所有梯度计算完成。这通过 PyTorch 的 autograd hook 机制实现。


## 4. Bucket 通信机制


DDP 使用桶（bucket）将多个小梯度打包成一个大通信操作，以减少通信次数、提高带宽利用率。


### 4.1 桶的工作原理


```
python
# DDP 桶配置参数
ddp_model = DDP(
    model,
    device_ids=[rank],
    bucket_cap_mb=25,           # 桶容量上限 (默认 25MB)
    gradient_as_bucket_view=True, # 直接使用梯度作为桶视图, 减少拷贝
    static_graph=False,          # 是否使用静态图优化
    find_unused_parameters=False # 是否查找未使用参数
)
```


### 4.2 反向桶排序（Reverse Bucketing）


DDP 按照**参数被使用的反向顺序**组织桶，这样最后一次使用的参数会被放入第一个桶，最先触发 All-Reduce：


- 前向传播时记录参数的使用顺序
- 反向传播时，最后一个被使用的参数（即梯度最先计算完的）放在第一个桶
- 这样可以在反向传播的早期就开始通信，最大化通信-计算重叠


### 4.3 桶大小调优


| 桶大小 | 优势 | 劣势 | 适用场景 |
| --- | --- | --- | --- |
| 大桶 (>50MB) | 通信效率高, 次数少 | 等待时间长, 重叠差 | 大模型, 高带宽网络 |
| 中桶 (25-50MB) | 平衡通信与重叠 | 中等 | 通用场景 (默认) |
| 小桶 (<25MB) | 重叠好, 延迟低 | 通信次数多, 开销大 | 小模型, 低带宽 |


> **Note:** **Static Graph 优化：**
> 当模型的计算图在每次迭代中保持不变时（即不使用 if/for 等动态控制流），设置
> `static_graph=True`
> 可以让 DDP 在第一次迭代后缓存梯度同步的顺序，后续迭代直接复用，消除同步开销。


## 5. 梯度累积策略


梯度累积（Gradient Accumulation）是一种在不增加显存占用的情况下，等效增大 batch size 的技术。在 DDP 环境下需要特别注意同步的时机。


### 5.1 基本原理


```
梯度累积流程 (accum_steps = 4):

  Step 1: 前向 → 反向 (累积梯度)
  Step 2: 前向 → 反向 (累积梯度)
  Step 3: 前向 → 反向 (累积梯度)
  Step 4: 前向 → 反向 (累积梯度)
                            │
                            ▼
  All-Reduce 同步累积后的梯度
                            │
                            ▼
  optimizer.step() 更新参数
                            │
                            ▼
  optimizer.zero_grad() 清零
```


### 5.2 DDP 中的梯度累积实现


```
python
model = DDP(model, device_ids=[rank])
accum_steps = 4
for step, batch in enumerate(dataloader):
    output = model(batch)
    loss = criterion(output) / accum_steps  # 损失缩放
    loss.backward()

    if (step + 1) % accum_steps == 0:
        # 同步梯度并更新
        optimizer.step()
        optimizer.zero_grad()
```


### 5.3 no_sync 上下文管理器


在累积步骤中，前 (accum_steps - 1) 步不需要梯度同步，使用 `no_sync()` 禁用 All-Reduce：


```
python
for step, batch in enumerate(dataloader):
    if (step + 1) % accum_steps != 0:
        with ddp_model.no_sync():  # 禁用梯度同步
            output = ddp_model(batch)
            loss = criterion(output) / accum_steps
            loss.backward()
    else:
        # 最后一步正常同步
        output = ddp_model(batch)
        loss = criterion(output) / accum_steps
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()
```


> **Tip:** **性能收益：**
> 使用
> `no_sync()`
> 后，通信次数从每次迭代都同步减少到每 accum_steps 次同步一次，通信开销降低为原来的 1/accum_steps。


## 6. DDP vs DP 对比


DataParallel（DP）和 DistributedDataParallel（DDP）是 PyTorch 中两种数据并行实现，但 DDP 在几乎所有方面都更优。


| 特性 | DataParallel (DP) | DistributedDataParallel (DDP) |
| --- | --- | --- |
| 进程模型 | 单进程多线程 | 多进程，每个 GPU 一个进程 |
| GIL 影响 | 受 GIL 限制 | 无 GIL 限制 |
| 通信模式 | 主 GPU 聚合再分发 | 点对点 Ring All-Reduce |
| 通信量 | O(N) per step | O(2K) total (与 N 无关) |
| 显存均衡 | 主 GPU 显存占用高 | 各 GPU 均衡 |
| 扩展性 | 仅单机 | 支持多机多卡 |
| 数据加载 | 自动拆分 | 需手动 DistributedSampler |
| 推荐度 | 不推荐 | 强烈推荐 |


### DP 的主 GPU 瓶颈


```
DataParallel 通信模式:

  GPU 0 (主)     GPU 1     GPU 2     GPU 3
    │             │         │         │
    │  前向+反向  │ 前向+反向│ 前向+反向│ 前向+反向
    │     │       │    │    │    │    │    │
    │     ▼       │    ▼    │    ▼    │    ▼
    │  梯度       │  梯度    │  梯度    │  梯度
    │     │       │    │    │    │    │    │
    │◄────┘       │◄───┘    │◄───┘    │◄───┘
    │  收集所有梯度 (Scatter-Gather)  ← 瓶颈!
    │     │
    │     ▼
    │  平均梯度 → 更新参数
    │     │
    │────►│──────►│──────►│
    │  广播新参数到所有 GPU ← 瓶颈!
    ▼     ▼       ▼       ▼
```


> **Warning:** **为什么 DP 主 GPU 显存高？**
> DP 的主 GPU (GPU 0) 需要将所有 GPU 的梯度汇总到自己这里，因此需要分配额外的显存来存储所有 GPU 的梯度副本。当 GPU 数量较多时，主 GPU 容易 OOM。


## 7. 最佳实践


### 7.1 启动方式


```
bash
# 方式一: torchrun (推荐)
torchrun --nproc_per_node=8 --nnodes=2 --node_rank=0 \
         --master_addr="192.168.1.1" --master_port=29500 \
         train.py

# 方式二: torch.multiprocessing
torch.multiprocessing.spawn(train, args=(world_size,), nprocs=world_size)

# 方式三: SLURM
# torchrun 会自动从 SLURM 环境变量读取配置
torchrun --rdzv_backend=c10d --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT train.py
```


### 7.2 常见问题排查


- **进程不同步：**
   确保所有进程调用
   `dist.barrier()`
   或使用 DDP 的梯度同步
- **显存不均衡：**
   检查是否所有 GPU 使用相同的 batch size 和模型
- **通信超时：**
   增大
   `NCCL_TIMEOUT`
   环境变量
- **未使用参数报错：**
   设置
   `find_unused_parameters=True`
   ，但会降低性能


### 7.3 性能调优清单


> **Tip:** 1. 使用
>    `torchrun`
>    而非
>    `mp.spawn`
> 2. 设置合适的
>    `bucket_cap_mb`
>    （默认 25MB）
> 3. 开启
>    `gradient_as_bucket_view=True`
> 4. 静态图场景开启
>    `static_graph=True`
> 5. 梯度累积使用
>    `no_sync()`
> 6. 确保
>    `DistributedSampler.set_epoch()`
>    每个 epoch 调用
> 7. 使用 NCCL 后端而非 Gloo（GPU 训练）
> 8. 设置
>    `OMP_NUM_THREADS`
>    避免线程竞争

大模型训练工程 - 数据并行DDP | 最后更新: 2025年


<!-- Converted from: 01_数据并行DDP.html -->
