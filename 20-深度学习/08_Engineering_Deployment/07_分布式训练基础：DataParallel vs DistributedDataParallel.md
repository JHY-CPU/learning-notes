# 07_分布式训练基础：DataParallel vs DistributedDataParallel

## 核心概念

- **DataParallel (DP)**：PyTorch 的简易数据并行方案。在前向传播时，将输入 batch 均匀切分到各 GPU，各 GPU 持有完整模型副本，主 GPU（rank 0）负责梯度汇总和参数更新。缺点是单进程多线程模型受限于 GIL，且主 GPU 的通信和计算负载远高于其他 GPU。
- **DistributedDataParallel (DDP)**：PyTorch 的推荐分布式训练方案。每个 GPU 由一个独立进程控制，通过 `torch.distributed` 通信后端（NCCL/MPI/GLOO）进行梯度同步。所有 GPU 在训练中角色对称，无负载热点。
- **NCCL 通信后端**：NVIDIA 提供的集体通信库，实现了 all-reduce、broadcast、all-gather 等操作的 GPU 间高效通信。在 GPU 集群训练中 NVIDIA NCCL 是标准选择。
- **Ring All-Reduce 算法**：DDP 梯度同步的核心算法。将 N 个 GPU 组成逻辑环，每个 GPU 只与相邻 GPU 通信，分 scatter-reduce 和 all-gather 两个阶段完成全局梯度归约。每个 GPU 的通信量为 $2(N-1)/N \cdot d$，当 N 很大时趋近于常数 $2d$，与 GPU 数量无关。
- **梯度同步与计算重叠**：DDP 在每个 backward pass 中注册了 gradient hook，当每个参数的梯度计算完成后立即启动异步 all-reduce，使通信与后续层的 backward 计算重叠，极大降低同步开销。
- **world_size / rank / local_rank**：分布式训练的基本术语。world_size 是总进程数（总 GPU 数），rank 是全局进程编号（0 到 world_size-1），local_rank 是节点内部的 GPU 编号。通过 `torch.distributed.init_process_group` 初始化。

## 数学推导

Ring All-Reduce 分为两个阶段。假设有 N 个 GPU，每个 GPU 上的梯度向量为 $g_i \in \mathbb{R}^d$，目标计算 $\bar{g} = \frac{1}{N}\sum_{i=1}^N g_i$。

**阶段 1: Scatter-Reduce**
将梯度向量切分为 N 个块 $g_i = [g_i^1, g_i^2, ..., g_i^N]$。经过 N-1 轮环形传递和累加后，每个 GPU i 持有第 i 个块的全局和：

$$
\text{GPU}_i \text{ 持有: } \sum_{j=1}^N g_j^i, \quad \forall i
$$

**阶段 2: All-Gather**
再做 N-1 轮环形传递，将每个 GPU 上的部分和广播到所有 GPU：

$$
\text{GPU}_i \text{ 最终持有: } \bar{g} = \frac{1}{N}\left[\sum_{j=1}^N g_j^1, \sum_{j=1}^N g_j^2, ..., \sum_{j=1}^N g_j^N\right]
$$

总通信时间对比：
- DP 的参数服务器模式：$T_{\text{DP}} = 2(N-1) \cdot \frac{d}{B}$
- DDP 的 Ring All-Reduce：$T_{\text{DDP}} = 2(N-1) \cdot \frac{d}{N \cdot B} = 2 \cdot \frac{N-1}{N} \cdot \frac{d}{B}$

其中 $B$ 是 GPU 间带宽。当 N 很大时，DDP 的通信时间趋近于常数 $\frac{2d}{B}$，而 DP 线性增长。

## 直观理解

- **DP vs DDP**：DP 就像是一个经理（主 GPU）分配任务给多个员工（GPU），所有员工把结果汇报给经理，经理一个人做总结——经理是瓶颈。DDP 则是一个 self-organized 的团队，每个人都和邻居交换信息，所有人同步完成汇总，没有单点瓶颈。
- **最佳实践**：始终使用 DDP 而非 DP。PyTorch 官方已明确 DDP 是推荐方案，DP 仅用于兼容性场景（如 Windows 缺少 NCCL 支持时）。
- **常见陷阱**：使用 DDP 时，batch_size 的含义是"每个 GPU 的 batch_size"，全局 batch_size = per_gpu_batch_size * world_size。学习率通常按全局 batch_size 线性缩放。
- **经验法则**：`torchrun --nproc_per_node=N train.py` 是最简单的 DDP 启动方式；`find_unused_parameters=True` 在模型中有条件分支时需启用。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader, Dataset, DistributedSampler

# ========== DDP 单机多卡示例 ==========

# 1. 初始化进程组
def setup(rank, world_size):
    dist.init_process_group(
        backend="nccl",  # 使用 NCCL 后端
        init_method="tcp://localhost:23456",
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()

# 2. 简单模型
class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 500),
            nn.ReLU(),
            nn.Linear(500, 10)
        )

    def forward(self, x):
        return self.net(x)

# 3. 每个 rank 的训练函数
def train_worker(rank, world_size):
    setup(rank, world_size)
    device = torch.device(f"cuda:{rank}")

    # 模型必须在初始化 DDP 前移动到正确的 GPU
    model = ToyModel().to(device)
    ddp_model = DDP(model, device_ids=[rank])

    # 每个 GPU 独立的 DataLoader (通过 DistributedSampler 分片)
    dataset = torch.randn(1000, 100), torch.randint(0, 10, (1000,))
    sampler = DistributedSampler(dataset[0], num_replicas=world_size, rank=rank)
    loader = DataLoader(
        list(zip(dataset[0], dataset[1])),
        batch_size=32,
        sampler=sampler
    )

    optimizer = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
    criterion = nn.CrossEntropyLoss()

    for epoch in range(3):
        sampler.set_epoch(epoch)  # 确保每个 epoch 数据打乱不同
        for data, target in loader:
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

        # 只在 rank 0 打印
        if rank == 0:
            print(f"Epoch {epoch} done")

    cleanup()

# 4. 启动（使用 torchrun 或手动 spawn）
if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    print(f"检测到 {world_size} 个 GPU")
    # mp.spawn(train_worker, args=(world_size,), nprocs=world_size)

# ========== 启动命令（注释） ==========
# torchrun --nproc_per_node=4 --master_port=29500 train_ddp.py
# 或在 slurm 集群中：
# srun --ntasks=8 --gres=gpu:8 torchrun --nproc_per_node=8 train_ddp.py

# ========== DP 对比（不推荐） ==========
class DPDemo:
    """DP 的使用方式，仅用于对比"""
    @staticmethod
    def run():
        model = ToyModel()
        if torch.cuda.device_count() > 1:
            dp_model = nn.DataParallel(model)
            print(f"使用 DataParallel，设备数: {torch.cuda.device_count()}")
        # dp_model 像一个普通模型一样使用
        # x = torch.randn(128, 100).cuda()
        # output = dp_model(x)

# ========== DDP 关键注意事项 ==========
# 1. batch_size 是每个 GPU 的，全局 = per_gpu * world_size
# 2. 学习率一般按 全局_batch / 参考_batch 线性缩放
# 3. 使用 SyncBatchNorm 代替 BatchNorm 提升多卡精度
# 4. 保存模型时只在 rank 0 保存
#    if rank == 0:
#        torch.save(ddp_model.module.state_dict(), "model.pt")
```

## 深度学习关联

- **超大规模训练中的混合并行**：当模型大到单卡无法容纳时，DDP 需要与模型并行（Tensor Parallelism、Pipeline Parallelism）结合。现代大模型训练通常采用 3D 并行（DP + TP + PP），而 DDP 是其中数据并行维度的核心实现。
- **MLOps 中的分布式训练编排**：在 Kubernetes + Kubeflow 环境中，DDP 作业通常通过 `torchrun` 或 `elastic training` 启动，配合 Volcano/Batch 调度器实现 GPU 资源的动态分配。每次实验的分布式配置（world_size、通信后端、节点数）需在 MLflow 中记录以分析扩展效率。
- **弹性训练 (Elastic Training)**：PyTorch 的 `torch.distributed.elastic` 支持训练过程中动态增删节点，提升了集群的容错性和资源利用率。在 Spot Instance 环境中，弹性训练可以在实例被回收时自动迁移到新节点继续训练。
