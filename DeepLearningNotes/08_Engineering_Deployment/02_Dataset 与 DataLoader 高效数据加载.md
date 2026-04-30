# 02_Dataset 与 DataLoader 高效数据加载

## 核心概念

- **Dataset 基类**：PyTorch 中所有数据集都必须继承 `torch.utils.data.Dataset` 并实现 `__len__` 和 `__getitem__` 两个方法。前者返回数据集大小，后者通过索引返回单个样本（特征 + 标签）。
- **DataLoader 封装器**：接收 Dataset 实例，自动实现批量采样 (batching)、打乱 (shuffling)、多进程加载 (num_workers) 和内存锁页 (pin_memory) 等功能。是训练循环中的数据供给入口。
- **Sampler 策略**：DataLoader 内部通过 Sampler 控制数据访问顺序，包括顺序采样 (SequentialSampler)、随机采样 (RandomSampler)、加权采样 (WeightedRandomSampler) 和分布式采样 (DistributedSampler) 等。
- **num_workers 与多进程加载**：设置 `num_workers > 0` 时，DataLoader 使用多个子进程预取数据，将数据加载与 GPU 计算流水线化。推荐的典型值为 CPU 核心数或 GPU 数量的 2-4 倍。
- **pin_memory 与锁页内存**：当 `pin_memory=True` 时，数据加载到 GPU 前的 CPU 内存会被锁定（不可换页），从而加速 CPU 到 GPU 的数据传输 (`D2H` 与 `H2D`)。搭配 `non_blocking=True` 可实现异步传输。
- **Prefetch Factor 与预取**：设置 `prefetch_factor=2` 表示每个 worker 预加载 2 个批次的样本，通过隐藏 I/O 延迟最大化 GPU 利用率。在数据吞吐量成为瓶颈的大规模训练中至关重要。

## 数学推导

DataLoader 的批处理流程可以抽象为以下数据流：

$$
\text{Disk/Network} \xrightarrow{\text{num\_workers}} \text{RAM (pin\_memory)} \xrightarrow{\text{async copy}} \text{GPU VRAM} \xrightarrow{\text{model}} \text{loss}
$$

流水线并行度 = 数据加载时间与单步训练时间的比值：

$$
\text{GPU 空闲率} = \frac{\max(0, T_{\text{load}} - T_{\text{train}})}{\max(T_{\text{load}}, T_{\text{train}})}
$$

当 $T_{\text{load}} < T_{\text{train}}$ 时，数据加载完全被计算掩盖，GPU 利用率接近 100%；反之则出现数据饥饿，需增加 num_workers 或使用更高效的数据格式（如 WebDataset、Mosaic 格式）。

在分布式训练中，每个 rank 的数据分片通过 DistributedSampler 实现：

$$
\text{rank\_r 的样本索引} = \{i \in [0, N) \mid i \bmod \text{world\_size} = r\}
$$

确保每个 GPU 处理不重叠的唯一数据子集。

## 直观理解

- DataLoader 就像是一个"智能传送带"：Dataset 定义了货架上的所有货物，DataLoader 决定拿货的顺序和每次拿多少，而 multi-process workers 是多个搬运工同时从仓库搬货。
- **最佳实践**：在数据加载成为瓶颈时，优先增加 `num_workers`（通常设置为 CPU 核心数，但避免过多导致 I/O 饱和），其次考虑 `pin_memory=True`。
- **常见陷阱**：`num_workers` 并非越大越好——过多的 workers 会导致磁盘 I/O 争抢、内存暴涨和进程间通信开销，反而降低性能。建议从 4 开始逐步尝试。
- **经验法则**：如果训练时 GPU 利用率低于 70%，先检查 DataLoader 配置；在 Windows 上建议 `num_workers=0`，因为多进程序列化开销远大于 Unix 系统。

## 代码示例

```python
import torch
from torch.utils.data import Dataset, DataLoader, DistributedSampler
import numpy as np
import time

# 1. 自定义 Dataset
class MyDataset(Dataset):
    def __init__(self, size=10000, transform=None):
        self.data = np.random.randn(size, 3, 224, 224).astype(np.float32)
        self.labels = np.random.randint(0, 10, size=(size,))
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.from_numpy(self.data[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        if self.transform:
            x = self.transform(x)
        return x, y

# 2. 基础 DataLoader 配置
dataset = MyDataset(size=50000)
dataloader = DataLoader(
    dataset,
    batch_size=64,
    shuffle=True,
    num_workers=4,          # 4 个子进程加载
    pin_memory=True,        # 使用锁页内存加速 GPU 传输
    prefetch_factor=2,      # 每个 worker 预取 2 个批次
    persistent_workers=True # epoch 间不销毁 worker 进程
)

# 3. 训练循环中的数据加载
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
for epoch in range(3):
    epoch_start = time.time()
    for batch_idx, (inputs, targets) in enumerate(dataloader):
        # 异步传输到 GPU
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        # ... forward/backward 省略 ...
        if batch_idx >= 50:  # 仅演示前 50 个 batch
            break
    elapsed = time.time() - epoch_start
    print(f"Epoch {epoch} took {elapsed:.2f}s")

# 4. 分布式训练中的 Sampler
if torch.distributed.is_available():
    sampler = DistributedSampler(
        dataset,
        num_replicas=torch.distributed.get_world_size(),
        rank=torch.distributed.get_rank(),
        shuffle=True
    )
    # 每个 epoch 需要调用 set_epoch 以保证打乱不同
    # sampler.set_epoch(epoch)
    dataloader_dist = DataLoader(dataset, batch_size=32, sampler=sampler)
    # 此时每个 GPU 只会处理 50000 / world_size 个样本

# 5. 性能测试工具函数
def benchmark_dataloader(dataset, num_workers_list=[0, 2, 4, 8]):
    for nw in num_workers_list:
        loader = DataLoader(dataset, batch_size=64, num_workers=nw, pin_memory=True)
        start = time.time()
        for i, (x, y) in enumerate(loader):
            if i >= 100:
                break
        print(f"num_workers={nw}: {time.time() - start:.3f}s for 100 batches")

# benchmark_dataloader(dataset)
```

## 深度学习关联

- **大规模训练的数据管道**：在现代 MLOps 实践中，DataLoader 通常与 WebDataset、FFRecord 或 Mosaic 格式配合使用，将海量小文件合并为 tar/二进制分片，大幅减少文件系统元数据开销。DataLoader 的 `num_workers` 与存储后端（本地 SSD / NFS / 对象存储）的 IOPS 能力需要匹配调优。
- **实验跟踪中的重现性**：在 MLflow/W&B 等实验管理平台中记录训练时，需固定 DataLoader 的随机种子并通过 Sampler 记录数据划分方式，确保实验结果可复现。`shuffle=True` 且不固定 seed 时，不同 run 的数据顺序不同会导致训练结果差异。
- **推理服务中的数据预处理**：在生产级推理服务（Triton/TorchServe）中，DataLoader 的思想被复用为模型的 Preprocess/Postprocess 阶段，通过自定义 Python Backend 或模型集成实现请求级别的高吞吐数据预处理流水线。
