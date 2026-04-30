# 03_自定义 Dataset 类实现与增强策略

## 核心概念

- **`__getitem__` 设计模式**：这是自定义 Dataset 的核心方法。每次调用返回一个样本（通常为 (input, label) 元组）。设计时需确保索引到样本的映射是 O(1) 操作，避免在 `__getitem__` 中做耗时操作。
- **数据增强 (Data Augmentation)**：在训练时对输入数据施加随机的、语义保持的变换，增加数据多样性，提高模型泛化能力。PyTorch 中通常通过 `torchvision.transforms` 或 `albumentations` 实现。
- **Lazy Loading vs Eager Loading**：Lazy Loading 只在 `__getitem__` 调用时从磁盘读取数据，节省内存但 I/O 波动大；Eager Loading 在初始化时全部加载到内存，速度快但受限于内存大小。实践中多采用前者 + 缓存策略。
- **缓存机制 (Caching)**：对重复读取的样本（如增强前的原始数据）使用 LRU 缓存（`functools.lru_cache`），避免重复磁盘 I/O。对于小数据集可考虑全量缓存到内存或 SSD。
- **多模态 Dataset**：当样本包含图像、文本、数值等多种模态时，`__getitem__` 需要返回结构化字典（如 `{"image": ..., "text": ..., "label": ...}`），并在 collate_fn 中处理不同模态的批量化逻辑。
- **增量式数据加载**：对于超大规模数据集（TB 级），使用 `IterableDataset` 替代 `Dataset`，以流式方式读取数据，不要求实现 `__len__` 和随机索引访问。

## 数学推导

数据增强可以被视为在训练分布上施加一个随机变换算子 $\mathcal{T}_\theta$，其中 $\theta \sim P_\Theta$ 是随机参数：

$$
\tilde{x} = \mathcal{T}_\theta(x), \quad \tilde{y} = y
$$

训练目标变为在增强后的分布上最小化期望风险：

$$
\min_{\mathbf{w}} \mathbb{E}_{(x,y)\sim P_{\text{data}}} \left[ \mathbb{E}_{\theta\sim P_\Theta} \left[ \mathcal{L}(f_{\mathbf{w}}(\mathcal{T}_\theta(x)), y) \right] \right]
$$

这等价于隐式地扩大了训练数据分布的支持集，起到一种数据驱动的正则化作用。以随机裁剪 (RandomResizedCrop) 为例：

$$
\mathcal{T}_{\theta}(x) = \text{Crop}(x, \theta), \quad \theta \sim \text{Uniform}([0, 1]^4)
$$

其中 $\theta$ 控制裁剪框的坐标和大小，这迫使模型学习目标的尺度不变性和平移不变性。

对于 MixUp 增强，其数学形式为：

$$
\tilde{x} = \lambda x_i + (1-\lambda) x_j, \quad \tilde{y} = \lambda y_i + (1-\lambda) y_j, \quad \lambda \sim \text{Beta}(\alpha, \alpha)
$$

MixUp 等价于在样本对之间进行线性插值，使模型学习到更平滑的决策边界。

## 直观理解

- **自定义 Dataset = 数据仓库的索引系统**：`__getitem__` 就是一张"地图"，告诉 PyTorch 如何根据编号找到对应的原始文件和标签；`transforms` 则是"加工流水线"，在取出原始材料后进行切割、上色、扭曲等处理。
- **最佳实践**：数据增强应该尽可能在 CPU 上异步完成，和 GPU 训练流水线重叠。使用 `albumentations` 库时，先在 `__init__` 中创建变换流水线，在 `__getitem__` 中调用，确保每次调用的随机性。
- **常见陷阱**：不要在 `__getitem__` 中做耗时同步 I/O（如从远程 S3 读取），这会导致 workers 长时间阻塞。应该先用预处理脚本将数据下载到本地存储。
- **经验法则**：在 `collate_fn` 中处理变长数据（如文本序列、不同尺寸的图像）的 padding 逻辑；对于需要固定 batch 内所有样本尺寸的模型，在 collate_fn 中进行统一 resizing 比在 `__getitem__` 中更高效。

## 代码示例

```python
import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
import numpy as np
from PIL import Image
import os
from functools import lru_cache

# 1. 带增强的图像分类 Dataset
class ImageClassificationDataset(Dataset):
    def __init__(self, image_paths, labels, transform=None):
        self.image_paths = image_paths
        self.labels = labels
        self.transform = transform or self._default_transform()

    def _default_transform(self):
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx]).convert("RGB")
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)
        return image, label


# 2. 使用缓存避免重复读取原始文件
class CachedImageDataset(Dataset):
    def __init__(self, paths, labels, transform=None):
        self.paths = paths
        self.labels = labels
        self.transform = transform

    @lru_cache(maxsize=512)
    def _load_image(self, path):
        return Image.open(path).convert("RGB")

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = self._load_image(self.paths[idx])  # 缓存原始图像
        label = self.labels[idx]
        if self.transform:
            image = self.transform(image)  # 增强每次不同，不缓存
        return image, label


# 3. 多模态 Dataset（返回字典）
class MultimodalDataset(Dataset):
    def __init__(self, image_paths, texts, labels, image_transform=None):
        self.image_paths = image_paths
        self.texts = texts
        self.labels = labels
        self.image_transform = image_transform

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return {
            "image": self._load_and_transform(self.image_paths[idx]),
            "text": self.texts[idx],
            "label": self.labels[idx]
        }

    def _load_and_transform(self, path):
        img = Image.open(path).convert("RGB")
        if self.image_transform:
            img = self.image_transform(img)
        return img


# 4. 自定义 collate_fn 处理变长数据
def multimodal_collate(batch):
    images = torch.stack([item["image"] for item in batch])
    texts = [item["text"] for item in batch]  # 保留为列表
    labels = torch.tensor([item["label"] for item in batch])
    return images, texts, labels


# 5. IterableDataset 用于流式数据
class StreamingDataset(IterableDataset):
    def __init__(self, file_list):
        self.file_list = file_list

    def __iter__(self):
        for fpath in self.file_list:
            data = np.load(fpath)
            for i in range(len(data)):
                yield data[i], 0

# 使用示例
# paths = [f"/data/img_{i}.jpg" for i in range(1000)]
# labels = np.random.randint(0, 10, 1000).tolist()
# dataset = ImageClassificationDataset(paths, labels)
# loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
# for images, labels in loader:
#     ...
```

## 深度学习关联

- **训练-测试分布一致性**：在 MLOps 流水线中，自定义 Dataset 的增强策略（train transforms）与验证/推理阶段的预处理（val/test transforms）必须严格区分。常见错误是在验证集上误用随机增强，导致评估指标不准确。建议使用 YAML 配置文件区分 train/val 的 transform 链。
- **数据版本管理 (DVC/Delta Lake)**：生产环境中，Dataset 的 `__init__` 通常接收数据集的版本哈希或 commit ID，通过 DVC (Data Version Control) 或 Hugging Face Datasets 进行版本追踪。每次训练实验可在 MLflow 中记录使用的数据集版本与 transform 参数，确保实验的可复现性。
- **特征存储 (Feature Store)**：在大型推荐系统或搜索排名模型中，Dataset 不再从原始文件读取，而是从特征存储服务（如 Feast、Tecton）批量拉取预计算特征。此时 `__getitem__` 需要实现特征拼接逻辑，将在线特征与离线批特征对齐，这是工程化部署中的典型挑战。
