# 21_Fast R-CNN 与 ROI Pooling 实现

## 核心概念

- **RoI Pooling 的核心思想**：将所有不同尺寸的候选区域（Region of Interest）映射到卷积特征图上，并池化为固定尺寸（如 $7\times7$）的特征图，使全连接层可以处理任意尺寸的输入。
- **单次特征提取**：与R-CNN（每个候选框独立通过CNN）不同，Fast R-CNN仅对整张图像做一次CNN前向传播，所有候选框共享特征图，推理速度提升约200倍。
- **多任务损失**：将分类（Softmax）和边界框回归（Smooth L1）合并到同一个网络中联合训练，无需分阶段训练SVM和回归器。
- **RoI Pooling 的量化问题**：将浮点数坐标量化为整数时会产生空间偏差，影响小目标的定位精度。后续RoI Align（Mask R-CNN）通过双线性插值解决此问题。
- **Truncated SVD加速全连接层**：对全连接层权重进行SVD分解，将 $u \times v$ 的矩阵分解为 $u \times t$ 和 $t \times v$ 两个矩阵（$t \ll \min(u,v)$），在检测阶段加速全连接层推理。

## 数学推导

**RoI Pooling 前向传播步骤：**

假设卷积特征图大小为 $H \times W$，RoI坐标为 $(x_1, y_1, x_2, y_2)$，目标输出尺寸为 $h_{out} \times w_{out}$：

- 将RoI映射到特征图尺度（除以特征图相对于输入图像的下采样倍数 $s$）：
   $$
   x_1' = \lfloor x_1 / s \rfloor, \quad y_1' = \lfloor y_1 / s \rfloor, \quad x_2' = \lceil x_2 / s \rceil, \quad y_2' = \lceil y_2 / s \rceil
   $$

- 将RoI区域划分为 $h_{out} \times w_{out}$ 个网格，每个网格尺寸：
   $$
   grid\_h = (y_2' - y_1' + 1) / h_{out}, \quad grid\_w = (x_2' - x_1' + 1) / w_{out}
   $$

- 对每个网格内的特征值执行最大池化：
   $$
   y_{i,j} = \max_{p=0,\dots,\lfloor grid\_h \rfloor} \max_{q=0,\dots,\lfloor grid\_w \rfloor} x_{y_1' + i \cdot grid\_h + p,\; x_1' + j \cdot grid\_w + q}
   $$

**多任务损失函数：**
$$
L = L_{cls}(p, u) + \lambda [u \ge 1] L_{loc}(t_u, v)
$$

其中 $L_{cls}(p, u) = -\log(p_u)$ 是对数损失，$L_{loc}(t_u, v) = \sum_{i\in\{x,y,w,h\}} \text{smooth}_{L1}(t_u^i - v^i)$。

**Smooth L1损失：**
$$
\text{smooth}_{L1}(x) = \begin{cases}
0.5x^2 & \text{如果 } |x| < 1 \\
|x| - 0.5 & \text{否则}
\end{cases}
$$

相比L2损失，Smooth L1对异常值更不敏感，训练更稳定。

## 直观理解

Fast R-CNN的核心创新可以理解为"算一次，用多次"和"边学边改"。整个图像只经过一次卷积网络提取特征（就像先拍一张全景照片），所有候选区域的特征都在这张"全景图"上裁剪得到。RoI Pooling像是用一个"万能取景框"——不管原图上的候选框多大、多小、多畸形，都能取出固定大小的局部特征图。

多任务损失则实现了"分类和定位同时学习"——网络既要判断框里是什么（分类），又要微调框的位置（回归），两个任务共享底层特征表示，互相促进。

## 代码示例

```python
import torch
import torch.nn as nn
import torchvision

class FastRCNN(nn.Module):
    """简化 Fast R-CNN"""
    def __init__(self, num_classes=21):
        super().__init__()
        # 骨干网络（VGG16的卷积层）
        vgg = torchvision.models.vgg16(pretrained=False)
        self.features = vgg.features
        # RoI Pooling（使用torchvision的API）
        self.roi_pool = torchvision.ops.RoIPool(output_size=(7, 7), spatial_scale=1.0)
        # 分类和回归头
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
        )
        self.cls_score = nn.Linear(4096, num_classes)
        self.bbox_pred = nn.Linear(4096, num_classes * 4)

    def forward(self, x, rois):
        # x: 整张图像, rois: (N, 5) [batch_idx, x1, y1, x2, y2]
        x = self.features(x)     # (1, 512, H/32, W/32)
        # 计算spatial_scale
        spatial_scale = x.shape[-1] / x.shape[-1]  # 实际应为 1/32
        self.roi_pool.spatial_scale = 1.0 / 32.0
        x = self.roi_pool(x, rois)  # (N, 512, 7, 7)
        x = x.view(x.size(0), -1)    # (N, 512*7*7)
        x = self.classifier(x)
        scores = self.cls_score(x)
        bboxes = self.bbox_pred(x)
        return scores, bboxes

# 模拟推理
model = FastRCNN()
img = torch.randn(1, 3, 224, 224)
# 模拟2个候选区域 [batch_id, x1, y1, x2, y2]
rois = torch.tensor([[0, 10, 10, 100, 100],
                      [0, 50, 50, 200, 200]], dtype=torch.float32)
scores, bboxes = model(img, rois)
print(f"分类得分: {scores.shape}")
print(f"边界框: {bboxes.shape}")
```

## 深度学习关联

- **R-CNN系列的效率革命**：Fast R-CNN将R-CNN的推理速度提升200倍以上，使两阶段检测达到了接近实时的速度，推动了目标检测从学术研究走向工业应用。其"共享特征图"的思想成为后续所有两阶段检测器的标准设计。
- **RoI Pooling的后续改进**：RoI Pooling中的量化误差启发了一系列改进——Mask R-CNN的RoI Align（双线性插值）、Cascade R-CNN的多阶段RoI refinement、Libra R-CNN的平衡特征金字塔等。
- **多任务学习范式**：Fast R-CNN的"分类+回归"多任务损失成为检测模型的标准配置，后续的Mask R-CNN进一步加入分割分支，YOLO则将分类、回归和置信度预测统一为单阶段输出。
