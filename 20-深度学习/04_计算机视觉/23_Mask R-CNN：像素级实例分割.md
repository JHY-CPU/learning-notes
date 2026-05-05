# 23_Mask R-CNN：像素级实例分割

## 核心概念

- **实例分割（Instance Segmentation）**：同时检测图像中的每个物体实例，并对其生成像素级的分割掩码。比语义分割更进一步——不仅要分割"人"类，还要区分"人1"和"人2"。
- **多任务输出**：Mask R-CNN在Faster R-CNN（分类+边界框）的基础上增加了一个并行的掩码预测分支，输出每个RoI的二进制分割掩码。三个任务共享骨干特征。
- **RoI Align**：解决RoI Pooling中两次量化（坐标量化和网格划分）导致的空间偏差问题，使用双线性插值计算浮点数坐标上的特征值，提升小目标和精细分割的精度。
- **掩码分支结构**：使用全卷积网络（FCN）预测 $m \times m$（如 $28\times28$）的二进制掩码，每个类别独立预测（$K$ 个通道）。独立预测避免类别间竞争。
- **掩码损失**：对每个RoI，仅使用其对应真实类别的掩码通道计算二值交叉熵损失（Binary CE），不与其他类别竞争。
- **分割与检测的解耦**：对每个RoI，分类和框回归由全连接层处理，而掩码分割由全卷积层处理，保持空间信息的完整性。

## 数学推导

**RoI Align 的双线性插值：**

对于RoI中采样点 $(x, y)$（浮点数坐标），取其周围4个整数坐标点的特征值进行双线性插值：
$$
f(x, y) = (1 - \alpha)(1 - \beta) f(x_1, y_1) + \alpha(1 - \beta) f(x_2, y_1) + (1 - \alpha)\beta f(x_1, y_2) + \alpha\beta f(x_2, y_2)
$$

其中 $\alpha = x - x_1,\; \beta = y - y_1$。

**RoI Align vs RoI Pooling 的量化差异：**
- RoI Pooling: 将 $[x_{float}, y_{float}]$ 量化为 $[x_{int}, y_{int}]$ → 两次量化误差
- RoI Align: 保持浮点数坐标，通过插值计算特征值 → 无量化误差

**Mask R-CNN 总损失函数：**
$$
L = L_{cls} + L_{box} + L_{mask}
$$

其中 $L_{mask}$ 是对每个RoI中每个像素的二值交叉熵损失：
$$
L_{mask} = -\frac{1}{m^2} \sum_{i=1}^{m^2} [y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i)]
$$

掩码损失只在真实类别 $k$ 上计算，其他类别的掩码不贡献损失。

## 直观理解

Mask R-CNN在Faster R-CNN的基础上增加了一个"精雕细琢"的分支。如果说Faster R-CNN的任务是在图像上画出"这里有一辆车"的方框（检测），那么Mask R-CNN还要额外"沿着车的边缘精确裁剪"（分割）。三个分支共享同一个视觉特征提取器（骨干网络），但各自的任务不同：

- 分类分支："这是什么？"
- 框回归分支："它在图像的大概位置在哪？"  
- 掩码分支："它的精确轮廓是怎样的？"

RoI Align相比RoI Pooling的关键改进在于"精细度"——就像用高精度电子秤代替普通台秤称量微小物体。对于像素级的分割任务，即使是1-2个像素的偏差也会导致分割质量明显下降。

## 代码示例

```python
import torch
import torch.nn as nn
import torchvision
from torchvision.ops import RoIAlign

class MaskRCNN(nn.Module):
    """简化版 Mask R-CNN"""
    def __init__(self, num_classes=81):
        super().__init__()
        # ResNet-50 骨干
        backbone = torchvision.models.resnet50(pretrained=False)
        self.features = nn.Sequential(*list(backbone.children())[:-2])

        # RPN
        self.rpn_conv = nn.Conv2d(2048, 256, 3, padding=1)
        self.rpn_cls = nn.Conv2d(256, 9 * 2, 1)  # 9个锚点×2（前景/背景）
        self.rpn_reg = nn.Conv2d(256, 9 * 4, 1)  # 9个锚点×4坐标

        # RoI Align
        self.roi_align = RoIAlign(output_size=(7, 7), spatial_scale=1.0/32.0, sampling_ratio=2)

        # 分类+框回归头
        self.fc = nn.Sequential(
            nn.Linear(2048 * 7 * 7, 1024),
            nn.ReLU(),
        )
        self.cls_score = nn.Linear(1024, num_classes)
        self.bbox_pred = nn.Linear(1024, num_classes * 4)

        # 掩码头（全卷积，保持空间结构）
        self.mask_head = nn.Sequential(
            nn.Conv2d(2048, 256, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(256, 256, 2, stride=2),  # 上采样到14x14
            nn.ReLU(),
            nn.Conv2d(256, num_classes, 1),  # 每类一个掩码通道
        )

    def forward(self, x, rois):
        feat = self.features(x)  # (B, 2048, H/32, W/32)

        # 掩码预测（用高分辨率特征图）
        mask_logits = self.mask_head(feat)  # (B, num_classes, 14, 14)

        # RoI Align
        roi_feat = self.roi_align(feat, rois)  # (N, 2048, 7, 7)
        roi_feat = roi_feat.view(roi_feat.size(0), -1)
        fc_feat = self.fc(roi_feat)
        scores = self.cls_score(fc_feat)
        bboxes = self.bbox_pred(fc_feat)
        return scores, bboxes, mask_logits

model = MaskRCNN()
img = torch.randn(1, 3, 224, 224)
rois = torch.tensor([[0, 10, 10, 100, 100], [0, 50, 50, 200, 200]], dtype=torch.float32)
scores, bboxes, masks = model(img, rois)
print(f"分类: {scores.shape}, 框: {bboxes.shape}, 掩码: {masks.shape}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **实例分割的标准框架**：Mask R-CNN成为实例分割领域的事实标准，后续的Cascade Mask R-CNN、Hybrid Task Cascade、SOLOv2等模型都在其基础上改进。COCO实例分割竞赛的绝大多数参赛方案都基于Mask R-CNN。
- **RoI Align的广泛影响**：RoI Align替代RoI Pooling成为两阶段检测器的标准配置，在视频目标分割（MaskTrack R-CNN）、全景分割（Panoptic FPN）、姿态估计（Keypoint R-CNN）等任务中被广泛采用。
- **从检测到像素级理解的演进**：Mask R-CNN代表了计算机视觉从"框级别"理解（检测）到"像素级别"理解（分割）的关键跨越，推动了自动驾驶（道路目标实例分割）、医学影像（器官和病灶分割）、工业质检（缺陷区域提取）等应用的发展。
