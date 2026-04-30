# 24_YOLO 系列：单阶段检测的网格划分思想

## 核心概念

- **单阶段检测（One-Stage Detection）**：YOLO（You Only Look Once）将目标检测视为一个端到端的回归问题——将图像分为 $S \times S$ 的网格，每个网格直接预测边界框和类别概率，一步到位，无需候选区域阶段。
- **网格划分（Grid Division）**：将输入图像分为 $S\times S$ 网格（如YOLOv1使用 $7\times7$），每个网格负责检测"中心点落在该网格内"的物体。每个网格预测 $B$ 个边界框和 $C$ 个类别概率。
- **统一检测流程**：单次CNN前向传播直接输出 $S \times S \times (B \times 5 + C)$ 的张量，其中5表示 $(x, y, w, h, confidence)$。整个过程"一气呵成"。
- **YOLOv1的局限性**：每个网格只能预测一个类别（当多个物体落入同一网格时出错），对小物体检测效果差，对不规则长宽比物体定位不准。
- **YOLOv2/v3的改进**：引入锚点机制（基于K-Means聚类先验框）、批归一化、多尺度训练；YOLOv3使用FPN-like的多尺度检测头（3个尺度），使用Logistic分类器替代Softmax支持多标签。
- **速度-精度权衡**：YOLO系列的核心优势在于极致的推理速度（YOLOv3在Titan X上可达30-60 FPS），使其成为视频实时检测和边缘部署的首选。

## 数学推导

**YOLOv1的损失函数：**
$$
L = \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{obj} [(x_i - \hat{x}_i)^2 + (y_i - \hat{y}_i)^2]
+ \lambda_{coord} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{obj} [(\sqrt{w_i} - \sqrt{\hat{w}_i})^2 + (\sqrt{h_i} - \sqrt{\hat{h}_i})^2]
$$
$$
+ \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{obj} (C_i - \hat{C}_i)^2
+ \lambda_{noobj} \sum_{i=0}^{S^2} \sum_{j=0}^B \mathbb{1}_{ij}^{noobj} (C_i - \hat{C}_i)^2
+ \sum_{i=0}^{S^2} \mathbb{1}_i^{obj} \sum_{c \in classes} (p_i(c) - \hat{p}_i(c))^2
$$

其中 $\mathbb{1}_{ij}^{obj}$ 指示网格 $i$ 的第 $j$ 个锚点是否负责检测物体。

**YOLOv3的预测头：**
输出 $S \times S \times [3 \times (5 + C)]$，其中3是每个尺度的锚点数，5是 $(t_x, t_y, t_w, t_h, obj)$：
$$
b_x = \sigma(t_x) + c_x, \quad b_y = \sigma(t_y) + c_y
$$
$$
b_w = p_w e^{t_w}, \quad b_h = p_h e^{t_h}
$$

其中 $(c_x, c_y)$ 是网格左上角坐标，$(p_w, p_h)$ 是锚点的宽高先验值。$\sigma$ 函数确保中心坐标落在网格内。

**YOLOv3三个尺度的检测：**
- 大尺度检测：$13 \times 13$ 特征图（下采样32倍），负责大物体
- 中尺度检测：$26 \times 26$ 特征图（下采样16倍），负责中物体
- 小尺度检测：$52 \times 52$ 特征图（下采样8倍），负责小物体

## 直观理解

YOLO的设计哲学是"一眼看遍全局"，与R-CNN系列"先看局部再定全局"截然不同。YOLO将图像划分网格，就像用地毯式搜索的方式查找物体——每个网格对自己的"地盘"负责，预测该区域是否有物体。网格之间虽有分工但并非完全独立，因为网格在预测时会观察到周围区域的信息（感受野大于网格本身）。

"单阶段"的优势在于效率：不需要"先提候选框再分类"的两次处理，一次前向传播就能得到所有结果。这就像在人群找一个人——"逐个区域仔细比对"（两阶段）和"全场扫视一眼"（单阶段）的区别。

## 代码示例

```python
import torch
import torch.nn as nn

class YOLOv3Head(nn.Module):
    """YOLOv3 单尺度检测头"""
    def __init__(self, in_channels, num_classes=80, num_anchors=3):
        super().__init__()
        self.num_anchors = num_anchors
        self.num_classes = num_classes
        # 检测头：若干卷积层后直接输出预测
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, 512, 3, padding=1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.1),
            nn.Conv2d(512, 256, 3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1),
        )
        self.predict = nn.Conv2d(
            256, num_anchors * (5 + num_classes), 1
        )

    def forward(self, x):
        x = self.conv(x)
        x = self.predict(x)
        # (B, anchors*(5+C), H, W) -> (B, H, W, anchors, 5+C)
        B, _, H, W = x.shape
        x = x.view(B, self.num_anchors, -1, H, W)
        x = x.permute(0, 3, 4, 1, 2).contiguous()
        return x  # (B, H, W, anchors, 5+C)

# 模拟YOLOv3的多尺度输出
x_small = torch.randn(2, 1024, 13, 13)   # 小物体分支
x_medium = torch.randn(2, 512, 26, 26)   # 中物体分支
x_large = torch.randn(2, 256, 52, 52)    # 大物体分支

head_small = YOLOv3Head(1024)
head_medium = YOLOv3Head(512)
head_large = YOLOv3Head(256)

out_small = head_small(x_small)
out_medium = head_medium(x_medium)
out_large = head_large(x_large)

print(f"大尺度输出: {out_small.shape}")   # (2, 13, 13, 3, 85)
print(f"中尺度输出: {out_medium.shape}")  # (2, 26, 26, 3, 85)
print(f"小尺度输出: {out_large.shape}")   # (2, 52, 52, 3, 85)
```

## 深度学习关联

- **实时检测的标准**：YOLO系列定义了"实时目标检测"的标准——在保证可用精度的前提下追求极致速度。YOLOv3成为工业界最广泛使用的检测器之一，被集成到OpenCV、TensorFlow Object Detection API等框架中。后续的YOLOv4-v7在速度和精度上持续刷新记录。
- **单阶段检测的普及**：YOLO的成功证明了"简洁即高效"——单阶段检测在速度上远超两阶段检测，且随着技术改进（FPN、Focal Loss、GIoU等），精度差距也在缩小。单阶段检测成为边缘计算和移动部署的首选方案。
- **统一框架的趋势**：YOLO的"统一网络、统一损失"设计思想影响了后续的DETR（Transformer端到端检测）、YOLOS（纯ViT检测）等模型，体现了计算机视觉从多阶段流水线向端到端统一框架演进的大趋势。
