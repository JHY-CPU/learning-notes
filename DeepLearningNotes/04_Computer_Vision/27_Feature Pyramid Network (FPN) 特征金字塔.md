# 27_Feature Pyramid Network (FPN) 特征金字塔

## 核心概念

- **特征金字塔的动机**：检测不同尺度的物体是视觉任务的核心挑战之一。CNN的深层特征语义强但空间信息少，浅层特征空间信息丰富但语义弱。FPN通过自顶向下的路径将高层的语义信息与浅层的细节信息融合。
- **自底向上路径（Bottom-up Pathway）**：标准CNN的前向传播，生成多尺度的特征图层次（如ResNet的C2, C3, C4, C5），每个阶段的尺度递减2倍。
- **自顶向下路径（Top-down Pathway）**：从最高层特征开始，通过上采样（最近邻插值2倍）逐步恢复空间分辨率，与对应的自底向上特征进行横向连接（lateral connection）融合。
- **横向连接（Lateral Connection）**：使用 $1\times1$ 卷积将自底向上路径的特征图通道数统一为 $d=256$，然后与自顶向下的上采样特征逐元素相加。
- **特征金字塔的输出**：经过融合生成P2, P3, P4, P5（对应不同尺度），每层均可独立用于目标检测。通常还会在P5基础上通过步长2的最大池化生成P6用于大尺度检测。
- **FPN的普适性**：FPN是一个与骨干网络无关的通用模块，可以接在任何层次化CNN（ResNet、VGG等）之后，显著提升多尺度检测性能。

## 数学推导

**FPN的构建过程：**

设自底向上生成的特征图为 $\{C_2, C_3, C_4, C_5\}$，对应输入图像的下采样倍数分别为 $\{4, 8, 16, 32\}$。

1. 对 $C_5$ 使用 $1\times1$ 卷积得到 $P_5$（投影到256通道）
2. 对 $P_5$ 进行2倍最近邻上采样，与经过 $1\times1$ 卷积的 $C_4$ 相加，得到 $P_4$
3. 对 $P_4$ 重复上述过程得到 $P_3$、$P_2$

$$
P_5 = \text{Conv}_{1\times1}(C_5)
$$
$$
P_4 = \text{Conv}_{1\times1}(C_4) + \text{Upsample}_{2\times}(P_5)
$$
$$
P_3 = \text{Conv}_{1\times1}(C_3) + \text{Upsample}_{2\times}(P_4)
$$
$$
P_2 = \text{Conv}_{1\times1}(C_2) + \text{Upsample}_{2\times}(P_3)
$$

每个 $P$ 层再经过一个 $3\times3$ 卷积消除上采样的混叠效应，生成最终特征图。

**FPN在Faster R-CNN中的应用：**
将RPN和RoI Pooling分别应用到所有金字塔层：
- RPN在每层 $P_k$ 上独立预测物体提议
- 不同尺度的RoI被分配到对应的金字塔层：$k = k_0 + \log_2(\sqrt{wh}/224)$

## 直观理解

FPN好比一个"双向信息高速公路"。传统的CNN只有一条"自底向上"的单行道——信息从浅层流到深层，浅层保留了精细的空间位置但"看不懂"是什么（语义弱），深层"看懂了"是什么但位置信息已经模糊不清。FPN增加了一条"自顶向下"的反向通道——让深层学到的"是什么"知识反向传播到浅层，在每个尺度上都产生"既懂是什么又知道在哪"的特征。

这就像侦探破案：一线警员（浅层特征）了解现场的每一个细节但不知道全貌，而总指挥（深层特征）掌握了全局但看不到细节。FPN建立了总指挥到一线警员的沟通渠道，让每个层级的分析都能兼顾全局和细节。

## 代码示例

```python
import torch
import torch.nn as nn

class FPN(nn.Module):
    """特征金字塔网络"""
    def __init__(self, in_channels_list=[256, 512, 1024, 2048], out_channels=256):
        super().__init__()
        # 横向连接: 1x1 卷积统一通道数
        self.lateral_convs = nn.ModuleList([
            nn.Conv2d(in_c, out_channels, 1)
            for in_c in in_channels_list
        ])
        # 输出卷积: 3x3 消除混叠效应
        self.output_convs = nn.ModuleList([
            nn.Conv2d(out_channels, out_channels, 3, padding=1)
            for _ in range(len(in_channels_list))
        ])

    def forward(self, features):
        # features: [C2, C3, C4, C5] 从浅到深
        # 自顶向下路径
        laterals = [
            lateral_conv(f)
            for lateral_conv, f in zip(self.lateral_convs, features)
        ]

        # 从高层开始逐步上采样融合
        for i in range(len(laterals) - 1, 0, -1):
            laterals[i-1] = laterals[i-1] + nn.functional.interpolate(
                laterals[i], size=laterals[i-1].shape[-2:], mode='nearest'
            )

        # 输出卷积
        outputs = []
        for i, output_conv in enumerate(self.output_convs):
            outputs.append(output_conv(laterals[i]))
        return outputs  # [P2, P3, P4, P5]

# 测试
fpn = FPN()
c2 = torch.randn(1, 256, 56, 56)   # 1/4
c3 = torch.randn(1, 512, 28, 28)   # 1/8
c4 = torch.randn(1, 1024, 14, 14)  # 1/16
c5 = torch.randn(1, 2048, 7, 7)    # 1/32

outputs = fpn([c2, c3, c4, c5])
for i, out in enumerate(outputs):
    print(f"P{i+2} 特征图尺寸: {out.shape}")

# 验证所有输出通道数=256
assert all(out.shape[1] == 256 for out in outputs)
print("FPN输出验证通过: 所有层通道数均为256")
```

## 深度学习关联

- **目标检测的标准组件**：FPN已成为目标检测器的标准组件，几乎所有现代检测器（Faster R-CNN + FPN、YOLOv3（类似FPN的多尺度预测）、RetinaNet + FPN、EfficientDet等）都使用特征金字塔来检测多尺度物体。
- **超越检测的广泛应用**：FPN不仅用于目标检测，还被广泛应用于语义分割（Panoptic FPN）、实例分割（Mask R-CNN + FPN）、姿态估计、全景分割等密集预测任务中，成为计算机视觉领域最通用的模块之一。
- **双向特征融合的演变**：FPN的"自顶向下"特征融合启发了后续一系列改进——PANet增加了"自底向上"的补充路径，BiFPN（EfficientDet）引入了加权跨尺度连接和反复双向融合，NAS-FPN通过架构搜索自动设计最优的特征融合拓扑。
