# 17_EfficientNet：复合缩放系数 (Compound Scaling)

## 核心概念

- **复合缩放（Compound Scaling）**：同时缩放网络的深度（depth）、宽度（width）和输入分辨率（resolution），而非传统的单一维度缩放。复合缩放系数 $\phi$ 控制三个维度的缩放比例。
- **缩放维度的相互依赖**：更高的分辨率需要更深的网络来增加感受野，也需要更宽的网络来捕获更多细粒度特征。三个维度不是独立的，需要联合优化。
- **基线网络 EfficientNet-B0**：通过神经架构搜索（NAS）设计的基础网络，使用MBConv（Mobile Inverted Bottleneck Conv）和SE注意力模块，在ImageNet上以极低的计算量达到高精度。
- **MBConv结构**：基于MobileNet V2的倒置残差，额外添加Squeeze-and-Excitation（SE）通道注意力模块，提升特征表达能力。
- **缩放公式**：深度 $d = \alpha^\phi$，宽度 $w = \beta^\phi$，分辨率 $r = \gamma^\phi$，满足 $\alpha \cdot \beta^2 \cdot \gamma^2 \approx 2$（确保总FLOPs随 $\phi$ 增长约 $2^\phi$ 倍）。
- **SOTA性能**：EfficientNet-B7在ImageNet上达到84.4% top-1准确率（当时最高），参数量和FLOPs却远低于同类精度模型。

## 数学推导

**复合缩放公式：**
$$
\text{深度: } d = \alpha^\phi
$$
$$
\text{宽度: } w = \beta^\phi
$$
$$
\text{分辨率: } r = \gamma^\phi
$$
$$
\text{约束: } \alpha \cdot \beta^2 \cdot \gamma^2 \approx 2
$$

**FLOPs增长分析：**
卷积层的计算量约正比于 $d \cdot w^2 \cdot r^2$：
- 深度 $d$ → 计算量 $\times d$
- 宽度 $w$ → 计算量 $\times w^2$（输入/输出通道都缩放）
- 分辨率 $r$ → 计算量 $\times r^2$（特征图HW都缩放）

约束条件 $\alpha \cdot \beta^2 \cdot \gamma^2 = 2$ 保证当 $\phi=1$ 时总FLOPs翻倍约 $2^\phi$ 倍。

**EfficientNet-B0 到 B7 的缩放：**

| 模型 | $\phi$ | 深度 | 宽度 | 分辨率 | Top-1 Acc |
|---|---|---|---|---|---|
| B0 | 0 | 1.0× | 1.0× | 224 | 77.1% |
| B1 | 0.5 | 1.1× | 1.1× | 240 | 79.1% |
| B2 | 1 | 1.2× | 1.2× | 260 | 80.1% |
| B3 | 2 | 1.4× | 1.4× | 300 | 81.6% |
| B4 | 3 | 1.8× | 1.8× | 380 | 82.9% |
| B5 | 4 | 2.2× | 2.2× | 456 | 83.6% |
| B6 | 5 | 2.6× | 2.6× | 528 | 84.0% |
| B7 | 6 | 3.1× | 3.1× | 600 | 84.4% |

## 直观理解

复合缩放的思想好比设计一套"儿童-成人-巨人"的服装。你不能只放大衣服的长度（深度）而不放大宽度和袖长，否则衣服会变得畸形。同样，当你提高输入分辨率（让网络看更精细的图像）时，也需要增加网络深度（让网络有足够的感受野）和宽度（让网络能捕获更多细节）。传统的单维度缩放就像只加大衣服的长度——虽然数据变多了（深度增加），但宽度和分辨率没有匹配提升，网络能力受限。

## 代码示例

```python
import torch
import torch.nn as nn

class SEBlock(nn.Module):
    """Squeeze-and-Excitation 通道注意力"""
    def __init__(self, channels, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, channels // reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return x * self.fc(x)

class MBConvWithSE(nn.Module):
    """EfficientNet 的 MBConv 块 (含SE)"""
    def __init__(self, in_c, out_c, kernel_size, stride, expand_ratio, se_ratio=0.25):
        super().__init__()
        hidden_c = round(in_c * expand_ratio)
        self.use_res = stride == 1 and in_c == out_c

        layers = []
        if expand_ratio != 1:
            layers.extend([
                nn.Conv2d(in_c, hidden_c, 1, bias=False),
                nn.BatchNorm2d(hidden_c),
                nn.SiLU(inplace=True)  # Swish激活
            ])
        layers.extend([
            nn.Conv2d(hidden_c, hidden_c, kernel_size, stride,
                      padding=kernel_size//2, groups=hidden_c, bias=False),
            nn.BatchNorm2d(hidden_c),
            nn.SiLU(inplace=True),
            SEBlock(hidden_c, int(1/se_ratio)),
            nn.Conv2d(hidden_c, out_c, 1, bias=False),
            nn.BatchNorm2d(out_c)
        ])
        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res:
            return x + self.conv(x)
        return self.conv(x)

# 复合缩放模拟: 计算不同phi值下的理论FLOPs
import math
alphas = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]  # 深度缩放
betas  = [1.0, 1.1, 1.2, 1.4, 1.8, 2.2, 2.6, 3.1]  # 宽度缩放
gammas = [1.0, 1.07, 1.16, 1.34, 1.70, 2.04, 2.36, 2.68]  # 分辨率缩放
for i, (a, b, g) in enumerate(zip(alphas, betas, gammas)):
    flops_ratio = a * (b**2) * (g**2)
    print(f"B{i}: depth={a:.1f}x, width={b:.1f}x, res={224*g:.0f}, FLOPs≈{flops_ratio:.1f}x baseline")
```

## 深度学习关联

- **自动化模型缩放**：EfficientNet将模型缩放从"人工试错"提升为"系统化工程"，通过复合缩放公式可以在任意计算预算下生成最优模型。这一思想被广泛应用于移动端视觉模型（EfficientNet-Lite）和云端推理优化。
- **NAS + 缩放的两阶段范式**：EfficientNet验证了"先用NAS在小计算量下搜索最佳基线网络，再用复合缩放到大模型"的有效性，成为后来很多模型设计（如RegNet、MobileNet V3）的标准流程。
- **Backbone网络的效率标准**：EfficientNet-B0到B7系列提供了丰富的精度-效率权衡选项，被广泛用作目标检测（EfficientDet）、语义分割的骨干网络，以及Google Cloud TPU视觉推理服务的默认模型。
