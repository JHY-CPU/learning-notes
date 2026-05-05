# 07_深度可分离卷积 (Depthwise Separable Conv)

## 核心概念

- **深度可分离卷积的定义**：将标准卷积分拆为两个独立的步骤——深度卷积（Depthwise Convolution）和逐点卷积（Pointwise Convolution），大幅减少参数量和计算量。
- **深度卷积（Depthwise Conv）**：对每个输入通道独立使用一个单通道卷积核进行空间卷积，输出通道数与输入通道数相同，不进行跨通道信息融合。
- **逐点卷积（Pointwise Conv）**：使用 $1\times1$ 卷积在深度卷积的输出上进行跨通道线性组合，将通道数变换到目标输出通道数。
- **计算效率对比**：标准卷积的计算量为 $C_{in} \times C_{out} \times K^2 \times H \times W$，深度可分离卷积的计算量为 $C_{in} \times K^2 \times H \times W + C_{in} \times C_{out} \times H \times W$，计算量比约为 $1/C_{out} + 1/K^2$。
- **MobileNet 的核心构建块**：深度可分离卷积是MobileNet系列模型的基石，使移动设备上的实时视觉应用成为可能。
- **与分组卷积的关系**：深度卷积是分组卷积的特例——当分组数等于输入通道数时的分组卷积即为深度卷积。

## 数学推导

**标准卷积计算量：**
$$
\text{Cost}_{std} = C_{in} \times C_{out} \times K^2 \times H_{out} \times W_{out}
$$

**深度可分离卷积计算量：**
$$
\text{Cost}_{depthwise} = C_{in} \times K^2 \times H_{out} \times W_{out} \quad (\text{深度卷积})
$$
$$
\text{Cost}_{pointwise} = C_{in} \times C_{out} \times H_{out} \times W_{out} \quad (1\times1 \text{ 逐点卷积})
$$

**计算量压缩比：**
$$
\frac{\text{Cost}_{sep}}{\text{Cost}_{std}} = \frac{C_{in} \times K^2 \times H_{out} \times W_{out} + C_{in} \times C_{out} \times H_{out} \times W_{out}}{C_{in} \times C_{out} \times K^2 \times H_{out} \times W_{out}}
= \frac{1}{C_{out}} + \frac{1}{K^2}
$$

当 $C_{out}=256,\; K=3$ 时，压缩比为 $1/256 + 1/9 \approx 0.117$，即计算量减少约 88.3%。

**参数量对比：**
- 标准卷积参数量：$C_{in} \times C_{out} \times K^2 + C_{out}$
- 深度可分离卷积参数量：$C_{in} \times K^2 + C_{in} \times C_{out} + C_{out}$

## 直观理解

标准卷积可以看作同时做两件事：在每个通道内提取空间特征（"在哪里有什么"）和融合不同通道的特征（"不同的特征如何组合"）。深度可分离卷积把这两件事拆开做：先用深度卷积在每个通道内独立提取空间模式（每个通道单独处理，相当于3个独立的灰度图卷积），再用逐点卷积将不同通道的信息混合（像是一个小型全连接层在通道维度上做组合）。这就像做饭时先分别洗菜切菜（深度卷积），再把所有食材一起翻炒（逐点卷积）——分开做更高效。

## 代码示例

```python
import torch
import torch.nn as nn

class DepthwiseSeparableConv(nn.Module):
    """深度可分离卷积模块"""
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()
        # 深度卷积: groups=in_channels 使每个通道独立卷积
        self.depthwise = nn.Conv2d(
            in_channels, in_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
            groups=in_channels,  # 核心: 分组数=输入通道数
            bias=False
        )
        # 逐点卷积: 1x1 卷积做通道融合
        self.pointwise = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=1,
            bias=True
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.depthwise(x)
        x = self.pointwise(x)
        x = self.bn(x)
        x = self.relu(x)
        return x

# 参数量对比
in_c, out_c, k = 64, 128, 3
std_conv = nn.Conv2d(in_c, out_c, k, padding=k//2)
sep_conv = DepthwiseSeparableConv(in_c, out_c, k)

std_params = sum(p.numel() for p in std_conv.parameters())
sep_params = sum(p.numel() for p in sep_conv.parameters())
print(f"标准卷积参数量: {std_params}")
print(f"深度可分离卷积参数量: {sep_params}")
print(f"压缩比: {sep_params / std_params:.3f}")

# 验证输出尺寸一致
x = torch.randn(1, in_c, 32, 32)
print(f"标准卷积输出: {std_conv(x).shape}")
print(f"深度可分离卷积输出: {sep_conv(x).shape}")
```

## 深度学习关联

- **MobileNet系列**：MobileNet V1首次将深度可分离卷积引入图像分类，在ImageNet上以极低的计算量达到接近VGG16的准确率。MobileNet V2在此基础上加入倒置残差结构（Inverted Residual），V3则进一步结合了NAS搜索。
- **Xception架构**：Xception（Extreme Inception）将Inception模块中的标准卷积全部替换为深度可分离卷积，认为跨通道相关性和空间相关性可以完全解耦，在ImageNet上取得了优于Inception V3的效果。
- **EfficientNet中的MBConv**：EfficientNet使用的MBConv（Mobile Inverted Bottleneck Conv）基于深度可分离卷积构建，结合SE注意力机制和复合缩放策略，实现了SOTA精度与效率的平衡。
