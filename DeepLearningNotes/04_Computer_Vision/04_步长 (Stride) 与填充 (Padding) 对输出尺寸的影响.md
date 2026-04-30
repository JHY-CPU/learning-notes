# 04_步长 (Stride) 与填充 (Padding) 对输出尺寸的影响

## 核心概念

- **步长（Stride）定义**：卷积核每次滑动时移动的像素数。步长为1时输出尺寸与输入接近，步长为2时将空间尺寸减半，实现下采样。
- **填充（Padding）定义**：在输入特征图的边缘填充额外的像素（通常为0），用于控制输出特征图的空间尺寸。常见的有"valid"（无填充）和"same"（输出尺寸与输入相同）两种模式。
- **输出尺寸公式**：给定输入尺寸 $H_{in} \times W_{in}$，卷积核尺寸 $K$，填充 $P$，步长 $S$，输出尺寸为 $H_{out} = \lfloor (H_{in} + 2P - K) / S \rfloor + 1$。
- **Same Padding**：当 $P = (K - 1) / 2$ 且 $S = 1$ 时，输出尺寸等于输入尺寸，称为same padding，是ResNet等现代网络的默认选择。
- **Valid Padding（无填充）**：$P = 0$，每个像素只被卷积核覆盖相同次数，输出尺寸小于输入尺寸，边缘信息会丢失。
- **填充方式**：除零填充（zero padding）外，还有反射填充（reflect padding）、复制填充（replicate padding）等，适用于不同的边界处理需求。

## 数学推导

**输出尺寸公式（二维）：**
$$
H_{out} = \left\lfloor \frac{H_{in} + 2P_h - K_h}{S_h} \right\rfloor + 1
$$
$$
W_{out} = \left\lfloor \frac{W_{in} + 2P_w - K_w}{S_w} \right\rfloor + 1
$$

其中 $P_h, P_w$ 分别为高和宽方向的填充像素数，$S_h, S_w$ 为步长。

**示例计算：**
- 输入 $32\times32$，卷积核 $5\times5$，填充 $P=0$，步长 $S=1$：
  $H_{out} = (32 - 5)/1 + 1 = 28$
- 输入 $32\times32$，卷积核 $3\times3$，填充 $P=1$，步长 $S=2$：
  $H_{out} = (32 + 2 - 3)/2 + 1 = 16$

**等价于下采样的组合：**
当使用 $3\times3$ 卷积、$P=1$（same padding）、$S=2$ 时，输出尺寸恰好为输入尺寸的一半，等价于一个池化层加一个卷积层。

## 直观理解

步长和填充共同决定了卷积输出特征图的"分辨率"。步长控制卷积核的"跳跃幅度"——步长越大，采样越稀疏，输出图越小，类似于用大间距的网格来抽样。填充则控制边界的"扩充幅度"——填充相当于在图像四周加上一圈空白边距，让卷积核在边缘处也能正常计算（否则边缘像素被访问的次数比中心像素少）。将步长和填充配合使用，可以精确控制特征图的尺寸变化，这在设计需要多尺度特征的金字塔结构（如FPN、U-Net）时至关重要。

## 代码示例

```python
import torch
import torch.nn.functional as F

# 定义输入: batch=1, channel=1, height=32, width=32
x = torch.randn(1, 1, 32, 32)

# 3x3卷积, padding=0, stride=1 (Valid)
conv_valid = F.conv2d(x, torch.randn(1, 1, 3, 3), padding=0, stride=1)
print(f"Valid 输出尺寸: {conv_valid.shape[2]}x{conv_valid.shape[3]}")  # 30x30

# 3x3卷积, padding=1, stride=1 (Same)
conv_same = F.conv2d(x, torch.randn(1, 1, 3, 3), padding=1, stride=1)
print(f"Same 输出尺寸: {conv_same.shape[2]}x{conv_same.shape[3]}")    # 32x32

# 3x3卷积, padding=1, stride=2 (下采样)
conv_down = F.conv2d(x, torch.randn(1, 1, 3, 3), padding=1, stride=2)
print(f"Stride=2 输出尺寸: {conv_down.shape[2]}x{conv_down.shape[3]}")  # 16x16

# 5x5卷积, padding=2, stride=1 (Same)
conv_5 = F.conv2d(x, torch.randn(1, 1, 5, 5), padding=2, stride=1)
print(f"5x5 Same 输出尺寸: {conv_5.shape[2]}x{conv_5.shape[3]}")  # 32x32
```

## 深度学习关联

- **下采样策略设计**：在ResNet、VGGNet等分类网络中，通常使用步长=2的卷积或池化来逐步降低特征图分辨率（从 $224\times224$ 降到 $7\times7$），形成层级式特征表示。
- **全卷积网络（FCN）的空间精度**：语义分割任务中需要保持较高的空间分辨率，通常使用步长=1和空洞卷积的组合来避免过度下采样，同时扩大感受野。
- **转置卷积的尺寸恢复**：在U-Net、GAN等生成式模型中，转置卷积使用步长>1来上采样恢复分辨率，其输出尺寸由步长和填充共同决定，需要与对应的下采样尺寸精确匹配。
