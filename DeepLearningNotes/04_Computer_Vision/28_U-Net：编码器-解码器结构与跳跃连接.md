# 28_U-Net：编码器-解码器结构与跳跃连接

## 核心概念

- **U-Net架构**：对称的"U"形编码器-解码器结构，由下采样（收缩）路径和上采样（扩展）路径组成，最早用于生物医学图像分割。
- **编码器（收缩路径）**：由4个block组成，每个block包含2个 $3\times3$ 卷积（ReLU）和1个 $2\times2$ 最大池化（步长2），逐步降低空间分辨率并增加通道数（64→128→256→512→1024）。
- **解码器（扩展路径）**：由4个block组成，每个block先进行 $2\times2$ 转置卷积上采样（通道数减半），然后与编码器对应层（经过裁剪）拼接（skip connection），再进行2个 $3\times3$ 卷积。
- **跳跃连接（Skip Connection）**：将编码器每层的特征图直接拼接到解码器的对应层，为高分辨率分割提供精细的空间信息。这与ResNet的残差连接（相加）不同，U-Net使用**通道拼接**。
- **Overlap-Tile策略**：通过镜像填充处理边界像素，使U-Net可以处理任意大小的图像并输出完整的像素级预测。
- **数据增强**：使用弹性变形（elastic deformation）等医学图像特有的数据增强方法，在少量标注数据上大幅提升分割性能。

## 数学推导

**U-Net中的尺寸变化（输入 $572\times572$ 灰度图）：**

| 层级 | 操作 | 尺寸变化 | 通道数 |
|---|---|---|---|
| 输入 | - | $572\times572$ | 1 |
| Encoder 1 | 2×Conv3×3, MaxPool | $572\to570\to568\to284$ | 1→64 |
| Encoder 2 | 2×Conv3×3, MaxPool | $284\to280\to276\to136$ | 64→128 |
| Encoder 3 | 2×Conv3×3, MaxPool | $136\to132\to128\to64$ | 128→256 |
| Encoder 4 | 2×Conv3×3, MaxPool | $64\to60\to56\to28$ | 256→512 |
| Bridge | 2×Conv3×3 | $28\to24\to20$ | 512→1024 |
| Decoder 4 | UpConv, Cat(Skip), 2×Conv | $20\to40,\text{cat}56\to56\to52\to48$ | 1024→512→512 |
| Decoder 3 | UpConv, Cat(Skip), 2×Conv | $48\to96,\text{cat}128\to128\to124\to120$ | 512→256→256 |
| Decoder 2 | UpConv, Cat(Skip), 2×Conv | $120\to240,\text{cat}280\to280\to276\to272$ | 256→128→128 |
| Decoder 1 | UpConv, Cat(Skip), 2×Conv | $272\to544,\text{cat}570\to570\to566\to562$ | 128→64→64 |
| 输出 | 1×Conv1×1 | $562\to562$ | 64→2 |

**跳跃连接中的拼接操作：**
解码器第 $l$ 层输入 = [上采样(解码器$_{l+1}$), 编码器$_{l}$特征]

其中 $[\cdot, \cdot]$ 表示通道维度的拼接。假设编码器第 $l$ 层特征 $E_l \in \mathbb{R}^{C_E \times H \times W}$，上采样后的解码器特征 $D_{l+1} \in \mathbb{R}^{C_D \times H \times W}$，则拼接后通道数为 $C_E + C_D$。

**分割损失函数（像素级交叉熵）：**
$$
L = -\frac{1}{N} \sum_{i=1}^N \sum_{c=1}^C y_{i,c} \log(p_{i,c})
$$

其中 $y_{i,c}$ 是像素 $i$ 在类别 $c$ 上的真值，$p_{i,c}$ 是预测概率。对于医学图像，通常使用加权交叉熵（给边界像素更高的权重）或Dice Loss来缓解类别不平衡。

## 直观理解

U-Net的"U"形结构，可以理解为一个"先压缩再解压"的信息处理过程。编码器阶段像是"做摘要"：逐渐缩小图像的尺寸，提取越来越抽象的特征（从边缘→纹理→器官形状）。解码器阶段像是"解压缩"：将抽象的语义特征逐步还原为原始分辨率的分割图。

跳跃连接是U-Net的精髓所在：在解码器还原的过程中，直接从编码器的对应层"抄近路"把高分辨率的细节特征传递过来。这确保了解码器在做精细分割时既有高层语义指导（"这是一个肾脏"），又有低层位置信息（"它的边界精确在哪些像素上"）。没有跳跃连接，只靠上采样很难恢复丢失的空间细节。

## 代码示例

```python
import torch
import torch.nn as nn

class DoubleConv(nn.Module):
    """双卷积块: Conv3x3 + BN + ReLU (两次)"""
    def __init__(self, in_c, out_c):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_c, out_c, 3, padding=1),
            nn.BatchNorm2d(out_c),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.conv(x)

class UNet(nn.Module):
    """U-Net"""
    def __init__(self, in_channels=1, out_channels=2, features=[64, 128, 256, 512]):
        super().__init__()
        # 编码器
        self.encoders = nn.ModuleList()
        self.pools = nn.ModuleList()
        in_ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(in_ch, f))
            self.pools.append(nn.MaxPool2d(2))
            in_ch = f

        # 桥接层
        self.bridge = DoubleConv(features[-1], features[-1] * 2)

        # 解码器
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        for f in reversed(features):
            self.upconvs.append(nn.ConvTranspose2d(f * 2, f, 2, stride=2))
            self.decoders.append(DoubleConv(f * 2, f))

        # 输出
        self.final = nn.Conv2d(features[0], out_channels, 1)

    def forward(self, x):
        # 编码
        skip_connections = []
        for encoder, pool in zip(self.encoders, self.pools):
            x = encoder(x)
            skip_connections.append(x)
            x = pool(x)

        x = self.bridge(x)

        # 解码（反向遍历）
        skip_connections = skip_connections[::-1]
        for i in range(len(self.upconvs)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]
            # 尺寸匹配（如有需要则裁剪）
            if x.shape != skip.shape:
                diff_y = skip.shape[2] - x.shape[2]
                diff_x = skip.shape[3] - x.shape[3]
                x = nn.functional.pad(x, [diff_x//2, diff_x - diff_x//2,
                                           diff_y//2, diff_y - diff_y//2])
            x = torch.cat([skip, x], dim=1)  # 跳跃连接拼接
            x = self.decoders[i](x)

        return self.final(x)

model = UNet(in_channels=1, out_channels=2)
x = torch.randn(1, 1, 572, 572)
y = model(x)
print(f"U-Net输出: {y.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **医学图像分割的标准架构**：U-Net是医学图像分割领域引用量最高的论文之一，几乎所有医学影像分割任务（CT、MRI、显微镜图像）都以U-Net或其变体为基础，如3D U-Net、Attention U-Net、nnU-Net（自动配置的U-Net框架）。
- **编码器-解码器设计的通用性**：U-Net的编码器-解码器+跳跃连接设计被广泛用于各类密集预测任务：深度估计（Monodepth2）、图像修复（Image Inpainting）、图像去噪、超分辨率重建等。
- **跳跃连接的泛化**：U-Net的跳跃连接思想启发了后续大量特征融合设计——FPN使用"相加"融合（而非拼接），DenseNet的密集连接（拼接所有前层特征），残差网络中的恒等快捷连接。
