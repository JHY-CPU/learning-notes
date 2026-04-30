# 34_超分辨率重建 (Super-Resolution) 基础

## 核心概念

- **超分辨率（Super-Resolution, SR）定义**：从低分辨率（LR）图像重建高分辨率（HR）图像的过程。这是一个病态逆问题——同一个LR图像可能对应多个合理的HR图像。
- **SRCNN（Super-Resolution CNN）**：Dong et al. (2014) 首次将CNN用于超分辨率，使用一个三层CNN（Patch提取+非线性映射+重建）直接从LR图像插值后预测HR图像。
- **上采样方法**：SRCNN使用双三次插值先放大LR图像（"预上采样"），再通过CNN细化。后续SRResNet和ESPCN使用"后上采样"——先提取特征，最后在网络的末尾进行上采样。
- **亚像素卷积（Pixel Shuffle）**：ESPCN提出的高效上采样方法——通过卷积生成 $r^2 \times C$ 个通道的特征图（$r$ 是上采样倍数），再用Pixel Shuffle操作将空间重排为 $H \times W \times r^2 \to rH \times rW \times C$。
- **感知损失（Perceptual Loss）**：使用预训练VGG网络的特征空间的MSE损失替代像素空间的MSE损失，使重建结果在语义上更真实。
- **GAN用于超分辨率（SRGAN）**：引入对抗训练，生成器产生"看起来真实"的高分辨率图像，判别器区分真实HR和生成的SR图像，使结果在感知质量上大幅提升。

## 数学推导

**SRCNN的架构（三层卷积）：**
$$
Y_1 = \max(0, W_1 * X + B_1)
$$
$$
Y_2 = \max(0, W_2 * Y_1 + B_2)
$$
$$
Y_3 = W_3 * Y_2 + B_3
$$

其中 $X$ 是插值放大后的LR图像，$Y_3$ 是输出的HR图像。$W_1$: $9\times9$, $W_2$: $1\times1$（实际上是 $5\times5$）, $W_3$: $5\times5$。

**PSNR（峰值信噪比）指标：**
$$
\text{PSNR} = 10 \cdot \log_{10} \left(\frac{I_{max}^2}{\text{MSE}}\right)
$$

其中 $I_{max}$ 是最大像素值（通常为255），MSE是HR和SR之间的均方误差。PSNR越高表示重建越精确。但PSNR与人类感知质量并不完全一致。

**亚像素卷积（Pixel Shuffle）：**
$$
\mathcal{PS}(T)_{x,y,c} = T_{\lfloor x/r \rfloor, \lfloor y/r \rfloor, c \cdot r \cdot \text{mod}(y,r) + c \cdot \text{mod}(x,r) + c}
$$

输入张量 $T$ 形状为 $(H, W, C \cdot r^2)$，输出形状为 $(rH, rW, C)$。

**ESPCN的高效上采样：**
特征提取（卷积层，在LR空间进行）→ 亚像素卷积上采样 → HR输出

相比SRCNN（先插值放大再卷积），ESPCN的计算量减少 $r^2$ 倍，因为卷积在低分辨率空间进行。

## 直观理解

超分辨率可以想象为"看图猜细节"。低分辨率图像丢失了高频细节信息（如纹理、边缘），超分辨率任务就是根据低分辨率图像中的"线索"（如物体的形状、颜色过渡），合理推测出丢失的高频细节。

SRCNN的预上采样方法，就像先用傻瓜放大镜把图片放大（双三次插值），再用专家手绘师（CNN）对其中的模糊部分进行精细描绘。ESPCN的亚像素卷积则更加巧妙——它不是在放大后的图像上操作，而是在小图上分析，最后"重新排列像素"形成大图，就像用马赛克拼出大幅画作。

## 代码示例

```python
import torch
import torch.nn as nn

class SRCNN(nn.Module):
    """SRCNN: 预上采样超分辨率网络"""
    def __init__(self, num_channels=1):
        super().__init__()
        self.conv1 = nn.Conv2d(num_channels, 64, kernel_size=9, padding=4)
        self.conv2 = nn.Conv2d(64, 32, kernel_size=1, padding=0)
        self.conv3 = nn.Conv2d(32, num_channels, kernel_size=5, padding=2)
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        # x 是经过插值放大的LR图像 (与目标HR同尺寸)
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.conv3(x)
        return x

class ESRCNN(nn.Module):
    """增强型：后上采样超分辨率 (使用Pixel Shuffle)"""
    def __init__(self, upscale_factor=2, in_channels=3):
        super().__init__()
        # 在低分辨率空间进行特征提取
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 64, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, in_channels * (upscale_factor ** 2), 3, padding=1),
        )
        # Pixel Shuffle 上采样
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

    def forward(self, x):
        # x: 原始LR图像 (未插值)
        x = self.features(x)
        x = self.pixel_shuffle(x)
        return x

# 测试
srcnn = SRCNN()
x_bicubic = torch.randn(1, 1, 224, 224)  # 插值后的LR
out = srcnn(x_bicubic)
print(f"SRCNN 输出: {out.shape}")

espcn = ESRCNN(upscale_factor=2)
x_lr = torch.randn(1, 3, 112, 112)  # 原始LR
out_espcn = espcn(x_lr)
print(f"ESPCN (x2) 输出: {out_espcn.shape}")  # (1, 3, 224, 224)

# 参数量对比
print(f"SRCNN参数量: {sum(p.numel() for p in srcnn.parameters()):,}")
print(f"ESPCN参数量: {sum(p.numel() for p in espcn.parameters()):,}")
```

## 深度学习关联

- **底层视觉的基础任务**：超分辨率是底层视觉领域的基础任务之一，其方法（特别是感知损失和GAN训练）被推广到图像去噪、去模糊、图像修复等任务中，形成了"图像复原"的统一框架。
- **从PSNR到感知质量的转变**：SRGAN标志着超分辨率评价从追求PSNR（数值指标）转向追求感知质量（人眼看起来更真实）。这一转变与GAN在图像生成领域的整体趋势一致，推动了LPIPS（Learned Perceptual Image Patch Similarity）等感知指标的发展。
- **扩散模型在超分辨率中的应用**：最近的SR3、Stable Diffusion Upscaler等使用扩散模型进行超分辨率，通过迭代去噪生成高质量的高分辨率图像，在感知质量上超越了GAN方法，代表了超分辨率的最新发展方向。
