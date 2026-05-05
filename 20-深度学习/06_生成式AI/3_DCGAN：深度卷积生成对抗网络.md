# 04_DCGAN：深度卷积生成对抗网络

## 核心概念

- **DCGAN** (Deep Convolutional GAN)：将卷积神经网络引入 GAN 框架的开创性工作，标志着 GAN 能够稳定生成高质量的图像。
- **转置卷积 (Transposed Convolution)**：生成器中使用转置卷积（反卷积）将低维噪声映射到高维图像空间，逐步上采样。
- **架构约束**：DCGAN 提出了一系列架构设计准则——移除全连接层、使用 Batch Normalization、用 ReLU（生成器）和 LeakyReLU（判别器）作为激活函数。
- **潜空间算术**：DCGAN 展示了对潜空间 $z$ 的向量运算可以对应图像语义变换，例如"戴眼镜的男性 - 不戴眼镜的男性 + 不戴眼镜的女性 ≈ 戴眼镜的女性"。
- **卷积判别器**：判别器使用步长卷积（Strided Convolution）替代池化层进行下采样，让网络自己学习空间下采样方式。
- **无监督表征学习**：DCGAN 的判别器在训练后可以作为优秀的图像特征提取器，用于分类等下游任务。

## 数学推导

**转置卷积的前向传播**：

设输入特征图为 $X \in \mathbb{R}^{H_{in} \times W_{in} \times C_{in}}$，卷积核 $K \in \mathbb{R}^{k \times k \times C_{in} \times C_{out}}$，步长 $s$，填充 $p$，输出尺寸为：

$$
H_{out} = (H_{in} - 1) \times s - 2p + k
$$

$$
W_{out} = (W_{in} - 1) \times s - 2p + k
$$

**DCGAN 生成器结构**（以 64x64 图像为例）：

输入 $z \in \mathbb{R}^{100}$，经过投影和 reshape 得到 $4 \times 4 \times 1024$ 的特征图，然后经过 4 层转置卷积：

$$
4 \times 4 \times 1024 \xrightarrow{4\times4, s=2} 8 \times 8 \times 512 \xrightarrow{4\times4, s=2} 16 \times 16 \times 256 \xrightarrow{4\times4, s=2} 32 \times 32 \times 128 \xrightarrow{4\times4, s=2} 64 \times 64 \times 3
$$

**BN 层的训练与推理差异**：

训练时：$\hat{x} = \frac{x - \mu_B}{\sqrt{\sigma_B^2 + \epsilon}}$，推理时使用运行均值 $\mu_{\text{running}}$ 和运行方差 $\sigma_{\text{running}}^2$。

## 直观理解

- **生成器就像一个雕塑家**：先有大致的轮廓（低分辨率特征图），然后逐步细化细节（更高分辨率），最终完成一件精细作品。
- **转置卷积可以理解为"在学习如何上采样"**：不像双线性插值那样固定规则，转置卷积的参数是可学习的，网络自动学习最好的上采样方式。
- **潜空间算术**表明 GAN 的潜空间不是随机的噪声池，而是一个有结构的语义空间——类似于 word2vec 中词向量的语义算术性质。
- **Batch Normalization 的作用**：在对抗训练中，BN 防止了生成器或判别器的内部协变量偏移，相当于在"军备竞赛"中为双方都提供了稳定的内部状态。

## 代码示例

```python
import torch
import torch.nn as nn

class DCGANGenerator(nn.Module):
    """DCGAN 生成器：从噪声生成 64x64 图像"""
    def __init__(self, latent_dim=100, feature_map_size=64, channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        self.net = nn.Sequential(
            # 输入: latent_dim x 1 x 1
            nn.ConvTranspose2d(latent_dim, feature_map_size * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.ReLU(True),
            # 状态: (feature_map_size*8) x 4 x 4
            nn.ConvTranspose2d(feature_map_size * 8, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.ReLU(True),
            # 状态: (feature_map_size*4) x 8 x 8
            nn.ConvTranspose2d(feature_map_size * 4, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.ReLU(True),
            # 状态: (feature_map_size*2) x 16 x 16
            nn.ConvTranspose2d(feature_map_size * 2, feature_map_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size),
            nn.ReLU(True),
            # 状态: feature_map_size x 32 x 32
            nn.ConvTranspose2d(feature_map_size, channels, 4, 2, 1, bias=False),
            nn.Tanh()
            # 输出: channels x 64 x 64
        )
    
    def forward(self, z):
        z = z.view(z.size(0), self.latent_dim, 1, 1)
        return self.net(z)

class DCGANDiscriminator(nn.Module):
    """DCGAN 判别器：使用步长卷积下采样"""
    def __init__(self, feature_map_size=64, channels=3):
        super().__init__()
        self.net = nn.Sequential(
            # 输入: channels x 64 x 64
            nn.Conv2d(channels, feature_map_size, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: feature_map_size x 32 x 32
            nn.Conv2d(feature_map_size, feature_map_size * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_map_size*2) x 16 x 16
            nn.Conv2d(feature_map_size * 2, feature_map_size * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_map_size*4) x 8 x 8
            nn.Conv2d(feature_map_size * 4, feature_map_size * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(feature_map_size * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态: (feature_map_size*8) x 4 x 4
            nn.Conv2d(feature_map_size * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.net(x).view(-1, 1)

# 潜空间算术演示
def latent_space_arithmetic(G, z1, z2, z3):
    """演示潜空间向量运算"""
    # z1: 戴眼镜的男性, z2: 不戴眼镜的男性, z3: 不戴眼镜的女性
    z_result = z1 - z2 + z3  # 理论上 ≈ 戴眼镜的女性
    with torch.no_grad():
        img_result = G(z_result)
    return img_result

# 初始化
G = DCGANGenerator()
D = DCGANDiscriminator()
print(f"生成器参数: {sum(p.numel() for p in G.parameters()):,}")
print(f"判别器参数: {sum(p.numel() for p in D.parameters()):,}")

# 验证输出形状
z = torch.randn(1, 100)
img = G(z)
score = D(img)
print(f"输入噪声形状: {z.shape} -> 生成图像: {img.shape} -> 判别分数: {score.shape}")
```

## 深度学习关联

- **StyleGAN 的基础**：DCGAN 奠定了用卷积网络做图像生成的基础，StyleGAN 在此基础上引入了风格调制和自适应实例归一化（AdaIN）。
- **渐进式增长 (ProGAN)**：ProGAN 借鉴了 DCGAN 的结构，但将训练过程改为从低分辨率渐进增加到高分辨率，大幅提升了稳定性。
- **条件图像生成**：cGAN（条件 GAN）将条件信息（如类别标签）注入到 DCGAN 的生成器和判别器中，实现了可控生成。
- **图像超分辨率**：SRGAN 使用 DCGAN 架构进行超分辨率重建，用感知损失（Perceptual Loss）替代像素级损失，生成视觉上更真实的高分辨率图像。
