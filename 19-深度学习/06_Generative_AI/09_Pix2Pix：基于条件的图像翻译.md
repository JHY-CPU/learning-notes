# 09_Pix2Pix：基于条件的图像翻译

## 核心概念

- **条件 GAN (cGAN)**：Pix2Pix 基于 cGAN 框架，生成器接收输入图像（条件）而不是随机噪声，判别器同时看到输入和输出对来判断是否"匹配"。
- **配对数据训练**：与 CycleGAN 不同，Pix2Pix 需要成对的训练数据（如：边缘图 ↔ 实物照片、卫星图 ↔ 地图、黑白图 ↔ 彩色图）。
- **U-Net 生成器**：生成器采用 U-Net 架构——编码器-解码器之间有跳跃连接（Skip Connections），保留输入图像的底层细节。
- **PatchGAN 判别器**：判别器不对整张图做判断，而是对 70x70 的图像块做真假判断，最后取平均。这强制生成器关注高频细节，而 L1 损失负责低频结构。
- **联合损失函数**：L1 损失 + cGAN 对抗损失。L1 损失确保生成图像在结构上与输入接近，cGAN 损失确保生成图像真实。
- **多任务能力**：同一个 Pix2Pix 框架可以处理多种任务——语义分割→照片、黑白上色、草图→实物等。

## 数学推导

**Pix2Pix 目标函数**：

cGAN 对抗损失：

$$
\mathcal{L}_{\text{cGAN}}(G, D) = \mathbb{E}_{x, y}[\log D(x, y)] + \mathbb{E}_{x, z}[\log(1 - D(x, G(x, z)))]
$$

其中 $x$ 是输入图像（条件），$y$ 是目标图像，$G(x, z)$ 是生成器的输出。

L1 损失（结构保真度）：

$$
\mathcal{L}_{L1}(G) = \mathbb{E}_{x, y, z}[\|y - G(x, z)\|_1]
$$

总损失：

$$
G^* = \arg\min_G \max_D \mathcal{L}_{\text{cGAN}}(G, D) + \lambda \mathcal{L}_{L1}(G)
$$

其中 $\lambda$ 控制 L1 损失的权重（通常设为 100），因为实验表明仅靠对抗损失生成的图像在结构上不够准确。

**PatchGAN 的原理**：判别器 $D$ 将输入图像分割为 $N \times N$ 个 Patch，对每个 Patch 独立判断真假：

$$
\mathcal{L}_{\text{PatchGAN}}(D) = \frac{1}{N^2} \sum_{i,j} \left[ \log D(x_{ij}, y_{ij}) + \log(1 - D(x_{ij}, G(x)_{ij})) \right]
$$

当 Patch 大小为 70x70 时，判别器的感受野能覆盖关键的高频纹理信息，而结构信息由 L1 损失提供。

## 直观理解

- **Pix2Pix = 在监督下临摹**：U-Net 生成器像是一个画家，在给定线稿（输入条件）的情况下填色渲染。L1 损失像导师检查"形状是否像"，而判别器检查"是否真实自然"。
- **为什么需要 U-Net 的跳跃连接**：如果没有跳跃连接，信息需要经过"压缩-解压"的瓶颈，会丢失输入的精细结构——就像是画家只看了一眼线稿就要默画出来，跳跃连接相当于画家随时可以回头再看线稿。
- **PatchGAN 为什么有效**：整图判别器容易被"整体图像分布"欺骗，而 PatchGAN 关注每一个局部区域——这等同于说"你不仅要整体画得像，每个局部细节也要经得起推敲"。
- **为什么加 L1 损失**：纯 cGAN 生成的图像常常有"创造性失真"（比如把边缘理解错了），L1 损失像一个锚，把生成器固定在"不能偏离输入太远"的范围内。

## 代码示例

```python
import torch
import torch.nn as nn

# U-Net 生成器（简化版）
class UNetBlock(nn.Module):
    """U-Net 的下采样/上采样块"""
    def __init__(self, in_ch, out_ch, down=True, use_dropout=False):
        super().__init__()
        if down:
            self.block = nn.Sequential(
                nn.Conv2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.LeakyReLU(0.2, inplace=True),
            )
        else:
            layers = [
                nn.ConvTranspose2d(in_ch, out_ch, 4, 2, 1, bias=False),
                nn.BatchNorm2d(out_ch),
                nn.ReLU(inplace=True),
            ]
            if use_dropout:
                layers.append(nn.Dropout(0.5))
            self.block = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.block(x)

class Pix2PixGenerator(nn.Module):
    """U-Net 生成器：128x128 输入输出"""
    def __init__(self, in_channels=3, out_channels=3):
        super().__init__()
        # 编码器（下采样）
        self.e1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1), nn.LeakyReLU(0.2))
        self.e2 = UNetBlock(64, 128, down=True)
        self.e3 = UNetBlock(128, 256, down=True)
        self.e4 = UNetBlock(256, 512, down=True)
        self.e5 = UNetBlock(512, 512, down=True)
        self.e6 = UNetBlock(512, 512, down=True)
        self.e7 = UNetBlock(512, 512, down=True)
        self.e8 = nn.Sequential(
            nn.Conv2d(512, 512, 4, 2, 1), nn.ReLU())
        
        # 解码器（上采样）+ 跳跃连接
        self.d1 = UNetBlock(512, 512, down=False, use_dropout=True)
        self.d2 = UNetBlock(1024, 512, down=False, use_dropout=True)
        self.d3 = UNetBlock(1024, 512, down=False)
        self.d4 = UNetBlock(1024, 512, down=False)
        self.d5 = UNetBlock(1024, 256, down=False)
        self.d6 = UNetBlock(512, 128, down=False)
        self.d7 = UNetBlock(256, 64, down=False)
        self.d8 = nn.Sequential(
            nn.ConvTranspose2d(128, out_channels, 4, 2, 1),
            nn.Tanh())
    
    def forward(self, x):
        # 编码
        e1 = self.e1(x)
        e2 = self.e2(e1)
        e3 = self.e3(e2)
        e4 = self.e4(e3)
        e5 = self.e5(e4)
        e6 = self.e6(e5)
        e7 = self.e7(e6)
        e8 = self.e8(e7)
        # 解码 + 跳跃连接
        d1 = self.d1(e8)
        d1 = torch.cat([d1, e7], dim=1)
        d2 = self.d2(d1)
        d2 = torch.cat([d2, e6], dim=1)
        d3 = self.d3(d2)
        d3 = torch.cat([d3, e5], dim=1)
        d4 = self.d4(d3)
        d4 = torch.cat([d4, e4], dim=1)
        d5 = self.d5(d4)
        d5 = torch.cat([d5, e3], dim=1)
        d6 = self.d6(d5)
        d6 = torch.cat([d6, e2], dim=1)
        d7 = self.d7(d6)
        d7 = torch.cat([d7, e1], dim=1)
        d8 = self.d8(d7)
        return d8

class PatchGANDiscriminator(nn.Module):
    """70x70 PatchGAN 判别器"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels * 2, 64, 4, 2, 1),  # 输入和条件拼接
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )
    
    def forward(self, x, cond):
        # 沿通道维度拼接输入和条件
        x = torch.cat([x, cond], dim=1)
        return self.net(x)

# Pix2Pix 损失函数
def pix2pix_loss(G, D, real_x, real_y, lambda_l1=100):
    """计算 Pix2Pix 的生成器和判别器损失"""
    # 生成器前向
    fake_y = G(real_x)
    
    # 判别器判断
    d_real = D(real_y, real_x)
    d_fake = D(fake_y.detach(), real_x)
    
    # 判别器损失（LSGAN 形式：最小二乘损失）
    loss_D = 0.5 * (torch.mean((d_real - 1) ** 2) + torch.mean(d_fake ** 2))
    
    # 生成器损失
    d_fake_for_G = D(fake_y, real_x)
    loss_G_adv = 0.5 * torch.mean((d_fake_for_G - 1) ** 2)
    loss_G_l1 = torch.mean(torch.abs(fake_y - real_y)) * lambda_l1
    loss_G = loss_G_adv + loss_G_l1
    
    return loss_D, loss_G, loss_G_adv.item(), loss_G_l1.item()

# 初始化
G = Pix2PixGenerator()
D = PatchGANDiscriminator()
x = torch.randn(1, 3, 128, 128)  # 输入条件（如边缘图）
y = torch.randn(1, 3, 128, 128)  # 目标图像（如照片）
output = G(x)
print(f"输入: {x.shape} -> 输出: {output.shape}")
print(f"生成器参数: {sum(p.numel() for p in G.parameters()):,}")
print(f"判别器参数: {sum(p.numel() for p in D.parameters()):,}")
```

## 深度学习关联

- **pix2pixHD**：Pix2Pix 的高分辨率版本，使用多尺度生成器和判别器，以及实例图（Instance Map）作为额外条件，支持 2048x1024 分辨率的图像翻译。
- **SPADE ( spatially-adaptive denormalization)**：在 Pix2Pix 的基础上，用空间自适应的归一化层替代传统的 BN 层，使得语义布局信息更好地传递到生成器各层。
- **GauGAN / NVIDIA Canvas**：基于 SPADE 的交互式图像生成工具，用户画几笔语义标签（树、山、天空），GauGAN 就能生成逼真的风景照——这是 Pix2Pix 思想的产品化体现。
- **ControlNet**：在现代扩散模型中实现了类似于 Pix2Pix 的条件控制——通过将边缘图、深度图、姿态图等条件注入到预训练扩散模型中来实现可控生成。
