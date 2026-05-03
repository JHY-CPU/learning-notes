# 07_StyleGAN：风格混合与自适应实例归一化

## 核心概念

- **StyleGAN**：NVIDIA 提出的先进 GAN 架构，通过解耦"内容"和"风格"实现了对生成图像的精细控制，是 2018-2020 年间图像生成质量最高的模型。
- **映射网络 (Mapping Network)**：将输入的潜码 $z \in \mathcal{Z}$ 映射到一个中间潜空间 $\mathcal{W}$，这个解耦的 $\mathcal{W}$ 空间使得不同特征之间的纠缠减少，更容易进行语义编辑。
- **自适应实例归一化 (AdaIN)**：将风格信息注入生成过程的核心机制，通过对特征图的均值和方差进行调制来实现风格控制：$\text{AdaIN}(x_i, y) = \sigma(y) \frac{x_i - \mu(x_i)}{\sigma(x_i)} + \mu(y)$。
- **风格混合 (Style Mixing)**：在生成过程中，使用不同的潜码控制不同分辨率层的风格——低分辨率层控制粗粒度特征（姿势、脸型），高分辨率层控制细粒度特征（肤色、纹理）。
- **噪声注入**：在每个层引入随机噪声，用于控制随机细节（头发纹理、毛孔、雀斑等），使生成图像更加自然多样。
- **渐进式增长**：StyleGAN 在训练时从低分辨率（4x4）逐渐过渡到高分辨率（1024x1024），稳定性显著提升。

## 数学推导

**AdaIN 公式**：

给定输入特征图 $x \in \mathbb{R}^{B \times C \times H \times W}$ 和风格特征 $y \in \mathbb{R}^{B \times C}$（来自风格映射网络的输出）：

$$
\text{AdaIN}(x_i, y) = \sigma(y) \left( \frac{x_i - \mu(x_i)}{\sigma(x_i)} \right) + \mu(y)
$$

其中 $x_i$ 是第 $i$ 个特征通道，$\mu(x_i)$ 和 $\sigma(x_i)$ 是该通道的均值和标准差，$\mu(y)$ 和 $\sigma(y)$ 是风格向量 $y$ 通过学习的仿射变换（Affine Transform）得到的尺度和偏置。

**风格解耦的量化**：

StyleGAN 用感知路径长度（Perceptual Path Length）和线性可分性（Linear Separability）衡量 $\mathcal{W}$ 空间的解耦程度：

$$
\text{Path Length} = \mathbb{E} \left[ \frac{1}{\epsilon^2} d(G(\text{lerp}(w_1, w_2, t)), G(\text{lerp}(w_1, w_2, t+\epsilon))) \right]
$$

较短的路径长度表示 $\mathcal{W}$ 空间更加平滑和解耦。

**Mixing Regularization**：训练时以一定概率使用两个不同的潜码 $z_1, z_2$ 分别控制粗层和细层，迫使网络不要依赖各层之间的相关性。

## 直观理解

- **映射网络的作用**：可以理解为将混乱的原始噪声 $z$（熵编码的 $\mathcal{Z}$ 空间）整理成有序的语义空间 $\mathcal{W}$——类似于把一堆乱放的工具分类放入工具箱的不同隔层。
- **AdaIN = 给 AI 画家调色**：实例归一化把当前图层"漂白"（归一化到标准均值和方差），然后用风格向量重新上色（调制均值和方差），就像在 Photoshop 中调整图层的颜色曲线。
- **风格混合**：想象一个人脸由不同艺术家绘制——脸型是毕加索的风格（粗层），皮肤纹理是梵高的笔触（中层），眼睛细节是达芬奇的技法（细层）。
- **噪声注入**相当于在画布上撒细沙——这些随机细节不影响整体构图，但让画面看起来更自然真实。
- **$\mathcal{W}$ 空间的线性性**：在 $\mathcal{W}$ 空间中做插值和向量运算可以得到平滑且语义一致的图像变换，这使得 StyleGAN 成为图像编辑的理想工具。

## 代码示例

```python
import torch
import torch.nn as nn

class AdaIN(nn.Module):
    """自适应实例归一化层"""
    def __init__(self, style_dim, channels):
        super().__init__()
        self.norm = nn.InstanceNorm2d(channels, affine=False)
        # 从风格向量学习仿射变换参数
        self.style_to_scale = nn.Linear(style_dim, channels)
        self.style_to_bias = nn.Linear(style_dim, channels)
    
    def forward(self, x, style):
        # x: 特征图 [B, C, H, W]
        # style: 风格向量 [B, style_dim]
        B, C, H, W = x.shape
        
        # 实例归一化
        x_norm = self.norm(x)
        
        # 从风格向量生成尺度和偏置
        scale = self.style_to_scale(style).view(B, C, 1, 1)
        bias = self.style_to_bias(style).view(B, C, 1, 1)
        
        # 调制
        return scale * x_norm + bias

class StyleGANGenerator(nn.Module):
    """简化的 StyleGAN 生成器"""
    def __init__(self, latent_dim=512, style_dim=512, img_channels=3):
        super().__init__()
        self.latent_dim = latent_dim
        
        # 映射网络：Z -> W
        self.mapping_network = nn.Sequential(
            nn.Linear(latent_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
            nn.LeakyReLU(0.2),
            nn.Linear(style_dim, style_dim),
        )
        
        # 合成网络简化版（只包含一层作为示例）
        # 实际有 6-18 层，从 4x4 逐步上采样到 1024x1024
        self.generator = nn.Sequential(
            nn.ConvTranspose2d(style_dim, 512, 4, 1, 0),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(512, 256, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(128, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.ConvTranspose2d(64, img_channels, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, z, return_w=False):
        w = self.mapping_network(z)  # Z -> W 映射
        # 简化：将 w reshape 为初始特征图
        img = self.generator(w.view(w.size(0), -1, 1, 1))
        if return_w:
            return img, w
        return img
    
    def style_mixing(self, z1, z2, mix_layer=3):
        """风格混合：不同层使用不同风格"""
        w1 = self.mapping_network(z1)
        w2 = self.mapping_network(z2)
        img1 = self.forward(z1)
        return img1  # 完整实现需要在每层注入不同的 w

# 验证
G = StyleGANGenerator()
z = torch.randn(4, 512)
img, w = G(z, return_w=True)
print(f"潜码形状: {z.shape}")
print(f"W 空间形状: {w.shape}")
print(f"生成图像形状: {img.shape}")

# W 空间插值示例
def interpolate_w(G, z1, z2, steps=8):
    """在 W 空间中进行插值"""
    w1 = G.mapping_network(z1)
    w2 = G.mapping_network(z2)
    imgs = []
    for alpha in torch.linspace(0, 1, steps):
        w_interp = (1 - alpha) * w1 + alpha * w2
        # 简化：实际生成时需要将 w 注入网络
        imgs.append(w_interp)
    return torch.stack(imgs)
```

## 深度学习关联

- **StyleGAN2**：改进了 StyleGAN 中的"水滴伪影"问题，使用权重解调（Weight Demodulation）替代 AdaIN，并引入了路径长度正则化（Path Length Regularization）。
- **StyleGAN3**：解决了"纹理粘连"问题，通过对网络进行傅里叶特征对齐和滤波，使得生成图像的细节在平移和旋转时能够自然地跟随。
- **EditGAN/GAN Inversion**：利用 StyleGAN 解耦的 $\mathcal{W}$ 空间进行图像编辑——将真实图像反演到 $\mathcal{W}$ 空间，然后修改特定维度的值来编辑属性。
- **Diffusion 时代的 StyleGAN**：虽然扩散模型在质量上超越了 StyleGAN，但 StyleGAN 的 $\mathcal{W}$ 空间仍然是图像编辑领域最受青睐的潜空间之一。
