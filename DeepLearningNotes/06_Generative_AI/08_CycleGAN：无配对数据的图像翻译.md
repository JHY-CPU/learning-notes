# 08_CycleGAN：无配对数据的图像翻译

## 核心概念
- **无配对图像翻译 (Unpaired Image-to-Image Translation)**：在不需要成对训练数据的情况下，将图像从一个域转换到另一个域。
- **循环一致性 (Cycle Consistency)**：核心思想是 $x \rightarrow G(x) \rightarrow F(G(x)) \approx x$，即图像翻译到目标域后再翻译回来应该与原始图像一致，这提供了无配对训练所需的监督信号。
- **两个生成器 + 两个判别器**：CycleGAN 包含两个生成器 $G: X \rightarrow Y$ 和 $F: Y \rightarrow X$，以及两个判别器 $D_Y$（区分 $Y$ 域真假）和 $D_X$（区分 $X$ 域真假）。
- **循环一致性损失**：$\mathcal{L}_{\text{cyc}}(G,F) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[\|G(F(y)) - y\|_1]$。
- **身份一致性损失 (Identity Loss)**：可选损失，要求 $G(y) \approx y$ 和 $F(x) \approx x$，确保输入输出颜色/风格一致。
- **应用场景**：将照片转换为莫奈风格画作、把斑马变成马、夏季照片变冬季、卫星图转换为地图等。

## 数学推导

**CycleGAN 完整目标函数**：

对抗损失（两个方向）：

$$
\mathcal{L}_{\text{GAN}}(G, D_Y, X, Y) = \mathbb{E}_{y \sim p_{\text{data}}(y)}[\log D_Y(y)] + \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log(1 - D_Y(G(x)))]
$$

$$
\mathcal{L}_{\text{GAN}}(F, D_X, Y, X) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\log D_X(x)] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[\log(1 - D_X(F(y)))]
$$

循环一致性损失：

$$
\mathcal{L}_{\text{cyc}}(G, F) = \mathbb{E}_{x \sim p_{\text{data}}(x)}[\|F(G(x)) - x\|_1] + \mathbb{E}_{y \sim p_{\text{data}}(y)}[\|G(F(y)) - y\|_1]
$$

总目标（$\lambda$ 控制循环一致性的权重，通常设为 10）：

$$
\mathcal{L}(G, F, D_X, D_Y) = \mathcal{L}_{\text{GAN}}(G, D_Y) + \mathcal{L}_{\text{GAN}}(F, D_X) + \lambda \mathcal{L}_{\text{cyc}}(G, F)
$$

**为什么 L1 优于 L2**：实验表明 $\ell_1$ 范数比 $\ell_2$ 范数产生更少的模糊结果，因为 $\ell_2$ 损失倾向于惩罚大误差而允许较小的高频误差，导致图像模糊。

## 直观理解
- **无配对图像翻译**就像你描述一种味道（草莓味）给没有吃过草莓的人，让他调出草莓味，然后你再尝他调出的味道，告诉他是否像草莓——即使没有"草莓标准样本"也可以迭代改进。
- **循环一致性**的核心洞察：虽然我们不知道 $X$ 域的图对应 $Y$ 域的哪张图，但我们知道"变过去再变回来"应该保持不变。这个朴素的约束提供了强大的监督信号。
- 形象地说：把一张橘猫的照片变成虎斑猫（不知道具体对应关系），再变回橘猫——如果变回来的是原来的橘猫，说明变虎斑猫的过程做对了。
- **CycleGAN 的局限性**：它倾向于保留图像的整体布局和结构，因为大幅改变布局会导致循环一致性损失过大——这就是为什么 CycleGAN 不太擅长"猫变狗"（形状变化大）但擅长"夏天变冬天"（色彩/纹理变化）。

## 代码示例

```python
import torch
import torch.nn as nn

# 简化版的 CycleGAN 生成器（ResNet 风格）
class ResidualBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, 1, 1),
            nn.InstanceNorm2d(channels),
        )
    
    def forward(self, x):
        return x + self.block(x)

class CycleGANGenerator(nn.Module):
    """CycleGAN 生成器：编码器-转换器-解码器"""
    def __init__(self, in_channels=3, out_channels=3, n_residual=6):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 7, 1, 3),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, 2, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 256, 3, 2, 1),
            nn.InstanceNorm2d(256),
            nn.ReLU(inplace=True),
        )
        # 转换器（ResNet 残差块）
        self.transformer = nn.Sequential(
            *[ResidualBlock(256) for _ in range(n_residual)]
        )
        # 解码器
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(256, 128, 3, 2, 1, 1),
            nn.InstanceNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 3, 2, 1, 1),
            nn.InstanceNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, out_channels, 7, 1, 3),
            nn.Tanh()
        )
    
    def forward(self, x):
        enc = self.encoder(x)
        trans = self.transformer(enc)
        dec = self.decoder(trans)
        return dec

class CycleGANPatchDiscriminator(nn.Module):
    """70x70 PatchGAN 判别器"""
    def __init__(self, in_channels=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, 64, 4, 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 1, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1, 4, 1, 1),
        )
    
    def forward(self, x):
        return self.net(x)

# 循环一致性损失计算
def cycle_consistency_loss(G, F, real_x, real_y):
    """计算循环一致性损失"""
    # X -> Y -> X
    fake_y = G(real_x)
    recovered_x = F(fake_y)
    
    # Y -> X -> Y
    fake_x = F(real_y)
    recovered_y = G(fake_x)
    
    # L1 循环一致性损失
    loss_forward = torch.mean(torch.abs(recovered_x - real_x))
    loss_backward = torch.mean(torch.abs(recovered_y - real_y))
    
    return loss_forward + loss_backward

# 初始化
G_XtoY = CycleGANGenerator()  # 将 X 域图片转换为 Y 域
F_YtoX = CycleGANGenerator()  # 将 Y 域图片转换为 X 域
D_X = CycleGANPatchDiscriminator()  # 判断 X 域真假
D_Y = CycleGANPatchDiscriminator()  # 判断 Y 域真假

print("CycleGAN 初始化完成")
print(f"生成器参数量: {sum(p.numel() for p in G_XtoY.parameters()):,}")
print(f"判别器参数量: {sum(p.numel() for p in D_X.parameters()):,}")

# 验证输出形状
x = torch.randn(1, 3, 256, 256)
y_fake = G_XtoY(x)
x_recovered = F_YtoX(y_fake)
print(f"输入: {x.shape} -> 翻译后: {y_fake.shape} -> 循环恢复: {x_recovered.shape}")
```

## 深度学习关联
- **CUT (Contrastive Unpaired Translation)**：CycleGAN 的改进版本，用对比学习（InfoNCE 损失）替代或补充循环一致性损失，通过最大化输入和输出图像块之间的互信息来实现更精准的翻译。
- **UNIT/MUNIT**：基于 VAE 和共享潜空间假设的无配对图像翻译方法，认为不同域的图像可以映射到共享的内容空间，区别仅在于风格编码。
- **Stable Diffusion 中的 Image-to-Image**：现代文本到图像模型（如 SDEdit）通过扩散模型的反向去噪过程实现图像翻译，不需要配对训练数据，灵活度远高于 CycleGAN。
- **视频翻译**：RecycleGAN 等扩展将循环一致性引入视频领域，通过时序约束确保帧间的连续性。
