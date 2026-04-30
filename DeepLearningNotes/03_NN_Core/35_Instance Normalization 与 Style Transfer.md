# 35 Instance Normalization 与 Style Transfer

## 核心概念

- **Instance Normalization 定义**：Instance Normalization（IN）对单个样本的单个通道进行标准化。对输入 $x \in \mathbb{R}^{C \times H \times W}$，IN 对每个通道 $c$ 独立计算均值和方差：$\mu_c = \frac{1}{HW}\sum_{h,w} x_{c,h,w}$。

- **与 BN/LN 的区别**：BN 在 batch 维度标准化，LN 在特征维度标准化，IN 在空间维度（每个通道）标准化。IN 假设不同样本和不同通道应该独立处理，特别适合图像风格化任务。

- **风格迁移中的核心作用**：IN 被发现在图像风格迁移中效果显著优于 BN。原因是风格迁移的目标是改变图像的"风格"（颜色、纹理等统计信息），IN 通过移除每个通道的均值和方差，可以消除原始图像的风格信息，使模型更容易学习目标风格。

- **对比 BN 的差异**：BN 会"模糊"不同样本之间的风格差异（因为统计量跨样本计算），而 IN 保留并独立处理每个样本的风格信息。这正是风格迁移所需——对不同输入图像的风格进行独立归一化。

## 数学推导

**IN 前向传播**：

对输入 $x \in \mathbb{R}^{N \times C \times H \times W}$，对每个样本 $n$ 的每个通道 $c$：

$$
\mu_{nc} = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} x_{nchw}
$$

$$
\sigma_{nc}^2 = \frac{1}{HW} \sum_{h=1}^{H} \sum_{w=1}^{W} (x_{nchw} - \mu_{nc})^2
$$

$$
\hat{x}_{nchw} = \frac{x_{nchw} - \mu_{nc}}{\sqrt{\sigma_{nc}^2 + \epsilon}}
$$

$$
y_{nchw} = \gamma_c \hat{x}_{nchw} + \beta_c
$$

**四种归一化方式的对比**（对 $N \times C \times H \times W$ 张量）：

| 方法 | 统计量维度 | 归一化维度 | 适用场景 |
|------|-----------|-----------|---------|
| BN | $N \times H \times W$ | $C$ | CNN 通用 |
| LN | $C \times H \times W$ | $N$ | Transformer |
| IN | $H \times W$ | $N \times C$ | 风格迁移 |
| GN | $G \times H \times W$ | $N \times (C/G)$ | 小 batch |

**Adaptive Instance Normalization (AdaIN)**：

AdaIN 将 IN 扩展到风格迁移中，将输入内容的均值和方差与目标风格对齐：

$$
\text{AdaIN}(x, y) = \sigma(y) \cdot \frac{x - \mu(x)}{\sigma(x)} + \mu(y)
$$

其中 $x$ 是内容图像的特征，$y$ 是风格图像的特征。AdaIN 将 $x$ 的通道统计量替换为 $y$ 的统计量，实现了内容的风格化。

## 直观理解

Instance Normalization 可以理解为"移除风格信息"的操作。在图像中，一个通道的均值和方差大致对应了该通道的亮度和对比度——这可以看作是一种"风格"信息。IN 通过标准化移除这些信息，保留的是"内容"信息（边缘、形状等结构）。

在风格迁移中，AdaIN 做了两件事：
1. **移除内容图的风格**：用 IN 标准化的内容特征
2. **注入目标风格的统计量**：将标准化后的内容特征乘以目标风格的方差再加目标风格的均值

这就像是"换装"——先脱掉自己的衣服（移除原始风格），再穿上目标服装（注入目标风格）。内容（人的体型）保持不变，但外观（风格）彻底改变。

IN 与 BN 的核心区别：BN 在 batch 上计算均值/方差，假设 batch 中的样本共享统计量。但在风格迁移中，不同图像可能有完全不同的风格，这个假设不成立。IN 独立处理每个实例，更为合理。

## 代码示例

```python
import torch
import torch.nn as nn

# 手动实现 Instance Normalization
class InstanceNormManual(nn.Module):
    def __init__(self, num_channels, eps=1e-5, affine=True):
        super().__init__()
        self.num_channels = num_channels
        self.eps = eps
        self.affine = affine

        if affine:
            self.gamma = nn.Parameter(torch.ones(num_channels))
            self.beta = nn.Parameter(torch.zeros(num_channels))
        else:
            self.register_parameter('gamma', None)
            self.register_parameter('beta', None)

    def forward(self, x):
        # x: (N, C, H, W)
        # 在 (H, W) 维度上计算均值和方差
        mean = x.mean(dim=(2, 3), keepdim=True)  # (N, C, 1, 1)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        x_norm = (x - mean) / (var + self.eps).sqrt()

        if self.affine:
            shape = (1, self.num_channels, 1, 1)
            x_norm = x_norm * self.gamma.view(*shape) + self.beta.view(*shape)
        return x_norm

# 验证手动实现与 PyTorch 内置 IN 的一致性
torch.manual_seed(42)
x = torch.randn(4, 3, 32, 32)

in_manual = InstanceNormManual(3)
in_pytorch = nn.InstanceNorm2d(3, affine=True)
in_pytorch.weight.data = in_manual.gamma.data.clone()
in_pytorch.bias.data = in_manual.beta.data.clone()

out_manual = in_manual(x)
out_pytorch = in_pytorch(x)
print(f"IN 输出一致: {torch.allclose(out_manual, out_pytorch, atol=1e-5)}")

# 演示 BN vs IN 在风格迁移场景的差异
print("\nBN vs IN 每个样本的统计量:")

x = torch.randn(4, 3, 8, 8)  # batch=4

bn = nn.BatchNorm2d(3)
inn = nn.InstanceNorm2d(3)

with torch.no_grad():
    bn_out = bn(x)
    in_out = inn(x)

    for i in range(4):
        print(f"  样本 {i}:")
        print(f"    原始: mean={x[i].mean():.4f}, std={x[i].std():.4f}")
        print(f"    BN后: mean={bn_out[i].mean():.4f}, std={bn_out[i].std():.4f}")
        print(f"    IN后: mean={in_out[i].mean():.4f}, std={in_out[i].std():.4f}")

# Adaptive Instance Normalization (AdaIN) 实现
class AdaIN(nn.Module):
    """Adaptive Instance Normalization"""
    def forward(self, content, style):
        # content, style: (N, C, H, W)
        # 计算 content 的均值和方差
        content_mean = content.mean(dim=(2, 3), keepdim=True)
        content_std = content.std(dim=(2, 3), keepdim=True)

        # 计算 style 的均值和方差
        style_mean = style.mean(dim=(2, 3), keepdim=True)
        style_std = style.std(dim=(2, 3), keepdim=True)

        # 将 content 的风格替换为 style 的风格
        normalized = (content - content_mean) / (content_std + 1e-5)
        stylized = normalized * style_std + style_mean
        return stylized

# 演示 AdaIN
content = torch.randn(2, 3, 64, 64)  # 内容图像特征
style = torch.randn(2, 3, 64, 64)    # 风格图像特征

adain = AdaIN()
output = adain(content, style)

print(f"\nAdaIN 测试:")
print(f"  输出 shape: {output.shape}")
print(f"  原始 content mean: {content[0, 0].mean():.4f}")
print(f"  原始 style mean: {style[0, 0].mean():.4f}")
print(f"  AdaIN 输出 mean: {output[0, 0].mean():.4f}")
```

## 深度学习关联

- **风格迁移的核心组件**：Instance Normalization 是图像风格迁移的核心技术。从最早的 Neural Style Transfer（Gatys et al.）到实时的 Adaptive Style Transfer（Huang & Belongie, 2017），IN 及其变体（AdaIN）始终是风格迁移的标准组件。

- **GAN 中的应用**：IN 被广泛应用于生成对抗网络（GAN）的生成器中。StyleGAN 系列使用了调制解调（Modulation/Demodulation）机制，这本质上可以看作是 IN 的一种特殊形式。IN 帮助生成器更好地解耦内容和风格。

- **CycleGAN 等图像翻译模型**：在 CycleGAN、UNIT、MUNIT 等无监督图像翻译模型中，IN 被用于将图像表示为"内容编码 + 风格编码"的形式。内容编码通过 IN 标准化获得，风格编码通过统计量表示，使得模型可以灵活地组合不同图像的风格和内容。
