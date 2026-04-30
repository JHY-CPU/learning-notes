# 16_MobileNet V1-V2-V3：移动端轻量化设计

## 核心概念

- **MobileNet V1 核心：深度可分离卷积**：将标准卷积分解为深度卷积（Depthwise Conv）+ 逐点卷积（Pointwise Conv $1\times1$），计算量减少约 88.3%（$3\times3$ 核，64通道）。详见第7章。
- **宽度乘数（Width Multiplier $\alpha$）**：缩放通道数，使网络在精度和速度之间灵活调节。如 $\alpha=0.75$ 表示将通道数缩减为原来的 75%。计算量与 $\alpha^2$ 成正比。
- **分辨率乘数（Resolution Multiplier $\rho$）**：缩放输入图像分辨率，进一步调节计算量。计算量与 $\rho^2$ 成正比。
- **MobileNet V2 创新：倒置残差 + 线性瓶颈**：在深度卷积前后使用 $1\times1$ 卷积进行"扩张-过滤-压缩"。不同于ResNet先压缩再扩张，V2先扩张（6倍）后压缩。瓶颈层使用线性激活（无ReLU），因为ReLU会破坏低维信息。
- **MobileNet V3 创新：NAS搜索 + SE注意力 + Hard-Swish**：使用NetAdapt和MnasNet自动搜索最佳结构，引入Squeeze-and-Excitation通道注意力，使用Hard-Swish激活函数（$x \cdot \text{ReLU6}(x+3)/6$）提升移动端推理速度。
- **计算量 vs 参数量 vs 延迟**：MobileNet系列始终关注实际推理延迟（latency）而非仅看FLOPs，因为FLOPs与实际推理速度并不完全一致（受内存访问成本、缓存效率等因素影响）。

## 数学推导

**MobileNet V1 计算量公式：**
$$
\text{Cost} = H \times W \times (\alpha C_{in}) \times K^2 \quad (\text{深度卷积})
$$
$$
+ H \times W \times (\alpha C_{in}) \times (\alpha C_{out}) \quad (\text{逐点卷积})
$$

**MobileNet V2 倒置残差块的计算量：**
输入 $C_{in}$ → $1\times1$ 扩张 $t$ 倍 → 深度卷积 $3\times3$ → $1\times1$ 压缩回 $C_{out}$：
$$
\text{Cost} = H \times W \times C_{in} \times (t \cdot C_{in}) \quad (\text{扩张})
$$
$$
+ H \times W \times (t \cdot C_{in}) \times K^2 \quad (\text{深度卷积})
$$
$$
+ H \times W \times (t \cdot C_{in}) \times C_{out} \quad (\text{压缩})
$$

其中 $t$ 是扩张系数（通常 $t=6$），使得中间层的通道数是输入/输出的6倍。

**线性瓶颈的含义：**
假设输入是 $d$ 维的低维流形，经过ReLU后，若 $d$ 较小则信息可能被破坏：
$$
\text{ReLU}(x) \text{ 将负值置零} \Rightarrow \text{信息丢失}
$$
因此在瓶颈层（低维）使用线性激活，保留完整信息。

## 直观理解

MobileNet V2的倒置残差结构像一个"沙漏"变形——不是传统的"先压缩再扩张"（沙漏），而是"先扩张再压缩"（菱形）。这么做的理由是：深度卷积在低维空间做特征提取效率低（信息不够丰富），所以先用 $1\times1$ 卷积将通道扩展到6倍，在"信息充足"的高维空间做深度卷积提取特征，再压缩回去。

这与ResNet的bottleneck形成鲜明对比——ResNet是先压缩（为了减少计算量）再扩张，而MobileNet V2是先扩张（为了让深度卷积有足够信息）再压缩（为了控制输出大小）。

## 代码示例

```python
import torch
import torch.nn as nn

class InvertedResidual(nn.Module):
    """MobileNet V2 倒置残差块"""
    def __init__(self, in_channels, out_channels, stride, expand_ratio):
        super().__init__()
        hidden_dim = round(in_channels * expand_ratio)
        self.use_res_connect = stride == 1 and in_channels == out_channels

        layers = []
        if expand_ratio != 1:
            # 1x1 扩张
            layers.extend([
                nn.Conv2d(in_channels, hidden_dim, 1, bias=False),
                nn.BatchNorm2d(hidden_dim),
                nn.ReLU6(inplace=True)
            ])
        # 3x3 深度卷积
        layers.extend([
            nn.Conv2d(hidden_dim, hidden_dim, 3, stride, padding=1,
                      groups=hidden_dim, bias=False),  # groups=hidden_dim 实现深度卷积
            nn.BatchNorm2d(hidden_dim),
            nn.ReLU6(inplace=True)
        ])
        # 1x1 线性压缩 (无激活函数)
        layers.extend([
            nn.Conv2d(hidden_dim, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels)
        ])

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        return self.conv(x)

# 验证
block = InvertedResidual(32, 64, stride=2, expand_ratio=6)
x = torch.randn(1, 32, 56, 56)
out = block(x)
print(f"倒置残差输出: {out.shape}")
params = sum(p.numel() for p in block.parameters())
print(f"参数量: {params}")
```

## 深度学习关联

- **移动视觉的基石**：MobileNet系列使深度学习视觉应用在手机、嵌入式设备等资源受限平台上成为可能，催生了移动端拍照优化、实时视频分析等应用场景。
- **架构搜索（NAS）的工程实践**：MobileNet V3是将神经架构搜索（NAS）应用于实际产品中的典型案例，证明了自动搜索的网络结构在效率上可以超越人工设计。这推动了后续EfficientNet（使用NAS搜索基线网络）和RegNet（使用设计空间分析）的发展。
- **轻量化设计原则的普适性**：深度可分离卷积和倒置残差的设计思想被广泛应用于YOLO的轻量检测头、语义分割的快速骨干网络（如Fast-SCNN、BiSeNet），以及Transformer轻量化（MobileViT、EfficientFormer）等领域。
