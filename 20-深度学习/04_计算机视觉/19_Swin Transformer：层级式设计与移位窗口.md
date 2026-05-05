# 19_Swin Transformer：层级式设计与移位窗口

## 核心概念

- **层级式特征图（Hierarchical Feature Map）**：与CNN类似，Swin Transformer通过Patch Merging逐层下采样，生成多尺度特征图（$H/4 \times W/4 \rightarrow H/8 \times W/8 \rightarrow H/16 \times W/16 \rightarrow H/32 \times W/32$），便于作为各种视觉任务的骨干网络。
- **移位窗口（Shifted Window）**：将自注意力计算限制在局部窗口内（如 $7\times7$ 个Patch），并在相邻层之间移动窗口的划分位置，实现跨窗口信息交流。
- **窗口注意力（Window Multi-Head Self-Attention, W-MSA）**：将特征图划分为不重叠的窗口，在每个窗口内独立计算自注意力，复杂度从 $\mathcal{O}(N^2)$ 降为 $\mathcal{O}(M^2 \cdot N/M^2) = \mathcal{O}(M^2 \cdot \text{num\_win})$。
- **移位窗口注意力（Shifted Window Attention, SW-MSA）**：在W-MSA之后，将窗口划分位置移动 $(\lfloor M/2 \rfloor, \lfloor M/2 \rfloor)$，使相邻层窗口覆盖不同区域，间接实现跨窗口信息融合。
- **相对位置偏置（Relative Position Bias）**：在注意力计算中添加一个可学习的相对位置编码，为自注意力提供位置信息。$B \in \mathbb{R}^{(2M-1) \times (2M-1)}$ 是每个相对位置对的偏置。
- **Patch Merging**：将 $2\times2$ 邻域的4个Patch拼接为1个Patch，通道数变为4倍，再通过线性层压缩为2倍。实现类似于CNN中步长2卷积的下采样效果。

## 数学推导

**标准自注意力 vs 窗口自注意力的复杂度：**
$$
\Omega(\text{MSA}) = 4N \cdot D^2 + 2N^2 \cdot D \quad (\text{全局注意力})
$$
$$
\Omega(\text{W-MSA}) = 4N \cdot D^2 + 2M^2 \cdot N \cdot D \quad (\text{窗口注意力})
$$

其中 $N = H/P \times W/P$ 是Patch总数，$M$ 是窗口大小（默认7），$D$ 是嵌入维度。

当 $N \gg M$ 时，W-MSA的复杂度从 $\mathcal{O}(N^2)$ 降为 $\mathcal{O}(N)$，对于高分辨率图像效率提升显著。

**相对位置偏置的注意力计算：**
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}} + B\right)V
$$

其中 $B = \hat{B}[\text{relative\_idx}]$ 是从可学习参数表 $\hat{B} \in \mathbb{R}^{(2M-1)^2 \times 1}$ 中查表得到的相对位置偏置。

**阶段间分辨率变化（Swin-Tiny）：**
- Stage 1: $H/4 \times W/4$，嵌入维度 $C=96$
- Stage 2: $H/8 \times W/8$，嵌入维度 $2C=192$ （Patch Merging后）
- Stage 3: $H/16 \times W/16$，嵌入维度 $4C=384$
- Stage 4: $H/32 \times W/32$，嵌入维度 $8C=768$

## 直观理解

Swin Transformer的设计灵感来源于CNN的层级式多尺度特征表示。窗口注意力就像CNN中的局部感受野，每个Patch只看自己小邻域内的其他Patch。但仅有局部窗口是不够的——如果层与层之间窗口划分不变，信息就永远无法跨窗口流动。Swin的巧妙之处在于"移位窗口"：当前层的窗口边界在下层被移动了，使得原本属于不同窗口的Patch在下一层进入同一个窗口，间接实现了跨窗口信息交流。

这就像你在一个大厅里和附近的人交谈（窗口注意力），然后所有人重新分组（移位窗口），你之前聊的内容通过与你组队的新人传播到了之前不相关的组。经过多层这样的操作，信息可以传播到整个图像。

## 代码示例

```python
import torch
import torch.nn as nn

class WindowAttention(nn.Module):
    """窗口多头自注意力（含相对位置偏置）"""
    def __init__(self, dim, window_size, num_heads):
        super().__init__()
        self.dim = dim
        self.window_size = window_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # 相对位置偏置表
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * window_size - 1) * (2 * window_size - 1), num_heads)
        )
        # 相对位置索引
        coords_h = torch.arange(window_size)
        coords_w = torch.arange(window_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))
        coords_flatten = torch.flatten(coords, 1)
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()
        relative_coords[:, :, 0] += window_size - 1
        relative_coords[:, :, 1] += window_size - 1
        relative_coords[:, :, 0] *= 2 * window_size - 1
        relative_position_index = relative_coords.sum(-1)
        self.register_buffer("relative_position_index", relative_position_index)

        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        nn.init.trunc_normal_(self.relative_position_bias_table, std=.02)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads)
        qkv = qkv.permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale

        # 添加相对位置偏置
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.window_size ** 2, self.window_size ** 2, -1)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        attn = attn + relative_position_bias.unsqueeze(0)

        attn = attn.softmax(dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        return x

# 验证窗口注意力
win_attn = WindowAttention(dim=96, window_size=7, num_heads=3)
x = torch.randn(64, 49, 96)  # (num_windows*B, M*M, D)
print(f"窗口注意力输出: {win_attn(x).shape}")
```

## 深度学习关联

- **通用视觉骨干网络**：Swin Transformer是第一个可以作为通用骨干网络（backbone）用于各类视觉任务（分类、检测、分割）的Transformer，在COCO目标检测和ADE20K语义分割上超越了CNN时代的SOTA。
- **跨窗口信息交互的设计模式**：Swin的移位窗口策略启发了后续多种局部-全局交互设计，如CSWin Transformer（十字窗口注意力）、Focal Transformer（粗细粒度注意力）等，推动了视觉Transformer的发展。
- **工业应用的有效性**：Swin Transformer被广泛应用于视频理解（Video-Swin）、医学图像分割（Swin-UNet）、遥感图像分析等实际场景，证明了层级式Transformer在密集预测任务中的强大能力，已成为OpenMMLab等开源框架的标配骨干网络。
