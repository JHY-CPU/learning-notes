# 38_TimeSformer：视频中的时空注意力

## 核心概念

- **TimeSformer（Time-Space Transformer）**：Bertasius et al. (2021) 提出的纯Transformer视频理解架构，将ViT扩展到时域，使用分解的时空注意力机制处理视频帧序列。
- **分解的时空注意力（Divided Space-Time Attention）**：将完整的3D自注意力分解为两个步骤——先在时间维度上计算注意力（同一空间位置不同帧之间），再在空间维度上计算注意力（同一帧内不同空间位置之间）。
- **视频Patch化**：将每帧图像分割为 $P\times P$ 的Patch，所有帧的Patch拼成一个Token序列。对于 $T$ 帧 $H\times W$ 图像，Token总数为 $T \times (H/P) \times (W/P)$。
- **时间注意力的复杂度优势**：完整3D注意力的复杂度为 $\mathcal{O}(T^2H^2W^2P^{-4})$，分解时空注意力将复杂度降为 $\mathcal{O}(TH^2W^2P^{-4} + T^2HWP^{-2})$，大幅降低了计算量。
- **位置编码**：分别使用空间位置编码（每个空间位置共享所有帧）和时间位置编码（每个时间位置共享所有空间位置），或者使用可学习的3D位置编码。
- **多种注意力模式的消融**：TimeSformer实验了多种时空注意力的组合方式：空间-时间、时间-空间、并行、完整3D等，发现"先空间后时间"的分解方式效果最好且计算量最低。

## 数学推导

**视频Token的构建：**
对于输入视频 $V \in \mathbb{R}^{T \times 3 \times H \times W}$，将每帧划分为 $N = H/P \times W/P$ 个Patch，每个Patch展平为 $3P^2$ 维向量，通过线性投影变为 $D$ 维嵌入：
$$
z_{(t,i)} = E \cdot \text{Flatten}(V_{t, :, :, (i)})
$$

其中 $t \in [1, T]$ 是帧索引，$i \in [1, N]$ 是空间位置索引。

**分解的时空注意力（以"时间→空间"为例）：**

时间注意力（同空间位置跨帧）：
$$
\hat{z}_{(t,i)} = \sum_{t'=1}^T \text{Softmax}\left(\frac{q_{(t,i)}^T k_{(t',i)}}{\sqrt{D_h}}\right) v_{(t',i)} + z_{(t,i)}
$$

空间注意力（同帧内跨位置）：
$$
z_{(t,i)}^* = \sum_{j=1}^N \text{Softmax}\left(\frac{\hat{q}_{(t,i)}^T \hat{k}_{(t,j)}}{\sqrt{D_h}}\right) \hat{v}_{(t,j)} + \hat{z}_{(t,i)}
$$

**复杂度对比：**

完整时空自注意力：
$$
\Omega(\text{Full}) = \mathcal{O}((TN)^2 D)
$$

分解时空注意力（时间+空间）：
$$
\Omega(\text{Divided}) = \mathcal{O}(T^2 N D + TN^2 D)
$$

当 $T$ 和 $N$ 都较大时，分解注意力节省了大量计算。

## 直观理解

TimeSformer的核心设计是"先把视频看作帧的集合，分析与时间相关的模式，再分析帧内的空间模式"。这就像理解一段视频中"人挥手"的动作：

- 时间注意力：追踪画面中"手"这个位置在不同帧之间的变化——手的位置是否在移动、移动方向如何（捕获运动信息）
- 空间注意力：在每一帧中，理解手的形状、位置以及和身体其他部位的关系（捕获外观信息）

将时空注意力分解为两个步骤，而不是在完整的时空立方体上做注意力，相当于把"在同一时间观察不同位置"和"在不同时间观察相同位置"分开做，大幅降低了计算复杂度。实验证明这种分解方式不仅高效，而且效果好于完整3D注意力。

## 代码示例

```python
import torch
import torch.nn as nn

class TimeSformerBlock(nn.Module):
    """TimeSformer 的时空注意力块"""
    def __init__(self, dim, num_heads, mlp_ratio=4.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)
        
        # 时间注意力 (在帧之间)
        self.temporal_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        # 空间注意力 (在帧内)
        self.spatial_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        
        # MLP
        hidden_dim = int(dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, dim),
        )

    def forward(self, x, num_frames, num_patches):
        """
        x: (B, T*N, D) 所有帧的Patch Token
        num_frames: T
        num_patches: N (每帧的Patch数)
        """
        B, total_tokens, D = x.shape
        N = num_patches
        
        # --- 时间注意力 ---
        # 将同一空间位置的Token聚在一起
        x = x.view(B, num_frames, N, D)
        x_time = x.permute(0, 2, 1, 3).reshape(B * N, num_frames, D)
        x_time = self.norm1(x_time)
        x_time, _ = self.temporal_attn(x_time, x_time, x_time)
        x_time = x_time.reshape(B, N, num_frames, D).permute(0, 2, 1, 3)
        x = x + x_time.reshape(B, num_frames, N, D)
        
        # --- 空间注意力 ---
        x_space = x.reshape(B, num_frames * N, D)
        x_space = self.norm2(x_space)
        x_space, _ = self.spatial_attn(x_space, x_space, x_space)
        x = x.reshape(B, num_frames * N, D) + x_space
        
        # MLP
        x = x + self.mlp(self.norm3(x))
        return x

# 简化的TimeSformer
class TimeSformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, num_frames=8,
                 in_channels=3, num_classes=400, dim=768, depth=12, num_heads=12):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size, stride=patch_size)
        # CLS Token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        # 位置编码 (空间 + 时间)
        self.pos_embed_spatial = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        self.pos_embed_temporal = nn.Parameter(torch.zeros(1, num_frames, dim))
        
        # Transformer块
        self.blocks = nn.ModuleList([
            TimeSformerBlock(dim, num_heads) for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def forward(self, x):
        # x: (B, C, T, H, W)  ->  (B*T, C, H, W)
        B, C, T, H, W = x.shape
        x = x.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W)
        x = self.patch_embed(x)  # (B*T, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B*T, N, D)
        
        # 添加CLS Token
        cls_tokens = self.cls_token.expand(B * T, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed_spatial
        x = x.view(B, T, -1, x.shape[-1])
        
        # 添加时间位置编码
        x = x + self.pos_embed_temporal.unsqueeze(2)
        x = x.view(B, -1, x.shape[-1])  # (B, T*(N+1), D)
        
        # 通过Transformer块
        for blk in self.blocks:
            x = blk(x, T, x.shape[1] // T)
        
        x = self.norm(x)
        # 取每个帧的CLS Token的平均
        cls_out = x[:, 0::x.shape[1] // T, :]  # 每帧的CLS
        x = cls_out.mean(dim=1)  # 平均池化
        x = self.head(x)
        return x

model = TimeSformer(num_frames=8, num_classes=400)
x = torch.randn(1, 3, 8, 224, 224)
out = model(x)
print(f"TimeSformer输出: {out.shape}")
print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **视频理解的Transformer化**：TimeSformer将ViT的成功经验扩展到视频领域，推动了视频理解从3D CNN到Transformer的转变。后续的Video Swin Transformer、MViTv2（多尺度ViT）等在其基础上进一步改进了时空建模效率。
- **分解式注意力设计的影响**：TimeSformer的"分解时空注意力"设计思想被广泛应用于视觉-语言模型（如Florence、InternVideo），在统一处理时空信息时采用类似的分解策略以提高效率。
- **长视频理解的挑战**：TimeSformer受限于自注意力的 $O(T^2)$ 复杂度，对长视频处理能力有限。后续工作（如Memory Bank、Token Merging）致力于解决长时视频建模问题，使Transformer可以处理分钟级的视频内容。
