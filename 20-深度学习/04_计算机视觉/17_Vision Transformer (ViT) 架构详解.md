# 18_Vision Transformer (ViT) 架构详解

## 核心概念

- **ViT的核心思想**：将图像分割为固定大小的Patch（如 $16\times16$），每个Patch经过线性投影变为Token，再将这些Token输入标准Transformer编码器中进行处理。
- **Patch Embedding**：将 $H \times W \times C$ 的图像划分为 $N = (H/P) \times (W/P)$ 个 $P\times P$ 的Patch，每个Patch展平后通过线性投影映射为 $D$ 维向量（即Token）。
- **位置编码（Position Embedding）**：由于自注意力机制不具备位置感知能力，ViT在Patch Token中添加可学习的位置编码来保留空间位置信息。通常使用1D位置编码（简单有效）。
- **[CLS] Token**：参考BERT的设计，在输入序列开头添加一个可学习的[CLS] Token，其对应的输出向量作为图像的全局表示，用于分类。
- **自注意力机制（Self-Attention）**：每个Token通过QKV（Query-Key-Value）计算与其他所有Token的注意力权重，捕获全局依赖关系。
- **多头注意力（Multi-Head Attention, MHA）**：将注意力计算拆分为多个头，每个头学习不同的注意力模式，最后拼接融合。
- **MLP层（FFN）**：每个Transformer块包含两层MLP（通常 $D \to 4D \to D$），使用GELU激活函数，提供非线性变换。
- **归纳偏置缺失**：与CNN不同，ViT没有平移不变性和局部性的内置偏置，因此需要在大规模数据集（如ImageNet-21k、JFT-300M）上预训练才能超越CNN。

## 数学推导

**ViT的前向传播过程：**

- **Patch Embedding：**
$$
x_p^i = \text{Flatten}(I_{patch_i}) \in \mathbb{R}^{P^2 \cdot C}
$$
$$
z_0^i = E \cdot x_p^i + e_{pos}^i, \quad E \in \mathbb{R}^{D \times (P^2 \cdot C)}
$$

- **Transformer编码器（L层）：**
$$
z_\ell' = \text{MSA}(\text{LN}(z_{\ell-1})) + z_{\ell-1}
$$
$$
z_\ell = \text{MLP}(\text{LN}(z_\ell')) + z_\ell'
$$

- **分类头：**
$$
y = \text{LN}(z_L^0) \cdot W_{cls}
$$

**多头注意力计算（单头）：**
$$
\text{Attention}(Q, K, V) = \text{Softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right)V
$$

其中 $Q = zW_Q,\; K = zW_K,\; V = zW_V$，$d_k = D / H$ 是每个头的维度。

**计算复杂度对比：**
- CNN（$3\times3$ 卷积）：$\mathcal{O}(K^2 \cdot C_{in} \cdot C_{out} \cdot H \cdot W)$
- ViT自注意力：$\mathcal{O}(N^2 \cdot D)$，其中 $N = HW/P^2$ 是Patch数量

当输入分辨率较高时 $N$ 很大，Transformer的计算量会远高于CNN。ViT的分辨率缩放受限于 $\mathcal{O}(N^2)$ 的计算复杂度。

## 直观理解

ViT的工作方式可以想象为"把图像当作句子处理"。它将图像切成一个个小方块（Patch），就像把句子切分成一个个单词（Token）。然后通过自注意力机制让每个Patch"观察"所有其他Patch，理解它们之间的关系——就像在句子中每个单词需要理解与所有其他单词的关系一样。图像中的物体识别不再是CNN式的"局部感受野逐步扩大"，而是一次性全局分析所有局部区域之间的关系。

代价是ViT失去了CNN天生的"偏置"——知道邻近像素更相关、平移物体仍是同一物体等知识。这些在CNN中是内置的，而ViT需要从大量数据中学习。

## 代码示例

```python
import torch
import torch.nn as nn

class ViT(nn.Module):
    """简化版 Vision Transformer"""
    def __init__(self, img_size=224, patch_size=16, in_channels=3,
                 num_classes=1000, embed_dim=768, depth=12, num_heads=12):
        super().__init__()
        num_patches = (img_size // patch_size) ** 2

        # Patch Embedding: 卷积实现
        self.patch_embed = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # [CLS] Token + 位置编码
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
            torch.randn(1, num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(0.1)

        # Transformer编码器
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dim_feedforward=embed_dim * 4,
            activation='gelu',
            dropout=0.1,
            batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        # 分类头
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        # Patch Embedding + Flatten
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)

        # 添加[CLS] Token
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)

        # Transformer编码
        x = self.encoder(x)

        # [CLS] Token输出
        x = self.norm(x[:, 0])
        x = self.head(x)
        return x

model = ViT()
x = torch.randn(2, 3, 224, 224)
print(f"ViT输出: {model(x).shape}")
print(f"ViT-Base参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **CNN到Transformer的范式转变**：ViT证明了纯Transformer架构在视觉任务中可以超越CNN，开启了视觉领域的"Transformer化"浪潮。此后涌现了DeiT（数据高效ViT）、CaiT（深度ViT）、CrossFormer等大量改进工作。
- **视觉预训练的新范式**：ViT的预训练范式从CNN的"图像分类预训练"转变为"自监督预训练"（如MAE、DINOv2），在大规模无标注数据上学习通用视觉表示，效果甚至优于有监督预训练。
- **多模态融合的基础**：ViT的Token化表示天然适合多模态融合——文本Token和图像Patch Token可以在统一的Transformer中进行联合处理，CLIP、Flamingo等视觉-语言模型正是基于这一思想构建的。
