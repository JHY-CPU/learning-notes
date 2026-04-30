# 45_自监督视觉预训练：MAE (Masked Autoencoders)

## 核心概念

- **MAE（Masked Autoencoders）**：He et al. (2022) 提出的自监督视觉预训练方法——随机掩盖图像中大部分Patch（如75%），仅用剩余的少量可见Patch重建被掩盖的内容。
- **非对称编码器-解码器**：编码器只处理可见的Patch（计算量小），轻量级解码器从编码特征和掩盖Token重建完整的图像（计算量大但只在预训练时使用）。
- **高掩盖比例**：MAE使用高达75%的掩盖比例（远超BERT中15%的掩盖比例），迫使模型学习图像的全局语义信息，而非仅依赖局部像素相关性。
- **重建目标**：解码器预测每个被掩盖Patch的像素值（归一化后的RGB值），使用均方误差（MSE）损失仅在掩盖Patch上计算。
- **预训练-微调范式**：在ImageNet-1K等数据集上自监督预训练后，丢弃解码器，用编码器加任务特定头进行下游任务微调。
- **MAE的扩展能力**：随着模型容量增大（ViT-Large/Huge），MAE预训练的性能持续提升，验证了自监督预训练的可扩展性。

## 数学推导

**MAE的前向传播：**

1. **Patch化与掩盖**：将图像分为 $N$ 个非重叠Patch，随机选择 $M = \lfloor r \cdot N \rfloor$ 个Patch被掩盖（$r=0.75$），剩余的 $N-M$ 个Patch保留。

2. **编码器处理可见Patch**：
$$
z_{visible} = \text{ViT-Encoder}(x_{visible}) \in \mathbb{R}^{(N-M) \times D}
$$

3. **完整序列重建**：将可见Patch的编码特征与可学习的掩盖Token按原始位置排列：
$$
z_{full} = \text{Reorder}(z_{visible}, z_{mask\_token}) \in \mathbb{R}^{N \times D}
$$

4. **轻量级解码器重建**：
$$
\hat{x} = \text{Decoder}(z_{full})
$$

解码器只在预训练阶段使用，且比编码器更小（如8个Transformer块 vs 编码器的24个块）。

**MAE的损失函数（MSE损失）：**
$$
\mathcal{L} = \frac{1}{M} \sum_{i \in \mathcal{M}} \|\hat{x}_i - x_i\|_2^2
$$

其中 $\mathcal{M}$ 是被掩盖的Patch集合，$\hat{x}_i$ 是预测的像素值，$x_i$ 是归一化后的真实像素值。

**为什么MAE有效？** 高掩盖比例创建了一个"极具挑战性的任务"——模型必须理解物体的全局语义信息才能成功重建被掩盖的部分。例如，要重建被掩盖的人脸下巴，模型必须先识别出"这是一张脸"以及脸部的全局结构。

## 直观理解

MAE像是一种"视觉完形填空"——给出一张被抠去75%区域的图片，要求补全整张图片。这迫使模型从局部线索中推断全局结构。比如，只看到一只猫的耳朵和尾巴，就要补全整个猫的身体——这比"看到大部分图像，只抠掉一小块补全"要难得多。

MAE的关键设计选择是"非对称"结构：编码器只看可见的Patch（意味着编码器只处理了25%的信息，计算效率高），解码器负责将被掩盖的Patch"填空"（解码器可以很大，但只在预训练阶段使用）。这就像考试时先做简单的题目（可见Patch，编码器处理），再根据简单的题目推理难的题目（被掩盖Patch，解码器生成）。

## 代码示例

```python
import torch
import torch.nn as nn

class MAE(nn.Module):
    """Masked Autoencoder (简化版)"""
    def __init__(self, img_size=224, patch_size=16, embed_dim=768,
                 encoder_depth=12, decoder_depth=4, num_heads=12, 
                 mask_ratio=0.75):
        super().__init__()
        self.patch_size = patch_size
        self.mask_ratio = mask_ratio
        num_patches = (img_size // patch_size) ** 2
        
        # Patch Embedding
        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        
        # 位置编码
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim))
        
        # 编码器 (只处理可见Patch)
        encoder_layer = nn.TransformerEncoderLayer(
            embed_dim, num_heads, embed_dim * 4, dropout=0.,
            activation='gelu', batch_first=True
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, encoder_depth)
        
        # 掩盖Token
        self.mask_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # 解码器 (轻量级)
        self.decoder_embed = nn.Linear(embed_dim, embed_dim // 2)
        decoder_layer = nn.TransformerEncoderLayer(
            embed_dim // 2, num_heads // 2, embed_dim, dropout=0.,
            activation='gelu', batch_first=True
        )
        self.decoder = nn.TransformerEncoder(decoder_layer, decoder_depth)
        self.decoder_pos = nn.Parameter(torch.zeros(1, num_patches, embed_dim // 2))
        
        # 重建头
        self.decoder_pred = nn.Linear(embed_dim // 2, patch_size * patch_size * 3)

    def random_masking(self, x, mask_ratio):
        """随机掩盖Patch"""
        B, N, D = x.shape
        len_keep = int(N * (1 - mask_ratio))
        noise = torch.rand(B, N, device=x.device)
        ids_shuffle = torch.argsort(noise, dim=1)
        ids_keep = ids_shuffle[:, :len_keep]
        x_masked = torch.gather(x, dim=1, index=ids_keep.unsqueeze(-1).expand(-1, -1, D))
        # 记录掩盖/保留的索引
        ids_restore = torch.argsort(ids_shuffle, dim=1)
        return x_masked, ids_restore

    def forward(self, x):
        B = x.shape[0]
        # Patch Embedding
        x = self.patch_embed(x)  # (B, D, H/P, W/P)
        x = x.flatten(2).transpose(1, 2)  # (B, N, D)
        x = x + self.pos_embed
        
        # 随机掩盖
        x_masked, ids_restore = self.random_masking(x, self.mask_ratio)
        
        # 编码器 (只处理可见Patch)
        encoded = self.encoder(x_masked)  # (B, N*(1-r), D)
        
        # 拼接掩盖Token
        mask_tokens = self.mask_token.expand(B, ids_restore.shape[1] - encoded.shape[1], -1)
        x_full = torch.cat([encoded, mask_tokens], dim=1)
        x_full = torch.gather(x_full, dim=1,
                              index=ids_restore.unsqueeze(-1).expand(-1, -1, encoded.shape[-1]))
        
        # 解码器
        x_dec = self.decoder_embed(x_full) + self.decoder_pos
        x_dec = self.decoder(x_dec)
        
        # 重建像素
        pred = self.decoder_pred(x_dec)
        return pred, ids_restore

# 测试MAE
model = MAE(mask_ratio=0.75)
x = torch.randn(2, 3, 224, 224)
pred, ids = model(x)
print(f"重建输出: {pred.shape}")
print(f"编码器参数量: {sum(p.numel() for p in model.encoder.parameters()):,}")
print(f"解码器参数量: {sum(p.numel() for p in model.decoder.parameters()):,}")
print(f"总参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **自监督视觉预训练的SOTA**：MAE代表了自监督视觉预训练的最前沿，在ImageNet-1K上仅使用自监督预训练就达到了与有监督预训练竞争甚至更好的性能。MAE在目标检测（使用ViT检测器）、语义分割（使用MAE预训练的骨干）等下游任务中也表现优异。
- **掩码信号建模（Masked Signal Modeling）的泛化**：MAE的"高比例掩盖+重建"范式被扩展到视频（VideoMAE——在视频中掩盖时空立方体）、3D点云（Point-MAE——掩盖3D点）、多模态数据等多种信号模态。
- **自监督学习的范式转变**：MAE标志着视觉自监督学习从"对比学习"（SimCLR、MoCo——区分不同样本）到"生成式学习"（MAE、MaskFeat——重建掩码内容）的重要转变，类似于NLP中BERT到GPT的转变。两者各有优势，生成式方法在密集预测任务中表现更优。
