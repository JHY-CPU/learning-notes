# 14_VQ-VAE：向量量化与离散潜在空间

## 核心概念
- **VQ-VAE** (Vector Quantized VAE)：将 VAE 的连续潜空间替换为离散的"码本"（Codebook），每个编码通过最近邻查找量化为码本中最接近的向量。
- **向量量化 (Vector Quantization)**：编码器的输出 $z_e(x)$ 不是直接作为潜变量，而是被替换为码本中最近的向量 $e_k$，即 $z_q(x) = e_k$ 其中 $k = \arg\min_j \|z_e(x) - e_j\|_2$。
- **离散潜空间**：码本包含 $K$ 个可学习的嵌入向量（codebook entries），每个 $d$ 维。潜空间是离散的 $K$ 个条目，而不是连续的 $\mathbb{R}^d$。
- **后验坍塌的解决**：标准 VAE 常出现"后验坍塌"——解码器忽略潜变量 $z$，导致编码器退化为输出恒定分布。VQ-VAE 的离散潜变量强制解码器必须依赖 $z$。
- **自回归先验 (Autoregressive Prior)**：VQ-VAE 不使用假设的高斯先验，而是用 PixelCNN/Transformer 学习离散编码的分布 $p(z)$，能捕捉长期依赖。
- **梯度直通估计 (Straight-Through Estimator)**：由于 $\arg\min$ 操作不可导，VQ-VAE 使用直通估计器将解码器梯度直接复制到编码器输出上。

## 数学推导

**VQ-VAE 的损失函数**由三部分组成：

$$
\mathcal{L} = \underbrace{\|\text{sg}[z_e(x)] - e\|_2^2}_{\text{Codebook Loss}} + \underbrace{\beta\|z_e(x) - \text{sg}[e]\|_2^2}_{\text{Commitment Loss}} + \underbrace{\|\text{decoder}(z_q(x)) - x\|_2^2}_{\text{Reconstruction Loss}}
$$

其中 $\text{sg}[\cdot]$ 表示 stop-gradient 操作（梯度不通过该节点传播）。

1. **Codebook Loss**：将码本向量 $e$ 拉向编码器输出 $z_e(x)$（只更新码本，不更新编码器）
2. **Commitment Loss**：将编码器输出 $z_e(x)$"承诺"给码本（只更新编码器，不更新码本），$\beta$ 控制承诺强度（通常设为 0.25）
3. **Reconstruction Loss**：标准的重建损失，通过直通估计器将梯度从解码器传回编码器

**梯度直通估计**：

前向传播：$z_q(x) = z_e(x) + \text{sg}[e_k - z_e(x)]$（等价于 $z_q(x) = e_k$）

反向传播：$\nabla_\theta \mathcal{L} = \nabla_\theta \mathcal{L}_{\text{recon}}$（梯度直接穿过量化层，仿佛 $z_q = z_e$）

**码本学习的 EMA 更新**（训练稳定性的改进）：

不用梯度下降更新码本，而是使用指数移动平均（Exponential Moving Average）：

$$
N_i^{(t)} = N_i^{(t-1)} \cdot \gamma + n_i^{(t)} \cdot (1 - \gamma)
$$

$$
m_i^{(t)} = m_i^{(t-1)} \cdot \gamma + \sum_j z_{e,ij}^{(t)} \cdot (1 - \gamma)
$$

$$
e_i^{(t)} = \frac{m_i^{(t)}}{N_i^{(t)}}
$$

其中 $n_i$ 是第 $i$ 个码本条目被选中的次数，$\gamma$ 是衰减率。

## 直观理解
- **VQ-VAE = 用词汇而不是笔画描述图像**：标准 VAE 的连续潜空间好比用无数种"笔画"的自由组合来描述图像，VQ-VAE 则像有一个固定词汇表（码本），只能用这些词汇来描述图像。
- **向量量化就像"找最相似的颜料色号"**：你想画一种颜色，但不能直接调色（连续），只能从 Pantone 色卡（码本）中选最接近的色号。随着训练，色卡的色彩越来越丰富（码本学习）。
- **为什么离散有帮助**：离散表示天然适合文本、语音等符号数据。在图像生成中，离散编码迫使模型抓住最核心的结构信息，忽略噪声。
- **自回归先验为什么必要**：连续 VAE 可以假设 $p(z) = \mathcal{N}(0,I)$，但离散编码没有简单的"标准分布"。用 PixelCNN/Transformer 学习 $p(z)$ 相当于"学会了码本的语法"——哪些编码序列才是合理的。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorQuantizer(nn.Module):
    """向量量化层"""
    def __init__(self, num_embeddings=512, embedding_dim=64, commitment_cost=0.25):
        super().__init__()
        self.num_embeddings = num_embeddings
        self.embedding_dim = embedding_dim
        self.commitment_cost = commitment_cost
        
        # 码本：K 个 d 维嵌入向量
        self.embedding = nn.Embedding(num_embeddings, embedding_dim)
        self.embedding.weight.data.uniform_(-1/num_embeddings, 1/num_embeddings)
    
    def forward(self, z_e):
        """
        前向传播：将编码器输出 z_e 量化为最近的码本向量
        
        参数:
            z_e: 编码器输出 [B, D, H, W]
        返回:
            z_q: 量化后的潜变量 [B, D, H, W]
            encoding_indices: 每个位置的码本索引 [B, H, W]
            loss: 码本损失 + 承诺损失
        """
        # 展平为向量序列
        B, D, H, W = z_e.shape
        flat_z_e = z_e.permute(0, 2, 3, 1).reshape(-1, D)  # [B*H*W, D]
        
        # 计算距离：||z_e - e_k||^2 对所有 k
        dist = torch.cdist(flat_z_e, self.embedding.weight)  # [B*H*W, K]
        
        # 最近邻编码
        encoding_indices = torch.argmin(dist, dim=1)  # [B*H*W]
        encoding_one_hot = F.one_hot(encoding_indices, self.num_embeddings).float()
        
        # 量化：用码本向量替换
        flat_z_q = torch.matmul(encoding_one_hot, self.embedding.weight)  # [B*H*W, D]
        z_q = flat_z_q.view(B, H, W, D).permute(0, 3, 1, 2)  # [B, D, H, W]
        
        # VQ 损失
        # Codebook Loss: sg[z_e] -> e（只更新码本）
        codebook_loss = F.mse_loss(z_q.detach(), z_e)
        # Commitment Loss: sg[e] -> z_e（只更新编码器）
        commitment_loss = F.mse_loss(z_q, z_e.detach())
        
        vq_loss = codebook_loss + self.commitment_cost * commitment_loss
        
        # 直通估计：前向用 z_q，反向梯度等价于 z_e
        z_q = z_e + (z_q - z_e).detach()
        
        return z_q, encoding_indices.view(B, H, W), vq_loss

class VQVAE(nn.Module):
    """简化的 VQ-VAE 模型"""
    def __init__(self, in_channels=3, hidden_dim=128, num_embeddings=512, embedding_dim=64):
        super().__init__()
        # 编码器
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim, embedding_dim, 3, 1, 1),
        )
        
        # 向量量化层
        self.vq = VectorQuantizer(num_embeddings, embedding_dim)
        
        # 解码器
        self.decoder = nn.Sequential(
            nn.Conv2d(embedding_dim, hidden_dim, 3, 1, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, hidden_dim, 4, 2, 1),
            nn.ReLU(),
            nn.ConvTranspose2d(hidden_dim, in_channels, 4, 2, 1),
            nn.Tanh(),
        )
    
    def forward(self, x):
        z_e = self.encoder(x)
        z_q, encoding_indices, vq_loss = self.vq(z_e)
        recon = self.decoder(z_q)
        return recon, vq_loss, encoding_indices
    
    def encode_to_indices(self, x):
        """将输入编码为离散索引序列"""
        z_e = self.encoder(x)
        B, D, H, W = z_e.shape
        flat_z_e = z_e.permute(0, 2, 3, 1).reshape(-1, D)
        dist = torch.cdist(flat_z_e, self.vq.embedding.weight)
        return torch.argmin(dist, dim=1).view(B, -1)
    
    def decode_from_indices(self, indices, latent_shape):
        """从索引序列解码重建图像"""
        z_q = self.vq.embedding(indices).permute(0, 3, 1, 2).contiguous()
        return self.decoder(z_q)

# 训练 VQ-VAE 的损失函数
def vqvae_loss(model, x):
    recon, vq_loss, _ = model(x)
    recon_loss = F.mse_loss(recon, x)
    total_loss = recon_loss + vq_loss
    return total_loss, recon_loss.item(), vq_loss.item()

# 初始化
vqvae = VQVAE(in_channels=3, hidden_dim=128, num_embeddings=512, embedding_dim=64)
x = torch.randn(4, 3, 32, 32)
recon, vq_loss, indices = vqvae(x)
print(f"输入形状: {x.shape}")
print(f"重建形状: {recon.shape}")
print(f"码本大小: 512 x 64")
print(f"编码索引形状: {indices.shape}")
print(f"VQ 损失: {vq_loss.item():.4f}")
print(f"VQ-VAE 总参数量: {sum(p.numel() for p in vqvae.parameters()):,}")
```

## 深度学习关联
- **VQ-GAN**：将 VQ-VAE 的量化与 GAN 的对抗训练结合，用感知损失和对抗损失替代纯 L2 重建损失，大幅提升了解码质量（去除了 VQ-VAE 常见的模糊问题）。
- **DALL-E / VQ-Diffusion**：VQ-VAE 的离散潜空间为文本到图像生成提供了统一的表示空间——文本和图像都被编码为离散 token，然后用 Transformer 学习从文本 token 到图像 token 的映射。DALL-E 使用 dVAE（离散 VAE），VQ-Diffusion 在 VQ 潜空间中执行扩散过程。
- **MAGVIT / VideoGPT**：将 VQ-VAE 扩展到视频领域，将视频帧序列编码为离散 token 序列，然后用自回归 Transformer 或扩散模型生成视频。
- **AudioLM / SpeechTokenizer**：在音频领域，VQ-VAE 将语音信号量化为离散 token，使语言模型可以直接生成语音——这是神经编解码器（如 EnCodec、SoundStream）的核心思想。
