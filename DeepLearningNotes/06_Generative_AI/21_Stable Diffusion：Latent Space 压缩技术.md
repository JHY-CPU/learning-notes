# 21_Stable Diffusion：Latent Space 压缩技术

## 核心概念
- **潜空间扩散 (Latent Diffusion)**：Stable Diffusion 不在像素空间执行扩散过程，而是在 VAE 编码器压缩后的潜空间中执行，计算量减少约 90%。
- **感知压缩 (Perceptual Compression)**：VAE 编码器 $E$ 将 $H \times W \times 3$ 的像素图像压缩为 $h \times w \times c$ 的潜表示（压缩因子 $f=H/h=8$ 或 $f=16$），保留感知上重要的信息，丢弃高频细节。
- **语义压缩 (Semantic Compression)**：扩散模型（U-Net + Cross-Attention）在潜空间中学习从文本描述生成语义内容，在这个阶段压缩的是"语义"层面的信息。
- **条件注入**：通过交叉注意力层将文本嵌入（来自 CLIP 文本编码器）注入到 U-Net 的每一层，控制生成内容符合文本描述。
- **开放式文本生成**：不同于 GAN 只能处理有限类别标签，Stable Diffusion 利用 CLIP 文本编码器可以理解任意自然语言描述，实现开放式的文本到图像生成。
- **分层架构**：整个系统由三个主要组件构成——VAE 编解码器（压缩/解压像素↔潜空间）、U-Net 去噪网络（在潜空间中扩散/去噪）、CLIP 文本编码器（将文本转换为条件嵌入）。

## 数学推导

**Stable Diffusion 的三阶段流程**：

1. **压缩**：$z_0 = E(x)$，像素 $x$ 压缩为潜变量 $z_0$
2. **扩散/去噪**：在潜空间中进行 DDPM 过程——$z_t = \sqrt{\bar{\alpha}_t} z_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，去噪网络 $\epsilon_\theta(z_t, t, \tau_\theta(c))$ 预测噪声
3. **解压**：$\hat{x} = D(z_0)$，去噪后的潜变量 $z_0$ 解码为像素图像

**训练损失**：

$$
L = \mathbb{E}_{z_0 \sim E(x), c, t, \epsilon \sim \mathcal{N}(0, I)} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(c)) \|_2^2 \right]
$$

其中 $\tau_\theta$ 是 CLIP 文本编码器（冻结），$c$ 是文本条件。

**潜空间维度对比**：

- 像素空间：$512 \times 512 \times 3 \approx 786,432$ 维
- 潜空间：$64 \times 64 \times 4 = 16,384$ 维（压缩比 48 倍）

**Stable Diffusion vs 原始扩散模型的计算量**：

$$
\text{计算量比} = \frac{\text{潜空间 U-Net 的 FLOPs}}{\text{像素空间 U-Net 的 FLOPs}} \approx \frac{f^2}{r^2}
$$

其中 $f$ 是 VAE 压缩因子（通常为 8），$r$ 是 U-Net stride 比例约为 4——实际加速约 4 倍，内存节省约 2 倍。

## 直观理解
- **潜空间扩散 = 用草稿纸而不是最终画布**：想象你需要在画布上完成一幅精美的画作。直接在画布上修改（像素空间扩散）每次都要处理海量的像素点。在草稿纸上打草稿（潜空间扩散）则高效得多——草稿纸上的每根线条对应了最终画的纹理图案。
- **VAE 压缩 = JPEG 压缩的学习版**：VAE 编码器学会了"哪些视觉信息重要，哪些可以丢弃"。它知道保留边缘、形状、语义结构，丢弃像素级的随机噪声。这就像智能 JPEG——压缩 48 倍但核心内容保留。
- **分层压缩的理解**：第一层压缩（VAE）去除了感知冗余（像素级的细节），第二层（扩散模型）在更紧凑的语义空间中学习——这比直接从像素学习更容易。
- **为什么 Stable Diffusion 能跑在消费级 GPU 上**：就是因为潜空间比像素空间小 48 倍——网络更小、训练更快、推理更快。这是 Stable Diffusion 成功"出圈"的关键工程创新。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class VAECompressor(nn.Module):
    """简化的 VAE 压缩/解压模型"""
    def __init__(self, in_channels=3, latent_channels=4, compression_factor=8):
        super().__init__()
        # 编码器：像素 -> 潜空间（下采样 factor 倍）
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 4, 2, 1),  # 下采样 2x
            nn.SiLU(),
            nn.Conv2d(128, 256, 4, 2, 1),  # 下采样 2x
            nn.SiLU(),
            nn.Conv2d(256, 256, 4, 2, 1),  # 下采样 2x
            nn.SiLU(),
            nn.Conv2d(256, latent_channels * 2, 3, 1, 1),  # mu + logvar
        )
        
        # 解码器：潜空间 -> 像素
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_channels, 256, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(256, 256, 4, 2, 1),  # 上采样 2x
            nn.SiLU(),
            nn.ConvTranspose2d(256, 128, 4, 2, 1),  # 上采样 2x
            nn.SiLU(),
            nn.ConvTranspose2d(128, 128, 4, 2, 1),  # 上采样 2x
            nn.SiLU(),
            nn.Conv2d(128, in_channels, 3, 1, 1),
            nn.Tanh(),
        )
    
    def encode(self, x):
        h = self.encoder(x)
        mu, logvar = h.chunk(2, dim=1)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)
    
    def forward(self, x):
        mu, logvar = self.encode(x)
        std = torch.exp(0.5 * logvar)
        z = mu + std * torch.randn_like(std)
        recon = self.decode(z)
        return recon, mu, logvar

class LatentDiffusionUNet(nn.Module):
    """在潜空间中运行的 U-Net（带交叉注意力）"""
    def __init__(self, latent_channels=4, text_emb_dim=768, time_emb_dim=256):
        super().__init__()
        # 简化的条件 U-Net
        self.input_proj = nn.Conv2d(latent_channels, 64, 3, 1, 1)
        
        # 时间嵌入
        self.time_embed = nn.Sequential(
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.SiLU(),
            nn.Linear(time_emb_dim, time_emb_dim),
        )
        
        # 交叉注意力（文本条件注入的简化实现）
        self.cross_attn = nn.MultiheadAttention(64, 4, batch_first=True)
        self.text_proj = nn.Linear(text_emb_dim, 64)
        
        self.output_proj = nn.Conv2d(64, latent_channels, 3, 1, 1)
    
    def forward(self, z_t, t, text_emb):
        # 简化的前向
        h = self.input_proj(z_t)
        
        # 时间嵌入注入
        t_emb = self.time_embed(t.float())
        t_emb = t_emb[:, :, None, None].expand(-1, -1, h.size(2), h.size(3))
        h = h + t_emb
        
        # 交叉注意力（文本条件的注入）
        B, C, H, W = h.shape
        h_flat = h.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        text_emb_proj = self.text_proj(text_emb)  # [B, seq_len, C]
        h_attn, _ = self.cross_attn(h_flat, text_emb_proj, text_emb_proj)
        h = h_attn.permute(0, 2, 1).reshape(B, C, H, W)
        
        return self.output_proj(h)

# 验证 Stable Diffusion 的潜空间流程
def stable_diffusion_pipeline(text_encoder, vae, unet, text_prompt="a cat"):
    """简化的 Stable Diffusion 推理流程"""
    # 1. 编码文本
    with torch.no_grad():
        text_emb = text_encoder(text_prompt)  # 实际使用 CLIP
    
    # 2. 生成初始噪声（在潜空间中）
    z_T = torch.randn(1, 4, 64, 64)
    
    # 3. 在潜空间中逐步去噪（DDIM 采样）
    z_t = z_T
    for t in range(999, -1, -1):  # 简化：实际用 DDIM 跳步
        t_tensor = torch.full((1,), t)
        predicted_noise = unet(z_t, t_tensor, text_emb)
        
        # 简化的去噪更新
        alpha_bar = 0.5 ** (t / 1000)
        z_t = (z_t - torch.sqrt(1 - alpha_bar) * predicted_noise) / torch.sqrt(alpha_bar)
    
    # 4. 将潜变量解码为像素图像
    z_0 = z_t
    x_0 = vae.decode(z_0)
    
    return x_0

print("=== Stable Diffusion 潜空间架构 ===")
print("像素空间: 512x512x3 = 786,432 维")
print("潜空间:      64x64x4 =  16,384 维")
print(f"压缩比: {786432/16384:.1f}x")
print()
print("三阶段结构:")
print("  1. VAE 编码器:  像素空间 -> 潜空间")
print("  2. U-Net:       在潜空间中扩散/去噪 (带文本条件)")
print("  3. VAE 解码器:  潜空间 -> 像素空间")
```

## 深度学习关联
- **计算效率突破**：Stable Diffusion 的核心贡献不是新的数学理论，而是工程上的"将扩散过程搬到潜空间"。这使扩散模型从需要数百 GPU 天的训练（如 DALL-E 2 需要 1000+ GPU）降低到普通研究团队可负担的水平。
- **Text-to-Image 的普及**：Stable Diffusion 的开源发布（2022 年 8 月）彻底改变了生成式 AI 的生态——它使得文生图技术不仅限于少数大公司的 API，普通用户可以在自己的 GPU 上运行。
- **社区生态与模型微调**：潜空间的低维度使得模型微调（LoRA、DreamBooth、Textual Inversion）变得可行——微调一个小模型就能在潜空间中改变生成风格或添加新概念。
- **Stable Diffusion 2.0 / 3.0 的演进**：SD 2.0 改用 OpenCLIP 文本编码器、引入深度先验；SDXL 使用更大的 U-Net 和双文本编码器；SD 3.0 采用 MMDiT（Multimodal Diffusion Transformer）架构，进一步提升了文本理解和生成质量。
