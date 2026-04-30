# 29_音频生成：AudioLDM 基础

## 核心概念
- **AudioLDM**：一种基于潜在扩散模型的文本到音频生成方法，将 Stable Diffusion 的思想迁移到音频领域——在梅尔频谱图的潜空间中执行扩散过程。
- **梅尔频谱图 (Mel-spectrogram)**：音频的标准二维表示——时间轴（横坐标）× 频率轴（纵坐标），值表示能量强度。AudioLDM 将音频生成视为"图像生成"问题（生成频谱图）。
- **声学 VAE**：类似于 Stable Diffusion 的 VAE，将梅尔频谱图压缩到潜空间，再解码恢复为频谱图。最后一个声码器（Vocoder，如 HiFi-GAN）将频谱图转换为波形音频。
- **CLAP 对比学习**：AudioLDM 使用 CLAP（Contrastive Language-Audio Pretraining）模型将文本和音频对齐到同一语义空间，作为条件注入的文本编码器。
- **无条件生成 + 条件控制**：训练时通过随机丢弃条件（类似 CFG 中的空文本），使模型同时支持有声条件（文本→音频）和无声条件（纯音频生成）的模式。
- **推理速度**：AudioLDM 在潜空间中扩散，一次生成几秒的音频只需几秒推理时间，远快于之前基于自回归的音频生成模型。

## 数学推导

**AudioLDM 的三阶段流程**：

1. **压缩**：梅尔频谱图 $S \in \mathbb{R}^{F \times T}$ 通过 VAE 编码器压缩到潜空间 $z_0 = E(S)$
2. **扩散/去噪**：在潜空间中执行条件扩散 $p_\theta(z_{t-1}|z_t, c)$，$c$ 是文本描述
3. **解码为波形**：VAE 解码器恢复频谱图 $\hat{S} = D(z_0)$，声码器转换为波形 $w = \text{Vocode}(\hat{S})$

**模型训练损失**：

$$
L = \mathbb{E}_{z_0, c, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(c)) \|_2^2 \right]
$$

其中 $\tau_\theta$ 是 CLAP 文本编码器，$c$ 是音频内容的文本描述（如"雨声打在窗户上"）。

**CLAP 对比损失**：

与 CLIP 类似，CLAP 在 batch 中最大化配对的文本-音频嵌入相似度：

$$
\mathcal{L}_{\text{CLAP}} = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{\exp(t_i \cdot a_i / \tau)}{\sum_j \exp(t_i \cdot a_j / \tau)} + \log \frac{\exp(a_i \cdot t_i / \tau)}{\sum_j \exp(a_i \cdot t_j / \tau)} \right]
$$

**Latent Audio Diffusion 的关键参数**：

- 梅尔频谱图尺寸：$64 \times 256$（频率 bin × 时间帧，对应约 5 秒音频）
- 潜空间尺寸：$16 \times 64 \times 4$（压缩因子 4x）
- 潜在扩散的步数：1000 步训练，200 步 DDIM 采样

## 直观理解
- **音频生成 = 画声音的"照片"**：梅尔频谱图就像声音的"照片"——横轴是时间，纵轴是频率，颜色深浅表示音量。AudioLDM 做的事情本质上就是"文生图"，只不过这张"图"可以转换为声音。
- **声码器 = 照片翻译成声音**：模型生成的是一张频谱图（视觉格式），需要声码器把它"读"出来变成波形。这就像一个人能看懂乐谱（频谱图）并演奏出来（波形）。
- **CLAP 的作用**：CLAP 学到了"雨声"对应的频谱图是什么样子，"钢琴声"是什么样子。就像 CLIP 懂"猫"图片长什么样一样，CLAP 懂"雨声"的频谱图长什么样。
- **为什么在潜空间中扩散**：和 Stable Diffusion 一样——音频的频谱图分辨率也很大（$64 \times 256$），直接在像素级扩散计算量大，潜空间扩散高效得多。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class AudioVAE(nn.Module):
    """声学 VAE：音频梅尔频谱图 <-> 潜空间"""
    def __init__(self, n_mels=64, time_steps=256, latent_dim=4):
        super().__init__()
        # 编码器：频谱图 [B, 1, n_mels, T] -> 潜变量 [B, latent_dim, h, w]
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1),  # 32x32x128
            nn.SiLU(),
            nn.Conv2d(32, 64, 3, 2, 1),  # 64x16x64
            nn.SiLU(),
            nn.Conv2d(64, latent_dim * 2, 3, 1, 1),  # 输出 mu + logvar
        )
        
        # 解码器：潜变量 -> 频谱图
        self.decoder = nn.Sequential(
            nn.Conv2d(latent_dim, 64, 3, 1, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1),
            nn.SiLU(),
            nn.ConvTranspose2d(32, 1, 4, 2, 1),
        )
    
    def encode(self, mel):
        mu_logvar = self.encoder(mel)
        mu, logvar = mu_logvar.chunk(2, dim=1)
        return mu, logvar
    
    def decode(self, z):
        return self.decoder(z)

class AudioLDMDiffusion(nn.Module):
    """简化的 AudioLDM 条件扩散模型"""
    def __init__(self, latent_channels=4, text_emb_dim=512, time_emb_dim=128):
        super().__init__()
        # 条件注入
        self.text_encoder = nn.Linear(text_emb_dim, 128)
        self.time_embed = nn.Embedding(1000, time_emb_dim)
        
        # 简化的 U-Net
        self.net = nn.Sequential(
            nn.Conv2d(latent_channels + 128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, 1, 1),
        )
    
    def forward(self, z_t, t, text_emb):
        B, C, H, W = z_t.shape
        
        # 文本条件注入
        t_emb = self.time_embed(t)[:, :, None, None].expand(-1, -1, H, W)
        text_feat = self.text_encoder(text_emb)[:, :, None, None].expand(-1, -1, H, W)
        
        # 拼接条件
        z_in = torch.cat([z_t, t_emb + text_feat], dim=1)
        return self.net(z_in)

# 声码器（使用 HiFi-GAN 的简化版本）
class SimpleVocoder(nn.Module):
    """将梅尔频谱图转换为音频波形"""
    def __init__(self, n_mels=64, hop_length=256):
        super().__init__()
        self.hop_length = hop_length
        # 简化的波形生成（实际使用 HiFi-GAN 或 Griffin-Lim）
        self.generator = nn.Sequential(
            nn.ConvTranspose1d(n_mels, 256, 8, 4, 2),
            nn.SiLU(),
            nn.ConvTranspose1d(256, 128, 8, 4, 2),
            nn.SiLU(),
            nn.Conv1d(128, 1, 7, 1, 3),
            nn.Tanh(),
        )
    
    def forward(self, mel):
        """mel: [B, n_mels, T] 梅尔频谱图 -> [B, 1, T*hop_length] 波形"""
        return self.generator(mel)

# 演示 AudioLDM 流程
print("=== AudioLDM 音频生成流程 ===")
print()
print("文本 -> CLAP 编码 -> 潜空间扩散 -> VAE 解码 -> 频谱图 -> 声码器 -> 波形")
print()
print("关键组件:")
print("1. CLAP: 文本-音频对比学习编码器")
print("2. 声学 VAE: 频谱图压缩 (压缩比 4x)")
print("3. Latent Diffusion: 在潜空间中扩散")
print("4. HiFi-GAN Vocoder: 频谱图 -> 波形")
print()

# 模拟流程
text_emb = torch.randn(1, 512)  # CLAP 编码
z_t = torch.randn(1, 4, 16, 64)  # 初始潜变量
t = torch.tensor([500])
model = AudioLDMDiffusion()

noise_pred = model(z_t, t, text_emb)
print(f"潜变量形状: {z_t.shape}")
print(f"预测噪声形状: {noise_pred.shape}")

# 频谱图生成
vae = AudioVAE()
latent = torch.randn(1, 4, 16, 64)
mel = vae.decode(latent)
print(f"生成梅尔频谱图形状: {mel.shape}")  # [1, 1, n_mels, T]

# 波形生成
vocoder = SimpleVocoder()
waveform = vocoder(mel.squeeze(1))  # [1, 1, T*256]
print(f"生成波形形状: {waveform.shape}")
print(f"对应音频时长: {waveform.size(-1) / 16000:.2f} 秒 (@16kHz)")
```

## 深度学习关联
- **AudioLDM 2**：改进版本，使用更强大的语言模型（FLAN-T5）作为文本编码器，并引入音频级和帧级的双条件机制，大幅提升了生成质量和文本对齐度。
- **Make-An-Audio / MusicGen**：Meta 的音乐生成模型，使用与 AudioLDM 相似的架构但针对音乐优化（支持更长的音频、更好的旋律控制）。
- **AudioLDM 与扩散模型的统一**：音频生成与图像生成共享相同的基础架构（条件扩散模型），这体现了扩散模型作为通用生成范式的潜力——同样的方法可以应用于不同的模态。
- **实时音频生成**：最新的研究（如 FastDiff, WaveGrad）探索了用更少的采样步数（6-10 步）生成高质量音频，向实时生成方向迈进。
