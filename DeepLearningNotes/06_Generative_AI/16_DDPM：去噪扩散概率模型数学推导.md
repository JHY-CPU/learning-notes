# 16_DDPM：去噪扩散概率模型数学推导

## 核心概念
- **DDPM** (Denoising Diffusion Probabilistic Models)：Ho et al. 2020 年的里程碑工作，首次证明扩散模型能生成媲美 GAN 的高质量图像。
- **两个马尔可夫链**：前向过程 $q(x_t|x_{t-1})$（固定，无参数）将数据变为噪声；反向过程 $p_\theta(x_{t-1}|x_t)$（可学习）将噪声变回数据。
- **重新参数化的妙处**：DDPM 的训练目标从"预测图像"巧妙转换为"预测噪声"——训练网络 $\epsilon_\theta(x_t, t)$ 来预测添加到 $x_t$ 的噪声 $\epsilon$。
- **简化损失函数**：去掉权重系数后的简化 MSE 损失（只预测噪声）比原始的变分下界训练更稳定，生成质量更高。
- **方差调度**：使用线性噪声调度 $\beta_1 = 10^{-4}$ 到 $\beta_T = 0.02$，前向过程共 $T=1000$ 步。
- **非马尔可夫训练**：训练时不需要迭代所有步——直接从 $x_0$ 和随机 $t$ 计算 $x_t$ 和损失，单步完成训练。

## 数学推导

**前向过程**：

$$
q(x_t|x_{t-1}) = \mathcal{N}(x_t; \sqrt{1-\beta_t}x_{t-1}, \beta_t I)
$$

**重参数化**（任意 $t$ 步的 $x_t$ 可直接计算）：

$$
x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \quad \epsilon \sim \mathcal{N}(0, I)
$$

其中 $\alpha_t = 1-\beta_t$，$\bar{\alpha}_t = \prod_{s=1}^t \alpha_s$。

**反向过程**：

$$
p_\theta(x_{t-1}|x_t) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, t), \sigma_t^2 I)
$$

**关键推导：反向过程的均值**：

根据贝叶斯定理和条件高斯分布公式：

$$
q(x_{t-1}|x_t, x_0) = \mathcal{N}(x_{t-1}; \tilde{\mu}_t(x_t, x_0), \tilde{\beta}_t I)
$$

其中：

$$
\tilde{\mu}_t(x_t, x_0) = \frac{\sqrt{\bar{\alpha}_{t-1}}\beta_t}{1-\bar{\alpha}_t} x_0 + \frac{\sqrt{\alpha_t}(1-\bar{\alpha}_{t-1})}{1-\bar{\alpha}_t} x_t
$$

$$
\tilde{\beta}_t = \frac{1-\bar{\alpha}_{t-1}}{1-\bar{\alpha}_t} \beta_t
$$

将 $x_0 = \frac{x_t - \sqrt{1-\bar{\alpha}_t}\epsilon}{\sqrt{\bar{\alpha}_t}}$ 代入，得到用噪声 $\epsilon$ 表示的形式：

$$
\tilde{\mu}_t(x_t, t) = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon \right)
$$

**DDPM 简化损失函数**：

$$
L_{\text{simple}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ \|\epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, t)\|^2 \right]
$$

**采样（生成）过程**：

从 $x_T \sim \mathcal{N}(0, I)$ 开始，逆序遍历 $t = T, ..., 1$：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, t) \right) + \sigma_t z, \quad z \sim \mathcal{N}(0, I)
$$

其中当 $t=1$ 时 $z=0$（最后一步不加噪声）。

## 直观理解
- **训练 = 学习盲人摸象的逆向**：正向过程让大象逐渐消失（变成噪声），反向过程学习从噪声中重建大象。DDPM 的训练目标特别巧妙——不是直接学"如何重建大象"，而是学"每一步消失了什么"（预测噪声）。
- **为什么预测噪声比预测图像更容易**：因为 $x_t$ 里大部分是噪声（大 $t$ 时）或大部分是信号（小 $t$ 时），预测图像需要在不同信噪比下都准确。而预测噪声 $\epsilon$ 在各个时间步上分布一致（标准正态），网络更容易学习。
- **从棋盘到纯色**：想象你在看一个棋盘（原图），逐渐有人把白纸盖上去（噪声），最后看不到棋盘只剩白纸（纯噪声）。反向过程就是一步步揭开白纸——每次揭开一点，方向由网络决定。
- **DDPM 的关键洞察**：Ho et al. 发现去掉变分下界中的权重项，仅用 $\|\epsilon - \epsilon_\theta\|^2$ 反而效果更好——这等价于在所有时间步上给损失赋予了等权重，而不是变分推导出的非均匀权重。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionEmbeddings(nn.Module):
    """时间步 t 的正弦位置编码"""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
    
    def forward(self, time):
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=time.device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        return torch.cat([torch.sin(embeddings), torch.cos(embeddings)], dim=-1)

class SimpleUNet(nn.Module):
    """简化的 U-Net（用于 DDPM 去噪）"""
    def __init__(self, in_channels=3, time_emb_dim=128):
        super().__init__()
        self.time_mlp = nn.Sequential(
            SinusoidalPositionEmbeddings(time_emb_dim),
            nn.Linear(time_emb_dim, time_emb_dim),
            nn.ReLU(),
        )
        
        # 简化的 U-Net 结构
        self.enc1 = nn.Conv2d(in_channels, 64, 3, 1, 1)
        self.enc2 = nn.Conv2d(64, 128, 3, 2, 1)
        self.enc3 = nn.Conv2d(128, 256, 3, 2, 1)
        
        self.bottleneck = nn.Sequential(
            nn.Conv2d(256 + time_emb_dim, 256, 3, 1, 1),
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, 1, 1),
        )
        
        self.dec3 = nn.ConvTranspose2d(256, 128, 4, 2, 1)
        self.dec2 = nn.ConvTranspose2d(128, 64, 4, 2, 1)
        self.dec1 = nn.Conv2d(64, in_channels, 3, 1, 1)
    
    def forward(self, x, t):
        t_emb = self.time_mlp(t)
        
        e1 = torch.relu(self.enc1(x))
        e2 = torch.relu(self.enc2(e1))
        e3 = torch.relu(self.enc3(e2))
        
        # 将时间嵌入拼接到瓶颈层
        t_emb_expanded = t_emb[:, :, None, None].expand(-1, -1, e3.size(2), e3.size(3))
        b = torch.cat([e3, t_emb_expanded], dim=1)
        b = self.bottleneck(b)
        
        d3 = torch.relu(self.dec3(b))
        d2 = torch.relu(self.dec2(d3 + e2))
        d1 = self.dec1(d2 + e1)
        return d1

class DDPM(nn.Module):
    """去噪扩散概率模型"""
    def __init__(self, model, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        super().__init__()
        self.model = model
        self.timesteps = timesteps
        
        # 噪声调度
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1. - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
    
    def forward_diffusion(self, x_0, t, noise=None):
        """q(x_t|x_0): 从 x_0 一步计算 x_t"""
        if noise is None:
            noise = torch.randn_like(x_0)
        sqrt_alpha_bar = self.alpha_bars[t].sqrt()[:, None, None, None]
        sqrt_one_minus_alpha_bar = (1. - self.alpha_bars[t]).sqrt()[:, None, None, None]
        return sqrt_alpha_bar * x_0 + sqrt_one_minus_alpha_bar * noise, noise
    
    def sample(self, batch_size, img_channels=3, img_size=32, device='cpu'):
        """反向去噪生成新样本"""
        self.eval()
        x_t = torch.randn(batch_size, img_channels, img_size, img_size).to(device)
        
        for t in reversed(range(self.timesteps)):
            t_tensor = torch.full((batch_size,), t, device=device, dtype=torch.long)
            predicted_noise = self.model(x_t, t_tensor)
            
            # DDPM 更新公式
            alpha = self.alphas[t]
            alpha_bar = self.alpha_bars[t]
            beta = self.betas[t]
            
            coeff = 1 / torch.sqrt(alpha)
            noise_term = (beta / torch.sqrt(1 - alpha_bar)) * predicted_noise
            
            x_t_minus_1 = coeff * (x_t - noise_term)
            
            # 除最后一步外添加随机噪声
            if t > 0:
                noise = torch.randn_like(x_t)
                sigma_t = torch.sqrt(beta)
                x_t_minus_1 += sigma_t * noise
            
            x_t = x_t_minus_1
        
        return x_t

# 训练步骤
def train_ddpm_step(ddpm, optimizer, x_0, device='cpu'):
    """DDPM 单步训练"""
    batch_size = x_0.size(0)
    t = torch.randint(0, ddpm.timesteps, (batch_size,), device=device)
    
    noise = torch.randn_like(x_0)
    x_t, _ = ddpm.forward_diffusion(x_0, t, noise)
    
    predicted_noise = ddpm.model(x_t, t)
    loss = nn.functional.mse_loss(predicted_noise, noise)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 初始化
model = SimpleUNet(in_channels=3)
ddpm = DDPM(model, timesteps=1000)
optimizer = torch.optim.Adam(model.parameters(), lr=2e-4)

x_dummy = torch.randn(4, 3, 32, 32)
loss = train_ddpm_step(ddpm, optimizer, x_dummy)
print(f"DDPM 训练损失: {loss:.6f}")
print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联
- **Improved DDPM**：对 DDPM 的改进——使用余弦噪声调度（减少早期噪声添加过快的问题）、学习方差 $\Sigma_\theta$（不只是均值）、将 $T$ 从 1000 减到 50 步仍保持不错质量。
- **Diffusion Beat GANs**：2021 年 OpenAI 的工作通过改进架构（使用更大的 U-Net、自适应 GroupNorm）证明了扩散模型能在图像质量（FID）上超过 GAN。
- **Latent Diffusion (Stable Diffusion)**：将 DDPM 的扩散过程从像素空间搬到 VAE 潜空间，计算量减少一个数量级，同时保持生成质量。
- **DDPM 的采样加速**：原始 DDPM 需要 1000 步采样，后续的 DDIM、DPM-Solver、LCM 等技术将采样步数减少到 1-50 步，使得扩散模型可用于实时应用。
