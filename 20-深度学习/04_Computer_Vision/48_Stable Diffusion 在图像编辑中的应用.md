# 48_Stable Diffusion 在图像编辑中的应用

## 核心概念

- **Stable Diffusion (SD)**：Rombach et al. (2022) 提出的潜在扩散模型（Latent Diffusion Model, LDM），在压缩的潜在空间（而非像素空间）中执行扩散过程，大幅降低计算开销。
- **潜在扩散模型的核心思想**：使用预训练的自编码器（VAE）将图像压缩到低维潜在空间，在潜在空间中进行前向加噪和反向去噪过程。
- **UNet + 交叉注意力**：去噪UNet通过交叉注意力层（cross-attention）接收文本条件（CLIP文本编码器的输出），实现文本引导的图像生成。
- **图像编辑应用**：Stable Diffusion不仅用于文本到图像生成（text-to-image），还通过多种技术实现图像编辑：Inpainting（填充）、Image-to-Image（图到图翻译）、InstructPix2Pix（指令编辑）、DreamBooth（主体定制）等。
- **SD编辑的核心机制**：通过**反转（Inversion）** 将输入图像编码到噪声空间中，然后在去噪过程中修改条件引导（文本提示、注意力控制等），从而编辑生成结果。
- **注意力控制（Attention Control）**：通过编辑UNet中的交叉注意力图（cross-attention maps），可以控制编辑的区域和强度，实现"指定区域编辑"或"保留特定物体结构"。

## 数学推导

**潜在扩散模型的前向（加噪）过程：**
$$
q(z_t | z_{t-1}) = \mathcal{N}(z_t; \sqrt{1-\beta_t} z_{t-1}, \beta_t I)
$$

其中 $z_t = \mathcal{E}(x)$ 是图像通过VAE编码器得到的潜在表示。

**反向（去噪）过程（UNet预测噪声）：**
$$
p_\theta(z_{t-1} | z_t, c) = \mathcal{N}(z_{t-1}; \mu_\theta(z_t, t, c), \Sigma_t)
$$

其中 $c$ 是条件信息（文本嵌入）。

**去噪UNet的训练目标（简化）：**
$$
\mathcal{L}_{LDM} = \mathbb{E}_{z, c, \epsilon \sim \mathcal{N}(0,1), t} \left[ \|\epsilon - \epsilon_\theta(z_t, t, c)\|_2^2 \right]
$$

**DDIM反转（用于图像编辑）：**
给定输入图像 $x_0$，编码为 $z_0 = \mathcal{E}(x_0)$，通过DDIM（Denoising Diffusion Implicit Models）反转找到对应的 $z_T$：
$$
z_{t+1} = \sqrt{\bar{\alpha}_{t+1}} f_\theta(z_t, t, c) + \sqrt{1 - \bar{\alpha}_{t+1}} \epsilon_\theta(z_t, t, c)
$$

其中 $f_\theta(\cdot)$ 是预测的 $z_0$。通过反转得到 $z_T$ 后，再用新的条件 $c'$ 进行去噪，实现编辑。

**Classifier-Free Guidance (CFG)：**
$$
\tilde{\epsilon}_\theta(z_t, t, c) = \epsilon_\theta(z_t, t, \emptyset) + w \cdot (\epsilon_\theta(z_t, t, c) - \epsilon_\theta(z_t, t, \emptyset))
$$

其中 $w > 1$ 是引导尺度（guidance scale），$\emptyset$ 是无条件输入（空文本）。$w$ 越大，生成结果越忠于文本条件但多样性降低。

## 直观理解

Stable Diffusion的图像生成可以想象为一个"从噪声中逐步浮现图像"的过程。初始状态是一张纯噪声图（就像电视雪花屏），通过反复应用"去噪神经网络"，噪声逐渐被移除，清晰的图像逐步显现。

LSDA在潜在空间而非像素空间操作——可以理解为先用VAE的编码器将图像"压缩"成一个紧凑的表示（类似于JPEG压缩，但更加语义化），在这个压缩表示空间中进行去噪，最后用VAE解码器将清洁的潜在表示还原为像素图像。

在图像编辑中，最关键的技术是"反转"——从一张真实图像出发，通过正向添加噪声得到噪声图，然后用编辑后的文本提示重新去噪。就像一个雕塑家先在原作品上覆盖一层泥（加噪），然后按照新的设计重新雕刻（条件去噪）。

## 代码示例

```python
import torch
import torch.nn as nn

class SimplifiedLDM(nn.Module):
    """简化的潜在扩散模型组件"""
    def __init__(self, latent_dim=4, text_dim=768, time_dim=256):
        super().__init__()
        # 简化的去噪UNet (用全连接替代)
        self.time_mlp = nn.Sequential(
            nn.Linear(1, time_dim), nn.SiLU(),
            nn.Linear(time_dim, time_dim),
        )
        self.denoise_net = nn.Sequential(
            nn.Linear(latent_dim + text_dim + time_dim, 512),
            nn.SiLU(),
            nn.Linear(512, 512),
            nn.SiLU(),
            nn.Linear(512, latent_dim),
        )

    def predict_noise(self, z_t, t, c):
        """预测噪声"""
        t_emb = self.time_mlp(t.view(-1, 1))
        x = torch.cat([z_t, c, t_emb], dim=-1)
        return self.denoise_net(x)

# DDIM采样 (简化)
def ddim_sampling(model, z_T, c, num_steps=50, eta=0.0):
    """DDIM采样过程"""
    z = z_T
    timesteps = torch.linspace(1.0, 0.0, num_steps + 1)
    for i in range(num_steps):
        t = timesteps[i].unsqueeze(0)
        t_next = timesteps[i + 1].unsqueeze(0)
        eps_pred = model.predict_noise(z, t, c)
        
        # DDIM更新
        alpha_t = t  # 简化的alpha调度
        alpha_next = t_next
        z0_pred = (z - torch.sqrt(1 - alpha_t) * eps_pred) / torch.sqrt(alpha_t)
        z = torch.sqrt(alpha_next) * z0_pred + torch.sqrt(1 - alpha_next) * eps_pred
    return z

# Classifier-Free Guidance 的实现
def cfg_denoise(model, z_t, t, c, uncond_c, guidance_scale=7.5):
    """CFG引导去噪"""
    # 准备条件和无条件输入
    z_double = torch.cat([z_t, z_t], dim=0)
    t_double = torch.cat([t, t], dim=0)
    c_double = torch.cat([c, uncond_c], dim=0)
    
    eps_pred = model.predict_noise(z_double, t_double, c_double)
    eps_cond, eps_uncond = eps_pred.chunk(2, dim=0)
    
    # CFG
    eps = eps_uncond + guidance_scale * (eps_cond - eps_uncond)
    return eps

# 演示
model = SimplifiedLDM()
z_T = torch.randn(1, 4)  # 随机潜在噪声
t = torch.tensor([0.5])
c = torch.randn(1, 768)  # 文本嵌入

z_0 = ddim_sampling(model, z_T, c, num_steps=10)
print(f"生成的潜在表示: {z_0.shape}")

# 模拟编辑流程
print("\nStable Diffusion 图像编辑流程:")
print("1. 原图 -> VAE编码 -> 潜在表示 z_0")
print("2. DDIM反转: z_0 -> z_T (加噪到纯噪声)")
print("3. 修改文本提示: '一只猫' -> '一只戴着帽子的猫'")
print("4. CFG去噪: z_T -> z_0' (用新文本条件)")
print("5. VAE解码: z_0' -> 编辑后的图像")
```

## 深度学习关联

- **图像生成与编辑的标准化工具**：Stable Diffusion已经成为文本到图像生成和图像编辑的事实标准工具，基于其生态（LoRA微调、ControlNet控制、DreamBooth主体定制）涌现了大量创作工具（Automatic1111 WebUI、ComfyUI等）。
- **扩散模型的高效化之路**：Stable Diffusion将扩散模型从像素空间迁移到潜在空间，大幅降低了计算成本。后续的SDXL（更大模型）、SD Turbo（一步生成）、FLUX等在此基础上持续提升质量、速度和分辨率。
- **可控生成与编辑技术**：基于Stable Diffusion的图像编辑技术持续涌现——InstructPix2Pix（自然语言指令编辑）、DragGAN/DragDiffusion（拖拽式编辑）、Freecontrol（无需训练的注意力控制）等，使图像编辑更加灵活和直观。
