# 26_超分辨率扩散模型 (Super-Resolution)

## 核心概念

- **超分辨率 (Super-Resolution, SR)**：从低分辨率（LR）图像重建高分辨率（HR）图像的任务，瓶颈在于"补全缺失的高频信息"（ill-posed inverse problem）。
- **SR3 / SRDiff / LDM-SR**：将扩散模型应用于超分辨率任务的代表性工作——以低分辨率图像作为条件，通过去噪扩散过程生成高分辨率图像。
- **条件扩散超分**：前向过程从 HR 图像 $x$ 逐步加噪，反向去噪过程以 LR 图像 $y$ 和噪声图像 $x_t$ 的拼接作为条件，训练网络 $p_\theta(x_{t-1}|x_t, y)$。
- **级联超分辨率 (Cascaded SR)**：将超分分解为多个阶段——4x → 4x → 4x（总放大 64x），每个阶段是一个独立的扩散模型，逐步提升分辨率。
- **退化模型 (Degradation Model)**：训练时需要模拟低分辨率图像的生成过程 $y = \text{Downsample}(x) + \text{noise}$，通常使用双三次下采样（bicubic）加高斯噪声。
- **与 GAN 超分的对比**：ESRGAN 等 GAN 超分方法往往产生"伪细节"（看起来很锐利但可能不忠实），扩散超分在感知质量和真实性之间有更好的平衡。

## 数学推导

**SR3 的条件扩散模型**：

前向过程（标准 DDPM，但以 $y$ 为中心）：

$$
q(x_t|x_{t-1}, y) = \mathcal{N}(x_t; \sqrt{1-\beta_t} x_{t-1}, \beta_t I)
$$

反向去噪过程（以低分辨率图像 $y$ 为条件）：

$$
p_\theta(x_{t-1}|x_t, y) = \mathcal{N}(x_{t-1}; \mu_\theta(x_t, y, t), \sigma_t^2 I)
$$

**低分辨率条件注入**：

将低分辨率图像 $y$ 上采样到与 $x_t$ 同尺寸，然后沿通道维度拼接：

$$
z_t = \text{Concat}(x_t, \text{Up}(y))
$$

其中 $\text{Up}$ 是双线性上采样或可学习的上采样层。

**训练损失**（与 DDPM 类似，预测噪声）：

$$
L = \mathbb{E}_{x_0, y, \epsilon, t} \left[ \left\| \epsilon - \epsilon_\theta(\sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon, \text{Up}(y), t) \right\|^2 \right]
$$

**采样过程**：

从纯噪声 $x_T \sim \mathcal{N}(\text{Up}(y), I)$ 开始（以 LR 图像为中心），逐步去噪：

$$
x_{t-1} = \frac{1}{\sqrt{\alpha_t}} \left( x_t - \frac{\beta_t}{\sqrt{1-\bar{\alpha}_t}} \epsilon_\theta(x_t, \text{Up}(y), t) \right) + \sigma_t z
$$

最终 $x_0$ 就是生成的高分辨率图像。

**级联超分的放大因子分配**：

若总放大倍数为 $64\times$（如 SR3），可以分解为：

$$
64\times = 4\times \times 4\times \times 4\times
$$

每个 $4\times$ 超分模型只需要学习从 $256^2 \to 1024^2$ 的映射，训练和数据存储都更加可行。

## 直观理解

- **超分扩散模型 = 痕迹专家复原老照片**：低分辨率照片（LR）就像一张模糊的嫌疑人画像，超分模型根据"这张脸应该在标准面部流形上"的先验知识，结合 LR 提供的结构约束，一步步恢复出高清人脸。
- **为什么条件拼接有效**：将 LR 图像上采样后与 $x_t$ 拼接，相当于给去噪网络一个"参考草图"。$x_t$ 提供当前的噪声状态，LR 提供固定的结构约束——网络学会在两者之间找到平衡。
- **级联超分 = 瀑布式精修**：先 4 倍放大（比如 64x64 → 256x256），再 4 倍放大（→ 1024x1024），最后 4 倍放大（→ 4096x4096）。每一级只需要学习增加"合理的高频细节"，而非一次性想象所有细节。
- **退化模型的重要性**：现实中的低分辨率图像可能是各种原因造成的（运动模糊、镜头像差、压缩失真）。如果训练时只用双三次下采样，模型在实际场景中效果会大打折扣。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class SuperResolutionDiffusion(nn.Module):
    """
    简化的超分辨率扩散模型
    
    以低分辨率图为条件，生成高分辨率图
    """
    def __init__(self, in_channels=3, out_channels=3, upscale_factor=4):
        super().__init__()
        self.upscale_factor = upscale_factor
        
        # LR 条件编码器
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, 128, 3, 1, 1),
            nn.SiLU(),
        )
        
        # 去噪 U-Net（接受拼接后的输入：噪声图像 + 条件特征）
        self.unet = nn.Sequential(
            nn.Conv2d(out_channels + 128, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 128, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(128, 64, 3, 1, 1),
            nn.SiLU(),
            nn.Conv2d(64, out_channels, 3, 1, 1),
        )
        
        self.time_embed = nn.Embedding(1000, 128)
    
    def forward(self, x_t, lr_img, t):
        """
        参数:
            x_t: 当前时间步的噪声图像 [B, C, H, W]（高分辨率尺寸）
            lr_img: 低分辨率图像 [B, C, h, w]
            t: 时间步 [B]
        """
        # 上采样低分辨率图像到高分辨率尺寸
        lr_up = F.interpolate(
            lr_img, 
            size=(x_t.size(2), x_t.size(3)),
            mode='bicubic', align_corners=False
        )
        
        # 编码条件
        cond_feat = self.cond_encoder(lr_up)
        
        # 时间步嵌入
        t_emb = self.time_embed(t)[:, :, None, None].expand(-1, -1, x_t.size(2), x_t.size(3))
        
        # 拼接噪声图像、条件特征和时间嵌入
        x_in = torch.cat([x_t, cond_feat + t_emb], dim=1)
        
        return self.unet(x_in)

# SR3 风格的级联超分辨率
class CascadedSR(nn.Module):
    """级联超分辨率：多个 4x 模型串联"""
    def __init__(self, stages=3, upscale_per_stage=4):
        super().__init__()
        self.stages = nn.ModuleList([
            SuperResolutionDiffusion(upscale_factor=upscale_per_stage)
            for _ in range(stages)
        ])
        self.upscale_per_stage = upscale_per_stage
    
    def forward(self, lr_img, t):
        """逐步超分"""
        x = lr_img
        for i, stage in enumerate(self.stages):
            # 初始化噪声图像（当前分辨率）
            H, W = x.size(2) * self.upscale_per_stage, x.size(3) * self.upscale_per_stage
            x_t = torch.randn(x.size(0), 3, H, W)  # 实际应该从扩散采样开始
            
            # 条件去噪（简化）
            noise_pred = stage(x_t, x, t)
            
            # 去噪（简化：实际需要完整的 DDPM 采样循环）
            x = x_t - noise_pred  # 简化的去噪
            
        return x

# 退化模型
def create_lr_image(hr_img, scale_factor=4, noise_std=0.05):
    """模拟低分辨率图像的退化过程"""
    # 1. 下采样
    lr = F.interpolate(hr_img, scale_factor=1/scale_factor, mode='bicubic')
    # 2. 加噪声
    if noise_std > 0:
        lr = lr + torch.randn_like(lr) * noise_std
    return lr

print("=== 超分辨率扩散模型 ===")
hr_img = torch.randn(1, 3, 256, 256)
lr_img = create_lr_image(hr_img, scale_factor=4)
print(f"高分辨率: {hr_img.shape}")
print(f"低分辨率: {lr_img.shape}")

sr_model = SuperResolutionDiffusion()
x_t = torch.randn(1, 3, 256, 256)
t = torch.tensor([500])
noise_pred = sr_model(x_t, lr_img, t)
print(f"预测噪声形状: {noise_pred.shape}")
print(f"级联超分参数量: {sum(p.numel() for p in sr_model.parameters()):,}")
```

## 深度学习关联

- **Stable Diffusion Upscaler**：Stable Diffusion 官方的 4x 超分模型，在潜空间中进行条件扩散超分——用 VAE 编码低分辨率图像的低频潜变量，扩散模型补充高频细节，然后 VAE 解码。
- **Real-ESRGAN vs 扩散超分**：Real-ESRGAN 使用更复杂的退化模型（模糊 + 噪声 + JPEG 压缩 + 下采样）训练 GAN 超分，速度快但伪细节多。扩散超分质量更高但速度慢。
- **图像修复 + 超分的结合**：实际应用中，超分常与去噪、去模糊、修复等任务联合——所有这些任务都可以用条件扩散模型的统一框架来处理。
- **视频超分**：Stable Video Diffusion 等模型将超分扩展到时间维度，利用相邻帧的信息来提升每一帧的分辨率。
