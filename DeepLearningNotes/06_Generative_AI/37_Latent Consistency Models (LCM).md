# 37_Latent Consistency Models (LCM)

## 核心概念
- **LCM (Latent Consistency Model)**：将一致性模型（Consistency Model）应用于潜空间扩散模型（Stable Diffusion）的加速推理技术，实现 1-4 步生成高质量图像。
- **潜空间中的一致性蒸馏**：LCM 在 VAE 潜空间中运行——一致性蒸馏在潜空间中执行，而非像素空间。这使得 LCM 可以直接应用到已有的 Stable Diffusion 模型。
- **LoRA 微调适配**：LCM 通常使用 LoRA 方式微调，只需要训练较小的参数集（约 50M 参数）就能将任意 Stable Diffusion 模型转换为 LCM 模型。
- **无分类器引导集成 (Guided Distillation)**：LCM 的蒸馏过程直接将无分类器引导（CFG）整合到一致性映射中——在训练时使用 CFG 强度 $w$ 作为条件，推理时一步生成即可获得引导效果。
- **小步数大跳步**：不同于一致性模型从 $T=1000$ 一步跳到 $t=0$，LCM 可以在 1-8 步之间灵活选择——步数越多质量越高。
- **实时交互应用**：LCM 的 1-4 步推理使得交互式图像生成成为可能——用户可以在几毫秒内获得生成结果，而非传统扩散模型的几秒。

## 数学推导

**LCM 的蒸馏目标**：

给定预训练的潜空间扩散模型 $\epsilon_\phi(z_t, t, c)$ 和 CFG 扩展：

$$
\epsilon_{\text{cfg}}(z_t, t, c, w) = \epsilon_\phi(z_t, t, \emptyset) + w \cdot (\epsilon_\phi(z_t, t, c) - \epsilon_\phi(z_t, t, \emptyset))
$$

LCM 学习一个一致性函数 $f_\theta(z_t, t, c, w)$，从噪声潜变量 $z_t$ 直接映射到干净的潜变量 $z_0$。

**蒸馏训练数据生成**：

为 ODE 轨迹上的相邻点 $(z_{t_{n+1}}, t_{n+1})$ 和 $(z_{t_n}, t_n)$，用教师模型的单步 ODE 从 $t_{n+1}$ 更新到 $t_n$。

**LCM 损失函数**：

$$
\mathcal{L}_{\text{LCM}}(\theta) = \mathbb{E}_{z_0, c, w, n} \left[ \| f_\theta(z_{t_{n+1}}, t_{n+1}, c, w) - f_\theta(\hat{z}_{t_n}, t_n, c, w) \|_2^2 \right]
$$

其中 $\hat{z}_{t_n}$ 是使用教师模型的得分从 $z_{t_{n+1}}$ 经过单步 ODE 得到的结果。

**边界条件**：

$$
f_\theta(z, 0, c, w) = z
$$

通过跳跃连接确保：对于任意 $t$，$f_\theta(z, 0, c, w) = \text{skip}(t) \cdot z + \text{out}(t) \cdot f_{\theta, \text{net}}(z, t, c, w)$，其中 $\text{skip}(0)=1$, $\text{out}(0)=0$。

**LCM 推理（1 步）**：

$$
z_T \sim \mathcal{N}(0, I), \quad z_0 = f_\theta(z_T, T, c, w_{\text{cfg}}), \quad \text{image} = \text{Decoder}(z_0)
$$

## 直观理解
- **LCM = 给 Stable Diffusion 装了火箭推进器**：普通 SD 需要 50 步 DDIM 采样（约 5 秒），LCM 只需要 1-4 步（约 0.05-0.2 秒）。在同一张 GPU 上，LCM 将生成速度提升了 10-50 倍。
- **引导蒸馏的特殊设计**：传统一致性蒸馏不包含 CFG，需要在推理时额外计算——意味着一个步需要两次网络前向推理。LCM 将 $w$ 作为条件直接学习"已经去过引导的去噪声"，一步到位。
- **为什么 LCM 不损失太多质量**：从潜空间到图像的 VAE 解码是一个"大容错"过程——潜变量上的小误差在解码后可能不明显。LCM 利用了这一特性，用更粗的去噪换取更大的速度。
- **1 步 vs 4 步**：1 步生成的图像在细节上不如 4 步精细，但构图和内容已经可用。4 步生成的图像与 50 步 DDIM 质量相当，但速度快 12.5 倍。

## 代码示例

```python
import torch
import torch.nn as nn

class LCMWrapper(nn.Module):
    """
    LCM 包装器：将预训练 SD 的 U-Net 转换为一致性模型
    
    实际使用中，LCM 通过 LoRA 微调 UNet 的交叉注意力层
    """
    def __init__(self, unet, lora_rank=4, lora_alpha=1.0):
        super().__init__()
        self.unet = unet
        
        # 冻结 U-Net 所有参数
        for param in unet.parameters():
            param.requires_grad = False
        
        # 添加 LoRA 适配层（简化：只添加到第一个交叉注意力）
        self.lora_layers = nn.ModuleDict()
        self._add_lora_to_attention(lora_rank, lora_alpha)
        
        # 边界条件所需的跳跃连接参数
        self.skip_weight = nn.Parameter(torch.tensor(0.0))
    
    def _add_lora_to_attention(self, rank, alpha):
        """为交叉注意力层添加 LoRA"""
        # 实际 LCM 使用完整的 LoRA 实现
        # 此处为概念演示
        pass
    
    def forward(self, z_t, t, text_emb, cfg_scale=7.5):
        """
        LCM 前向：从 z_t 预测 z_0
        
        包含 CFG 效果的时间嵌入
        """
        # 实际 LCM 已将 CFG 效果融入一致性映射
        # 这里进行简化处理
        
        # 时间步嵌入
        t_emb = t[:, None, None, None].float() / 1000.0
        
        # 从 z_t 接跳跃连接到 z_0 预测
        output = self.unet(z_t, t, text_emb).sample
        
        # 跳跃连接保证边界条件
        skip_factor = torch.sigmoid(self.skip_weight) * (1 - t_emb)
        z_0_pred = (1 - skip_factor) * output + skip_factor * z_t
        
        return z_0_pred
    
    @torch.no_grad()
    def sample(self, text_emb, num_steps=4, img_size=(1, 4, 64, 64), device='cpu'):
        """
        LCM 采样
        可以用 1-8 步
        """
        z_t = torch.randn(*img_size).to(device)
        
        if num_steps == 1:
            # 一步生成
            t = torch.full((img_size[0],), 999, dtype=torch.long, device=device)
            z_0 = self.forward(z_t, t, text_emb)
            return z_0
        
        # 多步生成（一致性模型的多步采样）
        # 均匀分布时间步
        timesteps = torch.linspace(999, 0, num_steps + 1, dtype=torch.long, device=device)
        
        for i in range(num_steps):
            t_cur = timesteps[i]
            t_next = timesteps[i + 1]
            
            t_tensor = torch.full((img_size[0],), t_cur, dtype=torch.long, device=device)
            z_0_pred = self.forward(z_t, t_tensor, text_emb)
            
            if i < num_steps - 1:
                # 根据一致性属性，从 z_0 前向到下一个步骤
                noise = torch.randn_like(z_t)
                alpha = 1 - t_next.float() / 1000.0
                z_t = torch.sqrt(alpha) * z_0_pred + torch.sqrt(1 - alpha) * noise
        
        return z_0_pred

# LCM 蒸馏训练步骤
def lcm_distillation_step(lcm_model, teacher_model, z_0, text_emb, cfg_scale=7.5, delta=10):
    """
    LCM 蒸馏训练
    
    参数:
        lcm_model: 学生模型（一致性模型）
        teacher_model: 教师模型（原始 SD U-Net）
        z_0: 干净的潜变量
        text_emb: 文本嵌入
        cfg_scale: CFG 强度
        delta: 时间步间隔
    """
    batch_size = z_0.size(0)
    device = z_0.device
    
    # 1. 随机选择时间步
    t = torch.randint(delta, 1000 - delta, (batch_size,), device=device)
    
    # 2. 生成 x_{t+delta}
    noise = torch.randn_like(z_0)
    alpha_bar_plus = (1 - (t + delta).float() / 1000.0).view(-1, 1, 1, 1)
    z_t_plus = torch.sqrt(alpha_bar_plus) * z_0 + torch.sqrt(1 - alpha_bar_plus) * noise
    
    # 3. 教师模型：单步 ODE 更新 t+delta -> t
    with torch.no_grad():
        # 无条件预测（简化）
        pred_uncond = teacher_model(z_t_plus, t + delta, torch.zeros_like(text_emb)).sample
        pred_cond = teacher_model(z_t_plus, t + delta, text_emb).sample
        
        # CFG 预测
        pred_cfg = pred_uncond + cfg_scale * (pred_cond - pred_uncond)
        
        # ODE 更新：z_t ≈ z_{t+delta} + delta * f_theta
        z_t = z_t_plus - (delta.float() / 1000.0) * pred_cfg  # 简化 ODE
    
    # 4. 学生模型：自一致性损失
    pred_1 = lcm_model(z_t_plus, t + delta, text_emb, cfg_scale)
    pred_2 = lcm_model(z_t, t, text_emb, cfg_scale)
    
    # 停止预测 1 的梯度（作为目标）
    loss = nn.functional.mse_loss(pred_2, pred_1.detach())
    
    return loss

print("=== Latent Consistency Models ===")
print()
print("LCM 的关键特性:")
print("1. 在 VAE 潜空间中运行（兼容 SD 生态）")
print("2. 1-4 步生成高质量图像")
print("3. 将 CFG 直接蒸馏到一致性映射中")
print("4. 可用 LoRA 高效微调")
print()
print("推理速度对比 (单张 GPU):")
print(f"  SD + DDIM (50步): ~5 秒")
print(f"  LCM (4步):        ~0.4 秒 (12.5x 加速)")
print(f"  LCM (1步):        ~0.1 秒 (50x 加速)")
print()

# 模拟
z_t = torch.randn(1, 4, 64, 64)
text_emb = torch.randn(1, 77, 768)
t = torch.tensor([500])
print(f"潜变量形状: {z_t.shape}")
print(f"文本嵌入形状: {text_emb.shape}")
print("LCM 使实时交互式图像生成成为可能！")
```

## 深度学习关联
- **LCM-LoRA**：LCM 的 LoRA 版本，只需要训练一组 LoRA 权重就能将任意 SD 模型快速转换为 LCM，无需全量微调。这使得 LCM 成为社区广泛采用的加速方案。
- **Turbo / LCM-LoRA / SDXL Turbo**：多种加速技术的竞争——Turbo 使用对抗蒸馏，LCM 使用一致性蒸馏，SDXL Turbo 使用渐进式蒸馏。它们都实现了 1-4 步生成，但各有优劣。
- **LCM 与实时视频生成**：LCM 的 1 步生成能力使其成为实时视频生成的关键技术——结合 AnimateDiff 的运动模块和 LCM 的一步采样，可以实现实时文本到视频的生成。
- **ControlNet-LCM 组合**：将 ControlNet 与 LCM 结合——使用 LCM 加速采样，同时用 ControlNet 保持空间控制力。这使得用户可以实时调整条件（如姿态、边缘）并立即看到生成结果。
