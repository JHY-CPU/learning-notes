# 46_Distilled Diffusion 蒸馏加速

## 核心概念

- **蒸馏加速 (Distilled Diffusion / Diffusion Distillation)**：将预训练的多步扩散模型（教师）的知识迁移到少步/单步生成模型（学生），显著减少推理时的采样步数。
- **Progress Distillation (渐进式蒸馏)**：分阶段减少采样步数——先将教师模型从 1000 步蒸馏到 100 步，再从 100 步到 10 步，最后到 1 步。
- **Score Distillation Sampling (SDS)**：DreamFusion 提出的方法——用预训练扩散模型的梯度信号来优化（非扩散）参数化生成器（如 NeRF），不需要真实图像。
- **对抗蒸馏 (Adversarial Diffusion Distillation, ADD)**：在蒸馏过程中引入对抗损失，让学生在少步生成时关注感知质量，弥补蒸馏中的细节损失。
- **Bootstrapping**：学生模型在每一步的预测"跳到"更远的时间步，目标函数强制学生从 $t$ 步直接预测 $t-2$ 步的结果（而非 $t-1$），从而加速收敛。
- **蒸馏 vs 直接训练**：蒸馏后的模型可以少步生成（1-8 步），但是单步生成质量通常略低于教师模型多步生成——这是质量与速度的经典权衡。

## 数学推导

**Progressive Distillation**：

教师模型 $\epsilon_\phi$ 在时间步 $t$ 和 $t+1$ 之间做一步去噪：

$$
\hat{x}_t = \text{denoise}(\epsilon_\phi, x_{t+1}, t+1)
$$

学生模型 $\epsilon_\theta$ 学习从 $x_{t+2}$ 直接跳到 $\hat{x}_t$（跳过中间步）：

$$
\mathcal{L}_{\text{distill}} = \mathbb{E}_{t, x_0, \epsilon} \left[ \| \epsilon_\theta(x_{t+2}, t+2) - \epsilon_\phi(x_{t+1}, t+1) \|^2 \right]
$$

每次蒸馏后，采样步数减半（1000→500→250→125...）。

**Score Distillation Sampling (SDS)**：

对于可微参数化生成器 $g(\psi)$（如 NeRF 的权重），SDS 的梯度为：

$$
\nabla_\psi \mathcal{L}_{\text{SDS}} = \mathbb{E}_{t, \epsilon} \left[ w(t) (\epsilon_\phi(g(\psi)_t, t, y) - \epsilon) \frac{\partial g(\psi)}{\partial \psi} \right]
$$

直观理解：SDS 用扩散模型的噪声预测误差作为"方向"，指导生成器参数 $\psi$ 的更新。

**ADD (对抗蒸馏) 损失**：

$$
\mathcal{L}_{\text{ADD}} = \mathcal{L}_{\text{distill}} + \lambda \mathcal{L}_{\text{adv}}
$$

其中 $\mathcal{L}_{\text{adv}}$ 是判别器 D 的对抗损失：

$$
\mathcal{L}_{\text{adv}} = \mathbb{E}_{x \sim p_{\text{data}}}[\log D(x)] + \mathbb{E}_{z, t}[\log(1 - D(g_\theta(z, t)))]
$$

## 直观理解

- **蒸馏 = 老师教学生"偷懒"的方法**：老师（教师模型）每一步都小心翼翼地走（1000 步去噪），学生学完说"哦，原来只要走 2 大步就能到"——虽然步子大了容易摔，但经过反复练习（蒸馏训练）也能走稳。
- **SDS = 用扩散模型做"教练"**：你不是让扩散模型自己生成图像，而是让它当教练——你在训练一个 3D 模型（运动员），扩散模型从旁指导"这个角度的渲染图不够像真图"，运动员据此改进。
- **为什么需要对抗损失**：只用蒸馏损失时，学生学会了大致的去噪方向，但细节经常模糊。对抗损失加入了一个"辨别真伪"的判官，强制学生关注细腻纹理。
- **渐进式蒸馏 = 先学会走再学会跑**：从 1000 步到 500 步（走➔快走），500 到 100（快走➔慢跑），100 到 10（慢跑➔快跑），10 到 1（快跑➔冲刺）。每一级都相对容易学习。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionDistillation:
    """扩散模型蒸馏"""
    
    @staticmethod
    def progressive_distillation_loss(student_model, teacher_model, 
                                       x_0, text_emb, step_factor=2):
        """
        渐进式蒸馏损失
        
        学生从 x_{t+step_factor} 直接预测到 x_tlevel
        老师提供"慢速"的目标
        """
        batch_size = x_0.size(0)
        
        # 随机时间步
        t = torch.randint(0, 1000 - step_factor, (batch_size,))
        
        noise = torch.randn_like(x_0)
        alpha_bar = 1 - t.float() / 1000.0
        alpha_bar_factor = 1 - (t + step_factor).float() / 1000.0
        
        # 老师的目标：从 x_{t+1} 到 x_t
        x_t_plus_1 = x_0 * alpha_bar_factor[:, None, None, None].sqrt() + \
                     noise * (1 - alpha_bar_factor[:, None, None, None]).sqrt()
        
        with torch.no_grad():
            teacher_target = teacher_model(x_t_plus_1, t + 1, text_emb)
        
        # 学生的输入：从 x_{t+2} 开始
        if t.max() < 1000 - step_factor * 2:
            alpha_bar_factor2 = 1 - (t + step_factor * 2).float() / 1000.0
            x_t_plus_2 = x_0 * alpha_bar_factor2[:, None, None, None].sqrt() + \
                         noise * (1 - alpha_bar_factor2[:, None, None, None]).sqrt()
            student_pred = student_model(x_t_plus_2, t + step_factor * 2, text_emb)
        else:
            student_pred = student_model(x_t_plus_1, t + 1, text_emb)
        
        # 蒸馏损失
        loss = F.mse_loss(student_pred, teacher_target)
        return loss
    
    @staticmethod
    def sds_loss(generator_params, diffusion_model, render_fn, 
                 text_emb, t=None):
        """
        Score Distillation Sampling (SDS) 损失
        
        参数:
            generator_params: 3D 生成器参数（NeRF 等）
            diffusion_model: 预训练扩散模型
            render_fn: 从参数生成渲染图的函数
            text_emb: 文本条件
        """
        # 渲染图像
        rendered_image = render_fn(generator_params)
        
        # 加噪
        noise = torch.randn_like(rendered_image)
        if t is None:
            t = torch.randint(0, 1000, (1,))
        
        alpha_bar = 1 - t.float() / 1000.0
        z_t = rendered_image * alpha_bar.sqrt() + noise * (1 - alpha_bar).sqrt()
        
        # 扩散模型预测噪声
        noise_pred = diffusion_model(z_t, t, text_emb)
        
        # SDS 梯度（不是损失值）
        sds_gradient = (noise_pred - noise)
        
        return sds_gradient

# 演示蒸馏加速
print("=== Distilled Diffusion 蒸馏加速 ===")
print()

print("常见蒸馏方法对比:")
print(f"  渐进式蒸馏:     1000→500→250→125→...→1-8 步")
print(f"  SDS:             用于 3D 生成 (DreamFusion)")
print(f"  对抗蒸馏 (ADD):  2-4 步, Stable Diffusion Turbo")
print(f"  一致性蒸馏:      1-4 步, LCM")
print()

# 速度-质量分析
print("蒸馏的速度与质量权衡:")
for steps in [1000, 100, 50, 10, 4, 1]:
    speedup = 1000 / steps
    quality = max(0, 1.0 - 0.02 * steps)  # 模拟质量评分
    print(f"  {steps:4d} 步: {speedup:5.1f}x 加速, 相对质量={quality:.2f}")

print()
print("关键结论:")
print("1. 蒸馏可以将 1000 步压缩到 1-8 步")
print("2. 1 步生成的质量约为多步生成的 80-90%")
print("3. 对抗损失显著改善少步生成的细节")
print("4. SDS 是实现文本到 3D 生成的关键技术")
```

## 深度学习关联

- **Stable Diffusion Turbo (SD Turbo)**：使用对抗蒸馏（ADD）将 SD 压缩到 1-4 步，质量接近 50 步 DDIM。这是目前最快的 SD 变体之一。
- **Latent Consistency Model (LCM)**：使用一致性蒸馏，通过 LoRA 微调将任意 SD 模型转换为 1-4 步生成模型。LCM + LoRA 仅需训练少量参数。
- **Instant3D / Zero123++**：将 SDS 应用于 3D 生成——用预训练的文本到图像扩散模型作为先验，通过 SDS 优化 3D 表示，实现从文本直接生成 3D 模型。
- **Video Diffusion Distillation**：视频生成的蒸馏更难（需要保持时间一致性），Align Your Latents 等方法通过对视频潜变量做蒸馏来实现快速视频生成。
