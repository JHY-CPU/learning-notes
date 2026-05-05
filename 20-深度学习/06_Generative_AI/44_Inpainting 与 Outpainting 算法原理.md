# 44_Inpainting 与 Outpainting 算法原理

## 核心概念

- **Inpainting (图像修复/内补全)**：用生成的内容智能填充图像中的缺失或遮挡区域，使修复后的图像看起来自然连贯。
- **Outpainting (图像外延/外补全)**：将图像向四周扩展，生成超出原始边界的内容，保持与原始图像的风格和内容的连贯性。
- **掩码引导 (Mask-Guided)**：Inpainting 使用二值掩码 $m$ 标注需要修复的区域（$m=1$ 表示缺失区域），模型在掩码条件下生成填充内容。
- **条件扩散修复**：扩散模型在每一步去噪时，将已知区域的 $x_t$ 固定（用加噪后的原图替换），只更新掩码区域——确保已知区域不受影响。
- **图像扩展的一致性**：Outpainting 要求在扩展区域生成的内容不仅在风格上匹配，还要在结构上合理——例如，延伸天空的渐变色、延伸建筑物的线条、保持地面纹理的连续性。
- **RePaint / Blended Diffusion**：代表性的扩散模型修复方法——RePaint 使用"重采样"技巧在掩码区域内进行额外的去噪-加噪循环来改善一致性；Blended Diffusion 使用背景信息引导掩码区域的颜色和风格匹配。

## 数学推导

**Inpainting 的条件扩散过程**：

给定原图像 $x_{\text{orig}}$ 和掩码 $m \in \{0, 1\}^{H \times W}$（1 表示修复区域）。

在时间步 $t$，已知区域应该被加噪后的原图覆盖：

$$
x_t^{\text{known}} = \sqrt{\bar{\alpha}_t} x_{\text{orig}} + \sqrt{1-\bar{\alpha}_t} \epsilon
$$

掩码区域使用模型生成的噪声预测 $x_t^{\text{generated}}$：

$$
x_t = m \odot x_t^{\text{generated}} + (1 - m) \odot x_t^{\text{known}}
$$

**RePaint 的 Resampling 策略**：

为了解决已知区域和修复区域之间的边界不一致问题，RePaint 在每一步做额外的"去噪-加噪"循环：

$$
x_{t-1} \to \tilde{x}_{t-1} \quad \text{（去噪步骤）}
$$

$$
\tilde{x}_{t-1} \to \hat{x}_{t-1} \quad \text{（已知区域用加噪原图替换）}
$$

$$
\hat{x}_{t-1} \to \hat{x}_t \quad \text{（重新加噪到 } t \text{，重做几次）}
$$

**Outpainting 的数学**：

将原始图像 $x_{\text{orig}} \in \mathbb{R}^{H \times W \times 3}$ 放置在大画布 $x_{\text{canvas}} \in \mathbb{R}^{H' \times W' \times 3}$ 的指定位置，掩码 $m$ 标记扩展区域。

$$
x_{\text{canvas}}[h_0:h_0+H, w_0:w_0+W] = x_{\text{orig}}
$$

扩散过程以 $x_{\text{canvas}}$ 为条件，在掩码区域生成新内容，非掩码区域保持原始内容。

**颜色/结构一致性损失**：

可以通过对边界区域的特征匹配进行约束来改善一致性：

$$
\mathcal{L}_{\text{boundary}} = \|\phi(x_{\text{canvas}})_{\text{boundary}} - \phi(x_{\text{orig}})_{\text{boundary}}\|^2
$$

其中 $\phi$ 是感知特征提取器（如 VGG）。

## 直观理解

- **Inpainting = AI 版本的修复老照片**：就像你有一张破损的老照片（缺失部分涂黑），AI 根据周围的上下文推断缺失区域的内容——"这里应该是墙纸"、"这里应该是桌角"。优秀的修复甚至能恢复出照片中的文字细节。
- **Outpainting = AI 版本的"看到画框外的世界"**：你给 AI 一张画着猫的图片，AI 猜测画框外应该是什么——"猫坐的这张桌子应该延伸到画框外，桌上可能还有一个花瓶"。
- **掩码区域固定 = 盲人摸象时一只手固定不动**：在修复时，已知部分被强制保持原样——模型只能修改掩码区域。这就像盲人摸象时，已经摸到的地方不再更改，只探索未知区域。
- **RePaint 的重采样 = 反复涂抹边界**：修复区域和原始区域的边界最容易出现不连贯。RePaint 的重新采样相当于在边界上来回涂抹——"这边再擦掉一点重画，那边再遮盖一点"，直到边界自然融合。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class DiffusionInpainting:
    """基于扩散模型的 Inpainting"""
    def __init__(self, noise_scheduler):
        self.scheduler = noise_scheduler
    
    def forward_diffusion_known(self, x_orig, t):
        """已知区域的加噪"""
        noise = torch.randn_like(x_orig)
        x_t = self.scheduler.add_noise(x_orig, noise, t)
        return x_t, noise
    
    def inpaint_step(self, model, x_t, t, mask, x_orig_noisy):
        """
        单步修复
        
        参数:
            x_t: 当前时间步的噪声图像
            t: 时间步
            mask: 修复区域掩码 (1=修复)
            x_orig_noisy: 原图的噪声版本（用于已知区域）
        """
        # 1. 模型预测噪声
        noise_pred = model(x_t, t)
        
        # 2. 从噪声预测计算 x_{t-1}
        x_t_minus_1 = self.scheduler.step(noise_pred, t, x_t).prev_sample
        
        # 3. 已知区域用原图加噪版本替换
        x_t_minus_1 = mask * x_t_minus_1 + (1 - mask) * x_orig_noisy
        
        return x_t_minus_1, noise_pred
    
    def repaint_resample(self, model, x_t, t, mask, x_orig_noisy, 
                          num_resamples=5):
        """RePaint 风格的重采样（改善边界一致性）"""
        x_t_minus_1, noise_pred = self.inpaint_step(
            model, x_t, t, mask, x_orig_noisy
        )
        
        # 边界区域的 Resampling
        for _ in range(num_resamples):
            # 从 x_{t-1} 回到 x_t
            noise = torch.randn_like(x_t_minus_1)
            x_t_recovered = self.scheduler.add_noise(
                x_t_minus_1, noise, t
            )
            # 再次修复
            x_t_minus_1, _ = self.inpaint_step(
                model, x_t_recovered, t, mask, x_orig_noisy
            )
        
        return x_t_minus_1
    
    @torch.no_grad()
    def inpaint(self, model, x_orig, mask, num_steps=50):
        """完整修复流程"""
        B, C, H, W = x_orig.shape
        device = x_orig.device
        
        # 从纯噪声开始（只初始化掩码区域）
        x_T = torch.randn(B, C, H, W, device=device)
        
        # 已知区域用加噪后的原图填充
        x_t, _ = self.forward_diffusion_known(x_orig, torch.tensor([999]))
        x_t = mask * x_T + (1 - mask) * x_t
        
        # 逐步去噪
        for t in range(num_steps - 1, -1, -1):
            t_tensor = torch.full((B,), t, device=device, dtype=torch.long)
            x_orig_noisy, _ = self.forward_diffusion_known(x_orig, t_tensor)
            
            x_t, _ = self.inpaint_step(model, x_t, t_tensor, mask, x_orig_noisy)
        
        return x_t

class ImageOutpainting:
    """图像外延（Outpainting）"""
    def __init__(self, base_model):
        self.model = base_model
    
    @torch.no_grad()
    def outpaint(self, x_orig, expand_pixels=64, direction='right'):
        """
        图像外延
        
        参数:
            x_orig: 原始图像 [B, C, H, W]
            expand_pixels: 扩展的像素数
            direction: 扩展方向
        """
        B, C, H, W = x_orig.shape
        device = x_orig.device
        
        # 创建更大的画布和掩码
        new_W = W + expand_pixels
        canvas = torch.zeros(B, C, H, new_W, device=device)
        mask = torch.ones(B, 1, H, new_W, device=device)  # 1=生成
        
        if direction == 'right':
            canvas[:, :, :, :W] = x_orig
            mask[:, :, :, :W] = 0  # 已知区域
        elif direction == 'left':
            canvas[:, :, :, expand_pixels:] = x_orig
            mask[:, :, :, expand_pixels:] = 0
        elif direction == 'all':
            # 四边扩展
            pass
        
        # 使用扩散模型在掩码区域生成
        x_t = torch.randn(B, C, H, new_W, device=device)
        
        # 渐进式修复（简化）
        # 实际需要完整的 DDPM/DDIM 采样循环
        
        return canvas  # 返回扩展后的图像

# 模拟 Inpainting 流程
print("=== Inpainting & Outpainting ===")
print()
print("Inpainting 流程:")
print("  1. 创建掩码标注缺失区域")
print("  2. 已知区域保持原图")
print("  3. 扩散模型在掩码区域生成")
print("  4. 边界一致性处理 (RePaint)")
print()
print("Outpainting 流程:")
print("  1. 创建更大的画布")
print("  2. 将原图放置在画布指定位置")
print("  3. 创建掩码标注扩展区域")
print("  4. 扩散模型生成扩展内容")
print()

# 模拟
x_orig = torch.randn(1, 3, 256, 256)
mask = torch.zeros(1, 1, 256, 256)
mask[:, :, 128:, :] = 1  # 下半部分需要修复
print(f"原始图像: {x_orig.shape}")
print(f"修复掩码: {mask.shape}")
print(f"修复区域占比: {mask.mean().item()*100:.1f}%")
```

## 深度学习关联

- **Stable Diffusion Inpainting**：Stable Diffusion 的专门 Inpainting 模型——输入是拼接了掩码和噪声的 9 通道张量（4 通道潜变量 + 4 通道噪声 + 1 通道掩码），在潜空间中执行 Inpainting。
- **Outpainting with SD 的局限性**：Stable Diffusion 的 Outpainting 通过修改掩码实现，但扩展区域的全局一致性有限——通常需要多次生成 + 后处理（如边界羽化、颜色匹配）来改善。
- **视频 Inpainting (E2FGVI, ProPainter)**：视频修复比图像修复更难——不仅要空间一致，还需要时间一致。常见方法利用光流将相邻帧的信息传播到缺失区域。
- **3D Inpainting / Scene Completion**：将 Inpainting 扩展到 3D 场景——基于 NeRF 或 3DGS 的场景修复，根据已有视角的信息推断被遮挡区域的 3D 结构。
