# 27_视频生成：AnimateDiff 原理

## 核心概念

- **AnimateDiff**：一种将预训练图像扩散模型扩展到视频生成的高效方法，通过插入轻量的"运动模块"（Motion Module）实现时间一致性。
- **运动模块 (Motion Module)**：在冻结的图像 U-Net 各层之间插入时序自注意力层（Temporal Self-Attention），建模帧间的时间依赖关系。
- **时序自注意力**：将同空间位置不同帧的特征做自注意力，让各帧在相同位置的信息互相参考，确保时间上的一致性。
- **冻结骨干 + 训练适配器**：与 ControlNet/LoRA 类似，AnimateDiff 锁定预训练的图像 U-Net（保留图像生成能力），只训练新增的运动模块（约 100M 参数）。
- **隐变量帧序列**：将视频视为帧序列，每帧用 VAE 编码为潜变量 $z^{(i)} = E(x^{(i)})$，帧序列构成 $[B, T, C, H, W]$ 的五维张量。
- **端到端视频扩散**：训练时随机采样视频片段，对整个片段同时加噪/去噪，运动模块将信息在时间维度上传播。

## 数学推导

**运动模块的结构**：

在图像 U-Net 的每层之后插入：

$$
\text{特征} \xrightarrow{\text{空间层（已锁定）}} \xrightarrow{\text{运动模块（可训练）}} \xrightarrow{\text{下一层}}
$$

运动模块的核心是时序自注意力：

$$
\text{TemporalAttention}(z) = \text{Softmax}\left(\frac{Q_t K_t^T}{\sqrt{d}}\right) V_t
$$

其中 $Q_t, K_t, V_t$ 是对同一空间位置 $(h,w)$ 在所有 $T$ 帧上的特征计算得到的：

$$
z_{hw} \in \mathbb{R}^{T \times C} \to Q_t, K_t, V_t \in \mathbb{R}^{T \times C}
$$

**整体训练损失**：

给定视频片段 $\{x^{(i)}\}_{i=1}^T$ 和文本条件 $c$：

$$
L = \mathbb{E}_{z_0^{(1:T)}, c, \epsilon, t} \left[ \sum_{i=1}^T \left\| \epsilon^{(i)} - \epsilon_\theta(z_t^{(1:T)}, t, c) \right\|^2 \right]
$$

其中 $\epsilon_\theta$ 包含空间层（锁定）和运动模块（可训练）。

**推理时的采样**：

- 从随机噪声 $z_T^{(1:T)}$ 开始（$T$ 帧独立采样）
- 每步去噪，运动模块在帧间传递信息，使帧间逐渐对齐
- 在潜空间中同步去噪 $T$ 帧
- VAE 解码为像素帧

## 直观理解

- **AnimateDiff = 给图像模型装上时间记忆**：Stable Diffusion 像一个会画单张画的天才，但画不出动画（每帧看起来很牛但连在一起会闪烁）。AnimateDiff 的运动模块告诉它"你刚才画的这个地方是红色的，下一帧也应该是红色"。
- **时序自注意力 = 帧间对照"大家画的是不是一样"**：想象有 $T$ 个画家同时画 $T$ 帧，但他们都蒙着眼。时序自注意力让他们每画几笔就互相看一下——"哦，你画的是红苹果，那我这一帧的苹果也该是红色"。
- **为什么冻结空间层**：图像 U-Net 已经知道如何生成高质量的单帧图像。冻结空间层 = "保留你画单帧的能力，我只需要教会你保持帧间一致性"。这使得 AnimateDiff 可以用很少的视频数据训练。
- **运动模块的位置**：运动模块插入在 U-Net 的瓶颈层和低分辨率层效果最好，因为这些层处理的是语义信息（高分辨率层处理纹理细节，帧间变化大）。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class TemporalSelfAttention(nn.Module):
    """
    时序自注意力模块
    
    对同空间位置不同帧的特征做注意力
    """
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = channels // n_heads
        
        self.to_q = nn.Linear(channels, channels, bias=False)
        self.to_k = nn.Linear(channels, channels, bias=False)
        self.to_v = nn.Linear(channels, channels, bias=False)
        self.to_out = nn.Linear(channels, channels)
    
    def forward(self, x):
        """
        x: [B, C, H, W, T] 或 [B, T, C, H, W]
        核心：在时间维度上做 attention，空间维度是 batch 的扩展
        """
        B, T, C, H, W = x.shape
        
        # 重新排列为 [B*H*W, T, C]
        x_flat = x.permute(0, 3, 4, 1, 2).reshape(-1, T, C)
        
        # 多头注意力
        Q = self.to_q(x_flat)  # [B*H*W, T, C]
        K = self.to_k(x_flat)
        V = self.to_v(x_flat)
        
        # 多头分割
        Q = Q.view(-1, T, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(-1, T, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(-1, T, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(-1, T, C)
        out = self.to_out(out)
        
        # 恢复原始形状
        out = out.reshape(B, H, W, T, C).permute(0, 4, 3, 1, 2)
        return out  # [B, C, T, H, W]

class MotionModule(nn.Module):
    """AnimateDiff 运动模块"""
    def __init__(self, channels, n_heads=8):
        super().__init__()
        self.norm = nn.LayerNorm(channels)
        self.attn = TemporalSelfAttention(channels, n_heads)
    
    def forward(self, x):
        """
        x: [B, C, H, W] (常规图像特征)
        被插入到 U-Net 中，处理时间维度
        """
        # 实际由 U-Net 调度器在帧维度上扩展
        return x + self.attn(self.norm(x))

class AnimateDiffUNet(nn.Module):
    """插入运动模块的 U-Net（简化版）"""
    def __init__(self, in_channels=4, base_channels=64):
        super().__init__()
        # 空间层（冻结）
        self.spatial_conv1 = nn.Conv2d(in_channels, base_channels, 3, 1, 1)
        self.spatial_conv2 = nn.Conv2d(base_channels, base_channels, 3, 1, 1)
        
        # 运动模块（可训练，插入在空间层之间）
        self.motion = MotionModule(base_channels)
        
        self.out_conv = nn.Conv2d(base_channels, in_channels, 3, 1, 1)
    
    def forward(self, z_t, t, context):
        """
        z_t: [B, T, C, H, W] 帧序列的潜变量
        """
        B, T, C, H, W = z_t.shape
        
        # 将帧序列视为 batch 处理空间层（保持帧独立）
        z_t_flat = z_t.reshape(B * T, C, H, W)
        h = self.spatial_conv1(z_t_flat)
        h = self.spatial_conv2(h)
        
        # 恢复时间维度
        h = h.reshape(B, T, -1, H, W)
        
        # 运动模块（时间维度交互）
        h = self.motion(h.permute(0, 2, 1, 3, 4))  # [B, C, T, H, W]
        h = h.permute(0, 2, 1, 3, 4).reshape(B * T, -1, H, W)
        
        out = self.out_conv(h)
        return out.reshape(B, T, -1, H, W)

# 演示 AnimateDiff
print("=== AnimateDiff 视频生成 ===")
B, T, C, H, W = 1, 16, 4, 32, 32
x = torch.randn(B, T, C, H, W)

model = AnimateDiffUNet()
motion_params = sum(p.numel() for p in model.motion.parameters())
total_params = sum(p.numel() for p in model.parameters())
print(f"总参数量: {total_params:,}")
print(f"运动模块参数量: {motion_params:,}")
print(f"可训练比例: {motion_params/total_params*100:.1f}%")
print(f"帧数: {T}")
print(f"每帧尺寸: {H}x{W}")

output = model(x, torch.tensor([500]), None)
print(f"输出形状: {output.shape}")
```

## 深度学习关联

- **VideoLDM / Stable Video Diffusion**：在 AnimateDiff 之前的工作，直接训练端到端的视频扩散模型（需要大量视频数据）。AnimateDiff 的优势在于可以复用海量的图像预训练模型。
- **CameraCtrl / MotionCtrl**：在 AnimateDiff 基础上增加了相机运动控制和物体运动控制，通过额外的条件编码器控制镜头轨迹和物体运动路径。
- **帧插值 (Frame Interpolation)**：AnimateDiff 生成的关键帧较少（如 16 帧），后续可以用帧插值模型（如 FILM, EMA-VFI）扩展到更长、更平滑的视频。
- **视频一致性微调**：虽然 AnimateDiff 改善了时间一致性，但长视频（>100 帧）中仍会出现闪烁和漂移。SparseCtrl 等方法通过稀疏的关键帧控制来保持长期一致性。
