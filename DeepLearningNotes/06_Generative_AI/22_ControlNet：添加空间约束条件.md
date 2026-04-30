# 22_ControlNet：添加空间约束条件

## 核心概念
- **ControlNet**：一种向预训练扩散模型添加空间条件控制（如边缘图、深度图、姿态骨架、分割图）的神经网络架构，无需重新训练基础模型。
- **零卷积 (Zero Convolution)**：ControlNet 的核心创新——一个初始化为零的 1x1 卷积层，训练开始时输出为零，保证不干扰预训练模型的初始行为。
- **锁定骨干 + 训练副本**：ControlNet 锁定原始扩散模型的权重，创建一个训练副本（Trainable Copy），在副本中添加条件控制路径。锁定骨干确保基础模型的知识不被遗忘。
- **多种条件类型**：Canny 边缘（边缘检测）、深度图（MiDaS 深度估计）、OpenPose 姿态骨架（人体姿态检测）、HED 软边缘、语义分割图、法线图等。
- **端到端可学习**：用户只需要提供输入条件图像和目标图像对，ControlNet 自动学习如何将空间条件映射到生成控制。
- **强条件与弱条件**：Canny 边缘是"强条件"（严格控制形状），深度图和姿态图是"中条件"（控制位置但允许创意），语义分割是"弱条件"（控制区域但不控制具体形状）。

## 数学推导

**ControlNet 的结构**：

假设 $F(x; \Theta)$ 是预训练扩散模型的 U-Net 层（锁定），ControlNet 添加一个可训练分支：

$$
y_c = F(x; \Theta) + \mathcal{Z}(F(x + \mathcal{Z}(c; \Theta_z1); \Theta_c); \Theta_z2)
$$

其中：
- $c$ 是空间条件（边缘图、深度图等）
- $\Theta_c$ 是可训练的 ControlNet 副本参数
- $\Theta_z1, \Theta_z2$ 是零卷积层参数（初始化为零）
- $\mathcal{Z}$ 表示零卷积操作

**训练目标**：

$$
L = \mathbb{E}_{z_0, c, t, \epsilon} \left[ \| \epsilon - \epsilon_\theta(z_t, t, \tau_\theta(y), c_f) \|_2^2 \right]
$$

其中 $y$ 是 ControlNet 输出的条件特征，$c_f$ 是无条件/文本条件（与标准 CFG 类似）。

**零卷积的设计原理**：

训练开始时，$\mathcal{Z}(x; \Theta_z) = 0$，因此 $y_c = F(x; \Theta)$——ControlNet 分支的输出为零，完全不影响原模型的输出。

训练过程中，$\Theta_z$ 逐渐学习到非零值，ControlNet 分支逐渐接管空间条件的控制。

**与微调的区别**：

全量微调：$\Theta_{\text{new}} = \Theta_{\text{pretrained}} + \Delta\Theta$（所有参数更新）

ControlNet：$\Theta_{\text{locked}} \text{冻结}, \Theta_c \text{可训练且从锁定副本初始化}, \Theta_z \text{从零初始化}$

## 直观理解
- **ControlNet = 给画家一个参照骨架**：你给 AI 画家一张参考图（边缘检测、姿态骨架），说"按这个姿势画一只猫"——ControlNet 确保画出来的猫的姿势和你的骨架图吻合。
- **零卷积 = 慢慢引入辅助轮**：训练刚开���时 ControlNet 不发挥作用（输出为零），就像辅助轮不着地。随着训练进行，ControlNet 逐渐学习如何利用条件，辅助轮慢慢接触地面起作用。这确保不会"惊吓"到已经训练好的基础模型。
- **锁定骨干的智慧**：很多人会问"为什么不直接微调整个模型？"原因是：微调整个模型会导致灾难性遗忘——模型会忘记怎么画海量的文本概念，只学会新任务。ControlNet 通过锁定骨干避免了这个问题。
- **多种条件类型的本质差异**：Canny 边缘控制位置形状但不控制颜色纹理，深度图控制物体距离但不控制具体外形，姿态图控制人体肢体位置但不控制穿着——它们像是从不同角度描述同一个场景的"约束条件"。

## 代码示例

```python
import torch
import torch.nn as nn

class ZeroConv2d(nn.Module):
    """零卷积层：初始化为零的卷积"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        # 初始化为零（权重和偏置）
        nn.init.zeros_(self.conv.weight)
        nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        return self.conv(x)

class ControlNetBlock(nn.Module):
    """ControlNet 的单个残差块"""
    def __init__(self, channels, cond_channels):
        super().__init__()
        self.zero_conv_in = ZeroConv2d(cond_channels, channels)
        
        self.norm1 = nn.GroupNorm(32, channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.norm2 = nn.GroupNorm(32, channels)
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        
        self.zero_conv_out = ZeroConv2d(channels, channels)
    
    def forward(self, x, hint):
        # hint: 空间条件特征
        hint_feat = self.zero_conv_in(hint)
        x = x + hint_feat
        
        h = self.conv1(torch.silu(self.norm1(x)))
        h = self.conv2(torch.silu(self.norm2(h)))
        
        return self.zero_conv_out(h) + x

class ControlNet(nn.Module):
    """
    ControlNet：为扩散模型添加空间条件控制
    
    简化实现：只展示核心结构，不包含完整的 U-Net
    """
    def __init__(self, in_channels=3, latent_channels=4, cond_channels=3):
        super().__init__()
        # 条件编码器：将空间条件映射到潜空间
        self.cond_encoder = nn.Sequential(
            nn.Conv2d(cond_channels, 16, 4, 2, 1),  # 下采样
            nn.SiLU(),
            nn.Conv2d(16, 32, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(32, 64, 4, 2, 1),
            nn.SiLU(),
            nn.Conv2d(64, latent_channels, 3, 1, 1),
        )
        
        # 零卷积连接
        self.zero_conv = ZeroConv2d(latent_channels, latent_channels)
        
        # ControlNet 块（锁定 U-Net 的副本）
        self.control_blocks = nn.ModuleList([
            ControlNetBlock(latent_channels, latent_channels)
            for _ in range(4)
        ])
    
    def forward(self, latent, t, text_emb, control_hint):
        """
        latent: 当前时间步的潜变量
        control_hint: 空间条件（如 Canny 边缘图、深度图）
        """
        # 编码条件
        hint_feat = self.cond_encoder(control_hint)
        hint_feat = self.zero_conv(hint_feat)
        
        # 注入条件到各层
        output = latent
        for block in self.control_blocks:
            output = block(output, hint_feat)
        
        return output

# ControlNet 训练配置（冻结主干，只训练 ControlNet）
def train_controlnet(controlnet, pretrained_unet, dataloader):
    """
    ControlNet 训练循环
    
    冻结预训练模型的所有参数，只训练 ControlNet
    """
    for param in pretrained_unet.parameters():
        param.requires_grad = False
    
    optimizer = torch.optim.Adam(controlnet.parameters(), lr=1e-5)
    
    # 示例训练步骤
    for batch in dataloader:
        image, condition = batch  # image: 目标图, condition: 条件图
        t = torch.randint(0, 1000, (image.size(0),))
        
        noise = torch.randn_like(image)
        # 加噪
        noisy_image = image + noise * t[:, None, None, None].float() / 1000
        
        # ControlNet 前向
        control_output = controlnet(noisy_image, t, None, condition)
        
        # 简化损失（完整需结合 U-Net）
        loss = torch.mean((control_output - noise) ** 2)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    return controlnet

print("=== ControlNet 示例 ===")
controlnet = ControlNet()
sample_hint = torch.randn(1, 3, 256, 256)  # 如 Canny 边缘图
sample_latent = torch.randn(1, 4, 32, 32)  # 潜变量
output = controlnet(sample_latent, torch.tensor([500]), None, sample_hint)
print(f"潜变量形状: {sample_latent.shape}")
print(f"条件输入形状: {sample_hint.shape}")
print(f"ControlNet 输出形状: {output.shape}")
print(f"ControlNet 参数量: {sum(p.numel() for p in controlnet.parameters()):,}")
```

## 深度学习关联
- **T2I-Adapter**：与 ControlNet 相似的思路，但结构更轻量——使用简单的 adapter 网络将条件特征压缩后注入 U-Net，参数量更小，适合快速适配。
- **IP-Adapter**：将 ControlNet 的条件注入思想用在图像提示（Image Prompt）上——将参考图像编码为条件，控制生成图像的风格或内容相似于参考图。
- **Composable Diffusion / Multi-ControlNet**：ControlNet 支持多个条件同时输入（如边缘图 + 姿态图 + 深度图），每个条件独立控制生成的不同方面，最终在 U-Net 特征层面叠加。
- **LoRA + ControlNet 的组合使用**：在实际应用中，LoRA 微调风格/角色，ControlNet 控制姿态/构图——两个正交的调控手段可以同时使用，互不干扰。
