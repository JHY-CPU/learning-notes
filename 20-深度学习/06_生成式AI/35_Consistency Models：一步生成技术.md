# 36_Consistency Models：一步生成技术

## 核心概念

- **一致性模型 (Consistency Model)**：由 Song et al. (2023) 提出的一种新生成模型范式，学习一个直接从噪声 $x_T$ 映射到数据 $x_0$ 的函数 $f_\theta(x_T, T)$，实现一步或少量步的生成。
- **自一致性 (Self-Consistency)**：一致性函数的核心属性——对于任意时间步 $t$，沿概率流 ODE 轨迹上的所有点映射到相同的起始点 $x_0$：$f_\theta(x_t, t) = f_\theta(x_{t'}, t')$。
- **蒸馏训练 (Distillation Training)**：利用预训练的扩散模型（教师）为每个 ODE 轨迹生成相邻时间步的配对数据 $(x_{t+\Delta t}, x_t, t)$，训练一致性模型（学生）满足自一致性。
- **独立训练 (Isolated Training)**：一致性模型也可以不使用预训练的教师模型，直接从数据分布训练——通过"从噪声到数据的直接映射"损失训练。
- **一步生成的意义**：传统扩散模型需要 10-1000 步采样，一致性模型仅需 1-2 步——推理速度提升 500-1000 倍，使得扩散模型可用于实时交互场景。
- **质量-速度权衡**：一致模型在一步生成下质量不如多步扩散，但多步（如 2-4 步）可以接近扩散模型的质量，而速度仍快 100-500 倍。

## 数学推导

**一致性函数**：

定义一致性函数 $f_\theta(x_t, t)$，它应该满足：

$$
f_\theta(x_t, t) = x_0 \quad \text{对于任意 } t \in [0, T]
$$

其中 $x_t$ 是沿着概率流 ODE 从 $x_0$ 演变到 $x_T$ 的中间状态。

**边界条件**：

当 $t=0$ 时，$x_0$ 就是数据本身，因此：

$$
f_\theta(x, 0) = x \quad \text{（边界条件）}
$$

**自一致性损失**：

对于 ODE 轨迹上的相邻点 $x_t$ 和 $x_{t'}$：

$$
\mathcal{L}_{\text{CD}}(\theta) = \mathbb{E}_{t, x_0, \epsilon} \left[ d\left(f_\theta(x_{t+\Delta t}, t+\Delta t), f_\theta(x_t, t)\right) \right]
$$

其中 $d$ 是距离度量（如 MSE 或 LPIPS），$f_\theta$ 的目标是使同一轨迹上不同时间步的输出一致。

**通过蒸馏学习一致性模型**：

给定预训练扩散模型的得分函数 $s_\phi(x_t, t)$，单步 ODE 更新：

$$
\hat{x}_t = f_\theta(x_{t+\Delta t}, t+\Delta t)
$$

$$
x_t' = x_{t+\Delta t} + \Delta t \cdot \left[ f(x_{t+\Delta t}, t+\Delta t) - \frac{1}{2} g(t+\Delta t)^2 s_\phi(x_{t+\Delta t}, t+\Delta t) \right]
$$

目标函数：

$$
\mathcal{L}_{\text{CD}}(\theta) = \mathbb{E}_{t, x_{t+\Delta t}} \left[ \| f_\theta(x_t', t) - f_\theta(x_{t+\Delta t}, t+\Delta t) \|^2 \right]
$$

**一步采样**：

$$
x_0 = f_\theta(x_T, T), \quad x_T \sim \mathcal{N}(0, I)
$$

## 直观理解

- **一致性模型 = 学公式而不是学过程**：扩散模型学会了"如何一步步去噪"，一致性模型直接学会"从噪声到图片的公式"。前者需要 1000 步计算过程，后者一步出结果。
- **自一致性的直觉 = 不同路线到同一目的地**：如果你从北京出发（$x_0$），无论坐高铁（快速 ODE）还是骑自行车（慢速 ODE），最终到海口（$x_T$）的路线上的每个 $x_t$ 点，都应该对应同一个起点北京。
- **蒸馏的理解**：老师（扩散模型）演示了 1000 步怎么走，学生（一致性模型）学会了"哦，一步就能到"。蒸馏过程让学生模仿老师在不同时间步的输出，学会"跳跃"的能力。
- **为什么一步生成有质量损失**：扩散模型的迭代生成可以一步步修正错误，一致性模型的一步映射必须在一个前向中完成所有决策——信息瓶颈相当于从"多次考试"变为"一次考完"。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ConsistencyModel(nn.Module):
    """
    一致性模型：从 (x_t, t) 直接映射到 x_0
    
    使用跳跃连接（skip connection）确保 t=0 时 f(x,0)=x
    """
    def __init__(self, input_dim=784, hidden_dim=1024):
        super().__init__()
        self.time_embed = nn.Embedding(1000, 256)
        
        self.net = nn.Sequential(
            nn.Linear(input_dim + 256, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),
        )
        
        # t=0 时的跳跃连接权重
        self.skip_weight = nn.Parameter(torch.tensor(0.0))
    
    def forward(self, x_t, t):
        """
        从噪声 x_t 和时间步 t 直接预测 x_0
        
        满足边界条件: f(x_0, 0) = x_0
        """
        t_emb = self.time_embed(t)
        x_t_flat = x_t.view(x_t.size(0), -1)
        
        h = torch.cat([x_t_flat, t_emb], dim=-1)
        pred = self.net(h)
        
        # 跳跃连接 + 时间衰减: t=0 时输出等于输入
        # 使用时间衰减权重
        t_normalized = t.float().unsqueeze(-1) / 1000.0
        skip = torch.sigmoid(self.skip_weight) * (1 - t_normalized)
        
        result = (1 - skip) * pred + skip * x_t_flat
        return result.view(x_t.shape)
    
    @torch.no_grad()
    def sample_one_step(self, img_shape=(1, 1, 28, 28), device='cpu'):
        """一步生成：从纯噪声到图像"""
        x_T = torch.randn(*img_shape).to(device)
        t = torch.full((img_shape[0],), 999, dtype=torch.long, device=device)
        return self.forward(x_T, t)


def consistency_distillation_loss(student_model, teacher_score_fn, x_0, t, delta=1):
    """
    一致性蒸馏损失
    
    用教师模型生成 ODE 轨迹上的配对数据，训练学生满足自一致性
    """
    # 生成 x_{t+delta}
    noise = torch.randn_like(x_0)
    x_t_plus_delta = x_0 + noise * (t + delta).float().sqrt()
    
    # 学生模型: 从 x_{t+delta} 预测 x_0
    pred_1 = student_model(x_t_plus_delta, t + delta)
    
    # 教师模型的单步 ODE 更新（简化版）
    with torch.no_grad():
        score = teacher_score_fn(x_t_plus_delta, t + delta)
        # ODE 步: x_t ≈ x_{t+delta} + delta * score * g^2
        x_t = x_t_plus_delta + delta * score * 0.5
    
    # 学生模型: 从 x_t 预测 x_0
    pred_2 = student_model(x_t, t)
    
    # 自一致性损失
    loss = F.mse_loss(pred_1, pred_2)
    return loss


# CM 训练损失（独立训练版本）
def consistency_training_loss(model, x_0, delta=1):
    """
    一致性训练损失（不依赖教师模型）
    
    利用数据增强 + 噪声扰动生成配对样本
    """
    batch_size = x_0.size(0)
    device = x_0.device
    
    # 随机时间步
    t = torch.randint(delta, 1000 - delta, (batch_size,), device=device)
    
    # 生成两个相邻噪声级别的样本
    noise_1 = torch.randn_like(x_0)
    noise_2 = torch.randn_like(x_0)
    
    x_t_plus = x_0 + noise_1 * (t + delta).float().sqrt()
    x_t = x_0 + noise_2 * t.float().sqrt()
    
    # 预测 x_0
    pred_1 = model(x_t_plus, t + delta)
    pred_2 = model(x_t, t)
    
    # 停止梯度：一致性目标是固定的
    target = pred_1.detach()
    loss = F.mse_loss(pred_2, target)
    
    return loss


print("=== Consistency Models: 一步生成 ===")
cm = ConsistencyModel()
x = torch.randn(4, 784)
t = torch.randint(0, 1000, (4,))
out = cm(x, t)
print(f"输入噪声形状: {x.shape}, 时间步: [tensor]")
print(f"输出形状: {out.shape}")
print(f"边界条件检查 (t=0):")
t_0 = torch.zeros(1, dtype=torch.long)
x_test = torch.randn(1, 784)
out_0 = cm(x_test, t_0)
print(f"  ||f(x,0) - x|| = {(out_0 - x_test).norm().item():.6f} (应接近 0)")

print(f"\n参数量: {sum(p.numel() for p in cm.parameters()):,}")
print(f"一步生成 vs 1000 步 DDPM 的速度比: ~1000x")
```

## 深度学习关联

- **LCM (Latent Consistency Model)**：将一致性模型应用到 Stable Diffusion 的潜空间中，通过 LoRA 微调实现 1-4 步生成高质量图像，是目前最流行的快速采样技术。
- **Rectified Flow / Reflow**：通过将 ODE 轨迹"拉直"为直线路径实现一步生成——与一致性模型不同，它从"两两配对"的视角出发，通过多次 reflow 操作逐步缩短路径。
- **Adversarial Diffusion Distillation**：使用对抗损失（判别器）替代 MSE 损失进行蒸馏，使得蒸馏后的一步生成模型质量更好——这是 Stable Diffusion 3 Turbo 的核心技术。
- **Progressive Distillation**：通过逐步减少采样步数（1000→100→10→1）来蒸馏扩散模型，是 Consistency Model 的早期先驱工作。
