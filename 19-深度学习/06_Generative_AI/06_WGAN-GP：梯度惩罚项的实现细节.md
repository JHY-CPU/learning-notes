# 06_WGAN-GP：梯度惩罚项的实现细节

## 核心概念

- **WGAN-GP** (WGAN with Gradient Penalty)：针对原始 WGAN 权重裁剪（Weight Clipping）的改进方案，用梯度惩罚项替代裁剪来满足 Lipschitz 约束。
- **梯度惩罚 (Gradient Penalty)**：在真实数据和生成数据的插值点上，对 Critic 的梯度范数施加 L2 惩罚，迫使其接近 1。
- **权重裁剪的问题**：裁剪会导致 Critic 倾向于学习简单的函数（权重集中在极端值），表达能力受限；且容易导致梯度消失或爆炸。
- **插值采样**：梯度惩罚不是在任意点计算，而是在真实分布 $p_r$ 和生成分布 $p_g$ 之间的随机插值点上计算，因为这些区域是 Critic 最需要满足约束的地方。
- **无参数调节**：WGAN-GP 比 WGAN 更少需要调参（不需要调整裁剪范围 $c$），训练更稳定，收敛更快。
- **适用范围**：WGAN-GP 适用于大多数 GAN 架构，特别是在高分辨率图像生成中表现优异。

## 数学推导

**WGAN-GP 的 Critic 损失函数**：

$$
L = \underbrace{\mathbb{E}_{\tilde{x} \sim p_g}[D(\tilde{x})] - \mathbb{E}_{x \sim p_r}[D(x)]}_{\text{Wasserstein 损失}} + \lambda \underbrace{\mathbb{E}_{\hat{x} \sim p_{\hat{x}}}[( \|\nabla_{\hat{x}} D(\hat{x})\|_2 - 1)^2]}_{\text{梯度惩罚项}}
$$

其中：

- $p_r$ 是真实数据分布，$p_g$ 是生成数据分布
- $\hat{x}$ 是从 $p_r$ 和 $p_g$ 之间均匀插值采样：$\hat{x} = \epsilon x + (1-\epsilon)\tilde{x}$，$\epsilon \sim U[0,1]$
- $\lambda$ 是梯度惩罚的权重系数（通常设为 10）

**梯度惩罚的理论依据**：对于一个最优 1-Lipschitz 函数，在 $p_r$ 和 $p_g$ 之间的直线上，梯度范数几乎处处为 1。因此惩罚项 $(\|\nabla D(\hat{x})\|_2 - 1)^2$ 直接将 Critic 推向最优 Lipschitz 函数。

**权重裁剪的梯度对比**：

权重裁剪：$f(x) = \text{clip}(w, -c, c)$ 导致 $f$ 趋向于二值化权重

梯度惩罚：$\nabla_{\hat{x}} f(\hat{x})$ 在插值路径上被约束为具有单位范数

## 直观理解

- **权重裁剪**就像规定运动员只能在一个狭小的范围内活动——他确实不会越界（Lipschitz 约束），但也施展不开（表达能力受限）。
- **梯度惩罚**就像给运动员装了一个速度限制器——只要速度不超过限制（梯度范数接近 1），可以自由发挥。
- **插值采样**的精妙之处：我们只在"真假之间"的过渡区域施加约束，因为这些区域是 Critic 最需要区分真假的地方，也是 Lipschitz 约束最容易违反的区域。
- 可以想象真实数据和生成数据之间有一条路径，梯度惩罚确保在这条路径上 Critic 的变化率是受控的（每秒最多变化 1 单位）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd

def compute_gradient_penalty(Critic, real_data, fake_data, lambda_gp=10):
    """
    计算 WGAN-GP 的梯度惩罚项
    
    参数:
        Critic: 评论家网络
        real_data: 真实数据样本
        fake_data: 生成数据样本
        lambda_gp: 梯度惩罚权重系数（默认 10）
    """
    batch_size = real_data.size(0)
    
    # 1. 生成随机插值系数
    epsilon = torch.rand(batch_size, 1)
    # 扩展 epsilon 到数据维度
    epsilon = epsilon.expand_as(real_data)
    
    # 2. 在真实和生成数据之间插值
    interpolates = epsilon * real_data + (1 - epsilon) * fake_data
    interpolates = interpolates.requires_grad_(True)
    
    # 3. 计算 Critic 对插值点的输出
    d_interpolates = Critic(interpolates)
    
    # 4. 计算梯度
    gradients = autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    
    # 5. 计算梯度范数的 L2 惩罚
    gradients = gradients.view(batch_size, -1)
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = lambda_gp * ((gradient_norm - 1) ** 2).mean()
    
    return gradient_penalty

class WGANGPCritic(nn.Module):
    """WGAN-GP 评论家（不带 BatchNorm，因为 GP 要求每批独立）"""
    def __init__(self, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)
        )
    
    def forward(self, x):
        return self.net(x)

# WGAN-GP 训练步骤
def train_wgangp_step(G, Critic, opt_G, opt_C, real_data, latent_dim, n_critic=5):
    batch_size = real_data.size(0)
    
    # 更新 Critic
    for _ in range(n_critic):
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z).detach()
        
        c_real = Critic(real_data)
        c_fake = Critic(fake_data)
        
        # WGAN 损失
        wasserstein_loss = torch.mean(c_fake) - torch.mean(c_real)
        
        # 梯度惩罚
        gp = compute_gradient_penalty(Critic, real_data, fake_data)
        
        loss_C = wasserstein_loss + gp
        
        opt_C.zero_grad()
        loss_C.backward()
        opt_C.step()
    
    # 更新生成器
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    loss_G = -torch.mean(Critic(fake_data))
    
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    
    return loss_C.item(), loss_G.item()

# 初始化
latent_dim = 100
G = nn.Sequential(
    nn.Linear(latent_dim, 256), nn.BatchNorm1d(256), nn.ReLU(),
    nn.Linear(256, 512), nn.BatchNorm1d(512), nn.ReLU(),
    nn.Linear(512, 784), nn.Tanh()
)
Critic = WGANGPCritic()
opt_G = optim.Adam(G.parameters(), lr=1e-4, betas=(0.5, 0.9))
opt_C = optim.Adam(Critic.parameters(), lr=1e-4, betas=(0.5, 0.9))

print("WGAN-GP 初始化完成")
print("超参数: lambda_gp=10, n_critic=5, lr=1e-4, betas=(0.5, 0.9)")
```

## 深度学习关联

- **StyleGAN2 中的 Path Length Regularization**：借鉴了梯度惩罚的思想，通过正则化生成器的映射网络来确保潜空间中的插值路径更加平滑。
- **Spectral Normalization**：另一种施加 Lipschitz 约束的方法，通过对每层权重矩阵进行奇异值分解约束来实现，计算效率比 GP 更高。
- **R1 正则化**：另一种流行的梯度正则化方法，只对真实数据点计算梯度惩罚：$R_1 = \frac{\gamma}{2} \mathbb{E}_{x \sim p_{\text{data}}}[\|\nabla D(x)\|^2]$，计算量比 WGAN-GP 更小。
- **EBM（能量基模型）**：梯度惩罚在能量基模型训练中也有广泛应用，用于约束能量函数的平滑性。
