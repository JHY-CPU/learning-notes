# 05_Wasserstein GAN (WGAN) 与 Earth Mover's Distance

## 核心概念
- **Earth Mover's Distance (EMD)**：也称为 Wasserstein-1 距离，度量将一个概率分布"搬运"到另一个概率分布所需的最小成本，类比于将一堆土搬到另一个地方需要的最小工作量。
- **WGAN**：用 EMD/Wasserstein 距离替代原始 GAN 中的 JS 散度作为优化目标，从根本上解决了梯度消失问题，使 GAN 训练更加稳定。
- **Critic 替代 Discriminator**：WGAN 中将判别器重命名为"评论家"（Critic），因为它不再输出真假概率（0-1 之间），而是输出一个实数值分数（无上限），用来衡量生成样本的"真实程度"。
- **Lipschitz 约束**：为了满足 Wasserstein 距离的对偶形式要求，Critic 函数 $f$ 必须满足 1-Lipschitz 连续性，即 $|f(x_1) - f(x_2)| \leq \|x_1 - x_2\|$。
- **权重裁剪 (Weight Clipping)**：原始 WGAN 通过对 Critic 的权重进行简单裁剪（如限制在 [-0.01, 0.01] 范���内）来近似满足 Lipschitz 约束。这种方法简单但粗糙。
- **训练信号的意义**：WGAN 的 Critic 损失值可以作为生成质量的真实指标——损失越小，生成质量越高（而原始 GAN 的 D 损失无此含义）。

## 数学推导

**Wasserstein-1 距离定义**：

$$
W(p_r, p_g) = \inf_{\gamma \sim \Pi(p_r, p_g)} \mathbb{E}_{(x,y) \sim \gamma} [\|x - y\|]
$$

其中 $\Pi(p_r, p_g)$ 是所有以 $p_r$ 和 $p_g$ 为边缘分布的联合分布集合。

**Kantorovich-Rubinstein 对偶形式**（实际使用的形式）：

$$
W(p_r, p_g) = \sup_{\|f\|_L \leq 1} \mathbb{E}_{x \sim p_r}[f(x)] - \mathbb{E}_{x \sim p_g}[f(x)]
$$

其中 $f$ 是 1-Lipschitz 函数。这就是 WGAN 的核心公式。

**WGAN 价值函数**：

$$
\min_G \max_{f \in \mathcal{F}} \mathbb{E}_{x \sim p_{\text{data}}}[f(x)] - \mathbb{E}_{z \sim p_z}[f(G(z))]
$$

其中 $\mathcal{F}$ 是所有 1-Lipschitz 函数的集合。

**与原始 GAN 的对比**：

原始 GAN 的最优 D 对应的 JS 散度在分布不重叠时为常数（梯度消失）。而 WGAN 的 Critic 在分布不重叠时仍然提供有意义的梯度——因为 Wasserstein 距离是连续且几乎处处可微的。

## 直观理解
- **EMD 的物理类比**：想象你有一堆土（分布 A），需要把它搬到另一个形状（分布 B）。EMD 就是搬土所需的最小工作量，等于"每粒土移动的距离 × 土量"的总和。
- **JS 散度的问题**：如果两堆土完全不重叠（没有交集），JS 散度就是常数，你看不到任何梯度指引如何搬土。就像在两个孤岛上，你不知道该往哪个方向游。
- **Wasserstein 距离的优势**：即使两堆土不重叠，它们之间的距离仍然提供了"该往哪个方向搬"的信号。这就像虽然隔着海，但你能看到对岸的位置，知道该往哪个方向游。
- **Critic 的意义**：Critic 输出的分数衡量了"这个样本有多真实"——这与原始 GAN 中 D 的 0/1 分类意义完全不同，它是一个连续的、有信息的度量。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

class WGANCritic(nn.Module):
    """WGAN 评论家（Critic）—— 输出实数值而非概率"""
    def __init__(self, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(data_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.2),
            nn.Linear(128, 1)  # 注意：没有 Sigmoid！
        )
    
    def forward(self, x):
        return self.net(x)

class WGANGenerator(nn.Module):
    def __init__(self, latent_dim=100, data_dim=784):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(latent_dim, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(),
            nn.Linear(512, data_dim),
            nn.Tanh()
        )
    
    def forward(self, z):
        return self.net(z)

def train_wgan(G, Critic, opt_G, opt_C, real_data, latent_dim, n_critic=5, clip_value=0.01):
    """WGAN 单步训练（含权重裁剪）"""
    batch_size = real_data.size(0)
    critic_loss = 0
    
    # Critic 更新 n_critic 次
    for _ in range(n_critic):
        z = torch.randn(batch_size, latent_dim)
        fake_data = G(z)
        
        c_real = Critic(real_data)
        c_fake = Critic(fake_data.detach())
        
        # Wasserstein 损失 = - (E[f_real] - E[f_fake])
        loss_C = -(torch.mean(c_real) - torch.mean(c_fake))
        
        opt_C.zero_grad()
        loss_C.backward()
        opt_C.step()
        
        # 权重裁剪（满足 Lipschitz 约束的朴素方法）
        for p in Critic.parameters():
            p.data.clamp_(-clip_value, clip_value)
        
        critic_loss += loss_C.item()
    
    # 生成器更新 1 次
    z = torch.randn(batch_size, latent_dim)
    fake_data = G(z)
    c_fake = Critic(fake_data)
    
    loss_G = -torch.mean(c_fake)  # 生成器试图最大化 Critic 对假数据的评分
    
    opt_G.zero_grad()
    loss_G.backward()
    opt_G.step()
    
    return critic_loss / n_critic, loss_G.item()

# 初始化
latent_dim = 100
G = WGANGenerator(latent_dim)
Critic = WGANCritic()
opt_G = optim.RMSprop(G.parameters(), lr=5e-5)
opt_C = optim.RMSprop(Critic.parameters(), lr=5e-5)

# WGAN 的训练信号可作为质量指标
print("WGAN 初始化完成")
print("训练配置: RMSprop, lr=5e-5, n_critic=5, weight_clip=±0.01")

# 生成样本并评估 Critic 分数
def evaluate_generation(G, Critic, n_samples=64):
    with torch.no_grad():
        z = torch.randn(n_samples, latent_dim)
        fake = G(z)
        scores = Critic(fake)
        print(f"生成样本的平均 Critic 分数: {scores.mean().item():.4f}")
        print(f"（分数越高表示生成质量越好）")
    return fake
```

## 深度学习关联
- **WGAN-GP**：针对权重裁剪的缺陷提出了梯度惩罚项（Gradient Penalty），在 Critic 的梯度上施加 L2 范数约束，是 WGAN 的改进版本，也是实际使用最广的版本。
- **Spectral Normalization (SN-GAN)**：另一种施加 Lipschitz 约束的方法，通过对每一层的权重矩阵进行谱归一化来满足 Lipschitz 条件，训练更稳定且不需要调超参数。
- **Wasserstein 距离在其他领域的应用**：Wasserstein Auto-Encoder (WAE)、Wasserstein 强化学习等将 Wasserstein 距离的思想推广到了生成模型之外。
- **理论意义**：WGAN 的一个重要贡献是揭示了 GAN 训练不稳定的根本原因——JS 散度在非重叠分布上的不良性质，为后续所有 GAN 改进提供了理论指导。
