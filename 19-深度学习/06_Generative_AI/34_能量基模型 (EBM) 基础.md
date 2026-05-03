# 34_能量基模型 (EBM) 基础

## 核心概念

- **能量基模型 (Energy-Based Model, EBM)**：用一个能量函数 $E_\theta(x) \in \mathbb{R}$ 来描述数据点 $x$ 的"能量"——低能量对应高概率（真实数据点），高能量对应低概率（非数据点）。
- **玻尔兹曼分布**：EBM 的概率密度定义为 $p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}$，其中 $Z(\theta) = \int \exp(-E_\theta(x)) dx$ 是归一化常数（配分函数）。
- **无归一化模型**：EBM 只定义未归一化的概率 $\tilde{p}_\theta(x) = \exp(-E_\theta(x))$，不需要知道 $Z(\theta)$ 就能计算概率比 $\frac{p_\theta(x_1)}{p_\theta(x_2)} = \exp(-(E_\theta(x_1) - E_\theta(x_2)))$。
- **配分函数的挑战**：$Z(\theta)$ 需要对所有可能的 $x$ 积分，在高维空间中极其困难（无法精确计算）。EBM 的训练方法必须绕开 $Z$ 的计算。
- **对比散度 (Contrastive Divergence, CD)**：通过 MCMC 采样生成"负样本"（高能量点），然后拉低真实点的能量、推高负样本的能量来训练 EBM。
- **EBM 与扩散模型的联系**：得分函数 $s_\theta(x) = -\nabla_x E_\theta(x)$——训练 EBM 等价于学习得分函数，而得分函数正是扩散模型的核心。

## 数学推导

**EBM 的概率定义**：

$$
p_\theta(x) = \frac{\exp(-E_\theta(x))}{Z(\theta)}, \quad Z(\theta) = \int \exp(-E_\theta(x)) dx
$$

**最大似然训练的梯度**：

$$
\nabla_\theta \log p_\theta(x) = -\nabla_\theta E_\theta(x) - \nabla_\theta \log Z(\theta)
$$

其中 $\nabla_\theta \log Z(\theta) = -\mathbb{E}_{x \sim p_\theta(x)}[\nabla_\theta E_\theta(x)]$

因此：

$$
\nabla_\theta \log p_\theta(x) = \underbrace{-\nabla_\theta E_\theta(x)}_{\text{正相位（降低 $x$ 的能量）}} + \underbrace{\mathbb{E}_{\tilde{x} \sim p_\theta(\tilde{x})}[\nabla_\theta E_\theta(\tilde{x})]}_{\text{负相位（提升采样点的能量）}}
$$

正相位（Positive Phase）：降低真实数据点的能量

负相位（Negative Phase）：从模型分布中采样（MCMC），提升这些样本的能量

**对比散度 (CD-k)**：

用 $k$ 步 MCMC（通常用 Langevin Dynamics 或 Gibbs Sampling）从当前模型分布采样，作为负相位样本的近似。

$$
\nabla_\theta \log p_\theta(x) \approx -\nabla_\theta E_\theta(x) + \nabla_\theta E_\theta(x_{\text{CD}})
$$

其中 $x_{\text{CD}}$ 是 $k$ 步 MCMC 后的样本。

**Langevin Dynamics 采样**：

从先验分布 $x_0 \sim \pi(x)$（如均匀分布）出发，迭代更新：

$$
x_{t+1} = x_t - \frac{\eta}{2} \nabla_x E_\theta(x_t) + \sqrt{\eta} \cdot \epsilon_t, \quad \epsilon_t \sim \mathcal{N}(0, I)
$$

当 $\eta \to 0$ 且 $t \to \infty$，$x_t \sim p_\theta(x)$。

## 直观理解

- **能量景观比喻**：想象一个地形图——山谷（低能量）代表数据所在的区域，山峰（高能量）代表不可能的数据区域。EBM 学习的就是这个地形图的形状。
- **训练过程 = 挖山谷 + 推山峰**：正相位在真实数据点处挖坑（降低能量），负相位在模型认为可能的点堆土（升高能量）。两者平衡后，真实数据的"山谷"就是模型学习到的分布。
- **为什么 EBM 难训练**：MCMC 采样（负相位）需要在每个训练步中运行——这在高维空间中非常慢，而且 MCMC 可能无法充分探索所有模式（模式混合问题）。
- **EBM vs 概率归一化模型**：归一化模型（如 PixelCNN）直接定义 $p(x)$ 且 $Z=1$，但需要自回归结构，不适合并行生成。EBM 可以定义任意复杂的能量函数，但代价是需要在采样时处理 $Z$。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EnergyNetwork(nn.Module):
    """能量函数网络：输入 x，输出能量（标量）"""
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 1),  # 输出能量（标量）
        )
    
    def forward(self, x):
        """返回能量 E(x)：-inf 到 +inf"""
        return self.net(x).squeeze(-1)

class EBM:
    """能量基模型训练器"""
    def __init__(self, energy_net, lr=1e-3):
        self.energy_net = energy_net
        self.optimizer = torch.optim.Adam(energy_net.parameters(), lr=lr)
    
    def sample_langevin(self, x_init, n_steps=60, step_size=0.1):
        """
        Langevin Dynamics 采样
        
        参数:
            x_init: 初始样本
            n_steps: MCMC 步数
            step_size: 步长
        """
        x = x_init.clone().detach().requires_grad_(True)
        
        for _ in range(n_steps):
            energy = self.energy_net(x).sum()
            grad = torch.autograd.grad(energy, x, create_graph=False)[0]
            
            x = x - 0.5 * step_size * grad + step_size * torch.randn_like(x)
            x = x.detach().requires_grad_(True)
        
        return x.detach()
    
    def train_step(self, real_data, n_steps=60, step_size=0.1):
        """
        EBM 训练步骤（对比散度）
        
        参数:
            real_data: 真实数据 [B, D]
            n_steps: MCMC 采样步数
            step_size: Langevin 步长
        """
        # 正相位：计算真实数据的能量
        real_energy = self.energy_net(real_data).mean()
        
        # 负相位：从模型分布采样
        noise = torch.randn_like(real_data) * 5  # 从大范围初始化为增加探索
        fake_samples = self.sample_langevin(noise, n_steps, step_size)
        fake_energy = self.energy_net(fake_samples).mean()
        
        # 损失：降低真实数据能量，提高采样点能量
        loss = real_energy - fake_energy
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return {
            'loss': loss.item(),
            'real_energy': real_energy.item(),
            'fake_energy': fake_energy.item(),
        }
    
    def generate(self, n_samples=100, input_dim=2, n_steps=100, step_size=0.1):
        """从训练好的 EBM 生成样本"""
        noise = torch.randn(n_samples, input_dim)
        samples = self.sample_langevin(noise, n_steps, step_size)
        return samples

# 验证 EBM 在简单数据上的训练
def train_ebm_on_2d_data():
    """在 2D 高斯混合数据上训练 EBM"""
    torch.manual_seed(42)
    
    # 生成 2D 数据（4 个高斯混合）
    n_per_mode = 250
    modes = [(3, 3), (-3, 3), (3, -3), (-3, -3)]
    data = []
    for mx, my in modes:
        samples = torch.randn(n_per_mode, 2) + torch.tensor([mx, my])
        data.append(samples)
    data = torch.cat(data, dim=0)
    
    # 训练 EBM
    energy_net = EnergyNetwork(input_dim=2)
    ebm = EBM(energy_net, lr=1e-2)
    
    print("=== EBM 训练（2D 高斯混合数据）===")
    for epoch in range(50):
        stats = ebm.train_step(data, n_steps=20, step_size=0.1)
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}: loss={stats['loss']:.4f}, "
                  f"real_E={stats['real_energy']:.4f}, fake_E={stats['fake_energy']:.4f}")
    
    print("\nEBM 训练完成！")
    print(f"真实数据点能量（应低）: {energy_net(data).mean().item():.4f}")
    
    # 生成样本
    generated = ebm.generate(100, input_dim=2)
    print(f"生成样本能量（应较低）: {energy_net(generated).mean().item():.4f}")
    print(f"随机噪声能量（应高）: {energy_net(torch.randn(100, 2) * 5).mean().item():.4f}")

# 运行演示
print("=== 能量基模型 (EBM) 基础 ===")
print()
print("核心公式: p(x) = exp(-E(x)) / Z")
print("训练: 降低真实数据能量 + 提升 MCMC 采样点能量")
print("采样: Langevin Dynamics")
print()

# 得分函数与能量函数的关系
print("得分函数 s(x) = -∇_x E(x)")
print("即 EBM 的能量梯度 = 负得分函数")
print("EBM ↔ Score-based Models ↔ Diffusion Models")
```

## 深度学习关联

- **EBM 与扩散模型的等价性**：扩散模型学习的是得分函数 $s_\theta(x_t, t) = -\nabla_{x_t} E_\theta(x_t, t)$，其中 $E_\theta$ 是随 $t$ 变化的能量函数（在不同噪声级别下的数据分布）。从 EBM 视角看，扩散模型就是一系列随时间退火的 EBM。
- **基于能量的 GAN (EBGAN)**：将判别器视为能量函数，生成器生成低能量样本（真实样本）。这统一了 GAN 和 EBM 的视角。
- **Implicit Generation versus EBMs**：GAN 和扩散模型是隐式生成模型（不显式定义 $p(x)$），而 EBM 是显式概率模型（定义 $p(x) \propto \exp(-E(x))$）。EBM 的优点是能计算似然，缺点是采样慢。
- **JEM (Joint Energy-based Model)**：将 EBM 与分类器结合——用分类器 $f(x)[y]$ 的能量 $E(x, y) = -f(x)[y]$ 同时学习分类和生成，在同一框架下处理判别和生成任务。
