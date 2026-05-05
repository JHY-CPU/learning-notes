# 19_Score Matching 与 Langevin Dynamics

## 核心概念

- **得分函数 (Score Function)**：定义为对数概率密度的梯度 $\nabla_x \log p(x)$，指向概率密度增长最快的方向。在低密度区域，得分指向高密度区域。
- **Score Matching**：一种不需要知道归一化常数 $Z$ 就能学习未归一化概率模型 $p(x) = \frac{\tilde{p}(x)}{Z}$ 的技术，直接学习得分函数 $\nabla_x \log \tilde{p}(x)$。
- **Langevin Dynamics**：一种马尔可夫链蒙特卡洛（MCMC）采样方法，利用得分函数从概率分布中生成样本：$x_{i+1} = x_i + \frac{\delta}{2} \nabla_x \log p(x_i) + \sqrt{\delta} \cdot z_i$。
- **得分匹配 + Langevin 采样**：Song & Ermon (2019) 提出用得分匹配训练网络 $s_\theta(x) \approx \nabla_x \log p(x)$，然后用 Langevin 采样生成——这就是 Score-based Generative Model。
- **多尺度噪声扰动**：因为单独一个得分函数在低密度区域不准确（缺少训练数据），需要添加多级噪声扰动，使得分布覆盖整个空间，每个噪声级别对应一个得分网络。
- **与 DDPM 的统一**：扩散模型和得分匹配在 SDE 框架下被统一——DDPM 学习的是 $\epsilon_\theta(x_t, t)$，而得分函数 $s_\theta(x_t, t) = -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$，两者存在简单的换算关系。

## 数学推导

**得分匹配损失**：

目标：学习 $s_\theta(x) \approx \nabla_x \log p_{\text{data}}(x)$

$$
\mathcal{L}_{\text{SM}}(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}} \left[ \left\| s_\theta(x) - \nabla_x \log p_{\text{data}}(x) \right\|^2 \right]
$$

但是 $\nabla_x \log p_{\text{data}}(x)$ 未知。

**去噪得分匹配 (Denoising Score Matching)**：

用扰动后的分布 $q_\sigma(\tilde{x}|x) = \mathcal{N}(\tilde{x}; x, \sigma^2 I)$ 替代，目标变为：

$$
\mathcal{L}_{\text{DSM}}(\theta) = \frac{1}{2} \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{\tilde{x} \sim q_\sigma(\tilde{x}|x)} \left[ \left\| s_\theta(\tilde{x}) - \nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) \right\|^2 \right]
$$

其中 $\nabla_{\tilde{x}} \log q_\sigma(\tilde{x}|x) = -\frac{\tilde{x} - x}{\sigma^2}$，这是已知的。

**Langevin 采样算法**：

给定得分函数 $s_\theta(x) \approx \nabla_x \log p(x)$，从先验 $x_0 \sim \pi(x)$（如均匀分布）开始，迭代：

$$
x_{t+1} = x_t + \frac{\epsilon}{2} s_\theta(x_t) + \sqrt{\epsilon} \cdot z_t, \quad z_t \sim \mathcal{N}(0, I)
$$

其中 $\epsilon$ 是步长。当 $\epsilon \to 0$ 且 $t \to \infty$ 时，$x_t$ 的分布收敛到 $p(x)$。

**噪声条件得分网络 (NCSN)**：

使用 $L$ 个不同级别的噪声 $\sigma_1 > \sigma_2 > ... > \sigma_L$：

$$
\mathcal{L}_{\text{NCSN}}(\theta) = \sum_{i=1}^L \lambda(\sigma_i) \mathbb{E}_{x \sim p_{\text{data}}} \mathbb{E}_{\tilde{x} \sim \mathcal{N}(x, \sigma_i^2 I)} \left[ \left\| s_\theta(\tilde{x}, \sigma_i) + \frac{\tilde{x} - x}{\sigma_i^2} \right\|^2 \right]
$$

其中 $\lambda(\sigma_i) = \sigma_i^2$ 用于平衡各噪声级别的损失尺度。

**得分函数与 DDPM 的关系**：

在 DDPM 中，$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$，$\epsilon \sim \mathcal{N}(0, I)$。

$$
\nabla_{x_t} \log p(x_t|x_0) = -\frac{x_t - \sqrt{\bar{\alpha}_t} x_0}{1-\bar{\alpha}_t} = -\frac{\epsilon}{\sqrt{1-\bar{\alpha}_t}}
$$

因此得分函数 $s_\theta(x_t, t) \approx \nabla_{x_t} \log p(x_t) \approx -\frac{\epsilon_\theta(x_t, t)}{\sqrt{1-\bar{\alpha}_t}}$。

## 直观理解

- **得分函数 = 地图上的坡度**：如果你站在一张概率分布的地图上，得分函数告诉你"往哪个方向走能找到更高密度的区域"。峰顶（数据所在区域）的得分为 0，山谷的得分指向峰顶。
- **Langevin 动力学 = 醉汉爬山**：一个醉汉（采样点）沿着"上坡方向"（得分方向）走，但又会随机晃悠（噪声项），最终他会停在峰顶附近（高密度区域）。多个醉汉从不同位置出发，就能探索整个分布。
- **为什么需要多级噪声**：在高维空间中，数据只集中在低维流形上，流形外几乎没有数据——得分函数在这些区域方向不准。给数据加噪声就像给地图"模糊化"，让数据"扩散"到整个空间，使得分函数处处有意义。
- **得分匹配 vs 最大似然**：最大似然需要知道归一化常数 $Z$（对所有可能的 $x$ 积分，不可行），得分匹配通过直接学习 $\nabla_x \log p(x)$ 巧妙地避开了 $Z$ 的计算，因为 $\nabla_x \log p(x) = \nabla_x \log \tilde{p}(x)$（$Z$ 与 $x$ 无关，梯度为 0）。

## 代码示例

```python
import torch
import torch.nn as nn
import numpy as np

class ScoreNetwork(nn.Module):
    """得分网络：学习数据的得分函数"""
    def __init__(self, input_dim=2, hidden_dim=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, input_dim),  # 输出与输入维度相同
        )
    
    def forward(self, x, sigma):
        # 将 sigma 作为额外输入拼接
        sigma_emb = torch.full((x.size(0), 1), sigma, device=x.device)
        x_cond = torch.cat([x, sigma_emb], dim=-1)
        return self.net(x_cond)

# 去噪得分匹配损失
def denoising_score_matching_loss(score_net, x, sigma=0.1):
    """
    去噪得分匹配损失
    
    参数:
        score_net: 得分网络 s_theta(x, sigma)
        x: 干净数据
        sigma: 噪声级别
    """
    # 添加噪声
    noise = torch.randn_like(x) * sigma
    x_noisy = x + noise
    
    # 网络预测的得分
    predicted_score = score_net(x_noisy, sigma)
    
    # 真实得分：grad_x log q_sigma(x_noisy|x) = -noise / sigma^2
    target_score = -noise / (sigma ** 2)
    
    # 损失：加权后等价于 ||预测噪声 - 真实噪声||^2
    loss = torch.mean((predicted_score - target_score) ** 2) * (sigma ** 2)
    return loss

# Langevin 采样
def langevin_sampling(score_net, n_samples=1000, n_steps=100, step_size=0.01, sigma=0.1):
    """
    Langevin 动力学采样
    
    参数:
        score_net: 得分网络
        n_steps: 采样步数
        step_size: 步长
        sigma: 噪声级别
    """
    # 从均匀分布初始化
    x = torch.randn(n_samples, 2) * 5  # 从大范围初始化
    
    for i in range(n_steps):
        # 计算得分
        score = score_net(x, sigma)
        
        # Langevin 更新
        noise = torch.randn_like(x)
        x = x + 0.5 * step_size * score + torch.sqrt(torch.tensor(step_size)) * noise
    
    return x

# NCSN：多级噪声训练
def train_ncsn(score_net, data, sigmas, optimizer):
    """训练噪声条件得分网络（NCSN）"""
    total_loss = 0
    
    for sigma in sigmas:
        loss = denoising_score_matching_loss(score_net, data, sigma)
        total_loss += loss
    
    optimizer.zero_grad()
    total_loss.backward()
    optimizer.step()
    
    return total_loss.item()

# 演示得分函数与 DDPM 噪声预测的关系
def score_to_noise(score, alpha_bar_t):
    """得分函数到噪声预测的转换"""
    # s_theta = -epsilon_theta / sqrt(1 - alpha_bar)
    # epsilon_theta = -s_theta * sqrt(1 - alpha_bar)
    return -score * torch.sqrt(1 - alpha_bar_t)

def noise_to_score(epsilon, alpha_bar_t):
    """噪声预测到得分函数的转换"""
    # score = -epsilon / sqrt(1 - alpha_bar)
    return -epsilon / torch.sqrt(1 - alpha_bar_t)

# 简单验证
print("=== Score Matching 与 Langevin Dynamics ===")
print()

# 得分函数与 DDPM 的联系
alpha_bar = torch.tensor(0.5)
epsilon_pred = torch.tensor([0.1, 0.2, -0.1])
score_pred = noise_to_score(epsilon_pred, alpha_bar)
epsilon_back = score_to_noise(score_pred, alpha_bar)
print(f"原始噪声预测: {epsilon_pred}")
print(f"转换为得分: {score_pred}")
print(f"转换回噪声: {epsilon_back}")
print(f"一致: {torch.allclose(epsilon_pred, epsilon_back)}")
print()
print("DDPM 的噪声预测与 Score-based 模型的得分函数是等价的！")
```

## 深度学习关联

- **Score SDE (Song et al., 2021)**：将去噪得分匹配和 DDPM 统一为随机微分方程框架——前向 SDE 加噪，反向 SDE 去噪，得分函数 $\nabla_x \log p_t(x)$ 是连接两者的桥梁。
- **Classifier Guidance**：条件生成时，需要 $\nabla_x \log p(x|y) = \nabla_x \log p(x) + \nabla_x \log p(y|x)$——无条件得分加上分类器的梯度（对抗梯度引导），这直观上就是 Langevin 动力学向分类器认为"对"的方向移动。
- **Energy-Based Models (EBM)**：得分匹配与 EBM 有直接联系——$s_\theta(x) = \nabla_x \log p_\theta(x) = -\nabla_x E_\theta(x)$，其中 $E_\theta(x)$ 是能量函数。得分匹配可以视为训练 EBM 的一种方式。
- **一致性模型 (Consistency Model)**：直接从得分函数 $s_\theta(x_t, t)$ 学习一个从任意噪声级别 $t$ 到 $t=0$ 的跳跃映射，避免了 Langevin 采样的迭代过程，实现一步生成。
