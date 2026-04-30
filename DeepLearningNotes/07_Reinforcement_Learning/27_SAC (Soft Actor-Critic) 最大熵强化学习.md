# 27_SAC (Soft Actor-Critic) 最大熵强化学习

## 核心概念

- **最大熵强化学习 (Maximum Entropy RL)**：在标准 RL 目标（最大化期望回报）基础上加入策略熵最大化项。目标是最大化 $\sum_t \mathbb{E}[r(s_t, a_t) + \alpha H(\pi(\cdot|s_t))]$，鼓励策略在保持高回报的同时尽可能随机。
- **SAC (Soft Actor-Critic)**：2018 年 Haarnoja 等人提出的 off-policy 最大熵 Actor-Critic 算法。结合了 最大熵 RL、off-policy 训练、软更新和重参数化技巧，是目前连续控制任务中最先进的算法之一。
- **软价值函数 (Soft Value Function)**：最大熵框架下的价值函数定义为 $V_{\text{soft}}(s) = \mathbb{E}[\sum_t \gamma^t (r_t + \alpha H(\pi(\cdot|s_t)))]$，同时编码了奖励和探索。
- **温度参数 $\alpha$**：控制熵项的相对重要性。$\alpha$ 越大，策略越随机（更多探索）；$\alpha$ 越小，策略越确定（专注利用）。SAC 可以自动调节 $\alpha$ 以维持目标熵水平。
- **Off-policy 训练**：SAC 使用经验回放，显著提高了样本效率。与 DQN 一样，SAC 从回放缓冲区采样 mini-batch 进行更新，但 SAC 同时学习策略（Actor）和价值函数（Critic）。
- **连续动作空间**：SAC 原生支持连续动作，使用重参数化技巧（reparameterization trick）计算策略梯度，Actor 输出高斯分布的均值和方差。

## 数学推导

$$
\text{最大熵 RL 目标: } J(\pi) = \sum_{t=0}^\infty \mathbb{E}_{(s_t,a_t) \sim \rho_\pi} \left[ r(s_t, a_t) + \alpha H(\pi(\cdot|s_t)) \right]
$$

$$
\text{软 Q 函数更新: } Q_{\text{soft}}(s, a) = r(s, a) + \gamma \mathbb{E}_{s' \sim P} \left[ V_{\text{soft}}(s') \right]
$$

$$
\text{软状态价值: } V_{\text{soft}}(s) = \mathbb{E}_{a \sim \pi} \left[ Q_{\text{soft}}(s, a) - \alpha \log \pi(a|s) \right]
$$

$$
\text{SAC Actor 损失: } L_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}} \left[ \mathbb{E}_{a \sim \pi_\phi} \left[ \alpha \log \pi_\phi(a|s) - Q_\theta(s, a) \right] \right]
$$

$$
\text{重参数化: } a = f_\phi(\epsilon; s), \quad \epsilon \sim \mathcal{N}(0, I)
$$

$$
\text{重参数化后的 Actor 损失: } L_\pi(\phi) = \mathbb{E}_{s \sim \mathcal{D}, \epsilon \sim \mathcal{N}} \left[ \alpha \log \pi_\phi(f_\phi(\epsilon; s)|s) - Q_\theta(s, f_\phi(\epsilon; s)) \right]
$$

$$
\text{SAC Critic 损失: } L_Q(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( Q_\theta(s, a) - (r + \gamma V_{\bar{\theta}}(s')) \right)^2 \right]
$$

$$
\text{自动调节 $\alpha$: } L(\alpha) = \mathbb{E}_{a \sim \pi_\phi} \left[ -\alpha \log \pi_\phi(a|s) - \alpha \bar{H} \right]
$$

**推导说明**：
- 软 Q 函数的贝尔曼方程引入了熵项，使 Q 函数的更新不仅考虑奖励，还考虑未来的策略熵。
- 重参数化技巧（类似 VAE）将采样过程变为确定性路径 + 噪声输入，使 Actor 的梯度可以穿过采样操作反向传播。
- 温度 $\alpha$ 的自动调节通过最小化 $-\alpha(\log\pi + \bar{H})$ 实现，$\bar{H}$ 是目标熵（通常设为 $-\dim(\mathcal{A})$）。

## 直观理解

SAC 的核心理念是"既要赚钱，又要保持好奇心"：

**标准 RL 目标**：最大化总奖励 = "只赚钱，其他不管"
**最大熵 RL 目标**：最大化总奖励 + 熵 = "赚钱的同时保持探索精神"

**类比：投资策略**
- **标准 RL**：发现比特币涨得好，All in 比特币。之后比特币涨了一段时间，突然暴跌——亏光。
- **最大熵 RL**：发现比特币涨得好，但配置了 60% 比特币、20% 黄金、20% 债券。比特币暴跌时，其他资产缓冲了损失，而且你有分散投资的"经验"（熵）来应对市场变化。

**为什么 SAC 在连续控制中表现优异？**

想象一个机器人手需要抓住一个杯子：
- **标准确定性策略（如 DDPG）**：学习一种"只有特定角度、特定力度"才能成功的抓取方式。如果环境稍有变化（杯子位置偏移 1cm），策略完全失效。
- **SAC 的最大熵策略**：学会了一系列"可能成功"的抓取方式。不仅一种方式成功，而且有多种"备份方式"。当环境变化时，策略能灵活适应。

SAC 相当于学会了"多个好方案"，而不仅仅是"一个最佳方案"。这带来了两个关键优势：
1. **更强的鲁棒性**：面对变化时不崩溃。
2. **更好的探索**：训练过程中自然探索更多样的行为。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

# SAC 网络定义
class GaussianActor(nn.Module):
    """连续动作的高斯策略网络"""
    def __init__(self, state_dim, action_dim, hidden=256, log_std_min=-20, log_std_max=2):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        self.action_dim = action_dim
    
    def forward(self, state, deterministic=False):
        features = self.net(state)
        mean = self.mean(features)
        log_std = torch.clamp(self.log_std(features), self.log_std_min, self.log_std_max)
        std = torch.exp(log_std)
        
        if deterministic:
            return torch.tanh(mean)
        
        # 重参数化采样
        normal = torch.distributions.Normal(mean, std)
        z = normal.rsample()  # reparameterization trick
        action = torch.tanh(z)
        
        # 计算 log 概率（考虑 tanh 缩放）
        log_prob = normal.log_prob(z) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def get_action(self, state, deterministic=False):
        if deterministic:
            action = self.forward(state, deterministic=True)
        else:
            action, _ = self.forward(state)
        return action

class SoftQNetwork(nn.Module):
    """软 Q 网络"""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class SACAgent:
    """SAC 主智能体"""
    def __init__(self, state_dim, action_dim, actor_lr=3e-4, critic_lr=3e-4,
                 alpha_lr=3e-4, gamma=0.99, tau=0.005, alpha=0.2,
                 automatic_alpha_tuning=True, target_entropy=None):
        self.actor = GaussianActor(state_dim, action_dim)
        self.q1 = SoftQNetwork(state_dim, action_dim)
        self.q2 = SoftQNetwork(state_dim, action_dim)  # 双 Q 缓解过估计
        self.q1_target = SoftQNetwork(state_dim, action_dim)
        self.q2_target = SoftQNetwork(state_dim, action_dim)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.q1_optimizer = optim.Adam(self.q1.parameters(), lr=critic_lr)
        self.q2_optimizer = optim.Adam(self.q2.parameters(), lr=critic_lr)
        
        self.gamma = gamma
        self.tau = tau
        
        # 自动调节 alpha
        self.automatic_alpha_tuning = automatic_alpha_tuning
        if automatic_alpha_tuning:
            self.target_entropy = target_entropy or -action_dim
            self.log_alpha = torch.zeros(1, requires_grad=True)
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=alpha_lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha
    
    def update(self, batch):
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            # 下一状态的动作和 log 概率
            next_actions, next_log_probs = self.actor(next_states)
            # 目标 Q 值（取两个 Q 的最小值）
            next_q1 = self.q1_target(next_states, next_actions)
            next_q2 = self.q2_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2) - self.alpha * next_log_probs
            # 软贝尔曼目标
            target_q = rewards + self.gamma * (1 - dones) * next_q
        
        # 更新 Q 网络
        current_q1 = self.q1(states, actions)
        current_q2 = self.q2(states, actions)
        q1_loss = F.mse_loss(current_q1, target_q)
        q2_loss = F.mse_loss(current_q2, target_q)
        
        self.q1_optimizer.zero_grad()
        q1_loss.backward()
        self.q1_optimizer.step()
        
        self.q2_optimizer.zero_grad()
        q2_loss.backward()
        self.q2_optimizer.step()
        
        # 更新 Actor（重参数化）
        new_actions, log_probs = self.actor(states)
        q1_new = self.q1(states, new_actions)
        q2_new = self.q2(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        actor_loss = (self.alpha * log_probs - q_new).mean()
        
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # 更新 alpha
        if self.automatic_alpha_tuning:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()
        
        # 软更新目标网络
        for target, source in [(self.q1_target, self.q1), (self.q2_target, self.q2)]:
            for tp, sp in zip(target.parameters(), source.parameters()):
                tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        
        return {
            'q1_loss': q1_loss.item(),
            'actor_loss': actor_loss.item(),
            'alpha': self.alpha,
            'entropy': -log_probs.mean().item()
        }

print("SAC (Soft Actor-Critic) 实现 - 准备就绪")
print("SAC 是当前连续控制任务的首选算法之一")
```

## 深度学习关联

1. **最大熵与探索**：SAC 的最大熵目标可以看作是一种"内在动机"（intrinsic motivation）的数学形式——通过熵正则化鼓励智能体保持随机性。这与深度学习中防止过拟合的正则化类似（如 Dropout、标签平滑），都是在训练目标中引入额外的分散度以防止过早收敛。
2. **重参数化技巧**：SAC 中使用的重参数化技巧（rsample）与 VAE（变分自编码器）中的技巧完全相同。两者都需在随机采样过程中传播梯度，重参数化将随机性转移到独立的噪声变量，使梯度可微。这体现了深度生成模型与深度强化学习之间的技术互通。
3. **双 Q 网络**：SAC 使用两个 Q 网络并取最小值来缓解价值过估计，与 TD3 的 clipped double Q-learning 相同。这种"冗余+保守估计"的策略在工程中广泛使用——它是强化学习中的"冗余系统"设计，类似于航空中的双引擎备份，提高了系统的整体稳定性。
