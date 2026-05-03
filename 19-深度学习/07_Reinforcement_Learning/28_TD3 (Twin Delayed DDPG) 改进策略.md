# 28_TD3 (Twin Delayed DDPG) 改进策略

## 核心概念

- **TD3 (Twin Delayed DDPG)**：2018 年 Fujimoto 等人提出的 DDPG 改进算法，针对连续动作空间中的价值过估计和策略更新不稳定问题，提出了三项关键改进。
- **改进一：裁剪双 Q 学习 (Clipped Double Q-Learning)**：使用两个独立的 Q 网络 $(Q_{\theta_1}, Q_{\theta_2})$，目标值取两者的最小值 $y = r + \gamma \min_{i=1,2} Q_{\theta_i'}(s', \tilde{a}')$。这有效缓解了 DDPG 中的 Q 值过估计问题。
- **改进二：延迟策略更新 (Delayed Policy Updates)**：策略（Actor）的更新频率低于价值（Critic）网络。通常每更新 2 次 Critic 才更新 1 次 Actor。这给价值网络足够的时间"稳定下来"，减少策略更新时的误差。
- **改进三：目标策略平滑 (Target Policy Smoothing)**：在目标动作中加入噪声 $a' \leftarrow \pi_{\phi'}(s') + \epsilon, \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)$，迫使策略对类似的 Q 值保持平滑，防止价值函数在动作空间的尖锐峰值处过拟合。
- **DDPG 的问题**：DDPG（Deep Deterministic Policy Gradient）是 DQN 在连续动作空间的扩展，但确定性策略 + 函数近似会导致过估计和策略脆弱性。TD3 的所有改进都针对这些问题。
- **与 SAC 的异同**：两者都使用双 Q 网络和连续控制，但 SAC 基于最大熵框架（关注随机策略），TD3 基于确定性策略梯度（关注改进稳定性）。TD3 的"平滑 + 双 Q + 延迟更新"是工程技巧，而非 SAC 那样的理论框架改造。

## 数学推导

$$
\text{TD3 Critic 目标（Clipped Double Q）: }
$$

$$
y = r + \gamma \min_{i=1,2} Q_{\theta_i'} \left( s', \pi_{\phi'}(s') + \epsilon \right), \quad \epsilon \sim \text{clip}(\mathcal{N}(0, \sigma), -c, c)
$$

$$
\text{TD3 Critic 损失: } L(\theta_i) = \mathbb{E} \left[ \left( y - Q_{\theta_i}(s, a) \right)^2 \right], \quad i = 1, 2
$$

$$
\text{TD3 Actor 更新（延迟）: } \nabla_\phi J(\phi) = \mathbb{E} \left[ \nabla_a Q_{\theta_1}(s, a) \big|_{a=\pi_\phi(s)} \nabla_\phi \pi_\phi(s) \right]
$$

$$
\text{延迟更新: 每 } d \text{ 步更新一次 Actor, 通常 } d = 2
$$

$$
\text{目标网络软更新: } \theta_i' \leftarrow \tau \theta_i + (1 - \tau) \theta_i', \quad \phi' \leftarrow \tau \phi + (1 - \tau) \phi'
$$

**推导说明**：
- Clipped Double Q-learning 从 Double Q-learning 中汲取灵感但做了连续控制的适配：两个 Q 网络同时训练，目标取 min，提供了一个下界估计，减少了正向偏差。
- 目标策略平滑的正则化效果类似于参数噪声或 Dropout——Q 函数在动作空间中的尖锐峰值被平滑了，使策略更新更加稳健。
- 延迟更新基于直觉：Critic 误差较大时更新 Actor 会导致高方差的策略梯度。等待 Critic 收敛到更稳定的状态后再更新 Actor。

## 直观理解

TD3 的三项改进可以类比为"审稿制度"：

**标准 DDPG**：一个作者（Actor）写好论文，一个审稿人（Critic）给出评分。但审稿人有时会给虚高的分数（Q 值过估计），作者根据虚高的评分改变研究方向（策略更新），最后研究方向越来越歪。

**TD3 的改进**：

- **双 Q 网络（双审稿人）**：聘请两个审稿人，最终评分取**两人中较低的那个**。这防止了某个审稿人因偏心给出虚高评分。$\min(Q_1, Q_2)$ 是一种保守的评分策略。
- **目标策略平滑（模糊评审标准）**：审稿人在看"未来论文方向"时，先给论文的评价标准加上一些噪声——"如果作者稍微改几个词（动作噪声），评价不应该剧烈波动"。这迫使 Q 函数对动作变化不敏感，像一个"平滑"的评分函数，防止过度拟合特定表达。
- **延迟策略更新（审稿人先熟悉领域再改方向）**：作者的策略改动频率只有审稿人评分标准更新频率的一半。审稿人先多次审稿、稳定自己的评分标准，然后作者再根据稳定的评分来调整研究方向。

**把三者合在一起**：TD3 就是用"保守双审稿 + 平滑标准 + 延迟修改"来避免学术方向跑偏的稳健研究策略。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

class Actor(nn.Module):
    """确定性策略网络（输出连续动作）"""
    def __init__(self, state_dim, action_dim, max_action, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Tanh()
        )
        self.max_action = max_action
    
    def forward(self, state):
        return self.net(state) * self.max_action

class Critic(nn.Module):
    """Q 网络"""
    def __init__(self, state_dim, action_dim, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim + action_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
    
    def forward(self, state, action):
        return self.net(torch.cat([state, action], dim=-1))

class TD3Agent:
    """TD3 智能体"""
    def __init__(self, state_dim, action_dim, max_action=1.0,
                 actor_lr=3e-4, critic_lr=3e-4, gamma=0.99,
                 tau=0.005, policy_noise=0.2, noise_clip=0.5, 
                 policy_delay=2):
        self.actor = Actor(state_dim, action_dim, max_action)
        self.actor_target = Actor(state_dim, action_dim, max_action)
        self.actor_target.load_state_dict(self.actor.state_dict())
        
        self.critic1 = Critic(state_dim, action_dim)
        self.critic2 = Critic(state_dim, action_dim)
        self.critic1_target = Critic(state_dim, action_dim)
        self.critic2_target = Critic(state_dim, action_dim)
        self.critic1_target.load_state_dict(self.critic1.state_dict())
        self.critic2_target.load_state_dict(self.critic2.state_dict())
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=actor_lr)
        self.critic_optimizer = optim.Adam(
            list(self.critic1.parameters()) + list(self.critic2.parameters()), 
            lr=critic_lr
        )
        
        self.gamma = gamma
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_delay = policy_delay
        self.max_action = max_action
        self.total_it = 0
    
    def select_action(self, state, noise=0.1):
        state = torch.FloatTensor(state.reshape(1, -1))
        action = self.actor(state).detach().numpy()[0]
        if noise:
            action += np.random.normal(0, noise * self.max_action, size=action.shape)
        return np.clip(action, -self.max_action, self.max_action)
    
    def update(self, batch):
        self.total_it += 1
        
        states, actions, rewards, next_states, dones = batch
        states = torch.FloatTensor(states)
        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones).unsqueeze(1)
        
        with torch.no_grad():
            # 目标策略平滑
            next_actions = self.actor_target(next_states)
            noise = (torch.randn_like(actions) * self.policy_noise)
            noise = torch.clamp(noise, -self.noise_clip, self.noise_clip)
            next_actions = torch.clamp(next_actions + noise, 
                                       -self.max_action, self.max_action)
            
            # 裁剪双 Q 目标
            target_q1 = self.critic1_target(next_states, next_actions)
            target_q2 = self.critic2_target(next_states, next_actions)
            target_q = torch.min(target_q1, target_q2)
            target_q = rewards + self.gamma * (1 - dones) * target_q
        
        # 更新 Critic
        current_q1 = self.critic1(states, actions)
        current_q2 = self.critic2(states, actions)
        critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)
        
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        
        # 延迟更新 Actor
        if self.total_it % self.policy_delay == 0:
            # 确定性策略梯度
            actor_loss = -self.critic1(states, self.actor(states)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
            
            # 软更新目标网络
            for target, source in [(self.actor_target, self.actor),
                                   (self.critic1_target, self.critic1),
                                   (self.critic2_target, self.critic2)]:
                for tp, sp in zip(target.parameters(), source.parameters()):
                    tp.data.copy_(self.tau * sp.data + (1 - self.tau) * tp.data)
        
        return {'critic_loss': critic_loss.item()}

print("TD3 (Twin Delayed DDPG) 实现 - 准备就绪")
print("TD3 的三项改进: 裁剪双Q + 策略延迟 + 目标平滑")
```

## 深度学习关联

- **目标平滑作为正则化**：TD3 的目标策略平滑等效于在 Q 函数的输入（动作）上添加噪声，迫使 Q 函数在动作空间中保持局部 Lipschitz 连续性。这类似于监督学习中的数据增强（如随机裁剪、颜色抖动）——通过扰动输入让模型对局部变化不敏感，从而提高泛化能力。
- **延迟更新的思想**：延迟策略更新的思想在深度学习中也有对应——在 GAN 训练中，判别器（Critic）通常需要多次更新后生成器（Actor）才更新一次。两个网络以不同频率更新，可以让"评价网络"先收敛到合理状态，再让"生成网络"据此改进。
- **确定性 vs 随机策略**：TD3 使用确定性策略（输出确定的动作值），而 SAC 使用随机策略（输出动作分布）。确定性策略在连续控制中更高效（无需采样），但更容易过拟合；随机策略更鲁棒但需要更多采样。这种权衡与深度学习中的点估计 vs 贝叶斯方法的对比类似。
