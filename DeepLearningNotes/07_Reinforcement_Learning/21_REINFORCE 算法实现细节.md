# 21_REINFORCE 算法实现细节

## 核心概念

- **REINFORCE 算法**：1992 年 Williams 提出的最基础的蒙特卡洛策略梯度算法。直接使用完整 episode 的回报 $G_t$ 作为策略梯度中的 Q 值估计，是策略梯度方法最简单直接的实现。
- **REINFORCE 更新公式**：$\theta \leftarrow \theta + \alpha \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t|S_t)$。注意 $\gamma^t$ 项用于"折扣调整"（每个时间步的更新权重不同）。
- **与 Policy Gradient 定理的关系**：REINFORCE 直接使用策略梯度定理，用 $G_t$（完整回报的无偏估计）替代 $Q_{\pi_\theta}(s,a)$。$G_t$ 是 $Q_{\pi_\theta}(s,a)$ 的无偏估计，但方差很大。
- **带 baseline 的 REINFORCE**：$G_t - b(S_t)$ 替代 $G_t$，可大幅降低方差。最优 baseline 是 $b(s) = \frac{\mathbb{E}[||\nabla \log \pi||^2 G]}{\mathbb{E}[||\nabla \log \pi||^2]}$，实际常用 $b(s) = V(s)$。
- **算法流程**：收集完整 episode -> 计算每一步的折扣回报 $G_t$ -> 计算每一步的损失 $- \log \pi(a_t|s_t) G_t$ -> 求和/平均 -> 反向传播更新策略。
- **REINFORCE 的不足**：高方差导致收敛慢；on-policy 性质意味着每个 episode 的数据用完即弃；不能处理连续动作（需要高斯策略输出均值和方差）。

## 数学推导

$$
\text{REINFORCE 梯度: } \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ G_t \nabla_\theta \log \pi_\theta(A_t | S_t) \right]
$$

$$
\text{更新规则: } \theta_{t+1} = \theta_t + \alpha \gamma^t G_t \nabla_\theta \log \pi_\theta(A_t | S_t)
$$

$$
\text{带 baseline 的 REINFORCE: } \theta_{t+1} = \theta_t + \alpha \gamma^t (G_t - b(S_t)) \nabla_\theta \log \pi_\theta(A_t | S_t)
$$

$$
\text{回报计算（反向递推）: } G_t = R_{t+1} + \gamma G_{t+1}
$$

$$
\text{损失函数形式（PyTorch 实现）: } L(\theta) = -\frac{1}{T} \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) \cdot G_t
$$

**推导说明**：
- 使用 $\gamma^t$ 项是因为 $J(\theta) = \mathbb{E}[\sum_t \gamma^t R_t]$，每个时间步的奖励带有折扣权重 $\gamma^t$。
- 但实践中，当把 $G_t$ 定义为折扣回报（本身已含折扣）时，更新中不再额外乘以 $\gamma^t$。
- 常用的实现技巧：将 $G_t$ 标准化（减去均值除以标准差）可以显著稳定训练，虽然理论上引入了偏差但实际效果很好。
- 损失函数的负号是因为我们需要最大化 $J(\theta)$，而梯度下降默认最小化。

## 直观理解

REINFORCE 遵循"好的要鼓励，坏的要惩罚"的最朴素直觉：

想象你在训练一只小狗（策略 $\pi_\theta$）玩接飞盘游戏：
1. **玩一局（collect episode）**：小狗追飞盘，过程中它做了各种动作——跑、跳、停、转身。最后它成功接住了飞盘（好结果）。
2. **复盘（compute returns）**：你回放整个过程，对每个时刻打分——"跑到飞盘落点"这个动作获得了高分（$G_t$ 大），"半路停下来闻花"获得了低分。
3. **强化（apply gradient）**：对于高分动作，你给小狗零食（增加概率）；对于低分动作，你轻微责备（降低概率）。$\log \pi$ 的梯度告诉你怎么调整小狗的"行为参数"。
4. **重复**：继续玩，继续调整，直到小狗学会最优策略。

**为什么 REINFORCE 效率低？**
因为它像"只看结果，不看过程"的教练。一局比赛赢了，所有动作都加分；输了，所有动作都减分——不管具体哪个动作真正贡献了胜负。这种"全局奖励分配"导致了高方差，需要大量 episode 才能收敛。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from collections import deque

class REINFORCEAgent:
    """带 Baseline 的 REINFORCE 算法实现"""
    def __init__(self, state_dim, action_dim, hidden=128, lr=0.001, gamma=0.99):
        # 策略网络 (Actor)
        self.policy = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
        # 价值网络 (Baseline) - 可选
        self.baseline = nn.Sequential(
            nn.Linear(state_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
            nn.Linear(hidden, 1)
        )
        
        self.policy_optimizer = optim.Adam(self.policy.parameters(), lr=lr)
        self.baseline_optimizer = optim.Adam(self.baseline.parameters(), lr=lr)
        
        self.gamma = gamma
        self.action_dim = action_dim
    
    def get_action(self, state):
        """从策略分布中采样动作"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        logits = self.policy(state_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample().item()
        log_prob = dist.log_prob(torch.tensor(action))
        return action, log_prob
    
    def get_value(self, state):
        """估计状态价值（baseline）"""
        state_t = torch.FloatTensor(state).unsqueeze(0)
        return self.baseline(state_t).squeeze()
    
    def update(self, episode):
        """使用一个完整 episode 更新策略"""
        states, actions, rewards, log_probs = episode
        
        # 计算折扣回报 G_t
        returns = []
        G = 0
        for r in reversed(rewards):
            G = r + self.gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # 可选：标准化回报
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算 baseline 预测（使用价值网络）
        states_t = torch.FloatTensor(np.array(states))
        values = self.baseline(states_t).squeeze()
        
        # 更新 baseline（MSE loss）
        baseline_loss = F.mse_loss(values, returns)
        self.baseline_optimizer.zero_grad()
        baseline_loss.backward()
        self.baseline_optimizer.step()
        
        # 计算优势 A = G - V(s)
        advantages = returns - values.detach()
        
        # 更新策略（REINFORCE with baseline）
        policy_loss = 0
        for log_prob, adv in zip(log_probs, advantages):
            policy_loss += -log_prob * adv
        
        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()
        
        return policy_loss.item(), baseline_loss.item()

# 训练测试
def train_reinforce():
    env = gym.make("CartPole-v1")
    agent = REINFORCEAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        lr=0.001
    )
    
    scores = []
    for episode in range(500):
        state, info = env.reset()
        done = False
        episode_data = [], [], [], []
        states, actions, rewards, log_probs = episode_data
        
        while not done:
            action, log_prob = agent.get_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            log_probs.append(log_prob)
            
            state = next_state
        
        policy_loss, value_loss = agent.update(episode_data)
        scores.append(sum(rewards))
        
        if (episode + 1) % 50 == 0:
            avg = np.mean(scores[-50:])
            print(f"Episode {episode+1}: 平均奖励={avg:.1f}")
    
    env.close()
    return scores

print("REINFORCE with Baseline 算法实现 - 准备就绪")
print("训练中... (取消注释下方代码以实际运行)")
# scores = train_reinforce()
```

## 深度学习关联

1. **REINFORCE 到 PPO 的演进路径**：REINFORCE（高方差、on-policy）-> Actor-Critic（引入价值函数降低方差）-> A2C/A3C（优势函数 + 并行训练）-> TRPO（信任区域约束）-> PPO（截断式剪裁，目前最主流的 DRL 算法）。理解 REINFORCE 是理解整个策略梯度家族的基础。
2. **Policy Gradient in LLM Fine-tuning**：策略梯度方法（包括 REINFORCE 的思想）被广泛应用于大语言模型的强化学习微调（如 RLHF）。PPO 是其主力算法，而 REINFORCE 风格的基于排名的奖励（如在线对比学习）也在涌现。
3. **PG 的自动微分实现**：深度学习框架（PyTorch/TF/JAX）使策略梯度的实现变得极为简洁——仅需定义 $L = -\log \pi(a|s) \cdot A$ 作为损失，"最大化似然乘以优势"这个操作被自动微分完美支持。这体现了自动微分在 RL 算法工程中的核心作用。
