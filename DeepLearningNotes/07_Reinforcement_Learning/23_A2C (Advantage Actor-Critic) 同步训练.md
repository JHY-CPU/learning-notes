# 23_A2C (Advantage Actor-Critic) 同步训练

## 核心概念

- **A2C (Advantage Actor-Critic)**：A3C 的同步版本。使用多个并行 worker 在环境中收集经验，所有 worker 同步计算梯度后统一更新全局网络参数。比 A3C 更简洁，且 GPU 利用率更高。
- **优势函数 (Advantage Function)**：A2C 的名称即源于"Advantage"——使用优势函数 $A(s,a) = Q(s,a) - V(s)$ 作为策略梯度的加权信号，而非原始的 Q 值或回报。优势函数大幅降低了梯度估计的方差。
- **多步优势估计 (N-step Advantage)**：A2C 通常使用 N 步回报计算优势，而非一步 TD 误差：$A_t = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)$。N 步估计平衡了偏差和方差。
- **同步并行训练**：多个 worker 分别在环境中采集 $T$ 步数据（$T=5$ 或 $T=20$ 等），然后同步计算梯度并累积，统一更新全局网络参数。所有 worker 使用相同的参数进行下一轮采集。
- **Entropy 正则化**：在 A2C 的损失函数中加入策略熵 $H(\pi) = -\sum_a \pi(a|s) \log \pi(a|s)$ 的负值作为正则项，鼓励探索、防止策略过早变得确定。
- **A2C vs A3C**：A2C (synchronous) 等所有 worker 完成后再更新；A3C (asynchronous) 每个 worker 独立异步更新。实践证明 A2C 通常比 A3C 更稳定、效果更好，且更容易在 GPU 上高效运行。

## 数学推导

$$
\text{N 步优势: } A(s_t, a_t) = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) - V(s_t)
$$

$$
\text{A2C 总损失: } L = \underbrace{L_{\text{Actor}} + L_{\text{Critic}}}_{\text{标准 AC 损失}} + \underbrace{\beta \, \mathbb{E}[H(\pi(\cdot|s))]}_{\text{熵正则}}
$$

$$
\text{Actor 损失: } L_{\text{Actor}} = -\frac{1}{T} \sum_{t=0}^{T-1} \log \pi_\theta(a_t|s_t) \, A(s_t, a_t)
$$

$$
\text{Critic 损失: } L_{\text{Critic}} = \frac{1}{T} \sum_{t=0}^{T-1} \left( G_t^{(n)} - V_\phi(s_t) \right)^2
$$

$$
\text{其中 } G_t^{(n)} = \sum_{i=0}^{n-1} \gamma^i r_{t+i} + \gamma^n V(s_{t+n}) \quad \text{(N 步回报目标)}
$$

$$
\text{熵正则: } H(\pi(\cdot|s)) = -\sum_{a \in \mathcal{A}} \pi_\theta(a|s) \log \pi_\theta(a|s)
$$

**推导说明**：
- 多 worker 同步采集：每个 worker 收集 $T$ 步经验（如 5 步或 20 步），所有 worker 数据拼接后计算梯度。
- $G_t^{(n)}$ 是介于 MC 回报（$n = \infty$）和一步 TD 目标（$n=1$）之间的 N 步回报。
- 熵系数 $\beta$（通常 0.01 到 0.001）控制探索程度——熵越大，策略越随机，探索越充分。
- 同步的平均梯度方差比单 worker 低 $\sqrt{N}$ 倍（$N$ 为 worker 数），因此可以使用更大的学习率。

## 直观理解

A2C 就像"多个实习生 + 一个导师"的团队协作：

- **多个实习生（并行 workers）**：每个实习生独立地在不同环境中工作（例如在不同副本中玩同一款游戏）。他们各自收集 $T$ 步经验。
- **统一汇报道（同步更新）**：每 $T$ 步后，所有实习生向导师汇报他们的经验（梯度）。导师汇总所有经验后，更新指导手册（全局网络参数）。
- **所有实习生拿到同一版手册**：导师更新后，把新版手册发给所有实习生，大家按照统一的新策略继续收集数据。

**为什么多 worker 有帮助？**

想象你教 8 个人同时玩《超级马里奥》。每个人在不同的关卡副本中玩了 5 分钟：
- 小 A 遇到了水关卡（$V(s)$ 低）
- 小 B 遇到了金币关（$V(s)$ 高）
- 小 C 发现了隐藏通道...

如果只让一个人玩 40 分钟，他可能一直在同一关卡徘徊。但 8 个人每人玩 5 分钟，覆盖了更多样的经验。A2C 的同步平均梯度相当于"8 个人的经验平均"，方差更小，更新方向更准确。

**N 步优势的直觉**：只看一步（TD）太短视，看完全程（MC）太漫长。N 步是折中——看未来 $N$ 步的实际奖励，再加上后续价值的估计。就像"观测 5 步后再做判断"，比 1 步更稳，比全程更快。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym
from multiprocessing import Pipe, Process

class A2CNetwork(nn.Module):
    """A2C 共享网络"""
    def __init__(self, state_dim, action_dim):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(state_dim, 128), nn.ReLU(),
            nn.Linear(128, 128), nn.ReLU(),
        )
        self.actor = nn.Linear(128, action_dim)
        self.critic = nn.Linear(128, 1)
    
    def forward(self, x):
        features = self.fc(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value

class A2CAgent:
    """同步 A2C 训练器"""
    def __init__(self, state_dim, action_dim, n_steps=5, gamma=0.99, 
                 lr=0.001, entropy_beta=0.01, device='cpu'):
        self.net = A2CNetwork(state_dim, action_dim).to(device)
        self.optimizer = optim.Adam(self.net.parameters(), lr=lr)
        self.n_steps = n_steps
        self.gamma = gamma
        self.entropy_beta = entropy_beta
        self.device = device
    
    def compute_advantages(self, rewards, values, dones, last_value):
        """计算 N 步优势"""
        advantages = []
        advantage = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                advantage = 0  # 终止状态后的优势重置
            td_error = rewards[t] + self.gamma * last_value * (1 - dones[t]) - values[t]
            advantage = td_error + self.gamma * advantage
            advantages.insert(0, advantage)
            last_value = values[t]
        return advantages
    
    def update(self, trajectories):
        """用多个 worker 的轨迹更新"""
        all_states, all_actions, all_advantages, all_returns = [], [], [], []
        
        for traj in trajectories:
            states, actions, rewards, dones = traj
            states_t = torch.FloatTensor(np.array(states)).to(self.device)
            
            with torch.no_grad():
                _, values = self.net(states_t)
                values = values.squeeze().cpu().numpy()
                _, last_value = self.net(
                    torch.FloatTensor(states[-1:]).to(self.device))
                last_value = last_value.item()
            
            advantages = self.compute_advantages(rewards, values, dones, last_value)
            returns = [adv + val for adv, val in zip(advantages, values)]
            
            all_states.extend(states)
            all_actions.extend(actions)
            all_advantages.extend(advantages)
            all_returns.extend(returns)
        
        # 转为 Tensor
        states_t = torch.FloatTensor(np.array(all_states)).to(self.device)
        actions_t = torch.LongTensor(all_actions).to(self.device)
        advantages_t = torch.FloatTensor(all_advantages).to(self.device)
        returns_t = torch.FloatTensor(all_returns).to(self.device)
        
        # 标准化优势（减小方差）
        advantages_t = (advantages_t - advantages_t.mean()) / (advantages_t.std() + 1e-8)
        
        # 前向传播
        logits, values = self.net(states_t)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        
        # Actor loss (policy gradient)
        log_probs = dist.log_prob(actions_t)
        actor_loss = -(log_probs * advantages_t).mean()
        
        # Critic loss (value function)
        critic_loss = F.mse_loss(values.squeeze(), returns_t)
        
        # Entropy bonus
        entropy = dist.entropy().mean()
        entropy_loss = -self.entropy_beta * entropy
        
        total_loss = actor_loss + critic_loss + entropy_loss
        
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()
        
        return {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'entropy': entropy.item()
        }

# 单 worker 训练示例（展示 A2C 核心逻辑）
def train_a2c_single(env_name="CartPole-v1", max_steps=10000, n_steps=5):
    env = gym.make(env_name)
    agent = A2CAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n,
        n_steps=n_steps
    )
    
    state, info = env.reset()
    episode_reward = 0
    episode_rewards = []
    
    states, actions, rewards, dones = [], [], [], []
    total_steps = 0
    
    while total_steps < max_steps:
        # 采集 N 步数据
        for _ in range(n_steps):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            logits, _ = agent.net(state_t)
            probs = F.softmax(logits, dim=-1)
            action = torch.multinomial(probs, 1).item()
            
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            
            states.append(state)
            actions.append(action)
            rewards.append(reward)
            dones.append(done)
            
            episode_reward += reward
            state = next_state
            total_steps += 1
            
            if done:
                episode_rewards.append(episode_reward)
                episode_reward = 0
                state, info = env.reset()
        
        # 每 N 步更新一次
        if len(states) >= n_steps:
            losses = agent.update([(states, actions, rewards, dones)])
            states, actions, rewards, dones = [], [], [], []
            
            if total_steps % 2000 == 0:
                avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0
                print(f"Steps {total_steps}: 平均奖励={avg:.1f}, "
                      f"熵={losses['entropy']:.3f}")
    
    env.close()
    return episode_rewards

print("A2C 同步训练框架 - 准备就绪")
print("训练中... (取消注释以实际运行)")
# train_a2c_single(max_steps=5000)
```

## 深度学习关联

1. **并行训练与批归一化**：A2C 的多 worker 同步采集天然产生多样化的 batch 数据，类似于监督学习中大数据集上的 mini-batch SGD。这使 A2C 可以利用 GPU 批处理加速——将多个 worker 的状态拼接成一个 batch 进行前向传播，大幅提高吞吐量。
2. **Entropy Regularization 的作用**：熵正则化在深度学习中类似于权重衰减或标签平滑——通过惩罚"过于自信"的预测来防止过拟合。在 DRL 中，它防止策略过早坍缩到次优的确定性策略，确保了持续的探索。
3. **从 A2C 到 IMPALA**：A2C 的同步更新保证了数据与策略的一致性（on-policy），但约束了吞吐量。IMPALA (Importance Weighted Actor-Learner Architecture) 通过引入 V-trace 校正，解耦了 actor 和 learner，允许异步、高吞吐量的经验收集，同时保持 on-policy 的理论保证。
