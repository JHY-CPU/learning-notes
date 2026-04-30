# 22_Actor-Critic 框架基础

## 核心概念

- **Actor-Critic (AC)**：将策略梯度（Actor）和价值函数（Critic）结合的统一框架。Actor 负责学习策略 $\pi_\theta(a|s)$，Critic 负责估计价值函数 $V_\phi(s)$ 或 $Q_\phi(s, a)$，两者协同训练。
- **Actor（策略网络）**：使用策略梯度更新，目标是最大化期望回报。更新方向由 Critic 提供的评分指导：$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot A(s, a)]$。
- **Critic（价值网络）**：使用 TD 学习或 MC 方法估计价值函数，为 Actor 提供低方差的优势估计。Critic 通常通过最小化 TD 误差来更新：$L(\phi) = \mathbb{E}[(r + \gamma V_\phi(s') - V_\phi(s))^2]$。
- **AC 的优势**：相比 REINFORCE 的"完全 episode 后才更新"，AC 可以每步或每 N 步即时更新（因为 Critic 提供即时的价值估计作为 baseline），大幅降低了方差，提高了学习效率。
- **偏差-方差权衡**：Actor-Critic 处于 REINFORCE（无偏但高方差）和完全自举方法（有偏但低方差）之间。Critic 的价值估计引入了偏差，但因为减少了回报的随机性而降低了方差。整体上，AC 比纯 PG 收敛更快。
- **One-step Actor-Critic**：最简单的 AC 形式，每一步都更新：$\delta = r + \gamma V(s') - V(s)$，然后 $\theta \leftarrow \theta + \alpha \nabla_\theta \log \pi(a|s) \delta$。

## 数学推导

$$
\text{Actor 更新（策略梯度）: } \nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, A(s, a) \right]
$$

$$
\text{Critic 更新（TD 学习）: } L(\phi) = \mathbb{E} \left[ (r + \gamma V_\phi(s') - V_\phi(s))^2 \right]
$$

$$
\text{优势函数估计（单步）: } A(s, a) = \delta = r + \gamma V_\phi(s') - V_\phi(s)
$$

$$
\text{整体训练循环: }
$$

$$
\begin{aligned}
&\text{观察 } s, \text{从 } \pi_\theta \text{ 采样 } a, \text{执行得到 } r, s' \\
&\delta \leftarrow r + \gamma V_\phi(s') - V_\phi(s) \quad \text{(TD 误差)} \\
&\phi \leftarrow \phi + \alpha_\phi \, \delta \, \nabla_\phi V_\phi(s) \quad \text{(更新 Critic)} \\
&\theta \leftarrow \theta + \alpha_\theta \, \delta \, \nabla_\theta \log \pi_\theta(a|s) \quad \text{(更新 Actor)}
\end{aligned}
$$

**推导说明**：
- $A(s, a) = r + \gamma V(s') - V(s)$ 是优势函数的一步估计，利用了贝尔曼方程：$\mathbb{E}[r + \gamma V(s')] = Q(s,a)$，所以 $A = Q - V$。
- Actor 和 Critic 的更新可以同时进行，也可以交替进行。实践中两者共享特征提取层以提升效率。
- 由于 Critic 也随时间更新，它传递给 Actor 的信号是非平稳的，这可能导致训练不稳定性——这也是 PPO 等算法需要进一步约束策略更新的原因。

## 直观理解

Actor-Critic 就像"运动员 + 教练"的组合：

**Actor（运动员）**：负责执行动作。他不断尝试新的打球方式，目标是赢得更多分数。
**Critic（教练）**：观察运动员的动作，给出即时反馈："刚才那个反手球（动作 A）打得不错，比你的平均水平好 0.3 分（优势为正）！"

**对比纯策略梯度（REINFORCE，无 Critic）**：
- 运动员自行复盘：打完一整局比赛后，回忆每个动作然后决定是否要调整。
- 问题：一局赢了，但分不清哪个具体动作贡献大——可能关键的正手球被漏掉了。

**有了 Critic**：
- 教练每步都给出即时评价："这个反手打得不错，比你平时好！"
- 运动员立即强化这个动作（增加概率）。
- 教练也在学习中——他不断调整自己的评价标准（更新价值网络），评价越来越准。

AC 框架的精髓在于**协同进化**：Actor 越来越会打球（策略提升），Critic 越来越会评价（价值准确），两者互相促进。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

class ActorCritic(nn.Module):
    """共享特征的 Actor-Critic 网络"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
        )
        # Actor head: 输出动作概率
        self.actor = nn.Linear(hidden, action_dim)
        # Critic head: 输出状态价值
        self.critic = nn.Linear(hidden, 1)
    
    def forward(self, x):
        features = self.shared(x)
        logits = self.actor(features)
        value = self.critic(features)
        return logits, value
    
    def get_action_and_value(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value

def train_actor_critic(env_name="CartPole-v1", num_steps=10000):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    model = ActorCritic(state_dim, action_dim)
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    gamma = 0.99
    
    state, info = env.reset()
    episode_rewards = []
    episode_reward = 0
    
    for step in range(num_steps):
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = model.get_action_and_value(state_t)
        
        next_state, reward, terminated, truncated, info = env.step(action.item())
        episode_reward += reward
        done = terminated or truncated
        
        # 计算 TD 误差（优势）
        if done:
            next_value = torch.tensor([0.0])
        else:
            with torch.no_grad():
                _, _, next_value = model.get_action_and_value(
                    torch.FloatTensor(next_state).unsqueeze(0))
            next_value = next_value.squeeze()
        
        # TD 误差 = r + gamma * V(s') - V(s)
        td_error = reward + gamma * next_value - value.squeeze()
        
        # Actor 损失: -log_prob * td_error
        actor_loss = -log_prob * td_error.detach()
        
        # Critic 损失: td_error^2
        critic_loss = td_error.pow(2)
        
        # 总损失
        total_loss = actor_loss + 0.5 * critic_loss
        
        # 更新
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        state = next_state
        
        if done:
            episode_rewards.append(episode_reward)
            state, info = env.reset()
            episode_reward = 0
        
        if (step + 1) % 2000 == 0:
            avg_reward = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Step {step+1}: 最近 10ep 平均奖励={avg_reward:.2f}")
    
    env.close()
    return episode_rewards

print("Actor-Critic 框架实现 - 准备就绪")
print("训练中... (取消注释以实际运行)")
# train_actor_critic(num_steps=5000)
```

## 深度学习关联

1. **共享网络架构**：现代 DRL 中的 Actor-Critic 通常共享底层的特征提取网络（如 CNN 或 ResNet），只在顶层分叉为 Actor 和 Critic 两个 head。这种"共享表示 + 分离输出"的设计在深度学习中广泛使用（如多任务学习），同时 AC 也面临"两个任务可能冲突"的挑战。
2. **从 AC 到现代 DRL 的演进**：A2C（同步并行）和 A3C（异步并行）是 AC 的并行化扩展；PPO 和 SAC 是 AC 的稳定性和效率升级版。所有现代 DRL 算法本质上都是 Actor-Critic 框架的变体。
3. **Actor-Critic 在序列决策中的应用**：AC 框架也被应用于 NLP 中的序列生成（如机器翻译、文本摘要），其中 Actor 是生成模型，Critic 评估生成结果的质量。这种"生成 + 评估"的框架是许多序列决策问题的基础范式。
