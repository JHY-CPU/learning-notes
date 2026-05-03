# 20_Policy Gradient 定理推导

## 核心概念

- **Policy Gradient 定理**：策略梯度方法的核心理论结果，给出了策略参数 $\theta$ 下期望回报 $J(\theta)$ 对策略参数的梯度表达式，不依赖于环境模型。
- **目标函数 $J(\theta)$**：策略 $\pi_\theta$ 的期望回报，通常有三种形式——初始状态价值 $J_1 = V_{\pi_\theta}(s_0)$、平均价值 $J_{avV} = \sum_s d_{\pi_\theta}(s) V_{\pi_\theta}(s)$、或每步平均奖励 $J_{avR} = \sum_s d_{\pi_\theta}(s) \sum_a \pi_\theta(a|s) R(s,a)$。
- **策略梯度定理表达式**：$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} [\nabla_\theta \log \pi_\theta(a|s) Q_{\pi_\theta}(s, a)]$，梯度的方向由"动作 $a$ 的对数概率梯度"乘以"该动作的 Q 值"决定。
- **REINFORCE 算法**：使用完整的 episode 回报 $G_t$ 作为 $Q_{\pi_\theta}(s, a)$ 的样本估计，是最简单的策略梯度实现。
- **与价值方法的区别**：策略梯度直接优化策略参数（在策略空间中搜索），而非通过价值函数间接确定策略。这使它能自然地处理连续动作空间和随机策略。
- **策略梯度的方差问题**：策略梯度估计通常方差很大，因为 $Q_{\pi_\theta}(s, a)$ 的采样估计不够稳定。引入 baseline $b(s)$ 减小方差（如 $A(s,a) = Q(s,a) - V(s)$）是标准做法。

## 数学推导

$$
\text{目标函数（起始状态形式）: } J(\theta) = V_{\pi_\theta}(s_0) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^\infty \gamma^t R_{t+1}\right]
$$

$$
\text{策略梯度定理: }
$$

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, Q_{\pi_\theta}(s, a) \right]
$$

$$
\text{带 baseline 的策略梯度: }
$$

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, (Q_{\pi_\theta}(s, a) - b(s)) \right]
$$

$$
\text{当 } b(s) = V_{\pi_\theta}(s) \text{ 时，优势函数形式: }
$$

$$
\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta} \left[ \nabla_\theta \log \pi_\theta(a|s) \, A_{\pi_\theta}(s, a) \right]
$$

**推导关键步骤**：
- 展开 $J(\theta) = \sum_{s \in \mathcal{S}} d_{\pi_\theta}(s) \sum_{a \in \mathcal{A}} \pi_\theta(a|s) R(s, a)$，这里 $d_{\pi_\theta}$ 是稳态分布。
- 两边取梯度，利用对数技巧：$\nabla \pi_\theta = \pi_\theta \nabla \log \pi_\theta$。
- 递归展开 $\nabla d_{\pi_\theta}$ 项，最终所有项合并为期望形式 $\mathbb{E}[\nabla \log \pi \cdot Q]$。
- 加入 baseline $b(s)$ 不改变梯度的期望值，因为 $\mathbb{E}[\nabla \log \pi \cdot b(s)] = b(s) \cdot \mathbb{E}[\nabla \log \pi] = b(s) \cdot 0 = 0$。

## 直观理解

策略梯度定理的直观理解可以浓缩为一句话：

**"好的动作要让它更可能发生，坏的动作要让它更不可能发生。"**

具体来说：
- 你让智能体在状态 $s$ 执行动作 $a$，得到了 Q 值（该动作的长远收益）。
- 如果 Q 值为正（好动作），希望 $\pi_\theta(a|s)$ 变大，所以梯度方向是"增加该动作的概率"。
- 如果 Q 值为负（坏动作），希望 $\pi_\theta(a|s)$ 变小，所以梯度方向是"减少该动作的概率"。
- 调整幅度与 Q 值大小成正比——"非常好"的动作多鼓励，"一般好"的动作少鼓励。

**类比：厨师改进菜谱**
- 策略 $\pi_\theta$ 就是菜谱，$\theta$ 是各种调料的用量。
- 你按照菜谱做菜（采样），客人品尝后打分（Q 值）。
- 如果客人说"这个菜真好吃！"（Q > 0），你就多加"这次加多的那个调料"。
- 如果客人说"太咸了！"（Q < 0），你就少放盐。
- 经过多次"做菜->品尝->调整"，菜谱越来越好。

**baseline 的直觉**：去掉基准分（baseline）。如果所有菜都有 5 分基础分（用餐体验本身），那么"5 分的菜"并不突出，只有"+"或"-"的偏差才代表这道菜（动作）的真正优势。这相当于 $A(s,a) = Q(s,a) - V(s)$。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import gymnasium as gym

# 策略网络
class PolicyNetwork(nn.Module):
    """输出动作概率分布的策略网络"""
    def __init__(self, state_dim, action_dim, hidden=128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim)
        )
    
    def forward(self, x):
        logits = self.net(x)
        return F.softmax(logits, dim=-1), logits
    
    def get_log_prob(self, state, action):
        probs, logits = self.forward(state)
        log_probs = F.log_softmax(logits, dim=-1)
        return log_probs.gather(1, action.unsqueeze(-1)).squeeze(-1)

# REINFORCE 算法（策略梯度的基本实现）
def reinforce(env_name="CartPole-v1", num_episodes=1000, gamma=0.99, lr=0.001):
    env = gym.make(env_name)
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n
    
    policy = PolicyNetwork(state_dim, action_dim)
    optimizer = torch.optim.Adam(policy.parameters(), lr=lr)
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        # 采集 episode
        states, actions, rewards = [], [], []
        state, info = env.reset()
        terminated = truncated = False
        
        while not (terminated or truncated):
            state_t = torch.FloatTensor(state).unsqueeze(0)
            probs, _ = policy(state_t)
            action = torch.multinomial(probs, 1).item()
            
            next_state, reward, terminated, truncated, info = env.step(action)
            
            states.append(state_t)
            actions.append(action)
            rewards.append(reward)
            state = next_state
        
        # 计算折扣回报
        G = 0
        returns = []
        for r in reversed(rewards):
            G = r + gamma * G
            returns.insert(0, G)
        returns = torch.FloatTensor(returns)
        
        # 标准化回报（减小方差）
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        # 计算策略梯度损失
        total_loss = 0
        for t, (s, a, G_t) in enumerate(zip(states, actions, returns)):
            log_prob = policy.get_log_prob(s, torch.tensor([a]))
            # 策略梯度: grad = log_prob * G_t
            loss = -log_prob * G_t
            total_loss += loss
        
        # 更新策略
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        
        episode_rewards.append(sum(rewards))
        
        if (ep + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            print(f"Episode {ep+1}: 平均奖励 = {avg_reward:.2f}")
    
    env.close()
    return episode_rewards

# 运行 REINFORCE
print("运行 REINFORCE (策略梯度基础算法)...")
# rewards = reinforce(num_episodes=500)
# print(f"最终平均奖励: {np.mean(rewards[-100:]):.2f}")
print("策略梯度定理已完成推导，REINFORCE 实现以上")
```

## 深度学习关联

- **策略梯度 + 深度网络**：策略梯度定理完美兼容深度神经网络——将策略 $\pi_\theta$ 参数化为神经网络，梯度 $\nabla_\theta \log \pi_\theta(a|s)$ 可以通过自动微分自动计算。这极大简化了实现，使得端到端策略学习成为可能。
- **Actor-Critic 架构**：策略梯度是 Actor（策略网络），价值函数是 Critic（价值网络）。Critic 提供低方差的 Q 值或优势估计，Actor 使用策略梯度更新。这种结合是几乎所有现代 DRL 方法（A2C、PPO、SAC）的基础架构。
- **Reparameterization Trick**：在连续动作空间中，策略梯度可以通过重参数化技巧进一步降低方差。SAC 和 TD3 中使用的"先采样噪声，再通过确定性函数变换"的技巧，将策略梯度转化为可以反向传播的确定性梯度，这是策略梯度定理在连续控制中的重要演进。
