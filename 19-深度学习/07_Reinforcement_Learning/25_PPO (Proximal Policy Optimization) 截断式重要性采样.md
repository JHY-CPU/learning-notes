# 25_PPO (Proximal Policy Optimization) 截断式重要性采样

## 核心概念

- **PPO (Proximal Policy Optimization)**：2017 年 Schulman 等人提出的策略梯度算法，目前最主流、最稳定的深度强化学习算法之一。核心思想是在每次更新中限制策略变化幅度，防止一步更新过大导致崩溃。
- **裁剪替代目标 (Clipped Surrogate Objective)**：$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$。限制重要性权重 $r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}$ 在一定范围内（$[1-\epsilon, 1+\epsilon]$）。
- **重要性采样 (Importance Sampling)**：PPO 是 off-policy 风格的 on-policy 方法——使用旧策略采集数据，但通过重要性权重修正分布差异，实现多次更新利用同一批数据。
- **信任区域 (Trust Region)**：PPO 通过裁剪隐式地约束了策略更新的信任区域，不需要像 TRPO 那样显式计算 KL 散度约束，实现更简单、计算效率更高。
- **PPO 的两种变体**：PPO-Clip（基于裁剪，最常用）和 PPO-Penalty（基于 KL 散度惩罚）。实践中 PPO-Clip 更受欢迎，因为超参数 $\epsilon$ 的调节更加直观。
- **PPO 的样本效率**：相比 A2C 等 on-policy 方法，PPO 可以多次（通常 3-10 个 epoch）使用同一批数据，大幅提高了样本效率。

## 数学推导

$$
\text{重要性采样权重: } r_t(\theta) = \frac{\pi_\theta(a_t|s_t)}{\pi_{\theta_{\text{old}}}(a_t|s_t)}
$$

$$
\text{PPO-Clip 目标函数: }
$$

$$
L^{CLIP}(\theta) = \mathbb{E}_t \left[ \min \left( r_t(\theta) A_t, \; \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t \right) \right]
$$

$$
\text{PPO 总损失: } L^{PPO}(\theta) = \mathbb{E}_t \left[ L^{CLIP}(\theta) - c_1 L^{VF}(\theta) + c_2 S[\pi_\theta](s_t) \right]
$$

$$
\text{其中：}
$$
$$
L^{VF}(\theta) = (V_\theta(s_t) - V_t^{\text{target}})^2 \quad \text{(价值函数损失)}
$$
$$
S[\pi_\theta](s_t) \quad \text{(策略熵正则项)}
$$

**裁剪机制的作用原理**：
- 当 $A_t > 0$（好动作）：鼓励提高 $\pi_\theta(a_t|s_t)$，但限制 $r_t(\theta) \le 1+\epsilon$，防止过度高估。
- 当 $A_t < 0$（坏动作）：鼓励降低 $\pi_\theta(a_t|s_t)$，但限制 $r_t(\theta) \ge 1-\epsilon$，防止过度规避。
- $\epsilon$ 通常取 0.1 或 0.2，控制信任区域的大小。

## 直观理解

PPO 就像"稳扎稳打"的学习策略，与 TRPO 的"安全第一"对比鲜明：

想象你在学习自由泳时调整姿势：

**策略梯度（无约束）**：教练说"你的打水太差了，要大幅改进！"——你拼命改变打水方式，结果游得更慢了（更新过大，策略崩溃）。

**TRPO**：教练说"改进你的打水，但任何新动作最多只能偏离旧动作 10%"。这很安全，但需要复杂的动作分析（计算 KL 散度和共轭梯度）。

**PPO-Clip**：教练说"你尽管改动作，但我会盯着你的改变幅度。如果改得太多我就帮你刹住。"——你尝试各种改进，PPO 自动确保任何改变不会超过 $1 \pm \epsilon$ 的范围。

**为什么裁剪有效？**

假设你现在行为（旧策略）有 50% 概率选择"大幅度打水"。教练观察后觉得它不好（$A_t < 0$）。如果更新后"大幅度打水"的概率降到了 10%——这是 5 倍的变化（$r = 0.1/0.5 = 0.2$），太激进了！

但 PPO 裁掉：$clip(0.2, 0.8, 1.2) = 0.8$。然后用裁剪后的值乘以负优势做梯度——相当于说"就降到这里，再多就不安全了"。

这好比安装了一个保险杠：你可以在安全范围内自由探索，但不会因为一步踏空而摔下悬崖。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import gymnasium as gym

class PPOAgent:
    """PPO-Clip 实现"""
    def __init__(self, state_dim, action_dim, lr=0.0003, gamma=0.99, 
                 clip_epsilon=0.2, value_coef=0.5, entropy_coef=0.01,
                 update_epochs=4, batch_size=64):
        self.policy = nn.Sequential(
            nn.Linear(state_dim, 64), nn.Tanh(),
            nn.Linear(64, 64), nn.Tanh(),
        )
        self.actor_head = nn.Linear(64, action_dim)
        self.critic_head = nn.Linear(64, 1)
        self.optimizer = optim.Adam(
            list(self.policy.parameters()) + 
            list(self.actor_head.parameters()) + 
            list(self.critic_head.parameters()), lr=lr
        )
        self.gamma = gamma
        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.update_epochs = update_epochs
        self.batch_size = batch_size
    
    def forward(self, x):
        features = self.policy(x)
        logits = self.actor_head(features)
        value = self.critic_head(features)
        return logits, value
    
    def get_action_and_value(self, state):
        logits, value = self.forward(state)
        probs = F.softmax(logits, dim=-1)
        dist = torch.distributions.Categorical(probs)
        action = dist.sample()
        log_prob = dist.log_prob(action)
        return action, log_prob, value
    
    def compute_gae(self, rewards, values, dones, last_value):
        """计算广义优势估计 (GAE)"""
        advantages = []
        gae = 0
        for t in reversed(range(len(rewards))):
            if dones[t]:
                delta = rewards[t] - values[t]
                gae = delta
            else:
                next_value = values[t+1] if t < len(rewards)-1 else last_value
                delta = rewards[t] + self.gamma * next_value - values[t]
                gae = delta + self.gamma * 0.95 * gae
            advantages.insert(0, gae)
        returns = [adv + val for adv, val in zip(advantages, values)]
        return advantages, returns
    
    def update(self, states, actions, old_log_probs, advantages, returns):
        """PPO 核心更新"""
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        old_log_probs = torch.FloatTensor(old_log_probs)
        advantages = torch.FloatTensor(advantages)
        returns = torch.FloatTensor(returns)
        
        # 标准化优势
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        
        for _ in range(self.update_epochs):
            # Mini-batch 训练
            indices = np.arange(len(states))
            np.random.shuffle(indices)
            
            for start in range(0, len(states), self.batch_size):
                idx = indices[start:start + self.batch_size]
                
                logits, values = self.forward(states[idx])
                probs = F.softmax(logits, dim=-1)
                dist = torch.distributions.Categorical(probs)
                
                # 新 log prob
                new_log_probs = dist.log_prob(actions[idx])
                
                # 重要性采样比率
                ratio = torch.exp(new_log_probs - old_log_probs[idx])
                
                # 裁剪目标
                surr1 = ratio * advantages[idx]
                surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 
                                    1 + self.clip_epsilon) * advantages[idx]
                actor_loss = -torch.min(surr1, surr2).mean()
                
                # 价值损失
                critic_loss = F.mse_loss(values.squeeze(), returns[idx])
                
                # 熵损失
                entropy = dist.entropy().mean()
                
                total_loss = actor_loss + self.value_coef * critic_loss - self.entropy_coef * entropy
                
                self.optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(self.policy.parameters()) + 
                    list(self.actor_head.parameters()) + 
                    list(self.critic_head.parameters()), 0.5
                )
                self.optimizer.step()

# 训练循环
def train_ppo(env_name="CartPole-v1", total_timesteps=100000):
    env = gym.make(env_name)
    agent = PPOAgent(
        state_dim=env.observation_space.shape[0],
        action_dim=env.action_space.n
    )
    
    state, info = env.reset()
    episode_rewards = []
    episode_reward = 0
    timestep = 0
    
    states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    
    while timestep < total_timesteps:
        # 采集一步
        state_t = torch.FloatTensor(state).unsqueeze(0)
        action, log_prob, value = agent.get_action_and_value(state_t)
        
        next_state, reward, terminated, truncated, _ = env.step(action.item())
        done = terminated or truncated
        episode_reward += reward
        
        states.append(state)
        actions.append(action.item())
        rewards.append(reward)
        dones.append(done)
        log_probs.append(log_prob.item())
        values.append(value.item())
        
        state = next_state
        timestep += 1
        
        if done:
            episode_rewards.append(episode_reward)
            episode_reward = 0
            state, info = env.reset()
        
        # 每 2048 步更新一次 PPO
        if len(states) >= 2048:
            with torch.no_grad():
                _, last_value = agent.get_action_and_value(
                    torch.FloatTensor(state).unsqueeze(0))
            
            advantages, returns = agent.compute_gae(
                rewards, values, dones, last_value.item())
            agent.update(states, actions, log_probs, advantages, returns)
            
            avg = np.mean(episode_rewards[-10:]) if episode_rewards else 0
            print(f"Timestep {timestep}: 平均奖励={avg:.1f}")
            
            states, actions, rewards, dones, log_probs, values = [], [], [], [], [], []
    
    env.close()
    return episode_rewards

print("PPO-Clip 实现 - 准备就绪")
print("（PPO 是当前工业界最常用的 DRL 算法）")
# train_ppo(total_timesteps=50000)
```

## 深度学习关联

- **裁剪机制与梯度裁剪**：PPO 的裁剪策略与深度学习中常用的梯度裁剪（gradient clipping）思想类似，都是通过"硬限制"来防止训练不稳定。不同之处在于 PPO 裁剪的是损失函数中的重要性权重，而不是梯度本身——这是一种更精细的稳定化手段。
- **PPO 在大语言模型微调中的应用**：PPO 是 RLHF（基于人类反馈的强化学习）中标准使用的算法。在 LLM 微调中，Actor 是被训练的语言模型，Critic 是一个独立的奖励/价值模型，KL 散度约束则被用来防止语言模型偏离预训练分布过远（防止"模式坍塌"）。
- **On-policy 与 Off-policy 的融合**：PPO 的"用旧数据多次更新"模糊了 on-policy 和 off-policy 的界限。这种灵活性使得 PPO 成为"通用 DRL 算法"——在机器人控制、游戏 AI、推荐系统、金融交易等广泛领域中，PPO 通常是首选的基准算法。
