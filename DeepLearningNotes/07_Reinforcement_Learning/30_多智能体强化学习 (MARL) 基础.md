# 30_多智能体强化学习 (MARL) 基础

## 核心概念

- **多智能体强化学习 (Multi-Agent Reinforcement Learning, MARL)**：多个智能体在共享环境中同时学习与交互的强化学习范式。每个智能体有自己的策略、观察和奖励，但环境动态受所有智能体联合动作的影响。
- **非平稳性 (Non-stationarity)**：在 MARL 中，从单个智能体的视角看，环境动态是不稳定的——因为其他智能体的策略也在变化。这违反了 MDP 的马尔可夫性质，是 MARL 的核心挑战。
- **部分可观测性**：通常每个智能体只能观测到全局状态的一部分（局部观测），无法获取其他智能体的完整信息。这使问题建模为 Dec-POMDP（Decentralized Partially Observable MDP）。
- **完全合作 vs 完全竞争 vs 混合场景**：
  - **完全合作**：所有智能体共享同一奖励（如多机器人搬箱子）。目标是最大化团队总回报。
  - **完全竞争**：零和博弈，一方的收益等于另一方的损失（如围棋、扑克）。
  - **混合场景**：既有合作也有竞争（如团队体育比赛、经济市场博弈）。
- **集中式训练-分散式执行 (CTDE)**：MARL 中最流行的范式。训练时可以访问所有智能体的信息（集中式 Critic），但执行时每个智能体仅根据局部观测独立决策（分散式 Actor）。
- **多智能体策略梯度 (MADDPG)**：CTDE 的代表算法。每个智能体有一个 Actor（基于局部观测），Critic 在训练时访问所有智能体的观测和动作来评估联合 Q 值。

## 数学推导

$$
\text{Multi-Agent MDP (Markov Game): } (\mathcal{S}, \{\mathcal{A}^i\}_{i=1}^N, P, \{R^i\}_{i=1}^N, \gamma)
$$

$$
N \text{ 个智能体，每个有独立动作空间 } \mathcal{A}^i \text{ 和奖励函数 } R^i
$$

$$
\text{联合策略: } \pi(\mathbf{a} \mid s) = \prod_{i=1}^N \pi^i(a^i \mid \tau^i)
$$

$$
\text{其中 } \tau^i \text{ 是智能体 } i \text{ 的观测历史}
$$

$$
\text{CTDE 框架下的集中式 Critic（MADDPG）: }
$$

$$
\text{Critic: } Q_i^\pi(\mathbf{x}, a^1, ..., a^N) = \mathbb{E}[R_t^i \mid s_t = \mathbf{x}, a_t^1, ..., a_t^N]
$$

$$
\text{Actor: } \nabla_{\theta_i} J(\theta_i) = \mathbb{E}_{\mathbf{x}, a^i \sim \pi_{\theta_i}} \left[ \nabla_{\theta_i} \log \pi_{\theta_i}(a^i|\tau^i) \, Q_i^\pi(\mathbf{x}, a^1, ..., a^N) \right]
$$

$$
\text{其中 } \mathbf{x} = (s, o^1, ..., o^N, a^1, ..., a^N) \text{（所有状态/观测/动作的拼接）}
$$

**推导说明**：
- Dec-POMDP 将部分可观测性建模引入多智能体场景，是最一般的 MARL 理论模型。
- CTDE 巧妙地解决了非平稳性：训练时 Critic 能观察到所有智能体的动作，使环境对单个 Agent 而言是平稳的；执行时 Actor 仅依赖局部观测，满足部署时的通信约束。
- MADDPG（Multi-Agent DDPG）是 CTDE 在连续控制中的实现：每个 Agent 有独立的 Actor-Critic 对，Critic 在训练时使用所有 Agent 的观测和动作。

## 直观理解

MARL 就像"一个团队里每个人都在学习"：

**单智能体 RL**：一个人学开车。路上是固定的规则和障碍物（环境静态），他只需要学自己的操作。

**多智能体 RL**：五个人同时学团队篮球。每一个人不仅要学自己的技术（运球、投篮），还要预测和适应其他四个人的行为。今天队友 A 决定多传球，明天他可能决定多投篮——环境在持续变化，因为队友也在学习。

**非平稳性**的具体表现：
- 你学会了"传球给 A，他在三分线总是接球投篮"。
- 但 A 在同时学习，他今天改变了策略——开始突破上篮而不是投篮了！
- 你之前的策略就不灵了。你需要重新适应。

**CTDE 的直觉**：
- **训练时（集中式）**：教练站在高处，能看到全场所有人——知道 A 要突破，B 在跑位，C 在防守。教练用全局信息指导每个人。
- **执行时（分散式）**：比赛中，每个球员只看到自己眼前的场景（局部观测），根据自己掌握的技能做决策。他们不需要实时和教练交流（通信约束）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from collections import deque
import random

# 简单的双智能体合作环境（简化）
class SimpleCoopEnv:
    """两个智能体协作抓取物体的简化环境"""
    def __init__(self):
        self.agent_pos = [[0.0, 0.0], [0.0, 0.0]]
        self.target_pos = [1.0, 1.0]
        self.done = False
    
    def reset(self):
        self.agent_pos = [[0.0, 0.0], [0.0, 0.0]]
        self.target_pos = [1.0, 1.0]
        self.done = False
        return self.get_obs()
    
    def get_obs(self):
        # 每个智能体看到自己的位置和物体的相对位置
        obs0 = self.agent_pos[0] + [
            self.target_pos[0] - self.agent_pos[0][0],
            self.target_pos[1] - self.agent_pos[0][1]
        ]
        obs1 = self.agent_pos[1] + [
            self.target_pos[0] - self.agent_pos[1][0],
            self.target_pos[1] - self.agent_pos[1][1]
        ]
        return [np.array(obs0, dtype=np.float32), 
                np.array(obs1, dtype=np.float32)]
    
    def step(self, actions):
        # 执行动作
        self.agent_pos[0][0] += actions[0][0] * 0.1
        self.agent_pos[0][1] += actions[0][1] * 0.1
        self.agent_pos[1][0] += actions[1][0] * 0.1
        self.agent_pos[1][1] += actions[1][1] * 0.1
        
        # 计算距离（两个智能体到目标的平均距离）
        dist0 = np.sqrt((self.agent_pos[0][0] - self.target_pos[0])**2 + 
                        (self.agent_pos[0][1] - self.target_pos[1])**2)
        dist1 = np.sqrt((self.agent_pos[1][0] - self.target_pos[0])**2 + 
                        (self.agent_pos[1][1] - self.target_pos[1])**2)
        
        avg_dist = (dist0 + dist1) / 2
        reward = -avg_dist  # 距离越小奖励越大
        
        done = avg_dist < 0.1
        return self.get_obs(), [reward, reward], done

# MADDPG 简化实现
class MADDPGAgent:
    """单个智能体的 Actor-Critic（CTDE 框架）"""
    def __init__(self, obs_dim, action_dim, n_agents, agent_id):
        self.agent_id = agent_id
        # Actor: 仅基于局部观测
        self.actor = nn.Sequential(
            nn.Linear(obs_dim, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, action_dim), nn.Tanh()
        )
        # Critic: 基于所有智能体的观测和动作
        total_obs = obs_dim * n_agents
        total_act = action_dim * n_agents
        self.critic = nn.Sequential(
            nn.Linear(total_obs + total_act, 64), nn.ReLU(),
            nn.Linear(64, 64), nn.ReLU(),
            nn.Linear(64, 1)
        )
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=0.001)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=0.001)
    
    def get_action(self, obs, noise=0.1):
        obs_t = torch.FloatTensor(obs).unsqueeze(0)
        action = self.actor(obs_t).detach().numpy()[0]
        action += np.random.randn(len(action)) * noise
        return np.clip(action, -1, 1)

class MADDPGTrainer:
    """MADDPG 训练器"""
    def __init__(self, n_agents, obs_dim, action_dim):
        self.agents = [MADDPGAgent(obs_dim, action_dim, n_agents, i) 
                      for i in range(n_agents)]
        self.n_agents = n_agents
        self.obs_dim = obs_dim
        self.action_dim = action_dim
    
    def update(self, experiences):
        """所有智能体联合更新"""
        # 从 replay buffer 采样（此处简化）
        states, actions, rewards, next_states, dones = experiences
        
        for i, agent in enumerate(self.agents):
            # 更新 Critic
            with torch.no_grad():
                next_actions = []
                for j, a_agent in enumerate(self.agents):
                    obs_t = torch.FloatTensor(next_states[j])
                    na = a_agent.actor(obs_t)
                    next_actions.append(na)
                next_actions = torch.cat(next_actions, dim=-1)
                
                all_next_obs = torch.cat(
                    [torch.FloatTensor(ns) for ns in next_states], dim=-1)
                target_q = torch.FloatTensor(rewards[i]).unsqueeze(1) + \
                    0.99 * agent.critic(torch.cat([all_next_obs, next_actions], dim=-1))
            
            all_obs = torch.cat(
                [torch.FloatTensor(s) for s in states], dim=-1)
            all_acts = torch.cat(
                [torch.FloatTensor(a) for a in actions], dim=-1)
            current_q = agent.critic(torch.cat([all_obs, all_acts], dim=-1))
            
            critic_loss = F.mse_loss(current_q, target_q)
            agent.critic_optimizer.zero_grad()
            critic_loss.backward()
            agent.critic_optimizer.step()
            
            # 更新 Actor（仅使用自己的局部观测）
            agent.actor_optimizer.zero_grad()
            # 保持其他智能体动作不变
            obs_t = torch.FloatTensor(states[agent.agent_id])
            new_actions = actions.copy()
            new_actions[agent.agent_id] = agent.actor(obs_t).detach().numpy()
            actor_loss = -agent.critic(
                torch.cat([all_obs, torch.cat([torch.FloatTensor(a) for a in new_actions], dim=-1)], dim=-1)
            ).mean()
            actor_loss.backward()
            agent.actor_optimizer.step()

# 训练示意
def train_marl():
    env = SimpleCoopEnv()
    trainer = MADDPGTrainer(n_agents=2, obs_dim=4, action_dim=2)
    
    for episode in range(100):
        obs = env.reset()
        done = False
        episode_reward = 0
        
        while not done:
            actions = []
            for i, agent in enumerate(trainer.agents):
                action = agent.get_action(obs[i])
                actions.append(action)
            
            next_obs, rewards, done = env.step(actions)
            episode_reward += rewards[0]
            
            # 存储经验（简化）
            obs = next_obs
        
        if (episode + 1) % 20 == 0:
            print(f"Episode {episode+1}: 团队奖励 = {episode_reward:.2f}")

print("MARL (多智能体强化学习) 基础 - 准备就绪")
print("训练中... (取消注释以实际运行)")
# train_marl()
```

## 深度学习关联

1. **CTDE 与多任务学习**：CTDE 中集中式 Critic 相当于一个"多任务价值评估器"——它接收所有智能体的信息并输出每个智能体的 Q 值。这与多任务学习中的共享表示类似，Critic 必须学会从联合信息中为不同智能体提取个性化的价值信号。
2. **通信学习与图神经网络**：在 MARL 中，智能体间的通信协议可以通过学习得到。通信学习（emerging communication）通常使用图神经网络（GNN）：智能体作为图节点，通信信道作为边，GNN 的消息传递机制天然适合多智能体间的信息交换和聚合。
3. **MARL 在博弈论和经济学中的应用**：MARL 不仅用于机器人协作，还广泛应用于博弈论中的纳什均衡求解、经济学中的市场模拟、以及社会科学的群体行为建模。深度网络为大规模博弈求解提供了函数近似能力，如 AlphaStar（星际争霸 II）和 OpenAI Five（Dota 2）都是 MARL + 深度学习的标志性成果。
