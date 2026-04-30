# 15_深度 Q 网络 (DQN) 架构详解

## 核心概念

- **DQN (Deep Q-Network)**：2013 年 Mnih 等人提出的里程碑工作，首次将深度学习与 Q-Learning 成功结合，在 Atari 2600 游戏上达到人类水平。核心创新包括经验回放和目标网络。
- **网络架构**：输入是预处理后的游戏画面（84x84x4 的堆叠灰度帧），经过三层卷积 + 两层全连接层，输出每个离散动作的 Q 值。卷积层提取空间特征，全连接层进行价值整合。
- **损失函数**：$L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$，其中 $\theta$ 是在线网络参数，$\theta^-$ 是目标网络参数。
- **输入预处理**：原始 Atari 画面（210x160x3）需要预处理：灰度化、下采样到 84x84、堆叠最近 4 帧作为输入（以感知运动信息）。
- **训练流程**：交互（$\epsilon$-greedy 探索）-> 存储 transition 到回放缓冲区 -> 从缓冲区随机采样 mini-batch -> 计算 TD 误差 -> 梯度下降更新 Q 网络 -> 定期复制参数到目标网络。
- **性能指标**：通常使用人类归一化分数（Human Normalized Score）来评估 DQN 的性能，衡量智能体相对于随机策略和人类玩家的表现水平。

## 数学推导

$$
\text{DQN 损失函数: } L(\theta) = \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right)^2 \right]
$$

$$
\text{梯度: } \nabla_\theta L(\theta) = -2 \mathbb{E}_{(s,a,r,s') \sim \mathcal{D}} \left[ \left( r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta) \right) \nabla_\theta Q(s, a; \theta) \right]
$$

$$
\text{卷积网络结构: }
$$

$$
84 \times 84 \times 4 \xrightarrow{\text{Conv: 8x8, stride 4, 32 filters}} 20 \times 20 \times 32
$$

$$
\xrightarrow{\text{Conv: 4x4, stride 2, 64 filters}} 9 \times 9 \times 64
$$

$$
\xrightarrow{\text{Conv: 3x3, stride 1, 64 filters}} 7 \times 7 \times 64
$$

$$
\xrightarrow{\text{FC: 512}} \xrightarrow{\text{FC: |A| (动作数)}}
$$

**推导说明**：
- 损失函数是 TD 误差的均方误差（MSE），其中 $r + \gamma \max_{a'} Q(s', a'; \theta^-)$ 是目标值（使用目标网络参数计算，不参与梯度计算）。
- 梯度公式中目标值 $r + \gamma \max Q$ 被视为常数（不对 $\theta$ 求导），这是 DQN 稳定训练的关键技巧。
- 网络输出层是线性层（无激活函数），输出维度等于动作数，因为 Q 值可以是任意实数。

## 直观理解

DQN 就像一个"边玩游戏边总结"的超级玩家：

想象你教一个新手打《吃豆人》（Pac-Man）：
1. **看屏幕（输入）**：新手看着游戏画面（84x84像素），注意到豆子、幽灵和自己的位置。
2. **直觉反应（Q 网络）**：大脑自然会产生直觉——"往左走能吃豆子"（Q 值高），"往右走会撞幽灵"（Q 值低）。
3. **实际体验（交互）**：新手真的操作了向左走，发现确实吃了豆子（正奖励），幽灵靠近了（恐惧）。
4. **复盘总结（训练）**：新手回想"刚才向左走的感觉——实际结果比预期好，下次要更相信向左走"（TD 误差被反向传播更新网络）。
5. **选择性记忆（经验回放）**：他不仅记得刚才这一步，还随机翻看过去的游戏记录，从中学习。
6. **照镜子（目标网络）**：每隔一段时间，他照照镜子，给自己做个"当前版本备份"，用之前的自己来帮助现在的自己稳定判断。

DQN 的三大创新正是为了解决深度网络做 Q-Learning 的三个问题：
- **灾难性遗忘** -> 经验回放（防止网络被最近的经历"冲昏头脑"）
- **目标不稳定** -> 目标网络（防止网络"追着自己的尾巴跑"）
- **相关性破坏** -> 随机采样（打乱时间相关性，使 mini-batch 近似独立同分布）

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random

# DQN 网络架构
class DQN(nn.Module):
    def __init__(self, n_actions):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, kernel_size=8, stride=4),  # 84x84 -> 20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2),  # 20x20 -> 9x9
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=1),  # 9x9 -> 7x7
            nn.ReLU(),
        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 7 * 7, 512),
            nn.ReLU(),
            nn.Linear(512, n_actions)
        )
    
    def forward(self, x):
        return self.fc(self.conv(x))

# 经验回放缓冲区
class ReplayBuffer:
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (np.array(states), np.array(actions), 
                np.array(rewards, dtype=np.float32),
                np.array(next_states), np.array(dones, dtype=np.float32))
    
    def __len__(self):
        return len(self.buffer)

# 训练循环伪代码
def dqn_train_step(q_net, target_net, optimizer, replay_buffer, batch_size=32, gamma=0.99):
    if len(replay_buffer) < batch_size:
        return
    
    states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
    
    # 转为 Tensor
    states = torch.FloatTensor(states)
    actions = torch.LongTensor(actions)
    rewards = torch.FloatTensor(rewards)
    next_states = torch.FloatTensor(next_states)
    dones = torch.FloatTensor(dones)
    
    # 当前 Q 值
    q_values = q_net(states).gather(1, actions.unsqueeze(1)).squeeze()
    
    # 目标 Q 值（用目标网络）
    with torch.no_grad():
        next_q = target_net(next_states).max(1)[0]
        target_q = rewards + gamma * next_q * (1 - dones)
    
    # 损失
    loss = nn.MSELoss()(q_values, target_q)
    
    # 更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    return loss.item()

# 网络实例化
n_actions = 4  # 以 CartPole 为例（作为示意，实际 DQN 用于 Atari）
q_net = DQN(n_actions)
target_net = DQN(n_actions)
target_net.load_state_dict(q_net.state_dict())
optimizer = optim.Adam(q_net.parameters(), lr=0.00025)

print("DQN 网络结构:")
print(q_net)
print(f"\n参数量: {sum(p.numel() for p in q_net.parameters()):,}")
```

## 深度学习关联

1. **CNN 特征提取**：DQN 中的卷积层与 CV 领域的 CNN 共享相同架构设计原理。区别在于：DQN 使用堆叠帧作为输入以捕获运动信息（类似光流），且训练目标不是分类而是回归 Q 值。
2. **目标网络的"冻结更新"机制**：这在深度学习中是一种独特的训练技巧——维护一个滞后更新的"备份"网络来计算目标值，与 Self-Supervised Learning 中 momentum encoder（如 MoCo、BYOL）有异曲同工之妙。两者都通过慢速更新的网络来提供稳定的训练目标。
3. **从表格到函数近似的泛化**：DQN 展示了深度网络的泛化能力——同一个网络架构、同样的超参数，在 49 个 Atari 游戏上通用，无需针对每个游戏调整。这种"通用架构 + 通用算法"的模式后来成为深度强化学习的标准范式。
