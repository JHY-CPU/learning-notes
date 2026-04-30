# 16_DQN 经验回放 (Experience Replay) 机制

## 核心概念

- **经验回放 (Experience Replay)**：智能体将交互产生的 transition $(s, a, r, s')$ 存储在一个回放缓冲区中，训练时从中随机采样 mini-batch 进行学习。这是 DQN 的第一个关键创新。
- **打破数据相关性**：在线 RL 中连续样本高度相关（时间序列的马尔可夫性质），直接使用连续样本训练网络会导致梯度更新方差大、容易发散。随机采样打破了这种时间相关性。
- **数据效率提升**：每个 experience 可以被多次使用（回放），而不是像 on-policy 方法那样用完即弃。这在深度学习中至关重要——因为神经网络训练需要大量数据。
- **回放缓冲区结构**：通常实现为固定大小的循环队列（ring buffer），新数据覆盖旧数据。缓冲区容量一般在 $10^5$ 到 $10^6$ 量级，具体取决于任务复杂度。
- **均匀采样 vs 优先采样**：标准 DQN 使用均匀采样（uniform sampling）。但并非所有 experience 同等重要——优先经验回放（Prioritized Experience Replay）根据 TD 误差大小来调整采样概率，显著提升学习效率。
- **Off-policy 的必要条件**：经验回放只在 off-policy 算法中可行，因为回放数据来自旧策略，而当前更新需要兼容旧数据。SARSA 等 on-policy 方法无法直接使用回放技术。

## 数学推导

$$
\text{Transition: } (s_t, a_t, r_{t+1}, s_{t+1}, d_t) \sim \mathcal{D}
$$

$$
\text{均匀采样损失: } L(\theta) = \frac{1}{|\mathcal{B}|} \sum_{i \in \mathcal{B}} \left( y_i - Q(s_i, a_i; \theta) \right)^2
$$

$$
\text{其中 } y_i = r_{i+1} + \gamma \max_{a'} Q(s_{i+1}, a'; \theta^-) \text{（目标值）}
$$

$$
\text{Prioritized Replay 采样概率: } P(i) = \frac{p_i^\alpha}{\sum_k p_k^\alpha}, \quad p_i = |\delta_i| + \epsilon
$$

$$
\text{重要性采样权重: } w_i = \left( \frac{1}{N} \cdot \frac{1}{P(i)} \right)^\beta
$$

**推导说明**：
- $\mathcal{B}$ 是 mini-batch，$\mathcal{D}$ 是回放缓冲区。均匀采样时每个 transition 的权重相同。
- Prioritized Replay 中，$p_i = |\delta_i| + \epsilon$（TD 误差绝对值 + 小常数），TD 误差大的 transition 被认为"更有学习价值"。
- 重要性采样权重 $w_i$ 用于修正因非均匀采样带来的分布偏差。$\beta$ 从 0.4 逐渐增加到 1。

## 直观理解

经验回放就像"错题本"和"记忆碎片"的结合：

**没有经验回放**（传统 on-policy 方法）：
好比一个健忘的学生，只看刚做完的一道题就马上考试——容易因为偶然的错题而影响整体理解。如果这道题恰好特别难（异常值），他的知识体系就会受到不必要的冲击。

**有经验回放**：
好比一个勤奋的学生，把所有做过的题（无论新旧）都整理到错题本中。每天随机翻看 32 道题复习——这样：
- 不会因为刚做错一道题就过度反应（打破时间相关性）
- 可以反复翻看之前的经典题型（数据重用）
- 复习时看到的题目是打乱顺序的（独立同分布）

**优先经验回放**则更像"重点错题本"——越是做错的题（TD 误差大），出现的频率越高。这比均匀翻看所有题目更高效，但需要注意不能完全忽视简单题（否则过拟合）。

## 代码示例

```python
import numpy as np
from collections import deque
import random

class ReplayBuffer:
    """标准经验回放缓冲区"""
    def __init__(self, capacity=100000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        return map(np.array, zip(*batch))
    
    def __len__(self):
        return len(self.buffer)

class PrioritizedReplayBuffer:
    """优先经验回放缓冲区"""
    def __init__(self, capacity=100000, alpha=0.6):
        self.capacity = capacity
        self.alpha = alpha
        self.buffer = []
        self.priorities = np.zeros(capacity, dtype=np.float32)
        self.pos = 0
        self.size = 0
    
    def push(self, state, action, reward, next_state, done):
        # 新 experience 赋予最大优先级
        max_prio = self.priorities[:self.size].max() if self.size > 0 else 1.0
        if len(self.buffer) < self.capacity:
            self.buffer.append((state, action, reward, next_state, done))
        else:
            self.buffer[self.pos] = (state, action, reward, next_state, done)
        self.priorities[self.pos] = max_prio
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
    
    def sample(self, batch_size, beta=0.4):
        if self.size < batch_size:
            return None
        
        # 计算采样概率
        priorities = self.priorities[:self.size] ** self.alpha
        probs = priorities / priorities.sum()
        
        # 根据概率采样
        indices = np.random.choice(self.size, batch_size, p=probs)
        batch = [self.buffer[idx] for idx in indices]
        
        # 重要性采样权重
        total = self.size
        weights = (total * probs[indices]) ** (-beta)
        weights /= weights.max()
        
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return states, actions, rewards, next_states, dones, indices, weights
    
    def update_priorities(self, indices, td_errors):
        """更新 TD 误差对应的优先级"""
        for idx, td in zip(indices, td_errors):
            self.priorities[idx] = abs(td) + 1e-6
    
    def __len__(self):
        return self.size

# 使用示例
rb = ReplayBuffer(capacity=50000)
for i in range(1000):
    rb.push(np.array([i]), i, float(i), np.array([i+1]), False)
    
batch = rb.sample(32)
print(f"Sampled batch size: {len(batch[0])}")
print(f"Buffer size: {len(rb)}")
```

## 深度学习关联

1. **Replay 与监督学习的平行**：经验回放使 RL 更接近监督学习的"数据集训练"范式——从固定数据集中随机采样 i.i.d. 数据进行训练。这是 DQN 能利用标准深度学习技术的根本原因（批归一化、dropout 等）。
2. **大型回放缓冲区（Replay Buffer）**：在 DRL 实践中，回放缓冲区占用大量内存（Atari 约 1-10GB）。Hindsight Experience Replay (HER) 进一步将"失败经验"修改为"成功经验"存入缓冲区，极大提升了稀疏奖励问题的样本效率。
3. **分布式回放（Distributed Replay）**：现代 DRL 系统（如 R2D2、Agent57）使用分布式架构：多个 actor 并行收集经验，存入中央回放缓冲区，一个 learner 从中采样训练。这种"生产者-消费者"模式是大规模 DRL 的标配架构。
