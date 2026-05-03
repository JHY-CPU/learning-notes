# 04_价值函数 (Value Function) 与状态-动作价值 (Q-function)

## 核心概念

- **状态价值函数 $V_\pi(s)$**：在策略 $\pi$ 下，从状态 $s$ 出发所能获得的期望折扣回报。$V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s]$，它衡量一个状态的"好坏"程度。
- **动作价值函数 $Q_\pi(s, a)$**：在策略 $\pi$ 下，从状态 $s$ 出发执行动作 $a$ 后所能获得的期望折扣回报。$Q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a]$，它衡量在某个状态下执行某个动作的"好坏"。
- **两者关系**：$V_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q_\pi(s, a)$。即状态价值是所有动作价值的策略加权平均。反过来，$Q_\pi(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) V_\pi(s')$。
- **最优价值函数**：$V^*(s) = \max_\pi V_\pi(s)$ 和 $Q^*(s, a) = \max_\pi Q_\pi(s, a)$ 分别表示在所有可能的策略下能达到的最大期望回报。一旦求出 $Q^*$，最优策略可直接通过 $\pi^*(s) = \arg\max_a Q^*(s, a)$ 得到。
- **优势函数 $A(s, a)$**：定义为 $A(s, a) = Q(s, a) - V(s)$，衡量某个动作相对于当前策略平均水平的优势。正的优势表示该动作比平均水平好，负值表示更差。这是 Actor-Critic 方法的核心概念。
- **函数近似**：当状态空间太大无法使用表格存储时，使用参数化函数（如神经网络）来近似价值函数：$V(s; \theta)$ 或 $Q(s, a; \theta)$。

## 数学推导

$$
V_\pi(s) = \mathbb{E}_\pi[G_t \mid S_t = s] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \;\Big|\; S_t = s\right]
$$

$$
Q_\pi(s, a) = \mathbb{E}_\pi[G_t \mid S_t = s, A_t = a] = \mathbb{E}_\pi\left[\sum_{k=0}^\infty \gamma^k R_{t+k+1} \;\Big|\; S_t = s, A_t = a\right]
$$

$$
V_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) Q_\pi(s, a)
$$

$$
Q_\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_\pi(s')
$$

$$
\text{最优函数关系: } V^*(s) = \max_a Q^*(s, a)
$$

$$
Q^*(s, a) = R(s, a) + \gamma \sum_{s'} P(s' \mid s, a) \max_{a'} Q^*(s', a')
$$

**推导说明**：
- 状态价值函数 $V_\pi(s)$ 是整个学习过程的"北极星"——它告诉智能体当前状态的长远期望收益。
- Q 函数的一个关键优势是：知道 $Q^*$ 后，选择最优动作不需要知道环境模型（model-free）——只需要在每一步计算 $\arg\max_a Q^*(s, a)$。
- 优势函数 $A(s, a)$ 在 Actor-Critic 方法中用于减少策略梯度的方差，是 PPO、A2C 等算法的关键组件。

## 直观理解

价值函数类比于评估一个人的职业前景：
- **$V(s)$（状态价值）** 相当于"在当前职位上的长期综合收益预期"。一个状态（职位）好，是因为从这个状态出发，按照常规策略发展，未来能获得的总回报高。例如，"高级工程师"的状态价值远高于"实习生"。
- **$Q(s, a)$（动作价值）** 相当于"在当前职位上选择某个具体行动的长期预期收益"。比如作为员工（状态），选择"努力加班"vs"摸鱼"——Q 函数告诉你每种选择对长期收益的影响。
- **$A(s, a)$（优势）** 相当于"这个行动比平均水平好多少"。如果加班带来的优势是正的，说明加班值得；如果是负的，说明不如躺平。

三种函数的递进关系：$V$ 告诉你"你在哪里"，$Q$ 告诉你"你该做什么"，$A$ 告诉你"这样做值多少"。

## 代码示例

```python
import numpy as np

# 模拟一个简单的 2 状态 2 动作环境
S = [0, 1]      # 状态: 0=低阶, 1=高阶
A = [0, 1]      # 动作: 0=摸鱼, 1=努力

# 转移概率: P(s' | s, a)
P = {
    (0, 0): {0: 0.9, 1: 0.1},  # 低阶+摸鱼: 90%保持低阶, 10%到高阶
    (0, 1): {0: 0.2, 1: 0.8},  # 低阶+努力: 20%保持低阶, 80%到高阶
    (1, 0): {0: 0.8, 1: 0.2},  # 高阶+摸鱼: 80%保持高阶, 20%到低阶
    (1, 1): {0: 0.1, 1: 0.9},  # 高阶+努力: 10%到低阶, 90%保持高阶
}
R = {(0, 0): 0, (0, 1): 1, (1, 0): 2, (1, 1): 3}  # 即时奖励

def policy_evaluation(policy, gamma=0.9, theta=1e-6):
    """策略评估: 计算给定策略下的 V 和 Q"""
    V = np.zeros(len(S))
    while True:
        delta = 0
        for s in S:
            v = V[s]
            new_v = 0
            for a in A:
                prob_a = policy[s][a]
                expected_q = R[(s, a)]
                for s_next, prob_s in P[(s, a)].items():
                    expected_q += gamma * prob_s * V[s_next]
                new_v += prob_a * expected_q
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    # 计算 Q 函数
    Q = np.zeros((len(S), len(A)))
    for s in S:
        for a in A:
            Q[s, a] = R[(s, a)]
            for s_next, prob_s in P[(s, a)].items():
                Q[s, a] += gamma * prob_s * V[s_next]
    return V, Q

# 假设均匀随机策略
policy = {0: [0.5, 0.5], 1: [0.5, 0.5]}
V, Q = policy_evaluation(policy)
print(f"状态价值: V(低阶)={V[0]:.2f}, V(高阶)={V[1]:.2f}")
print(f"动作价值: Q(低阶,摸鱼)={Q[0,0]:.2f}, Q(低阶,努力)={Q[0,1]:.2f}")
print(f"优势: A(低阶,努力)={Q[0,1]-V[0]:.2f}")
```

## 深度学习关联

- **DQN 与 Q 函数逼近**：DQN 使用深度卷积神经网络来逼近 $Q(s, a)$，输入是游戏画面的原始像素，输出是每个动作的 Q 值。这是深度学习与 Q-learning 的经典结合。
- **Dueling DQN**：通过将 Q 网络分解为 $Q(s, a) = V(s) + A(s, a)$，Dueling DQN 显式建模了价值函数和优势函数。这种分解使得网络可以独立学习哪些状态本身是有价值的，而不必为每个动作分别学习。
- **Actor-Critic 中的双函数**：现代深度强化学习（A2C、PPO、SAC）同时学习策略函数（Actor）和价值/优势函数（Critic）。Critic 提供的价值估计作为 baseline 来减小策略梯度的方差，是算法稳定训练的关键。
