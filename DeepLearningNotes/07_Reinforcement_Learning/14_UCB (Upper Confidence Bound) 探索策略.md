# 14_UCB (Upper Confidence Bound) 探索策略

## 核心概念

- **UCB 核心思想**：基于"乐观面对不确定性"（optimism in the face of uncertainty）原则。对于每个动作，根据当前估计和不确定性构建一个置信上界，选择上界最高的动作。
- **UCB1 算法**：多臂赌博机（Multi-Armed Bandit）问题中最著名的 UCB 算法。选择动作 $a = \arg\max_a [Q(a) + c\sqrt{\ln t / N(a)}]$，其中 $Q(a)$ 是当前平均奖励，$N(a)$ 是动作被选择的次数，$t$ 是总时间步。
- **探索项 $c\sqrt{\ln t / N(a)}$**：这衡量了动作 $a$ 价值估计的不确定性。当 $N(a)$ 较小时（探索不足），该项较大，鼓励选择这个动作；随着访问次数增加，该项趋近于 0。
- **后悔界 (Regret Bound)**：UCB1 保证了 $\log$ 级别的累积后悔——$O(\log T)$，即智能体因未选择最优动作而损失的奖励随时间对数增长，远优于 $\epsilon$-greedy 的线性后悔。
- **$\epsilon$-greedy vs UCB**：$\epsilon$-greedy 的探索是"盲目的"——等概率选择所有非最优动作。UCB 的探索是"智能的"——优先探索不确定性高的动作，忽略已经充分了解的低价值动作。
- **UCB 的局限性**：需要维护每个动作的访问计数（在状态-动作空间巨大时不可行）；最初每个动作需要至少被选一次；在非平稳环境中表现不佳（需要修改）。

## 数学推导

$$
\text{UCB1 动作选择: } a_t = \arg\max_{a \in \mathcal{A}} \left[ Q_t(a) + c \sqrt{\frac{\ln t}{N_t(a)}} \right]
$$

$$
\text{Hoeffding 不等式: } P\left( \mathbb{E}[X] - \bar{X}_n \ge u \right) \le \exp(-2nu^2)
$$

$$
\text{UCB 置信界推导: } \text{令 } u = \sqrt{\frac{2\ln t}{N_t(a)}} 
\Rightarrow P\left( Q^*(a) > Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}} \right) \le t^{-2c^2}
$$

$$
\text{累积后悔界: } R_T \le \left[ 8 \sum_{a: \Delta_a > 0} \frac{\ln T}{\Delta_a} \right] + \left(1 + \frac{\pi^2}{3}\right) \sum_{a} \Delta_a = O(\log T)
$$

**推导说明**：
- UCB 的置信上界来源于 Hoeffding 不等式：真实均值 $Q^*(a)$ 以高概率落在样本均值 $Q_t(a)$ 的某个半径内。
- $\sqrt{\ln t / N(a)}$ 中的 $\ln t$ 反映了"随着总步数 $t$ 增加，置信水平需要适当收紧"。
- 后悔界 $O(\log T)$ 是最优的——在信息论意义上，不存在渐近遗憾更低的算法。
- 探索系数 $c$ 控制探索强度：$c$ 越大越激进探索；理论最优值为 $c = \sqrt{2}$（假设奖励在 [0,1] 区间）。

## 直观理解

UCB 就像一个"好奇心驱动"的科学家，而 $\epsilon$-greedy 像一个"随机抽签"的赌徒：

想象你面前有 5 台老虎机，你玩了 100 轮。数据显示：
- 机器 A：玩了 80 次，平均收益 $9.0
- 机器 B：玩了 3 次，平均收益 $8.5
- 机器 C：玩了 4 次，平均收益 $8.8
- 机器 D：玩了 5 次，平均收益 $2.0
- 机器 E：玩了 8 次，平均收益 $8.7

**$\epsilon$-greedy 会怎么选**：90% 的概率选机器 A（当前最优），10% 的概率随机选 B/C/D/E 中的一台——甚至可能选已知很差的 D。

**UCB 会怎么选**：它计算每个机器的"上限估计"：
- 机器 A：$9.0 + c\sqrt{\ln 100/80} \approx 9.0 + 少量$（已经很确定了）
- 机器 B：$8.5 + c\sqrt{\ln 100/3} \approx 8.5 + 大量$（不确定性大，值得一试）
- 机器 D：$2.0 + c\sqrt{\ln 100/5} \approx 2.0 + 中量$（虽然不确定性大，但下界太低了）

UCB 会选择上限最高的——可能是机器 B（虽然均值不高，但存在暴涨的可能性）。这就是"乐观面对不确定"的含义。

## 代码示例

```python
import numpy as np

class UCBBandit:
    """使用 UCB1 的多臂赌博机"""
    def __init__(self, k=10, c=2.0):
        self.k = k
        self.c = c  # 探索系数
        self.q_true = np.random.normal(0, 1, k)
        self.q_est = np.zeros(k)
        self.n = np.zeros(k)
        self.t = 0
    
    def select_action(self):
        self.t += 1
        # 每个动作至少选一次
        for a in range(self.k):
            if self.n[a] == 0:
                return a
        
        # UCB 公式
        ucb_values = self.q_est + self.c * np.sqrt(np.log(self.t) / self.n)
        return np.argmax(ucb_values)
    
    def step(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.n[action] += 1
        self.q_est[action] += (reward - self.q_est[action]) / self.n[action]
        return reward

# 对比 UCB vs epsilon-greedy
def compare_strategies(runs=500, steps=1000):
    ucb_rewards = np.zeros(steps)
    eps_rewards = np.zeros(steps)
    
    for run in range(runs):
        # UCB
        ucb = UCBBandit(k=10, c=2.0)
        for step in range(steps):
            a = ucb.select_action()
            r = ucb.step(a)
            ucb_rewards[step] += r
        
        # Epsilon-greedy (epsilon=0.1)
        eg = EpsilonGreedyBandit(k=10, epsilon=0.1)
        for step in range(steps):
            a = eg.select_action()
            r = eg.step(a)
            eps_rewards[step] += r
    
    ucb_rewards /= runs
    eps_rewards /= runs
    
    print(f"UCB 平均奖励: {np.mean(ucb_rewards):.3f}")
    print(f"Epsilon-Greedy 平均奖励: {np.mean(eps_rewards):.3f}")
    
    # 计算累计遗憾
    optimal_values = [np.max(np.random.normal(0, 1, 10)) for _ in range(100)]
    best_mean = np.mean(optimal_values)
    print(f"\nUCB 后 500 步平均: {np.mean(ucb_rewards[500:]):.3f}")
    print(f"Eps-Greedy 后 500 步平均: {np.mean(eps_rewards[500:]):.3f}")

class EpsilonGreedyBandit:
    def __init__(self, k=10, epsilon=0.1):
        self.k = k
        self.epsilon = epsilon
        self.q_true = np.random.normal(0, 1, k)
        self.q_est = np.zeros(k)
        self.n = np.zeros(k)
    
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(self.k)
        return np.argmax(self.q_est)
    
    def step(self, action):
        reward = np.random.normal(self.q_true[action], 1)
        self.n[action] += 1
        self.q_est[action] += (reward - self.q_est[action]) / self.n[action]
        return reward

compare_strategies(runs=200, steps=500)
```

## 深度学习关联

1. **UCB 在 DRL 中的挑战**：标准 UCB 需要维护每个状态-动作对的访问计数，这在深度强化学习的海量状态空间中不可行。解决方案包括使用集成方法（Ensemble）来估计不确定性，如 Bootstrapped DQN 使用多个 Q 网络头部的方差作为置信区间。
2. **探索的现代视角**：在深度强化学习中，UCB 的思想启发了许多基于不确定性的探索方法。例如，RND (Random Network Distillation) 使用预测误差作为内在奖励——预测误差大的状态就是"不确定性高"的状态，类似 UCB 的探索项。
3. **信息论探索**：与 UCB 相关的还有信息增益（Information Gain）方法，如 VIME (Variational Information Maximizing Exploration)。这些方法将探索视作减少环境模型不确定性的过程，比 UCB 更通用但计算成本更高。
