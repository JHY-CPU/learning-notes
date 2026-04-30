# 02_马尔可夫决策过程 (MDP) 定义与性质

## 核心概念

- **马尔可夫性质 (Markov Property)**：未来状态只依赖于当前状态，与过去的历史无关。形式化为 $P(S_{t+1} \mid S_t, A_t) = P(S_{t+1} \mid S_t, A_t, S_{t-1}, A_{t-1}, ..., S_0, A_0)$，这是 MDP 的基石假设。
- **MDP 五元组定义**：一个马尔可夫决策过程由 $(S, A, P, R, \gamma)$ 五个元素构成，分别代表状态集、动作集、状态转移概率、奖励函数和折扣因子。
- **状态转移概率**：$P(s' \mid s, a) = \Pr(S_{t+1} = s' \mid S_t = s, A_t = a)$，描述了在状态 $s$ 执行动作 $a$ 后转移到 $s'$ 的概率，所有可能的状态转移概率之和为 1。
- **奖励函数**：$R(s, a, s')$ 表示在状态 $s$ 执行动作 $a$ 并转移到 $s'$ 时获得的即时奖励。通常也简化为 $R(s, a)$ 或 $R(s)$。
- **策略 (Policy)**：$\pi(a \mid s)$ 定义了智能体在每个状态选择各个动作的概率分布，是 MDP 中智能体的行为准则。
- **MDP 与 MP/MRP 的关系**：马尔可夫过程 (MP) 不含动作和奖励；马尔可夫奖励过程 (MRP) 含奖励但不含动作；MDP 是加入了动作选择的最完整形式。MDP 在固定策略下退化为 MRP。
- **各态历经性 (Ergodicity)**：指从任何状态出发都能到达其他所有状态，这对收敛性分析很重要。
- **最优策略存在性**：MDP 理论保证至少存在一个确定性最优策略，这保证我们只需要在确定性策略空间中搜索即可。

## 数学推导

$$
\text{MDP 五元组: } \mathcal{M} = (\mathcal{S}, \mathcal{A}, \mathcal{P}, \mathcal{R}, \gamma)
$$

$$
\text{状态转移概率: } \mathcal{P}_{ss'}^a = P(S_{t+1} = s' \mid S_t = s, A_t = a)
$$

$$
\text{奖励函数: } R(s, a) = \mathbb{E}[R_{t+1} \mid S_t = s, A_t = a]
$$

$$
\text{由 MDP 到 MRP 的转换（固定策略 $\pi$）: }
$$

$$
\mathcal{P}_{ss'}^\pi = \sum_{a \in \mathcal{A}} \pi(a \mid s) \mathcal{P}_{ss'}^a
$$

$$
R_s^\pi = \sum_{a \in \mathcal{A}} \pi(a \mid s) R(s, a)
$$

**推导说明**：
- 给定策略 $\pi$ 后，MDP 退化为 MRP：动作选择被策略"边际化"掉，转移概率和奖励都变成策略加权平均。
- 这一性质允许我们在固定策略下使用 MRP 的价值迭代方法。
- MDP 的核心问题是求解最优策略 $\pi^*$，使得期望回报最大化。

## 直观理解

MDP 就像一个有明确规则的棋盘游戏：
- **状态（棋盘布局）** 描述了当前所有棋子的位置，信息完全可观测。
- **动作（走棋）** 是玩家可以做出的合法移动。
- **转移概率（对手反应）** 虽然玩家行动是确定的，但对手的回应可能不确定（类似随机环境）。
- **奖励（得分变化）** 吃子加分，被吃减分，将死获胜。

"马尔可夫性质"意味着你不需要看棋谱（历史记录）来做出决策——只看当前盘面就足够了。在围棋这样的完美信息游戏中，这个性质天然成立；但在扑克等不完美信息游戏中，历史信息可能包含对手的欺骗模式，此时需要 POMDP 框架。

## 代码示例

```python
import numpy as np

class GridWorldMDP:
    """一个简单的 4x4 网格世界 MDP"""
    def __init__(self, size=4):
        self.S = list(range(size * size))  # 状态空间: 16 个格子
        self.A = [0, 1, 2, 3]  # 动作空间: 0=上, 1=下, 2=左, 3=右
        self.gamma = 0.9
        self.size = size
        self.terminal_states = [0, size * size - 1]  # 终止状态: 左上和右下
        
    def transition(self, s, a):
        """返回转移概率和奖励"""
        if s in self.terminal_states:
            return [(s, 0, 1.0)]  # 终止状态自循环
        
        row, col = s // self.size, s % self.size
        if a == 0:    # 上
            row = max(0, row - 1)
        elif a == 1:  # 下
            row = min(self.size - 1, row + 1)
        elif a == 2:  # 左
            col = max(0, col - 1)
        elif a == 3:  # 右
            col = min(self.size - 1, col + 1)
        
        s_next = row * self.size + col
        reward = -1 if s_next not in self.terminal_states else 0
        return [(s_next, reward, 1.0)]  # 确定性转移

# 验证 MDP 性质
mdp = GridWorldMDP(4)
s0 = 5  # 初始状态
print(f"状态 {s0} 下各动作的结果:")
for a in mdp.A:
    s_next, reward, prob = mdp.transition(s0, a)[0]
    print(f"  动作 {a} -> 状态 {s_next}, 奖励 {reward}")

# 固定随机策略下 MDP -> MRP
policy = {s: [0.25, 0.25, 0.25, 0.25] for s in range(16)}
s = 5
P_mrp = np.zeros(16)
for a, prob_a in enumerate(policy[s]):
    s_next, reward, _ = mdp.transition(s, a)[0]
    P_mrp[s_next] += prob_a
print(f"\n从状态 {s} 出发的 MRP 转移概率（边际化后）:\n{P_mrp.reshape(4,4).round(2)}")
```

## 深度学习关联

1. **深度 MDP 近似**：当状态空间连续或巨大时，深度神经网络作为函数近似器来估计 MDP 中的价值函数和策略，形成深度强化学习的理论基础。DQN 本质上是用深度网络拟合大型 MDP 的 Q 函数。
2. **POMDP 与循环神经网络**：部分可观测 MDP (POMDP) 需要记忆历史信息，这与 RNN/LSTM 的时序建模能力天然契合。DRQN (Deep Recurrent Q-Network) 就是结合 LSTM 处理部分观测的 MDP 问题。
3. **隐马尔可夫模型关联**：MDP 与 HMM 共享马尔可夫性质，区别在于 MDP 包含动作和奖励信号（控制问题），而 HMM 用于序列建模（无控制输入），两者在概率图模型中同属动态贝叶斯网络。
