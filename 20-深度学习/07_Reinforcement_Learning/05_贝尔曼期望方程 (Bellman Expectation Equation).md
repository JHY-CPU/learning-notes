# 05_贝尔曼期望方程 (Bellman Expectation Equation)

## 核心概念

- **贝尔曼方程**：由 Richard Bellman 提出的递归关系式，将价值函数分解为即时奖励与后继状态折扣价值之和。它是所有强化学习算法的理论基础。
- **状态价值贝尔曼期望方程**：$V_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s]$，将当前状态的价值表达为即时奖励加下一状态价值的期望。
- **动作价值贝尔曼期望方程**：$Q_\pi(s, a) = \mathbb{E}_\pi[R_{t+1} + \gamma Q_\pi(S_{t+1}, A_{t+1}) \mid S_t = s, A_t = a]$。
- **方程展开形式**：通过展开期望，贝尔曼期望方程可以写为关于转移概率、奖励和策略的显式求和形式，这是策略评估算法（动态规划）的直接依据。
- **回溯图 (Backup Diagram)**：贝尔曼方程的图形化表示，展示了价值信息如何在状态、动作和下一状态之间流动。树状结构清晰地表达了"当前价值 = 即时奖励 + 后继价值"的回溯关系。
- **策略评估**：利用贝尔曼期望方程反复迭代更新价值函数，直到收敛——这个过程称为"策略评估"或"预测问题"。每次迭代都使用 $V_{k+1}(s) = \sum_a \pi(a|s)[R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')]$。

## 数学推导

$$
\text{状态价值贝尔曼期望方程: }
V_\pi(s) = \mathbb{E}_\pi[R_{t+1} + \gamma V_\pi(S_{t+1}) \mid S_t = s]
$$

展开形式：

$$
V_\pi(s) = \sum_{a \in \mathcal{A}} \pi(a \mid s) \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_\pi(s') \right]
$$

$$
\text{动作价值贝尔曼期望方程: }
Q_\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) \sum_{a' \in \mathcal{A}} \pi(a' \mid s') Q_\pi(s', a')
$$

**推导关键步骤**：
- 从回报定义出发：$G_t = R_{t+1} + \gamma G_{t+1}$
- 两边取条件期望：$\mathbb{E}[G_t \mid S_t = s] = \mathbb{E}[R_{t+1} + \gamma \mathbb{E}[G_{t+1} \mid S_{t+1}] \mid S_t = s]$
- 注意 $\mathbb{E}[G_{t+1} \mid S_{t+1}] = V_\pi(S_{t+1})$，代入即得贝尔曼方程。
- 展开期望需对策略、转移概率和奖励分别加权求和。
- 理解贝尔曼方程的关键在于：它把"无穷和"问题转化为"一步 + 递归"问题，使动态规划求解成为可能。
- 贝尔曼方程的价值在于：它提供了价值函数之间的自洽约束条件，任何正确的价值函数必须满足该方程。

## 直观理解

贝尔曼期望方程可以理解为"如果明天更好，今天就更值钱"的数学表达：
- 想象你评估一片海滩的旅游价值。你问自己：今天在这片海滩能获得多少快乐（$R_{t+1}$）？如果明天转移到另一片海滩（$s'$），那片海滩的价值（$V(s')$）是多少？
- 贝尔曼方程告诉你：**当前海滩的总价值 = 今天的快乐 + 折扣后的明天海滩价值**。
- 这个递归可以一直延续下去——明天又有明天的快乐和后天的价值。最终，"如何评估一个状态"的问题就被简化为"如何评估它的下一刻"。

一个经典类比是"一美元的故事"：今天的 1 美元（即时奖励）在存到银行后会产生利息（未来回报）。用贝尔曼语言：这 1 美元的**总价值 = 1 美元 + 折扣后的未来总价值**。

回溯图的直觉：画出某个状态 $s$ 向下展开到动作 $a$，再到下一状态 $s'$，再到下一动作 $a'$ 的树形图。这棵树展示了价值估计是如何通过"自举"（bootstrap）从后续状态传播回来的——就像从山顶（终止状态）的温度值向山脚（初始状态）递推预报一样。

## 代码示例

```python
import numpy as np

# 小型 GridWorld: 3x3, 状态 0-8, 动作 0=上 1=下 2=左 3=右
class GridWorld:
    def __init__(self):
        self.nS = 9
        self.nA = 4
        self.gamma = 0.9
        # 确定性转移 + 边界碰撞不动
        self.P = {s: {a: [] for a in range(4)} for s in range(9)}
        for s in range(9):
            r, c = s // 3, s % 3
            for a, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                nr, nc = r + dr, c + dc
                if 0 <= nr < 3 and 0 <= nc < 3:
                    ns = nr * 3 + nc
                else:
                    ns = s  # 撞墙不动
                reward = -1 if ns != 8 else 0  # 状态 8 是目标
                self.P[s][a] = [(1.0, ns, reward, ns == 8)]

def bellman_expectation_evaluation(env, policy, iterations=100):
    """使用贝尔曼期望方程进行策略评估"""
    V = np.zeros(env.nS)
    for i in range(iterations):
        V_new = np.zeros(env.nS)
        for s in range(env.nS):
            total = 0
            for a in range(env.nA):
                prob_a = policy[s][a]
                for prob, s_next, reward, done in env.P[s][a]:
                    total += prob_a * prob * (reward + env.gamma * V[s_next])
            V_new[s] = total
        V = V_new
        if i % 20 == 0:
            print(f"Iter {i}: V 的范围 = [{V.min():.2f}, {V.max():.2f}]")
    return V

# 随机策略
policy = np.ones((9, 4)) / 4
V = bellman_expectation_evaluation(GridWorld(), policy)
print(f"\n最终 V 矩阵:\n{V.reshape(3, 3).round(2)}")
```

## 深度学习关联

- **自举 (Bootstrapping) 的本质**：贝尔曼方程的递归结构是所有基于自举的深度强化学习算法的核心。DQN 的损失函数 $L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$ 就是贝尔曼期望方程的深度版本。
- **TD 误差**：时间差分误差 $\delta = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 直接来源于贝尔曼期望方程的左右差。深度强化学习中梯度优化正是通过最小化这个 TD 误差来更新网络参数。
- **Residual 网络**：有趣的是，深度残差网络 $y = F(x) + x$ 的跨层连接结构与贝尔曼方程 $V(s) = R + \gamma V(s')$ 有异曲同工之妙——两者都将当前输出分解为"当前处理"加"后续传递"，这暗示了递归分解在深度架构中的普适性。
