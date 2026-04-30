# 07_策略迭代 (Policy Iteration) 算法流程

## 核心概念

- **策略迭代**：一种通过交替执行"策略评估"和"策略改进"来求解最优策略的动态规划算法。策略迭代保证在有限步内收敛到最优策略。
- **策略评估 (Policy Evaluation)**：给定策略 $\pi$，使用贝尔曼期望方程迭代计算该策略下的价值函数 $V_\pi$，直到收敛。评估阶段是"预测"问题——回答"当前策略有多好"。
- **策略改进 (Policy Improvement)**：基于当前价值函数 $V_\pi$，贪心地更新策略：$\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_\pi(s')]$。改进阶段是"控制"问题——回答"如何做得更好"。
- **策略改进定理**：如果在某个状态 $s$ 选择动作 $a$ 使得 $Q_\pi(s,a) > V_\pi(s)$，则新的策略在该状态严格优于原策略。将策略在所有状态贪心地选择 $Q$ 最大的动作，不会使性能变差。
- **收敛性**：由于策略空间是有限的（对有限 MDP），策略改进定理保证策略迭代在有限次迭代后收敛到最优策略。每次迭代策略单调改进。
- **广义策略迭代 (GPI)**：将策略评估和策略改进视为两个交替或并行的过程。几乎所有强化学习方法都可以视为 GPI 的实例。

## 数学推导

$$
\text{策略评估：} V_{k+1}(s) = \sum_{a \in \mathcal{A}} \pi(a|s) \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V_k(s') \right], \quad \forall s
$$

$$
\text{策略改进：} \pi'(s) = \arg\max_{a \in \mathcal{A}} \left[ R(s,a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) V_\pi(s') \right], \quad \forall s
$$

$$
\text{策略改进定理：} Q_\pi(s, \pi'(s)) \ge V_\pi(s), \quad \forall s \Rightarrow V_{\pi'}(s) \ge V_\pi(s), \quad \forall s
$$

**算法流程**：
1. 初始化策略 $\pi$（例如随机策略）
2. **重复直到收敛**：
   a. **策略评估**：求解线性方程组（或迭代）得到 $V_\pi$
   b. **策略改进**：使用 $\pi'(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P V_\pi(s')]$ 更新策略
   c. 如果策略在连续两次迭代中不变，则已收敛到最优策略 $\pi^*$

**推导说明**：
- 策略评估本质上是在求解一个线性方程组（对 MRP 的贝尔曼方程求解），可以使用迭代法或直接矩阵求逆。
- 策略改进步不要求策略评估完全收敛——部分评估后即改进也可以（modified policy iteration）。
- 策略迭代通常比价值迭代需要更少的迭代次数，但每次迭代的计算成本更高。

## 直观理解

策略迭代就像不断优化一个团队的工作流程：

- **策略评估**：给定现有流程（策略），测量其绩效指标（价值函数）。比如，评估"旧流程 A"下每个岗位的效率如何。
- **策略改进**：根据评估结果，找出每个环节可以改进的点。如果发现"改用新工具可以提升效率"，就更新流程。
- **迭代过程**：评估 -> 改进 -> 再评估 -> 再改进 -> ... 直到流程不再有改进空间（最优策略）。

这就像 Toyota 的"Kaizen（持续改进）"生产体系：先测量当前流程的产出（评估），然后找出瓶颈并调整（改进），再次测量确认改进有效（再评估），如此循环。每次改进都保证绩效不下降，最终达到局部乃至全局最优。

另一种直观类比是"爬山"：策略评估相当于测量当前海拔（确定当前位置的高度），策略改进相当于选择向上爬的方向（找到梯度方向），两者交替进行最终走到山顶。

## 代码示例

```python
import numpy as np

class GridWorld:
    def __init__(self, gamma=0.9):
        self.nS = 16
        self.nA = 4
        self.gamma = gamma
        self.P = {}
        for s in range(16):
            self.P[s] = {a: [] for a in range(4)}
            r, c = s // 4, s % 4
            if s == 0 or s == 15:  # 终止状态
                continue
            for a, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
                nr, nc = r + dr, c + dc
                ns = nr * 4 + nc if (0 <= nr < 4 and 0 <= nc < 4) else s
                reward = -1
                self.P[s][a] = [(1.0, ns, reward, False)]

def policy_evaluation(env, policy, theta=1e-6):
    V = np.zeros(env.nS)
    while True:
        delta = 0
        for s in range(env.nS):
            if s == 0 or s == 15:
                continue
            v = V[s]
            new_v = 0
            for a in range(env.nA):
                prob_a = policy[s, a]
                for prob, s_next, reward, _ in env.P[s][a]:
                    new_v += prob_a * prob * (reward + env.gamma * V[s_next])
            V[s] = new_v
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V

def policy_improvement(env, V, policy):
    """改进策略，返回是否稳定"""
    policy_stable = True
    for s in range(env.nS):
        if s == 0 or s == 15:
            continue
        old_action = np.argmax(policy[s])
        # 计算所有动作的 Q 值
        q_values = np.zeros(env.nA)
        for a in range(env.nA):
            for prob, s_next, reward, _ in env.P[s][a]:
                q_values[a] += prob * (reward + env.gamma * V[s_next])
        best_action = np.argmax(q_values)
        if old_action != best_action:
            policy_stable = False
        # 更新为确定性策略
        policy[s] = np.eye(env.nA)[best_action]
    return policy_stable

def policy_iteration(env):
    policy = np.ones((env.nS, env.nA)) / env.nA  # 初始随机
    iteration = 0
    while True:
        V = policy_evaluation(env, policy)
        print(f"迭代 {iteration}: 策略评估完成, V 范围=[{V.min():.2f}, {V.max():.2f}]")
        stable = policy_improvement(env, V, policy)
        if stable:
            print(f"策略已稳定, 共 {iteration+1} 次迭代")
            break
        iteration += 1
    return V, policy

env = GridWorld()
V_opt, policy_opt = policy_iteration(env)
print(f"\n最优策略（0=上,1=下,2=左,3=右）:")
print(np.argmax(policy_opt, axis=1).reshape(4, 4))
```

## 深度学习关联

1. **GPI 框架的普遍性**：几乎所有深度强化学习算法都遵循广义策略迭代（GPI）的思想。Actor-Critic 方法就是 GPI 的深度学习实例：Actor 负责策略改进，Critic 负责策略评估（价值函数估计）。
2. **部分评估与深度网络**：深度学习中策略评估不再精确求解贝尔曼方程，而是通过神经网络和梯度下降进行近似评估。这种"近似策略评估 + 近似策略改进"的范式是 DRL 算法（如 PPO、A2C）的基础。
3. **收敛速度权衡**：策略迭代的"完全评估 + 一步改进"范式在深度学习中演变为"几步 SGD 评估 + 一步梯度更新"的实践。Mini-batch 训练和 experience replay 可以看作是在评估精度和计算效率之间的现代权衡方案。
