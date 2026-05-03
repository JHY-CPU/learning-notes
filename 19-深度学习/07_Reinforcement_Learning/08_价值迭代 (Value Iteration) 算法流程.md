# 08_价值迭代 (Value Iteration) 算法流程

## 核心概念

- **价值迭代**：一种直接求解贝尔曼最优方程的动态规划算法。它将策略评估和策略改进合并为一步更新：$V_{k+1}(s) = \max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V_k(s')]$。
- **截断的策略评估**：价值迭代本质上是对策略迭代的截断——每次只进行一步策略评估就立即进行策略改进。这避免了策略迭代中需要完整收敛评估的高计算成本。
- **收缩映射**：贝尔曼最优算子 $\mathcal{T}^*$ 是一个 $\gamma$-收缩映射，满足 $\|\mathcal{T}^*V_1 - \mathcal{T}^*V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty$，这保证了价值迭代的线性收敛速度。
- **有限步终止条件**：当连续两次迭代的价值函数变化小于某个阈值 $\theta$ 时停止。此时可以保证 $\|V_k - V^*\|_\infty < \frac{2\gamma\theta}{1-\gamma}$。
- **与策略迭代的对比**：价值迭代每次更新的计算量小（只做一次扫描），但可能需要更多迭代次数。策略迭代每次更新计算量大但收敛更快（多项式 vs 指数级迭代次数）。
- **提取最优策略**：价值迭代完成后，通过 $\pi(s) = \arg\max_a [R(s,a) + \gamma \sum_{s'} P(s'|s,a) V^*(s')]$ 提取最优策略。

## 数学推导

$$
\text{价值迭代更新规则: }
V_{k+1}(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V_k(s') \right], \quad \forall s \in \mathcal{S}
$$

$$
\text{贝尔曼最优算子: } (\mathcal{T}^* V)(s) = \max_{a \in \mathcal{A}} \left[ R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s' \mid s, a) V(s') \right]
$$

$$
\text{收缩性质: } \|\mathcal{T}^* V_1 - \mathcal{T}^* V_2\|_\infty \le \gamma \|V_1 - V_2\|_\infty
$$

$$
\text{收敛保证: } V_k \to V^* \text{ as } k \to \infty, \text{ 且 } \|V_k - V^*\|_\infty \le \frac{\gamma^k}{1-\gamma} \|V_1 - V_0\|_\infty
$$

**推导说明**：
- 价值迭代可看作是对贝尔曼最优方程的不动点迭代（fixed-point iteration）。
- $\mathcal{T}^*$ 是收缩映射，根据巴拿赫不动点定理（Banach fixed-point theorem），从任意初始 $V_0$ 出发迭代都会收敛到唯一不动点 $V^*$。
- 价值迭代不仅计算 $V^*$，同时也隐含地计算了最优策略 $\pi^*$（因为 $\max$ 操作隐式选择了最优动作）。
- 终止条件 $\Delta < \theta$ 保证 $V_k$ 与 $V^*$ 的误差在可控范围内。

## 直观理解

价值迭代就像用"逆向推理"的方式找最优路径：

想象你在玩一个迷宫游戏，出口在右下角。价值迭代教会你：
- **从出口开始**：出口的价值是 0，因为到了就结束了。
- **逆向传播**：离出口最近的一圈格子，价值是"走一步到达出口的奖励（比如-1）+ 出口价值 0"。
- **一层层往外推**：每一轮迭代，价值信息就像波纹一样从出口向外扩散一圈。
- **每一轮每个格子都看最好的邻居**：不像策略迭代要绕一大圈评估完再更新，价值迭代每轮直接假设"我选了最好的邻居"来更新自己。

形象地说：价值迭代是"倒着推"到起点的过程，而策略迭代是"先走走看，再问怎么改进"。

一个经典比喻：价值迭代就像在山区重建等高线地图——你不关心具体走哪条路（策略），只关心每个位置的海拔高度（价值）。每一轮迭代，你都把每个点的海拔更新为"该点出发一步能到达的最高海拔"，经过多轮扩散，整张地图就反映了真实的山势。

## 代码示例

```python
import numpy as np

# 4x4 GridWorld
nS, nA = 16, 4
gamma = 0.9

# 构建状态转移
P = {s: {a: [] for a in range(nA)} for s in range(nS)}
for s in range(nS):
    r, c = s // 4, s % 4
    if s == 0 or s == 15:
        continue
    for a, (dr, dc) in enumerate([(-1,0),(1,0),(0,-1),(0,1)]):
        nr, nc = r + dr, c + dc
        ns = nr * 4 + nc if (0 <= nr < 4 and 0 <= nc < 4) else s
        reward = -1
        P[s][a] = [(1.0, ns, reward, False)]

def value_iteration(env_P, nS, nA, gamma=0.9, theta=1e-6):
    V = np.zeros(nS)
    iteration = 0
    while True:
        delta = 0
        V_new = np.zeros(nS)
        for s in range(nS):
            if s == 0 or s == 15:  # 终止状态
                continue
            max_value = -np.inf
            for a in range(nA):
                total = 0
                for prob, s_next, reward, _ in env_P[s][a]:
                    total += prob * (reward + gamma * V[s_next])
                max_value = max(max_value, total)
            V_new[s] = max_value
            delta = max(delta, abs(V_new[s] - V[s]))
        V = V_new
        iteration += 1
        if delta < theta:
            break
    
    # 提取最优策略
    policy = np.zeros(nS, dtype=int)
    for s in range(nS):
        if s == 0 or s == 15:
            continue
        q_values = np.zeros(nA)
        for a in range(nA):
            for prob, s_next, reward, _ in env_P[s][a]:
                q_values[a] += prob * (reward + gamma * V[s_next])
        policy[s] = np.argmax(q_values)
    
    return V, policy, iteration

V_opt, policy_opt, iters = value_iteration(P, nS, nA)
print(f"价值迭代在 {iters} 次迭代后收敛")
print(f"\n最优状态价值 V*:")
print(V_opt.reshape(4, 4).round(1))
print(f"\n最优策略 (0=上,1=下,2=左,3=右):")
print(policy_opt.reshape(4, 4))
```

## 深度学习关联

- **DQN 作为价值迭代的近似**：DQN 每次更新 $Q(s,a) \leftarrow Q(s,a) + \alpha(r + \gamma \max_{a'} Q(s',a') - Q(s,a))$ 可以看作是价值迭代的随机近似版本，其中 $\max_{a'} Q(s',a')$ 对应价值迭代中的 $\max$ 操作。
- **收敛速度与优化景观**：价值迭代使用收缩映射保证线性收敛，深度 Q 网络虽然失去了严格的收缩保证（因为函数近似噪声），但实践中通过目标网络和目标网络冻结（periodic copy）来近似保持更新的稳定性。
- **从表格到函数近似**：价值迭代对每个状态独立计算 $\max$，而 DQN 在一次前向传播中计算所有动作的 Q 值（batch 预测），这本质上是一种"并行化的价值迭代"，利用 GPU 加速了整个更新过程。
