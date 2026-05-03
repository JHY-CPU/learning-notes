# 10_时序差分学习 (TD Learning)：TD(0)

## 核心概念

- **时序差分学习 (Temporal Difference Learning, TD)**：将蒙特卡洛（MC）思想和动态规划（DP）思想结合的方法。TD 从 episode 的经验中学习，同时使用自举（bootstrap）——用当前的价值估计来更新价值估计。
- **TD(0) 更新规则**：$V(s) \leftarrow V(s) + \alpha [R_{t+1} + \gamma V(s') - V(s)]$。其中 $\delta = R_{t+1} + \gamma V(s') - V(s)$ 称为 TD 误差。
- **TD 误差 $\delta$**：衡量"实际观测到的转移"与"当前价值估计"之间的差距。$\delta > 0$ 表示实际结果比预期好；$\delta < 0$ 表示比预期差。TD 误差也是许多神经科学模型中的"预测误差"信号。
- **自举 (Bootstrapping)**：TD 使用当前的价值估计 $V(s')$ 来代替真实的回报，这引进了偏差（bias）但大幅降低了方差——因为不需要等到 episode 结束就能更新。
- **MC vs TD 对比**：MC 使用完整回报 $G_t$（无偏但高方差），TD(0) 使用 $R_{t+1} + \gamma V(s')$（有偏但低方差）。在有限数据下，TD 通常比 MC 学习更快。
- **批处理 TD (Batch TD)**：在有限数据集上，确定性等价估计（certainty equivalence）表明 TD 收敛到最大似然马尔可夫模型的解，而 MC 收敛到最小化均方误差的解。

## 数学推导

$$
\text{MC 目标: } G_t = R_{t+1} + \gamma R_{t+2} + \gamma^2 R_{t+3} + ... + \gamma^{T-t-1} R_T
$$

$$
\text{TD(0) 目标: } R_{t+1} + \gamma V(S_{t+1})
$$

$$
\text{TD(0) 更新: } V(S_t) \leftarrow V(S_t) + \alpha \underbrace{(R_{t+1} + \gamma V(S_{t+1}) - V(S_t))}_{\text{TD 误差 } \delta_t}
$$

$$
\text{TD 误差: } \delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)
$$

$$
\text{MC 与 TD 的偏差-方差权衡: }
$$
$$
\text{MC: } \mathbb{E}[G_t] = V_\pi(s), \quad \text{Var}(G_t) \propto O(T) \quad \text{(无偏, 高方差)}
$$
$$
\text{TD: } \mathbb{E}[R_{t+1} + \gamma V(S_{t+1})] \ne V_\pi(s) \quad \text{(有偏, 低方差)}
$$

**推导说明**：
- TD 的目标 $R_{t+1} + \gamma V(S_{t+1})$ 是一次"一步展开"的回报近似，它用当前的价值估计替代了真实但未知的 $V(S_{t+1})$。
- 这种一步近似的偏差来源于 $V(S_{t+1})$ 不是真实的 $V_\pi(S_{t+1})$——但正是这种偏差让 TD 能够以更少的样本有效学习。
- 在马尔可夫环境中，TD 的偏差会随学习收敛到 0（因为 $V \to V_\pi$）。

## 直观理解

TD 学习就像"边走边学"，而不像 MC 那样"走完再总结"：

- **MC**（事后诸葛亮）：等整局游戏结束，复盘"第 3 步那个状态，后面总共得了 10 分，所以它值 10 分"。
- **TD(0)**（即时反思）：走了一步后立刻想"从第 3 步到第 4 步，我得了 1 分，而第 4 步我预计值 8 分，所以第 3 步的实际价值是 1 + 0.9*8 = 8.2 分。比我之前想的 7 分高，所以调高它的估值。"

最生动的类比是开车导航：
- **MC 方法**：到达目的地后，回顾整个行程，"在路口 A 转弯后，总共花了 30 分钟"，下次在 A 路口就知道该不该转弯。
- **TD 方法**：在路口 A 转弯后，走了一段看到路况不错（即时奖励），加上对前方路段的预期（$V(s')$），立刻更新对这个路口的判断，不等到达目的地。

这就是为什么 TD 比 MC 学习更快——它不需要等到终点就能从每一步中学习。在神经科学中，多巴胺神经元的发放模式与 TD 误差惊人地一致，被认为是大脑中"预测误差信号"的实现。

## 代码示例

```python
import numpy as np

# 随机游走问题：状态 A-B-C-D-E，两端为终止状态
# 从左端终止得 0 分，从右端终止得 1 分
nS = 7  # 0 和 6 是终止状态
gamma = 1.0

# 生成一个 episode
def generate_episode():
    states = [3]  # 从 C (索引 3) 出发
    while True:
        s = states[-1]
        if s == 0 or s == 6:
            break
        # 等概率向左或向右
        s_next = s + np.random.choice([-1, 1])
        states.append(s_next)
    reward = 1.0 if states[-1] == 6 else 0.0
    return states, reward

# TD(0) 学习
V_td = np.zeros(nS)
V_td[0] = V_td[6] = 0  # 终止状态
alpha = 0.1

n_episodes = 1000
for ep in range(n_episodes):
    states, reward = generate_episode()
    # TD(0) 更新: 每一步都立即更新
    for t in range(len(states) - 1):
        s = states[t]
        s_next = states[t + 1]
        if t == len(states) - 2:  # 最后一步使用真实奖励
            actual_reward = reward
        else:
            actual_reward = 0  # 中间步没有即时奖励
        td_target = actual_reward + gamma * V_td[s_next]
        td_error = td_target - V_td[s]
        V_td[s] += alpha * td_error

print("TD(0) 学习的价值估计:")
for i, name in enumerate(["T_L", "A", "B", "C", "D", "E", "T_R"]):
    print(f"  {name}: {V_td[i]:.3f}")

# 与 MC 收敛值对比（理论值: 1/6, 2/6, 3/6, 4/6, 5/6）
print(f"\n理论真实值: A=1/6, B=2/6, C=3/6, D=4/6, E=5/6")
print(f"从 A 到 E: {[f'{i/6:.3f}' for i in range(1,6)]}")
```

## 深度学习关联

- **DQN 中的 TD 误差最小化**：DQN 的核心损失函数 $L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$ 正是 TD 误差的平方。整个深度 Q 网络就是通过最小化 TD 误差来学习的。这里的 $\theta^-$ 是目标网络参数，用以稳定 TD 目标。
- **TD 与残差学习**：TD 误差在深度学习中可以看作是一种"残差预测"——神经网络预测 $V(s)$，然后计算预测与实际观察 + 下一个预测之间的残差。这种预测纠错机制与残差网络（ResNet）有形式上的相似性。
- **多步 TD (N-step TD)**：$TD(\lambda)$ 通过引入 eligibility traces 在 MC 和 TD(0) 之间插值。在 DRL（如 A2C、PPO）中，N 步回报（N-step return）是标准实践，它介于 1 步 TD 和 MC 之间，提供了一个可调节偏差-方差权衡的实用工具。
