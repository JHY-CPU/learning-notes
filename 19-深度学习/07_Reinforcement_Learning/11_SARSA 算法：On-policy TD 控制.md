# 11_SARSA 算法：On-policy TD 控制

## 核心概念

- **SARSA 命名**：State-Action-Reward-State-Action 的缩写，表示每步更新需要 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$ 五个元素。这是典型的 on-policy TD 控制算法。
- **On-policy 学习**：用于评估和改进的策略与用于生成数据的策略是同一个。SARSA 使用当前策略选出的 $A_{t+1}$ 来构造 TD 目标，保证数据分布与学习目标一致。
- **SARSA 更新规则**：$Q(s, a) \leftarrow Q(s, a) + \alpha[R_{t+1} + \gamma Q(s', a') - Q(s, a)]$。注意目标中使用了 $a'$——这是当前策略在 $s'$ 下选择的动作。
- **收敛条件**：在有限 MDP 中，满足 GLIE（Greedy in the Limit with Infinite Exploration）条件时，SARSA 以概率 1 收敛到最优策略和最优 Q 函数。
- **与 Q-Learning 的核心区别**：SARSA 使用 $Q(s', a')$（实际执行的动作），而 Q-Learning 使用 $\max_a Q(s', a)$（最大可能的动作）。SARSA 更保守，Q-Learning 更激进。
- **现实世界适用性**：SARSA 在许多有安全顾虑的场景中表现更好，因为它考虑了探索带来的"代价"，学到的策略更谨慎。

## 数学推导

$$
\text{SARSA 更新公式: }
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t) \right]
$$

$$
\text{TD 目标（SARSA）: } \mathcal{T}_{\text{SARSA}} = R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})
$$

$$
\text{对比 Q-Learning TD 目标: } \mathcal{T}_{\text{QL}} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)
$$

$$
\text{SARSA 策略评估的贝尔曼方程: }
$$

$$
Q_\pi(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \sum_{a' \in \mathcal{A}} \pi(a'|s') Q_\pi(s', a')
$$

**推导说明**：
- SARSA 的更新目标 $R_{t+1} + \gamma Q(S_{t+1}, A_{t+1})$ 是 on-policy 的贝尔曼期望方程在 Q 函数上的直接应用。
- 与 MC 控制算法相比，SARSA 不需要等到 episode 结束即可更新，也不需要探索性初始化。
- SARSA 的行为策略和目标策略相同，因此在 Q 函数收敛过程中，$\epsilon$-greedy 策略本身也在不断改进。

## 直观理解

SARSA 就像一个"谨慎的学习者"：

想象你在森林里赶路，遇到岔路口（状态 $s$）。你选择了左边的路（动作 $a$），走了一段（获得奖励 $r$），到了下一个路口（状态 $s'$），然后你看了一下——如果你选择继续走左边（动作 $a'$），你会去哪里。

**SARSA 的关键特征**：你在更新当前决策时，用的是"你实际会怎么走下一步"（$Q(s', a')$），而不是"理论上最好怎么走"（$\max Q(s', a)$）。

这个比喻可以帮助理解 SARSA vs Q-Learning：
- **SARSA**（谨慎）：在悬崖边上走路时，会小心翼翼地保持距离，因为它知道自己的探索行为偶尔会"滑倒"靠近悬崖。
- **Q-Learning**（激进）：在悬崖边上走路时，会紧贴悬崖边缘走，因为它假设自己总是能做最优选择——但实际上如果探索时滑下去了就麻烦了。

这就是为什么 Cliff Walking 环境中 SARSA 通常比 Q-Learning 获得更高的累计奖励——SARSA 考虑了探索带来的风险。

## 代码示例

```python
import numpy as np
import gymnasium as gym

def sarsa(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        # 用 epsilon-greedy 选择第一个动作
        if np.random.random() < epsilon:
            action = env.action_space.sample()
        else:
            action = np.argmax(Q[state])
        
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # 执行动作
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # 用 epsilon-greedy 选择下一个动作
            if np.random.random() < epsilon:
                next_action = env.action_space.sample()
            else:
                next_action = np.argmax(Q[next_state])
            
            # SARSA 更新: Q(s,a) += alpha[r + gamma*Q(s',a') - Q(s,a)]
            td_target = reward + gamma * Q[next_state, next_action] * (not terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state, action = next_state, next_action
        
        episode_rewards.append(total_reward)
    
    return Q, episode_rewards

env = gym.make("CliffWalking-v0")
Q_sarsa, rewards_sarsa = sarsa(env, num_episodes=500)

print(f"SARSA 后 100 个 episode 的平均奖励: {np.mean(rewards_sarsa[-100:]):.2f}")
policy = np.argmax(Q_sarsa, axis=1)
print(f"学习到的策略（0=上,1=右,2=下,3=左）:")
print(policy.reshape(4, 12))
env.close()
```

## 深度学习关联

- **On-policy 限制与 DRL**：SARSA 的 on-policy 特性意味着它无法使用经验回放（Experience Replay），因为回放的数据来自旧策略，不满足 on-policy 要求。这正是 DQN 选择 off-policy Q-Learning 而非 SARSA 的关键原因——off-policy 允许重用历史数据。
- **A3C 与 on-policy 并行训练**：A3C 通过并行多个 worker 来解决 on-policy 方法的数据效率问题。每个 worker 采集自己的轨迹并计算梯度，异步更新共享参数。这种方式在保持 on-policy 性质的同时提高了数据吞吐量。
- **Expected SARSA**：将 SARSA 中的 $Q(s', a')$ 替换为 $\mathbb{E}_\pi[Q(s', \cdot)] = \sum_a \pi(a|s') Q(s', a)$，得到 Expected SARSA。它降低了方差（因为消除了 $a'$ 采样的随机性），可以作为 on-policy 和 off-policy 之间的桥梁算法。
