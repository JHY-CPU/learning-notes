# 12_Q-Learning 算法：Off-policy TD 控制

## 核心概念

- **Q-Learning**：由 Watkins (1989) 提出的突破性 off-policy TD 控制算法。它直接学习最优动作价值函数 $Q^*$，独立于智能体正在执行的策略。
- **Off-policy 本质**：行为策略（behavior policy，用于生成数据）和目标策略（target policy，被评估和改进的策略）相互分离。Q-Learning 总是使用 $\max_a Q(s', a)$ 作为目标，意味着目标策略是贪心策略。
- **Q-Learning 更新规则**：$Q(s, a) \leftarrow Q(s, a) + \alpha[R_{t+1} + \gamma \max_a Q(s', a) - Q(s, a)]$。关键的 $\max$ 操作使算法能直接从探索性行为中学习最优策略。
- **收敛保证**：在有限 MDP 中，当每个状态-动作对被无限频繁访问且学习率满足 Robbins-Monro 条件（$\sum \alpha_t = \infty, \sum \alpha_t^2 < \infty$）时，Q-Learning 以概率 1 收敛到 $Q^*$。
- **Off-policy 的优势**：可以安全地分离探索和学习。行为策略可以保持高探索性（如 $\epsilon$-greedy 较大的 $\epsilon$），而目标策略始终保持贪心。
- **Q-Learning 的激进性**：因为总是使用 $\max$ 操作，Q-Learning 倾向于高估 Q 值（maximization bias），尤其在早期学习阶段。Double Q-Learning 和 Double DQN 针对此问题做了改进。

## 数学推导

$$
\text{Q-Learning 更新公式: }
Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \left[ R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a) - Q(S_t, A_t) \right]
$$

$$
\text{TD 目标（off-policy）: } \mathcal{T}_{\text{Q}} = R_{t+1} + \gamma \max_{a} Q(S_{t+1}, a)
$$

$$
\text{收敛条件（Robbins-Monro）: } \sum_{t=1}^\infty \alpha_t = \infty, \quad \sum_{t=1}^\infty \alpha_t^2 < \infty
$$

$$
\text{Q-Learning 对应的贝尔曼最优算子: }
$$

$$
(\mathcal{T}^* Q)(s, a) = R(s, a) + \gamma \sum_{s' \in \mathcal{S}} P(s'|s,a) \max_{a'} Q(s', a')
$$

**推导说明**：
- Q-Learning 的关键创新是 off-policy 的 $\max$ 操作：目标 $r + \gamma \max_a Q(s', a)$ 不依赖于行为策略选择的动作。
- $\mathcal{T}^*$ 是收缩映射（压缩系数 $\gamma$），保证了 Q-Learning 收敛到不动点 $Q^*$。
- 实际更新中，行为策略通常使用 $\epsilon$-greedy，目标策略是 $\arg\max_a Q(s, a)$——两者不同，正是 off-policy 的定义。

## 直观理解

Q-Learning 就像一个"纸上谈兵"但高效的战略家：

**在岔路口，你选了左边的路**（行为策略在探索），但 Q-Learning 在更新时想的是：**"如果我在下一个路口选了最好的路会怎样？"**（目标策略用 $\max$）。

这意味着即使你因为"想探索一下"而走了右转，Q-Learning 在更新右转决策时仍然使用"如果下一步我走最佳路线能拿多少分"来评估。这就是为什么说 Q-Learning **"从别人的错误中学习"**——它不需要亲自走最优路线来学习最优路线。

**SARSA vs Q-Learning 的经典悬崖漫步例子**：
- 悬崖边有条窄路，一边是悬崖（掉下去扣 100 分），一边是安全的草坡。
- **SARSA**：因为有时候会"手滑"（$\epsilon$-greedy 探索），它学会离悬崖远一点——它把探索可能掉下悬崖的风险也纳入了评估。
- **Q-Learning**：知道理论上贴着悬崖走最快（最优策略），就勇敢地贴着悬崖走了——它假设自己能在关键时刻做出最优选择。

所以 Q-Learning 学到的策略是"纸面上最优的"，而 SARSA 学到的策略是"实际执行中最安全的"。

## 代码示例

```python
import numpy as np
import gymnasium as gym

def q_learning(env, num_episodes=10000, alpha=0.1, gamma=0.99, epsilon=0.1):
    nS = env.observation_space.n
    nA = env.action_space.n
    Q = np.zeros((nS, nA))
    
    episode_rewards = []
    
    for ep in range(num_episodes):
        state, info = env.reset()
        total_reward = 0
        terminated = truncated = False
        
        while not (terminated or truncated):
            # 行为策略: epsilon-greedy
            if np.random.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state])
            
            next_state, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            # Q-Learning 更新: Q(s,a) += alpha[r + gamma*max_a' Q(s',a') - Q(s,a)]
            td_target = reward + gamma * np.max(Q[next_state]) * (not terminated)
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            
            state = next_state
        
        episode_rewards.append(total_reward)
    
    return Q, episode_rewards

# Cliff Walking 对比（需配合 SARSA 实现使用）
# SARSA 是 on-policy，倾向于走远离悬崖的安全路径
# Q-Learning 是 off-policy，倾向于走最优（最短但危险）路径

env = gym.make("FrozenLake-v1", is_slippery=False)
Q, rewards = q_learning(env, num_episodes=2000)
print(f"Q-Learning 平均奖励（后 500ep）: {np.mean(rewards[-500:]):.2f}")
env.close()
```

## 深度学习关联

- **DQN = Q-Learning + 深度网络**：DQN (Deep Q-Network) 本质上是将 Q-Learning 的表格 Q 函数替换为深度神经网络。DQN 的损失函数 $L(\theta) = \mathbb{E}[(r + \gamma \max_{a'} Q(s', a'; \theta^-) - Q(s, a; \theta))^2]$ 与 Q-Learning 的更新目标完全一致。
- **Off-policy 学习与经验回放**：Q-Learning 的 off-policy 特性是经验回放（Experience Replay）得以成功的关键。因为目标 $r + \gamma \max Q$ 不依赖行为策略，历史经验可以被反复使用。这使得深度学习可以在大规模数据上通过 mini-batch 训练。
- **Double Q-Learning 到 Double DQN**：Q-Learning 的 $\max$ 操作会导致正偏差（overestimation bias）。Double Q-Learning 用两套 Q 表解耦选择和评估，Double DQN 将其推广到深度网络：使用在线网络选动作，目标网络评估价值。
