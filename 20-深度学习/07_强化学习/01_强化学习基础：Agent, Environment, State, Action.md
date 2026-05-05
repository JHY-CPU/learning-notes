# 01_强化学习基础：Agent, Environment, State, Action

## 核心概念

- **Agent（智能体）**：强化学习中的决策者，它观察环境状态并选择动作以最大化累计奖励。智能体是学习算法的载体，包含策略（policy）、价值函数（value function）和模型（model）三个核心组件。
- **Environment（环境）**：智能体交互的外部系统，定义了状态转移规则和奖励函数。环境可以是确定的（如网格世界）或随机的（如赌场老虎机），通常建模为马尔可夫决策过程（MDP）。
- **State（状态）**：环境在某一时刻的描述，包含智能体决策所需的所有信息。状态可以是完全可观测的（fully observable）或部分可观测的（partially observable），后者对应 POMDP 框架。
- **Action（动作）**：智能体在给定状态下可以执行的操作。动作空间可以是离散的（如左/右/上/下）或连续的（如机器人关节力矩），不同动作空间需要不同的算法处理。
- **奖励（Reward）**：环境对智能体动作的即时反馈信号，是强化学习优化的核心目标。奖励函数的设计直接决定了智能体学到的行为模式。
- **策略（Policy）**：智能体的行为函数，定义为从状态到动作的映射。策略分为确定性策略 $\pi(s) = a$ 和随机策略 $\pi(a|s) = P(A_t=a|S_t=s)$。
- **轨迹（Trajectory/Episode）**：智能体与环境交互产生的状态-动作-奖励序列 $S_0, A_0, R_1, S_1, A_1, R_2, ...$，是强化学习经验数据的基本单位。
- **交互循环**：在每个时间步，Agent 观察状态 $S_t$，选择动作 $A_t$，环境反馈奖励 $R_{t+1}$ 和新状态 $S_{t+1}$，循环往复直到终止。

## 数学推导

$$
\text{策略: } \pi(a|s) = \Pr(A_t = a \mid S_t = s)
$$

$$
\text{轨迹: } \tau = (S_0, A_0, R_1, S_1, A_1, R_2, ..., S_{T-1}, A_{T-1}, R_T, S_T)
$$

$$
\text{累计回报: } G_t = \sum_{k=0}^{\infty} \gamma^k R_{t+k+1}
$$

$$
\text{强化学习目标: } \max_\pi \mathbb{E}_{\tau \sim \pi} \left[ \sum_{t=0}^\infty \gamma^t R_{t+1} \right]
$$

**关键说明**：
- 随机策略 $\pi(a|s)$ 输出的是动作概率分布，而非直接动作值，这保证了探索的多样性。
- 轨迹中的 $S_{t+1}$ 仅依赖于 $S_t$ 和 $A_t$（马尔可夫性质），与历史状态无关。
- 累计回报 $G_t$ 加入了折扣因子 $\gamma \in [0,1]$，平衡近期奖励与远期奖励的重要性。
- 智能体的最终目标是找到最优策略 $\pi^*$，使得每个状态下的期望回报最大化。

## 直观理解

强化学习可以类比为训练一只宠物狗：
- **Agent（狗）** 需要学会在特定情境下做出正确行为。
- **Environment（家庭环境）** 包括客厅、院子、狗窝等，提供了狗行动的空间。
- **State（情境）** 包括"主人拿着零食"、"门外有陌生人"、"到吃饭时间了"等。
- **Action（动作）** 包括"坐下"、"趴下"、"叫"、"摇尾巴"等。
- **Reward（奖励）** 是狗粮或抚摸——好的行为得到奖励，坏的行为被忽视。

关键区别在于：狗的训练通常由主人明确指导（监督学习），而强化学习中 Agent 必须通过反复试错（trial and error）自己发现哪些动作会带来奖励。这就好比让狗自己摸索——"坐下"后有零食，"乱叫"后没有——最终它自己学会了坐下的好处。

## 代码示例

```python
import gymnasium as gym
import numpy as np

# 创建 CartPole 环境
env = gym.make('CartPole-v1')

# 重置环境，获取初始状态
state, info = env.reset()
print(f"初始状态: {state}")
# 输出示例: [杆子角度, 角速度, 小车位置, 小车速度]

# Agent-Environment 交互循环
total_reward = 0
for t in range(200):
    # Agent 选择动作（随机策略）
    action = env.action_space.sample()
    
    # 环境执行动作，返回新状态和奖励
    next_state, reward, terminated, truncated, info = env.step(action)
    
    total_reward += reward
    print(f"Step {t}: 状态={state}, 动作={action}, 奖励={reward}, 下一状态={next_state}")
    
    state = next_state
    
    if terminated or truncated:
        print(f"Episode 结束, 总奖励 = {total_reward}")
        break

env.close()
```

## 深度学习关联

- **深度强化学习基础**：深度神经网络充当强化学习中的函数近似器，解决了"维度灾难"问题。传统强化学习依赖表格来存储状态价值，但当状态空间巨大（如围棋 $10^{170}$ 种状态）时，必须使用深度网络进行泛化。
- **端到端学习**：深度学习使 Agent 可以直接从原始输入（如图像像素）学习策略，无需手工设计状态特征。DQN 直接从游戏画面像素学习玩 Atari 游戏，是现代深度强化学习的里程碑。
- **表示学习**：深度网络在训练过程中自动学习到有用的状态表示（representation），将高维感官输入压缩为低维语义特征，这比传统的手工特征工程更具表达力。
