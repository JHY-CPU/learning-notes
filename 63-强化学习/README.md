# 61-强化学习

本目录系统整理强化学习（Reinforcement Learning, RL）的核心理论与算法，从基础的马尔可夫决策过程到深度强化学习的前沿方法，覆盖完整的学习路线。

## 目录结构

| 序号 | 子目录 | 主要内容 |
|------|--------|----------|
| 01 | [MDP与值迭代](./01-MDP与值迭代/) | MDP形式化定义、值函数、Bellman方程、策略迭代、值迭代 |
| 02 | [时序差分与Q学习](./02-时序差分与Q学习/) | 蒙特卡洛方法、TD(0)、Sarsa、Q-Learning、资格迹、探索策略 |
| 03 | [策略梯度与Actor-Critic](./03-策略梯度与Actor-Critic/) | 策略梯度定理、REINFORCE、A2C/A3C、PPO、TRPO、DDPG、SAC |
| 04 | [深度强化学习](./04-深度强化学习/) | DQN及其变体、多智能体RL、模仿学习、AlphaGo/AlphaZero、工程实践 |

## 学习路线建议

1. **基础阶段**：先掌握MDP与动态规划（01），理解值函数和Bellman方程的核心思想。
2. **无模型方法**：学习TD学习和Q-Learning（02），掌握在不依赖环境模型情况下的学习方法。
3. **策略优化**：进入策略梯度与Actor-Critic（03），理解直接优化策略的方法和现代策略优化算法。
4. **深度学习结合**：最后学习深度强化学习（04），了解如何用神经网络逼近值函数和策略。

## 核心概念速查

- **强化学习**：智能体通过与环境交互，基于奖励信号学习最优行为策略的机器学习范式。
- **MDP**：马尔可夫决策过程，强化学习问题的标准数学框架。
- **值函数**：评估状态或状态-动作对的长期价值。
- **策略梯度**：直接参数化策略并沿梯度方向优化。
- **深度强化学习**：用深度神经网络逼近值函数或策略，解决高维状态/动作空间问题。

## 参考资料

- Sutton, R. S., & Barto, A. G. *Reinforcement Learning: An Introduction* (2nd Edition)
- Silver, D. *UCL Course on Reinforcement Learning*
- OpenAI Spinning Up in Deep RL
- Stable-Baselines3 文档
