# 07_强化学习

> 从MDP到PPO，系统地覆盖强化学习的理论基础与算法实现。包括基于价值的方法（DQN系列）、基于策略的方法（Policy Gradient、PPO）、Actor-Critic框架以及前沿的多智能体与稀疏奖励问题。

---

## 基础知识

- **前置知识**：03_神经网络核心; 01_数学基础（概率、优化）
- **关联目录**：05_NLP与序列模型（RLHF应用）; 04_计算机视觉（机器人视觉）
- **笔记数量**：共 30 篇

---

## 内容结构

#### RL基础与规划

Agent/MDP、回报与折扣、价值函数、贝尔曼方程、策略/价值迭代

| 编号 | 笔记 |
|------|------|
| 01 | [强化学习基础：Agent, Environment, State, Action](01_强化学习基础：Agent, Environment, State, Action.md) |
| 02 | [马尔可夫决策过程 (MDP) 定义与性质](02_马尔可夫决策过程 (MDP) 定义与性质.md) |
| 03 | [回报 (Return) 与折扣因子 (Gamma)](03_回报 (Return) 与折扣因子 (Gamma).md) |
| 04 | [价值函数 (Value Function) 与状态-动作价值 (Q-function)](04_价值函数 (Value Function) 与状态-动作价值 (Q-function).md) |
| 05 | [贝尔曼期望方程 (Bellman Expectation Equation)](05_贝尔曼期望方程 (Bellman Expectation Equation).md) |
| 06 | [贝尔曼最优方程 (Bellman Optimality Equation)](06_贝尔曼最优方程 (Bellman Optimality Equation).md) |
| 07 | [策略迭代 (Policy Iteration) 算法流程](07_策略迭代 (Policy Iteration) 算法流程.md) |
| 08 | [价值迭代 (Value Iteration) 算法流程](08_价值迭代 (Value Iteration) 算法流程.md) |

#### 无模型学习

蒙特卡洛方法、TD学习、SARSA、Q-Learning、Epsilon-Greedy、UCB

| 编号 | 笔记 |
|------|------|
| 09 | [蒙特卡洛方法 (MC) 与首次访问及每次访问](09_蒙特卡洛方法 (MC) 与首次访问及每次访问.md) |
| 10 | [时序差分学习 (TD Learning)：TD(0)](10_时序差分学习 (TD Learning)：TD(0).md) |
| 11 | [SARSA 算法：On-policy TD 控制](11_SARSA 算法：On-policy TD 控制.md) |
| 12 | [Q-Learning 算法：Off-policy TD 控制](12_Q-Learning 算法：Off-policy TD 控制.md) |
| 13 | [探索与利用 (Exploration vs Exploitation)：Epsilon-Greedy](13_探索与利用 (Exploration vs Exploitation)：Epsilon-Greedy.md) |
| 14 | [UCB (Upper Confidence Bound) 探索策略](14_UCB (Upper Confidence Bound) 探索策略.md) |

#### 深度Q网络

DQN、经验回放、目标网络、Double DQN、Dueling DQN

| 编号 | 笔记 |
|------|------|
| 15 | [深度 Q 网络 (DQN) 架构详解](15_深度 Q 网络 (DQN) 架构详解.md) |
| 16 | [DQN 经验回放 (Experience Replay) 机制](16_DQN 经验回放 (Experience Replay) 机制.md) |
| 17 | [DQN 目标网络 (Target Network) 稳定性分析](17_DQN 目标网络 (Target Network) 稳定性分析.md) |
| 18 | [Double DQN：解决过估计问题](18_Double DQN：解决过估计问题.md) |
| 19 | [Dueling DQN：分离价值与优势流](19_Dueling DQN：分离价值与优势流.md) |

#### 策略梯度与进阶

Policy Gradient定理、REINFORCE、Actor-Critic、A2C/A3C、PPO、TRPO、SAC、TD3

| 编号 | 笔记 |
|------|------|
| 20 | [Policy Gradient 定理推导](20_Policy Gradient 定理推导.md) |
| 21 | [REINFORCE 算法实现细节](21_REINFORCE 算法实现细节.md) |
| 22 | [Actor-Critic 框架基础](22_Actor-Critic 框架基础.md) |
| 23 | [A2C (Advantage Actor-Critic) 同步训练](23_A2C (Advantage Actor-Critic) 同步训练.md) |
| 24 | [A3C (Asynchronous Advantage Actor-Critic)](24_A3C (Asynchronous Advantage Actor-Critic).md) |
| 25 | [PPO (Proximal Policy Optimization) 截断式重要性采样](25_PPO (Proximal Policy Optimization) 截断式重要性采样.md) |
| 26 | [TRPO (Trust Region Policy Optimization) 约束优化](26_TRPO (Trust Region Policy Optimization) 约束优化.md) |
| 27 | [SAC (Soft Actor-Critic) 最大熵强化学习](27_SAC (Soft Actor-Critic) 最大熵强化学习.md) |
| 28 | [TD3 (Twin Delayed DDPG) 改进策略](28_TD3 (Twin Delayed DDPG) 改进策略.md) |

#### 高级主题

稀疏奖励、课程学习、多智能体RL

| 编号 | 笔记 |
|------|------|
| 29 | [稀疏奖励问题与课程学习 (Curriculum Learning)](29_稀疏奖励问题与课程学习 (Curriculum Learning).md) |
| 30 | [多智能体强化学习 (MARL) 基础](30_多智能体强化学习 (MARL) 基础.md) |
---

## 学习建议

1. 按编号顺序阅读每个子主题内的笔记，因为内部存在递进关系
2. 每个子主题完成后，尝试用「深度学习关联」部分串联知识点
3. 代码示例可以直接复制运行（需要 PyTorch 和 transformers 库）
4. 遇到数学推导不熟悉时，回到 01_数学基础 查阅对应基础

---

*本 README 由笔记元数据自动生成。*
