import os

titles = [
    "01_强化学习基础：Agent, Environment, State, Action",
    "02_马尔可夫决策过程 (MDP) 定义与性质",
    "03_回报 (Return) 与折扣因子 (Gamma)",
    "04_价值函数 (Value Function) 与状态-动作价值 (Q-function)",
    "05_贝尔曼期望方程 (Bellman Expectation Equation)",
    "06_贝尔曼最优方程 (Bellman Optimality Equation)",
    "07_策略迭代 (Policy Iteration) 算法流程",
    "08_价值迭代 (Value Iteration) 算法流程",
    "09_蒙特卡洛方法 (MC) 与首次访问及每次访问",
    "10_时序差分学习 (TD Learning)：TD(0)",
    "11_SARSA 算法：On-policy TD 控制",
    "12_Q-Learning 算法：Off-policy TD 控制",
    "13_探索与利用 (Exploration vs Exploitation)：Epsilon-Greedy",
    "14_UCB (Upper Confidence Bound) 探索策略",
    "15_深度 Q 网络 (DQN) 架构详解",
    "16_DQN 经验回放 (Experience Replay) 机制",
    "17_DQN 目标网络 (Target Network) 稳定性分析",
    "18_Double DQN：解决过估计问题",
    "19_Dueling DQN：分离价值与优势流",
    "20_Policy Gradient 定理推导",
    "21_REINFORCE 算法实现细节",
    "22_Actor-Critic 框架基础",
    "23_A2C (Advantage Actor-Critic) 同步训练",
    "24_A3C (Asynchronous Advantage Actor-Critic)",
    "25_PPO (Proximal Policy Optimization) 截断式重要性采样",
    "26_TRPO (Trust Region Policy Optimization) 约束优化",
    "27_SAC (Soft Actor-Critic) 最大熵强化学习",
    "28_TD3 (Twin Delayed DDPG) 改进策略",
    "29_稀疏奖励问题与课程学习 (Curriculum Learning)",
    "30_多智能体强化学习 (MARL) 基础"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
