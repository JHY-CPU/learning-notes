# 强化学习

## 一、基础

### 1.1 MDP

强化学习问题建模为马尔可夫决策过程 $(S, A, P, R, \gamma)$。

- **策略 $\pi(a|s)$**：状态到动作的映射
- **回报 $G_t = \sum_{k=0}^{\infty} \gamma^k r_{t+k+1}$**
- **价值函数 $V^\pi(s) = \mathbb{E}[G_t | S_t = s]$**

---

## 二、值函数方法

### 2.1 Q-Learning

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

Off-policy，不依赖行为策略。

### 2.2 DQN

用神经网络近似Q函数：
- 经验回放
- 目标网络
- Double DQN、Dueling DQN改进

---

## 三、策略梯度

### 3.1 REINFORCE

$$\nabla_\theta J = \mathbb{E}[\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$$

### 3.2 Actor-Critic

Actor选择动作，Critic评估价值。

### 3.3 PPO

剪辑策略目标，限制策略更新幅度，最流行的策略梯度算法。

---

## 四、应用

- 游戏AI（Atari、围棋、Dota）
- 机器人控制
- 推荐系统
- RLHF（大模型对齐）
