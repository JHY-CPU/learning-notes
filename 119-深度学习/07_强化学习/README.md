# 强化学习

## 一、基础概念

### 1.1 马尔可夫决策过程（MDP）

强化学习问题通常建模为MDP，由五元组定义：

- **S**：状态空间
- **A**：动作空间
- **P**：状态转移概率 $P(s'|s,a)$
- **R**：奖励函数 $R(s,a,s')$
- **$\gamma$**：折扣因子（0到1之间）

### 1.2 核心要素

- **策略 $\pi(a|s)$**：在状态s下选择动作a的概率
- **价值函数 $V^\pi(s)$**：从状态s出发，按策略π执行的期望累积回报
- **动作价值函数 $Q^\pi(s,a)$**：在状态s执行动作a后，按策略π执行的期望累积回报
- **优势函数 $A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$**

### 1.3 Bellman方程

$$V^\pi(s) = \sum_a \pi(a|s) \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^\pi(s')]$$

$$V^*(s) = \max_a \sum_{s'} P(s'|s,a) [R(s,a,s') + \gamma V^*(s')]$$

---

## 二、值函数方法

### 2.1 Q-Learning

$$Q(s,a) \leftarrow Q(s,a) + \alpha [r + \gamma \max_{a'} Q(s',a') - Q(s,a)]$$

- Off-policy：学习最优策略，不依赖行为策略
- 收敛性有理论保证（表格情况下）

### 2.2 DQN（Deep Q-Network）

用神经网络近似Q函数：
- **经验回放**：打破样本相关性
- **目标网络**：稳定训练目标
- **改进**：Double DQN、Dueling DQN、Prioritized Experience Replay

---

## 三、策略梯度方法

### 3.1 策略梯度定理

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(a|s) \cdot Q^\pi(s,a)]$$

### 3.2 REINFORCE

$$\nabla_\theta J(\theta) = \mathbb{E}_\pi [\nabla_\theta \log \pi_\theta(a|s) \cdot G_t]$$

$G_t$ 为从时刻t开始的实际回报。

### 3.3 Actor-Critic

- **Actor**：策略网络，选择动作
- **Critic**：价值网络，评估动作好坏

### 3.4 PPO（Proximal Policy Optimization）

$$L^{CLIP}(\theta) = \mathbb{E}[\min(r_t(\theta) A_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) A_t)]$$

其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{old}}(a_t|s_t)$

PPO是目前最流行的策略梯度算法，稳定且高效。

---

## 四、RLHF（人类反馈强化学习）

### 4.1 流程

1. **收集人类偏好数据**：人类对模型输出进行排序
2. **训练奖励模型**：学习人类偏好的评分函数
3. **PPO优化策略**：用奖励模型指导语言模型优化

### 4.2 应用

- 对话系统对齐
- 减少有害输出
- 提高回答质量

代表工作：InstructGPT、ChatGPT、Claude
