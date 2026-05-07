# 04-深度强化学习

## 1. DQN（Deep Q-Network）

### 1.1 核心思想

DQN由DeepMind于2013年提出（Nature 2015），首次成功将深度学习与Q-Learning结合，在Atari游戏上达到人类水平。用深度神经网络 $Q(s,a;\theta)$ 逼近Q值函数，解决高维状态空间（如图像输入）的问题。

### 1.2 经验回放（Experience Replay）

将转移 $(s, a, r, s')$ 存储在回放缓冲区 $D$ 中，训练时随机采样minibatch：

$$L(\theta) = \mathbb{E}_{(s,a,r,s') \sim D}\left[(r + \gamma \max_{a'} Q(s',a';\theta^-) - Q(s,a;\theta))^2\right]$$

经验回放的作用：
- **打破数据相关性**：随机采样消除时序相关性
- **提高数据效率**：每条经验可被多次使用
- **稳定训练**：平滑数据分布

### 1.3 目标网络（Target Network）

使用延迟更新的目标网络参数 $\theta^-$ 计算TD目标，每隔 $C$ 步才同步一次：

$$\theta^- \leftarrow \theta$$

目标网络解决了"追逐移动目标"的问题——如果不使用目标网络，TD目标和当前Q网络同时变化，导致训练不稳定。

### 1.4 DQN算法流程

```
初始化回放缓冲区 D，Q网络参数 θ，目标网络参数 θ^- ← θ

对每个回合:
    初始化状态 s（预处理为4帧堆叠）
    重复每一步:
        以概率 ε 选随机动作，否则 a = argmax_a Q(s,a;θ)
        执行 a，观察 r, s'
        存储 (s, a, r, s') 到 D
        从 D 采样 minibatch (s_i, a_i, r_i, s'_i)
        计算目标: y_i = r_i + γ·max_a Q(s'_i, a; θ^-)
        更新 θ 最小化 (y_i - Q(s_i, a_i; θ))^2
        每 C 步: θ^- ← θ
    直到回合结束
```

## 2. DQN变体

### 2.1 Double DQN

标准DQN存在过估计问题：max操作会系统性高估Q值。Double DQN解耦动作选择和价值评估：

$$Y_t = R_{t+1} + \gamma Q(S_{t+1}, \arg\max_a Q(S_{t+1}, a; \theta); \theta^-)$$

用在线网络选择动作，用目标网络评估价值。

### 2.2 Dueling DQN

将Q值分解为状态价值和优势函数：

$$Q(s,a;\theta,\alpha,\beta) = V(s;\theta,\beta) + A(s,a;\theta,\alpha) - \frac{1}{|A|}\sum_{a'}A(s,a';\theta,\alpha)$$

减去均值保证可识别性。在动作多但很多动作价值相近时特别有效。

### 2.3 Prioritized Experience Replay

根据TD误差的大小给经验赋予不同的采样优先级：

$$P(i) \propto |\delta_i|^\omega + \epsilon$$

TD误差大的样本（"惊喜"程度高的经验）被更频繁地采样，提高学习效率。需要重要性采样权重修正偏差。

### 2.4 Noisy Nets

用带参数噪声的线性层替代ε-greedy的随机探索：

$$y = (W + \sigma_W \odot \epsilon_W)x + (b + \sigma_b \odot \epsilon_b)$$

参数 $\sigma$ 通过梯度下降学习，探索程度随学习自适应调节。

### 2.5 Distributional RL

学习Q值的完整分布而非仅期望值：

- **C51**：将值分布离散化为51个原子，学习每个原子的概率
- **QR-DQN（Quantile Regression DQN）**：学习分位数而非固定原子
- 提供更丰富的梯度信号，帮助探索

### 2.6 Rainbow DQN

Rainbow将六种改进组合在一起：
1. Double DQN
2. Dueling DQN
3. Prioritized Experience Replay
4. Multi-step Learning（n步回报）
5. Distributional RL（C51）
6. Noisy Nets

实验证明每种改进都有独立贡献，组合后效果显著提升。

## 3. 连续动作空间的处理

Q-Learning类方法在连续动作空间面临 $\max_a Q(s,a)$ 难以计算的问题。解决方案包括：

- **离散化**：将连续空间划分为网格，但维度灾难严重
- **DDPG/SAC**：Actor-Critic架构，Actor提供连续动作
- **NAF（Normalized Advantage Functions）**：假设Q函数为动作的二次型，可解析求max
- **动作空间搜索**：在采样后用CMA-ES等优化方法近似求解

## 4. 多智能体强化学习基础

### 4.1 独立学习（Independent Learning）

每个智能体独立运行各自的RL算法，将其他智能体视为环境的一部分。

优点：简单直接，可复用单智能体算法。
缺点：环境非平稳（其他智能体也在学习），不收敛保证丧失。

### 4.2 集中式训练分布式执行（CTDE）

训练时可利用全局信息（所有智能体的观测和动作），执行时每个智能体只依赖自身观测。

核心思想：
- 训练时的Critic使用全局信息，提高学习效率
- 执行时的Actor仅依赖局部观测，保持分布式

### 4.3 MADDPG（Multi-Agent DDPG）

MADDPG将CTDE思想应用于连续动作空间：

- 每个智能体 $i$ 有自己的Actor $\mu_i$ 和Critic $Q_i$
- Critic $Q_i(o_1, \ldots, o_n, a_1, \ldots, a_n)$ 使用所有智能体的观测和动作
- Actor $\mu_i(o_i)$ 仅使用自己的观测

## 5. 模仿学习与逆强化学习

### 5.1 模仿学习（Imitation Learning）

**行为克隆（Behavioral Cloning）**：直接用专家演示数据训练策略，等价于监督学习。

缺点：分布偏移（Distribution Shift）——策略在训练中未见过的状态下会累积错误。

**DAgger（Dataset Aggregation）**：迭代收集策略自身产生的数据并标注，缓解分布偏移。

### 5.2 逆强化学习（Inverse RL）

从专家行为中推断奖励函数，再用推断的奖励函数训练策略。

常见方法：最大熵逆RL、GAIL（生成对抗模仿学习）。

GAIL将模仿学习建模为生成对抗框架：策略生成器模仿专家，判别器区分专家和策略生成的数据。

## 6. AlphaGo/AlphaZero的算法原理

### 6.1 AlphaGo

结合深度学习与蒙特卡洛树搜索（MCTS）：

1. **监督预训练**：用人类棋谱训练策略网络
2. **自我对弈强化学习**：策略网络通过自我对弈进一步提升
3. **价值网络**：评估棋盘局面的胜率
4. **MCTS**：结合策略网络（先验）和价值网络（评估）进行搜索

### 6.2 AlphaZero

AlphaGo的进化版，完全从零开始学习：

- 不使用人类棋谱，纯自我对弈
- 统一的网络同时输出策略和价值
- 通过MCTS与神经网络的迭代训练循环提升

关键创新：MCTS提供比原始神经网络更精确的策略，用MCTS的改进策略作为训练目标，形成"蒸馏-搜索"的迭代提升。

## 7. 强化学习的工程挑战

### 7.1 样本效率

深度RL通常需要大量环境交互（百万到亿级step），实际应用中数据收集成本高。

应对策略：
- 经验回放和数据增强
- 模型学习（Model-based RL）减少真实交互
- 迁移学习和预训练

### 7.2 奖励设计

稀疏奖励（如仅在最终成功时有奖励）导致学习困难。

应对策略：
- 奖励塑形（Reward Shaping）：添加中间奖励引导探索
- 课程学习（Curriculum Learning）：从简单任务逐步增加难度
- 内在奖励（Intrinsic Motivation）：基于好奇心或计数的探索奖励
- 逆强化学习：从专家数据中学习奖励函数

### 7.3 训练不稳定

深度RL训练过程不稳定，超参数敏感。

应对策略：
- 目标网络和软更新
- 梯度裁剪
- 学习率调度
- 多次重复实验取平均

## 8. 强化学习框架

### 8.1 OpenAI Gym / Gymnasium

标准RL环境接口，提供统一的 `step()`、`reset()` API，包含经典控制问题和Atari游戏等环境。

### 8.2 Stable-Baselines3

基于PyTorch的高质量RL算法实现库，提供PPO、SAC、TD3、A2C等算法的开箱即用实现，文档完善，便于研究和应用。

### 8.3 RLlib（Ray）

分布式RL框架，支持大规模并行训练，集成多种算法，适合工业级应用场景。基于Ray分布式计算框架，支持GPU集群训练。

### 8.4 其他常用框架

- **CleanRL**：单文件、可复现的RL实现
- **Tianshou**：PyTorch原生RL库
- **Acme**（DeepMind）：模块化RL研究框架
- **MuZero**相关实现：可用于棋类和Atari
