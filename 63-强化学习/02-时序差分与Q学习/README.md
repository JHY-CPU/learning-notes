# 02-时序差分与Q学习

## 1. 蒙特卡洛方法（MC）

### 1.1 基本思想

蒙特卡洛方法通过采样完整的回合（Episode），用实际回报的平均值来估计值函数，不依赖环境模型。

$$V(s) \leftarrow V(s) + \alpha[G_t - V(s)]$$

其中 $G_t$ 是从时刻 $t$ 开始的实际累积回报。

### 1.2 首次访问MC vs 每次访问MC

**首次访问MC（First-Visit MC）**：
- 每个回合中，仅在某状态第一次出现时记录回报
- 对每个状态的所有首次回报取平均
- 无偏估计，理论性质更好

**每次访问MC（Every-Visit MC）**：
- 每个回合中，状态每次出现都记录回报
- 实现更简单，但在小样本下有偏差
- 随着样本增多，两种方法渐近收敛

### 1.3 MC的优缺点

**优点**：
- 无偏估计：使用真实回报，不引入偏差
- 不需要环境模型
- 对马尔可夫性要求较低

**缺点**：
- 必须等到回合结束才能更新，不适合持续性任务
- 方差大：单条轨迹的回报波动很大
- 收敛速度慢：只能逐回合更新

## 2. 时序差分学习（TD Learning）

### 2.1 TD(0)更新规则

TD学习结合了MC的采样思想和DP的自举（Bootstrapping）思想：

$$V(S_t) \leftarrow V(S_t) + \alpha[\underbrace{R_{t+1} + \gamma V(S_{t+1})}_{\text{TD目标}} - V(S_t)]$$

其中 $R_{t+1} + \gamma V(S_{t+1})$ 称为**TD目标**，$R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$ 称为**TD误差**（TD Error），记为 $\delta_t$。

TD学习的"自举"含义：用当前对下一状态值函数的估计来更新当前状态的估计。

### 2.2 MC vs TD：偏差-方差权衡

| 特性 | 蒙特卡洛 | 时序差分 |
|------|----------|----------|
| 偏差 | 无偏 | 有偏（因自举） |
| 方差 | 高（完整回报波动大） | 低（单步奖励波动小） |
| 更新时机 | 回合结束 | 每步更新 |
| 收敛目标 | $V^\pi$ | $V^\pi$（在适当条件下） |
| 马尔可夫性 | 不依赖 | 依赖 |

- **MC** 方差高但无偏，适合非马尔可夫任务。
- **TD** 有偏但方差低，利用马尔可夫性收敛更快。

### 2.3 TD(0)算法

```
初始化 V(s)，对所有 s

对每个回合:
    初始化 S
    重复每一步:
        选择动作 A（如 ε-greedy）
        执行 A，观察 R, S'
        V(S) ← V(S) + α[R + γV(S') - V(S)]
        S ← S'
    直到 S 为终止状态
```

## 3. Sarsa算法（On-Policy TD控制）

### 3.1 Sarsa的更新规则

Sarsa（State-Action-Reward-State-Action）是on-policy的TD控制算法，使用当前策略选择的下一个动作来更新Q值：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma Q(S_{t+1}, A_{t+1}) - Q(S_t, A_t)]$$

名称来源于使用的五元组 $(S_t, A_t, R_{t+1}, S_{t+1}, A_{t+1})$。

```
初始化 Q(s, a)，对所有 s, a

对每个回合:
    初始化 S，选择 A（ε-greedy）
    重复每一步:
        执行 A，观察 R, S'
        选择 A'（ε-greedy 基于 Q）
        Q(S, A) ← Q(S, A) + α[R + γQ(S', A') - Q(S, A)]
        S ← S', A ← A'
    直到 S 为终止状态
```

### 3.2 Sarsa(λ)与资格迹

Sarsa(λ)通过资格迹（Eligibility Traces）实现多步回报的加权混合：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha \delta_t e_t(S_t, A_t)$$

- $\lambda = 0$：退化为TD(0)
- $\lambda = 1$：接近MC方法
- 中间值平衡偏差和方差

## 4. Q-Learning算法（Off-Policy TD控制）

### 4.1 Q-Learning更新规则

Q-Learning是经典的off-policy算法，使用贪婪策略选择目标，但行为策略可以不同：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha[R_{t+1} + \gamma \max_{a'} Q(S_{t+1}, a') - Q(S_t, A_t)]$$

关键区别：Sarsa用实际采取的下一个动作 $A_{t+1}$ 的Q值，而Q-Learning用最优动作的Q值 $\max_{a'} Q(S_{t+1}, a')$。

```
初始化 Q(s, a)，对所有 s, a

对每个回合:
    初始化 S
    重复每一步:
        选择 A（ε-greedy 基于 Q）
        执行 A，观察 R, S'
        Q(S, A) ← Q(S, A) + α[R + γ·max_a Q(S', a) - Q(S, A)]
        S ← S'
    直到 S 为终止状态
```

### 4.2 Q-Learning的收敛性

**Watkins定理**：如果满足以下条件，Q-Learning保证收敛到最优Q值：
1. 状态和动作空间有限
2. 所有状态-动作对被无限次访问
3. 学习率满足 Robbins-Monro 条件：$\sum_t \alpha_t = \infty$，$\sum_t \alpha_t^2 < \infty$

### 4.3 Sarsa vs Q-Learning的关键差异

在悬崖行走问题中：
- **Sarsa**（on-policy）会选择远离悬崖的安全路径
- **Q-Learning**（off-policy）学习到最优路径但可能在执行中冒险

## 5. Expected Sarsa

Expected Sarsa用策略下动作值的期望代替Sarsa中的采样：

$$Q(S_t, A_t) \leftarrow Q(S_t, A_t) + \alpha\left[R_{t+1} + \gamma \sum_a \pi(a|S_{t+1}) Q(S_{t+1}, a) - Q(S_t, A_t)\right]$$

优势：
- 消除了选择 $A_{t+1}$ 的随机性，方差更小
- 当策略为贪婪策略时，Expected Sarsa等价于Q-Learning
- 比Sarsa更稳定，计算开销增加有限

## 6. 资格迹（Eligibility Traces）

### 6.1 基本概念

资格迹是一种有效计算多步回报的机制，不需要存储完整轨迹。

### 6.2 前向视角（Forward View）

TD(λ)的目标是所有n步回报的加权平均：

$$G_t^\lambda = (1-\lambda)\sum_{n=1}^{\infty} \lambda^{n-1} G_t^{(n)}$$

其中 $G_t^{(n)}$ 是n步回报。前向视角理论上清晰但需要未来信息，不易在线实现。

### 6.3 后向视角（Backward View）

后向视角使用资格迹 $e_t(s)$ 实现在线更新：

**资格迹更新**（替换迹）：
$$e_t(s) = \begin{cases} \gamma \lambda e_{t-1}(s) + 1 & \text{if } s = S_t \\ \gamma \lambda e_{t-1}(s) & \text{otherwise} \end{cases}$$

**值函数更新**：
$$\delta_t = R_{t+1} + \gamma V(S_{t+1}) - V(S_t)$$
$$V(s) \leftarrow V(s) + \alpha \delta_t e_t(s), \quad \forall s$$

后向视角适用于在线增量更新，是实际实现的首选。

## 7. 探索与利用（Exploration vs Exploitation）

### 7.1 ε-greedy

以概率 $\epsilon$ 随机选择动作（探索），以概率 $1-\epsilon$ 选择当前最优动作（利用）：

$$A_t = \begin{cases} \text{uniform}(A) & \text{with prob } \epsilon \\ \arg\max_a Q(S_t, a) & \text{with prob } 1-\epsilon \end{cases}$$

常见策略：$\epsilon$ 从较高值逐渐衰减到较小值。

### 7.2 UCB（上置信界）

UCB鼓励探索那些访问次数少或不确定性高的动作：

$$A_t = \arg\max_a \left[Q_t(a) + c\sqrt{\frac{\ln t}{N_t(a)}}\right]$$

- $c$ 控制探索的程度
- 访问次数 $N_t(a)$ 越少，探索奖励越大
- 理论上比ε-greedy有更好的遗憾界

### 7.3 乐观初始值

将所有Q值初始化为较大的值（高于真实最优值），自然鼓励智能体先探索那些还未被访问过的动作。

优点是简单有效，缺点是不适合未知上界的问题。

### 7.4 Softmax/Boltzmann探索

根据Q值的大小按概率选择动作：

$$\pi(a|s) = \frac{\exp(Q(s,a)/\tau)}{\sum_{a'} \exp(Q(s,a')/\tau)}$$

- 温度参数 $\tau$ 控制探索程度
- $\tau \to \infty$：均匀随机（纯探索）
- $\tau \to 0$：贪婪策略（纯利用）
- 比ε-greedy更平滑地权衡探索与利用
