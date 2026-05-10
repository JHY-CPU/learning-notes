# SAC与DDPG


## SAC与DDPG


强化学习Actor-Critic连续控制


DDPG和SAC是两种主流的off-policy Actor-Critic算法，专为连续动作空间设计。


## DDPG (Deep Deterministic Policy Gradient)


```
DDPG (Lillicrap 2015)：
结合DQN的思想和确定性策略梯度

核心组件：
Actor: μ_θ(s) → 连续动作（确定性策略）
Critic: Q_φ(s, a) → 动作价值函数

确定性策略梯度 (DPG)：
∇_θ J(θ) = E[ ∇_a Q(s,a)|a=μ(s) × ∇_θ μ_θ(s) ]
= E[ ∇_θ Q(s, μ_θ(s)) ]

直觉：
- Actor选择最大化Q值的动作
- Critic评估Actor选择的动作有多好
- Actor沿着Critic梯度方向改进

关键技术（来自DQN）：
1. 经验回放：打破数据相关性
2. 目标网络：稳定训练
   θ' ← τθ + (1-τ)θ'  (软更新)
   φ' ← τφ + (1-τ)φ'

Critic更新（TD学习）：
y = r + γ × Q_φ'(s', μ_θ'(s'))
L = (Q_φ(s,a) - y)²

Actor更新：
∇_θ J = E[ ∇_θ Q_φ(s, μ_θ(s)) ]
```


## DDPG算法流程


```
初始化 Actor μ_θ, Critic Q_φ
初始化目标网络 θ' ← θ, φ' ← φ
初始化经验回放缓冲区 R

for episode = 1, ..., M:
    初始化噪声 N
    s = 初始状态

    for t = 1, ..., T:
        // 选择动作（加探索噪声）
        a = μ_θ(s) + N_t  （N_t是Ornstein-Uhlenbeck噪声）

        // 执行动作，观察奖励和新状态
        r, s' = env.step(a)

        // 存储经验
        R.push(s, a, r, s')

        // 采样batch更新
        if |R| > batch_size:
            从R采样 {(s_i, a_i, r_i, s'_i)}

            // 更新Critic
            y_i = r_i + γ × Q_φ'(s'_i, μ_θ'(s'_i))
            L = (1/N) × Σ (Q_φ(s_i, a_i) - y_i)²
            梯度下降更新 φ

            // 更新Actor
            ∇_θ J = (1/N) × Σ ∇_θ Q_φ(s_i, μ_θ(s_i))
            梯度上升更新 θ

            // 软更新目标网络
            θ' ← τθ + (1-τ)θ'
            φ' ← τφ + (1-τ)φ'

        s ← s'

关键参数：
- τ (软更新系数): 0.001~0.005
- 学习率: Actor 1e-4, Critic 1e-3
- OU噪声参数: σ=0.2, θ=0.15
```


## SAC (Soft Actor-Critic)


```
SAC (Haarnoja 2018)：
在最大熵框架下的Actor-Critic

核心思想：
不仅要最大化回报，还要最大化策略的熵
→ 鼓励探索，提高鲁棒性

目标函数：
J(θ) = E[ Σ γ^t × (Rt + α × H(π(·|St))) ]

H(π(·|s)) = -E_a~π[ log π(a|s) ]  策略熵

α：温度参数（自动调节）
- α大：鼓励探索（更随机）
- α小：利用已知好策略（更确定性）

Soft Bellman方程：
Q(s,a) = E[ R + γ × V(s') ]
V(s) = E_a~π[ Q(s,a) - α × log π(a|s) ]

Soft策略改进：
π_new = argmin_π D_KL(π || exp(Q(s,·)/α))
解析解：π(a|s) ∝ exp(Q(s,a)/α)

自动温度调节：
L(α) = E[ -α × log π(a_t|s_t) - α × H_target ]
H_target = -dim(A)（动作空间维度的负值）
```


> **Note:** SAC是目前连续控制任务中表现最好的off-policy算法之一，比DDPG更稳定，对超参数更鲁棒。


## SAC算法流程


```
初始化：Actor π_θ, 两个Critic Q_φ1, Q_φ2
        目标Critic Q_φ1', Q_φ2'（← φ1, φ2）
        温度α

for each iteration:
    用π_θ采集数据存入回放缓冲区

    for each gradient step:
        从缓冲区采样 {(s,a,r,s')}

        // 计算目标值（取两个Q的最小值）
        a' ~ π_θ(·|s')  // 从当前策略采样新动作
        Q_target = r + γ × min(Q_φ1'(s',a'), Q_φ2'(s',a'))
                      - α × log π_θ(a'|s')

        // 更新两个Critic
        L_Critic = (Q_φ1(s,a) - Q_target)² + (Q_φ2(s,a) - Q_target)²

        // 更新Actor
        a_sample ~ π_θ(·|s)  // 重参数化采样
        L_Actor = α × log π_θ(a_sample|s)
                - min(Q_φ1(s, a_sample), Q_φ2(s, a_sample))

        // 自动调节温度
        L_α = -α × (log π_θ(a_t|s_t) + H_target)

        梯度下降更新 θ, φ1, φ2, α
        软更新 φ1', φ2'

关键技术：
- 双Critic：缓解Q值过估计
- 自动温度调节：无需手动调α
- 重参数化：使随机策略可微
```


## DDPG vs SAC 对比


```
┌──────────────┬──────────────┬──────────────────┐
│              │ DDPG         │ SAC              │
├──────────────┼──────────────┼──────────────────┤
│ 策略类型     │ 确定性       │ 随机             │
│ 探索方式     │ 外部噪声     │ 策略熵内在探索   │
│ 理论框架     │ DPG          │ 最大熵RL         │
│ Q值估计     │ 单Critic     │ 双Critic(最小值) │
│ 超参数敏感   │ 敏感         │ 鲁棒             │
│ 温度参数     │ 无           │ 自动调节         │
│ 收敛稳定性   │ 一般         │ 好               │
│ 样本效率     │ 较高         │ 高               │
│ 最终性能     │ 好           │ 很好             │
│ 实现复杂度   │ 中等         │ 较复杂           │
└──────────────┴──────────────┴──────────────────┘

DDPG的问题：
- 对超参数敏感（学习率、噪声参数）
- 容易收敛到次优策略
- Q值过估计

SAC的改进：
- 最大熵框架自然鼓励探索
- 双Critic减少过估计
- 自动温度调节减少调参
- 随机策略更稳定
```


## 重参数化技巧 (Reparameterization Trick)


```
问题：从策略π(a|s)采样不可微
→ 无法直接对策略参数θ求梯度

解决方案：重参数化

高斯策略的重参数化：
原来：a ~ N(μ_θ(s), σ_θ(s)²)  不可微

重参数化：
ε ~ N(0, 1)     // 噪声与θ无关
a = μ_θ(s) + σ_θ(s) × ε  // 确定性变换，可微

梯度传播：
∇_θ a = ∇_θ μ_θ(s) + ε × ∇_θ σ_θ(s)
→ 梯度可以通过μ和σ反向传播到θ

实际实现：
class SquashedGaussianActor:
    def forward(self, s):
        mean, log_std = network(s)
        std = exp(log_std)
        ε = N(0,1)
        a_raw = mean + std × ε
        a = tanh(a_raw)  // 压缩到[-1,1]

        // log概率（tanh修正）
        log_prob = log_normal(a_raw) - log(1 - tanh(a_raw)²)
        return a, log_prob
```


> **Note:** 重参数化技巧是SAC等算法能够对随机策略求梯度的关键技术。tanh压缩确保动作在合理范围内。


<!-- Converted from: 03_SAC与DDPG.html -->
