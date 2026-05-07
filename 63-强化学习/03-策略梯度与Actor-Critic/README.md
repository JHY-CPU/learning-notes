# 03-策略梯度与Actor-Critic

## 1. 策略梯度定理的推导

### 1.1 动机

值函数方法（如Q-Learning）在连续动作空间中面临argmax难以计算的问题。策略梯度方法直接参数化策略 $\pi_\theta(a|s)$，通过梯度上升优化期望回报。

### 1.2 目标函数

定义目标函数为起始状态的期望回报：

$$J(\theta) = V^{\pi_\theta}(s_0) = \mathbb{E}_{\pi_\theta}[G_0]$$

对于折扣回报的连续情形：

$$J(\theta) = \mathbb{E}_{\pi_\theta}\left[\sum_{t=0}^{\infty} \gamma^t R_t\right]$$

### 1.3 策略梯度定理

**定理**：目标函数关于参数 $\theta$ 的梯度为：

$$\nabla_\theta J(\theta) = \mathbb{E}_{\pi_\theta}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot Q^{\pi_\theta}(s, a)\right]$$

**推导要点**：
1. 将轨迹概率写为 $P(\tau;\theta)$，回报为 $R(\tau)$
2. $\nabla_\theta J = \mathbb{E}_\tau[R(\tau) \nabla_\theta \log P(\tau;\theta)]$
3. 利用 $\nabla_\theta \log P(\tau;\theta) = \sum_t \nabla_\theta \log \pi_\theta(a_t|s_t)$
4. 整理得到策略梯度公式

直觉解释：$\nabla_\theta \log \pi_\theta(a|s)$ 指示如何调整参数使动作 $a$ 的概率增大，乘以 $Q(s,a)$ 后按回报大小加权。

## 2. REINFORCE算法（Monte Carlo策略梯度）

REINFORCE是最基本的策略梯度算法，使用采样回报替代Q值：

$$\nabla_\theta J(\theta) \approx \sum_{t=0}^{T} \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot G_t$$

```
初始化策略参数 θ

对每个回合:
    按 π_θ 采样完整轨迹 {S_0, A_0, R_1, ..., S_T}
    对每一步 t = 0, ..., T-1:
        计算回报 G_t = Σ γ^(k-t-1) R_k
        θ ← θ + α · ∇_θ log π_θ(A_t|S_t) · G_t
```

**缺点**：
- 高方差：完整轨迹的回报波动很大
- 必须等回合结束才能更新
- 收敛慢

## 3. 带基线的REINFORCE

引入基线 $b(s)$ 减少方差，不影响梯度的无偏性：

$$\nabla_\theta J(\theta) = \mathbb{E}\left[\nabla_\theta \log \pi_\theta(a|s) \cdot (Q^{\pi}(s,a) - b(s))\right]$$

常用基线选择：
- **状态值函数**：$b(s) = V^\pi(s)$，此时 $Q(s,a) - V(s) = A^\pi(s,a)$（优势函数）
- **常数**：$b = \mathbb{E}[G_t]$
- **移动平均**：$b \leftarrow \beta b + (1-\beta) G_t$

带基线后，梯度信号更集中在"比平均更好"的动作上。

## 4. Actor-Critic框架

### 4.1 基本思想

Actor-Critic结合了策略梯度（Actor，负责选择动作）和值函数（Critic，负责评估动作）：

- **Actor**：策略网络 $\pi_\theta(a|s)$，决定采取什么动作
- **Critic**：值函数网络 $V_w(s)$ 或 $Q_w(s,a)$，评估当前策略的好坏

### 4.2 更新规则

**Critic更新**（TD学习）：
$$\delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$$
$$w \leftarrow w + \alpha_w \delta_t \nabla_w V_w(S_t)$$

**Actor更新**（策略梯度）：
$$\theta \leftarrow \theta + \alpha_\theta \nabla_\theta \log \pi_\theta(A_t|S_t) \cdot \delta_t$$

用TD误差 $\delta_t$ 替代REINFORCE中的回报 $G_t$，降低方差同时保持在线更新能力。

## 5. A2C（Advantage Actor-Critic）

A2C使用优势函数作为梯度的权重：

$$A^\pi(s,a) = Q^\pi(s,a) - V^\pi(s)$$

实际中用TD误差估计优势：$\hat{A}_t = \delta_t = R_{t+1} + \gamma V_w(S_{t+1}) - V_w(S_t)$

A2C使用同步并行：多个环境并行采样，每个环境独立收集数据，合并后执行一次更新。这比单线程更稳定，数据利用率更高。

## 6. A3C（异步优势Actor-Critic）

A3C使用多个并行的worker在不同副本上异步执行：

```
全局共享参数 θ, w
每个worker:
    1. 复制全局参数
    2. 在本地环境执行若干步
    3. 计算梯度
    4. 异步更新全局参数
```

异步更新导致梯度可能略有过时（stale gradients），但在实践中表现良好。A3C不需要经验回放，因为并行采样本身提供了多样化的数据。

## 7. GAE（广义优势估计）

GAE结合了不同步数的优势估计，通过参数 $\lambda$ 控制偏差-方差权衡：

$$\hat{A}_t^{GAE(\gamma,\lambda)} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}$$

其中 $\delta_t = R_t + \gamma V(S_{t+1}) - V(S_t)$。

- $\lambda = 0$：单步TD优势（低方差，有偏）
- $\lambda = 1$：MC优势（无偏，高方差）
- $\lambda \in (0,1)$：偏差和方差的折中

GAE是PPO等现代算法的标准组件。

## 8. PPO（近端策略优化）

### 8.1 Clipped Surrogate Objective

PPO的核心思想是限制策略更新的幅度，防止破坏性更新。使用重要性采样比率：

$$r_t(\theta) = \frac{\pi_\theta(A_t|S_t)}{\pi_{\theta_{old}}(A_t|S_t)}$$

**裁剪目标函数**：

$$L^{CLIP}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta)\hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon)\hat{A}_t\right)\right]$$

- 当 $\hat{A}_t > 0$（好动作）：$r_t$ 被裁剪到 $1+\epsilon$，防止过度鼓励
- 当 $\hat{A}_t < 0$（坏动作）：$r_t$ 被裁剪到 $1-\epsilon$，防止过度惩罚

### 8.2 PPO算法流程

```
初始化参数 θ

对每次迭代:
    1. 用当前策略 π_θ 收集轨迹数据
    2. 计算优势估计（通常用GAE）
    3. 对数据做多个epoch的minibatch更新:
        L = L^{CLIP} - c_1 · L^{VF} + c_2 · S[π_θ]
    其中 L^{VF} 是值函数损失，S 是策略熵
```

PPO的优势：
- 实现简单，只需要一阶梯度
- 样本效率优于A2C/A3C
- 超参数鲁棒，易于调参
- 是目前最流行的策略梯度算法之一

## 9. TRPO（信赖域策略优化）

TRPO通过KL散度约束限制策略更新步长：

$$\max_\theta \mathbb{E}\left[\frac{\pi_\theta(a|s)}{\pi_{old}(a|s)}\hat{A}_t\right] \quad \text{s.t.} \quad \mathbb{E}[D_{KL}(\pi_{old}||\pi_\theta)] \leq \delta$$

需要计算Fisher信息矩阵并解共轭梯度，计算复杂度高于PPO。PPO可以看作TRPO的简化版本。

## 10. DPG（确定性策略梯度）

DPG使用确定性策略 $\mu_\theta(s)$ 而非随机策略：

$$\nabla_\theta J(\theta) = \mathbb{E}_{s \sim \rho^\mu}\left[\nabla_\theta \mu_\theta(s) \nabla_a Q^\mu(s,a)|_{a=\mu_\theta(s)}\right]$$

连续动作空间中不需要对动作积分，更高效。需要off-policy的Q函数学习配合。

## 11. DDPG（深度确定性策略梯度）

DDPG结合了DPG和DQN的思想，适用于连续动作空间：

```
Actor网络: μ_θ(s)    Critic网络: Q_w(s,a)
目标网络: μ_θ'(s)    目标网络: Q_w'(s,a)

对每个环境步:
    1. 执行动作 a = μ_θ(s) + 探索噪声
    2. 存储经验到回放缓冲区
    3. 采样minibatch
    4. 更新Critic: 最小化 (y_i - Q_w(s_i,a_i))^2
       y_i = r_i + γ Q_w'(s_i', μ_θ'(s_i'))
    5. 更新Actor: ∇_θ J ≈ ∇_θ Q_w(s, μ_θ(s))
    6. 软更新目标网络
```

## 12. SAC（Soft Actor-Critic）

SAC在目标函数中加入策略熵项，鼓励探索：

$$J(\pi) = \mathbb{E}\left[\sum_t \gamma^t (R_t + \alpha \mathcal{H}(\pi(\cdot|S_t)))\right]$$

$\alpha$ 是温度参数，自动调节探索程度。SAC具有以下特点：
- 自动熵调节
- 比DDPG更稳定
- 样本效率高
- 对超参数不敏感

## 13. 策略梯度 vs 值函数方法

| 维度 | 值函数方法 | 策略梯度方法 |
|------|------------|--------------|
| 动作空间 | 离散有限 | 离散/连续均可 |
| 收敛性 | 可能不收敛（函数逼近） | 局部收敛保证 |
| 方差 | 低 | 高（需基线/GAE降低） |
| 探索 | 需手动设计（ε-greedy等） | 策略本身包含探索 |
| 高维动作 | argmax困难 | 天然支持 |
| 最优性 | 可能收敛到次优 | 收敛到局部最优 |
