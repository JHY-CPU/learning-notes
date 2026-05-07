# 06-随机过程基础

> 马尔可夫链、泊松过程、布朗运动——连接概率论与时间序列、贝叶斯推断的桥梁

---

## 1. 随机过程的定义与分类

### 1.1 定义

随机过程 $\{X(t), t \in T\}$ 是一族定义在概率空间 $(\Omega, \mathcal{F}, P)$ 上的随机变量。

- **指标集** $T$：时间参数（离散或连续）
- **状态空间** $\mathcal{S}$：$X(t)$ 的可能取值集合
- 对每个固定的 $\omega \in \Omega$，$X(\cdot, \omega)$ 是一个**样本路径**

### 1.2 分类

| 分类维度 | 类型 | 示例 |
|----------|------|------|
| 时间 | 离散时间 | $\{X_n, n=0,1,2,\ldots\}$（马尔可夫链） |
| | 连续时间 | $\{X(t), t \geq 0\}$（泊松过程、布朗运动） |
| 状态空间 | 离散状态 | 马尔可夫链、随机游走 |
| | 连续状态 | 布朗运动、Ornstein-Uhlenbeck过程 |
| 统计特性 | 平稳过程 | 统计特性不随时间平移改变 |

### 1.3 有限维分布族

随机过程的完全刻画需要所有有限维联合分布：

$$F_{t_1,\ldots,t_n}(x_1,\ldots,x_n) = P(X(t_1) \leq x_1, \ldots, X(t_n) \leq x_n)$$

这些分布需满足**相容性条件**（Kolmogorov存在定理）。

---

## 2. 马尔可夫链（离散时间）

### 2.1 马尔可夫性质

$$P(X_{n+1} = j | X_n = i, X_{n-1} = i_{n-1}, \ldots) = P(X_{n+1} = j | X_n = i)$$

**无记忆性**：未来只依赖当前状态，与过去无关。

### 2.2 转移概率

**一步转移概率**：$p_{ij} = P(X_{n+1} = j | X_n = i)$

**转移矩阵**：$\mathbf{P} = [p_{ij}]$，每行之和为1（随机矩阵）。

**$n$步转移概率**：$p_{ij}^{(n)} = P(X_n = j | X_0 = i)$

**Chapman-Kolmogorov方程**：

$$p_{ij}^{(m+n)} = \sum_k p_{ik}^{(m)} \cdot p_{kj}^{(n)}$$

矩阵形式：$\mathbf{P}^{(n)} = \mathbf{P}^n$

### 2.3 状态分类

**可达**：从 $i$ 可达 $j$，若存在 $n$ 使得 $p_{ij}^{(n)} > 0$

**互通**：$i$ 与 $j$ 互通 $\Leftrightarrow$ $i$ 可达 $j$ 且 $j$ 可达 $i$

**返回时间**：$T_i = \min\{n \geq 1: X_n = i | X_0 = i\}$

| 状态类型 | 定义 | 直觉 |
|----------|------|------|
| **常返** | $P(T_i < \infty) = 1$ | 从 $i$ 出发一定返回 |
| **非常返（瞬态）** | $P(T_i < \infty) < 1$ | 可能永远不返回 |
| **正常返** | 常返且 $E[T_i] < \infty$ | 平均返回时间有限 |
| **零常返** | 常返但 $E[T_i] = \infty$ | 返回但平均时间无穷 |
| **遍历** | 非周期且正常返 | 最好的性质，有唯一平稳分布 |

### 2.4 周期性

状态 $i$ 的周期：$d(i) = \gcd\{n \geq 1: p_{ii}^{(n)} > 0\}$

- $d(i) = 1$：非周期（aperiodic）
- 互通的状态具有相同周期

---

## 3. 马尔可夫链的平稳分布

### 3.1 平稳分布定义

行向量 $\boldsymbol{\pi} = (\pi_1, \pi_2, \ldots)$ 满足：

$$\boldsymbol{\pi} \mathbf{P} = \boldsymbol{\pi}, \quad \sum_i \pi_i = 1, \quad \pi_i \geq 0$$

即 $\boldsymbol{\pi}$ 是转移矩阵 $\mathbf{P}$ 的左特征向量（特征值1）。

**含义**：若初始分布为 $\boldsymbol{\pi}$，则任意时刻的分布都是 $\boldsymbol{\pi}$（分布不变）。

### 3.2 遍历定理

**定理**：不可约、非周期、正常返的马尔可夫链有唯一平稳分布 $\boldsymbol{\pi}$，且：

$$\lim_{n \to \infty} p_{ij}^{(n)} = \pi_j, \quad \forall i,j$$

初始分布的影响逐渐消失，链收敛到平稳分布。

### 3.3 细致平衡条件

若存在 $\boldsymbol{\pi}$ 满足**细致平衡方程**（Detailed Balance）：

$$\pi_i p_{ij} = \pi_j p_{ji}, \quad \forall i,j$$

则 $\boldsymbol{\pi}$ 是平稳分布。

**意义**：细致平衡比平稳分布更强——它要求每对状态间的"流量"相等。

---

## 4. 隐马尔可夫模型（HMM）

### 4.1 模型结构

HMM包含两个随机过程：

- **隐状态序列** $\{S_t\}$：马尔可夫链，状态转移矩阵 $\mathbf{A}$，初始分布 $\boldsymbol{\pi}$
- **观测序列** $\{O_t\}$：每个时刻 $t$ 的观测 $O_t$ 依赖于 $S_t$，由发射概率 $\mathbf{B}$ 描述

**参数** $\lambda = (\mathbf{A}, \mathbf{B}, \boldsymbol{\pi})$

### 4.2 三个基本问题

| 问题 | 描述 | 算法 |
|------|------|------|
| 评估问题 | 给定 $\lambda$，计算 $P(O_1,\ldots,O_T | \lambda)$ | 前向算法（Forward） |
| 解码问题 | 给定 $\lambda$ 和观测序列，找最优隐状态序列 | Viterbi算法 |
| 学习问题 | 给定观测序列，估计 $\lambda$ | Baum-Welch算法（EM） |

### 4.3 在NLP中的应用

- **词性标注**：隐状态=词性，观测=单词
- **命名实体识别**
- **语音识别**（传统方法）

---

## 5. 泊松过程

### 5.1 定义

计数过程 $\{N(t), t \geq 0\}$ 是强度为 $\lambda$ 的泊松过程，若满足：

1. $N(0) = 0$
2. **独立增量**：不相交时间段内的增量独立
3. **平稳增量**：$N(t+s) - N(s) \sim \text{Poisson}(\lambda t)$

### 5.2 基本性质

**事件计数**：

$$P(N(t) = k) = \frac{(\lambda t)^k e^{-\lambda t}}{k!}$$

**到达间隔**：相邻事件的间隔 $T_1, T_2, \ldots$ 独立同分布于 $\text{Exponential}(\lambda)$

**到达时刻**：给定 $N(t) = n$，$n$ 个事件的到达时刻等价于 $[0,t]$ 上 $n$ 个独立均匀分布随机变量的次序统计量。

### 5.3 叠加与分解

**叠加**：独立泊松过程之和仍是泊松过程，强度为各过程强度之和

**随机分解**：每个事件以概率 $p$ 归入第一类，$1-p$ 归入第二类，得到两个独立泊松过程

### 5.4 非齐次泊松过程

强度随时间变化 $\lambda(t)$：

$$P(N(t+s) - N(s) = k) = \frac{(\int_s^{s+t} \lambda(u)\,du)^k}{k!} \exp\left(-\int_s^{s+t} \lambda(u)\,du\right)$$

**ML应用**：点过程建模、社交网络事件分析、地震预测。

---

## 6. 布朗运动与维纳过程

### 6.1 定义

标准布朗运动 $\{W(t), t \geq 0\}$ 满足：

1. $W(0) = 0$
2. **独立增量**：不相交时间段内增量独立
3. **正态增量**：$W(t) - W(s) \sim \mathcal{N}(0, t-s)$（$t > s$）
4. **连续路径**：样本路径 $t \mapsto W(t)$ 连续（几乎处处）

### 6.2 基本性质

- $E[W(t)] = 0$，$\text{Var}(W(t)) = t$
- $\text{Cov}(W(s), W(t)) = \min(s, t)$
- **自相似性**：对任意 $c > 0$，$\{c^{-1/2}W(ct)\}$ 也是布朗运动
- 路径处处连续但处处不可导（分形特征）
- **二次变差**：$\sum(W(t_{i+1})-W(t_i))^2 \to t$

### 6.3 广义布朗运动

$X(t) = \mu t + \sigma W(t)$

- 漂移 $\mu$：趋势项
- 扩散系数 $\sigma$：波动幅度
- $E[X(t)] = \mu t$，$\text{Var}(X(t)) = \sigma^2 t$

### 6.4 几何布朗运动

$$S(t) = S(0) \exp\left((\mu - \frac{\sigma^2}{2})t + \sigma W(t)\right)$$

**金融应用**：Black-Scholes期权定价模型的基础假设——股票价格服从几何布朗运动。

---

## 7. 在机器学习中的应用

### 7.1 马尔可夫链蒙特卡洛（MCMC）

利用马尔可夫链的平稳分布为目标分布 $\pi(\theta)$ 进行采样。

**Metropolis-Hastings算法**：

1. 从提议分布 $q(\theta^* | \theta^{(t)})$ 采样候选 $\theta^*$
2. 计算接受概率 $\alpha = \min\left(1, \frac{\pi(\theta^*)q(\theta^{(t)}|\theta^*)}{\pi(\theta^{(t)})q(\theta^*|\theta^{(t)})}\right)$
3. 以概率 $\alpha$ 接受 $\theta^{(t+1)} = \theta^*$，否则 $\theta^{(t+1)} = \theta^{(t)}$

**Gibbs采样**：MH的特例，依次从条件分布采样每个分量。

**收敛保证**：不可约+非周期+正常返 $\Rightarrow$ 链收敛到目标分布。

### 7.2 变分推断基础

当后验 $p(\theta | D)$ 难以计算时，用简单分布 $q(\theta)$ 近似：

$$q^*(\theta) = \arg\min_{q} \text{KL}(q(\theta) \| p(\theta | D))$$

等价于最大化证据下界（ELBO）：

$$\text{ELBO} = E_q[\ln p(D|\theta)] - \text{KL}(q(\theta) \| p(\theta))$$

**与随机过程的联系**：
- 变分自编码器（VAE）的训练
- 随机梯度变分推断（SVI）
- 扩散模型的训练目标本质上也是变分推断

### 7.3 扩散模型的随机过程视角

扩散模型可以理解为连续时间马尔可夫过程：

- **前向过程**：逐渐向数据添加高斯噪声（Ornstein-Uhlenbeck过程）
- **反向过程**：学习去噪（逆扩散），生成数据

$$d\mathbf{x} = -\frac{1}{2}\beta(t)\mathbf{x}\,dt + \sqrt{\beta(t)}\,d\mathbf{W}(t)$$

### 7.4 其他应用

| 应用领域 | 使用的随机过程 |
|----------|---------------|
| 强化学习 | 马尔可夫决策过程（MDP） |
| 时间序列 | ARMA过程、状态空间模型 |
| 自然语言 | n-gram模型、语言模型 |
| 图神经网络 | 随机游走、PageRank |
| 贝叶斯优化 | 高斯过程（连续状态的随机过程） |

---

## 参考资料

- 《Stochastic Processes》Sheldon Ross
- 《Pattern Recognition and Machine Learning》Bishop
- 《Deep Learning》Goodfellow et al., Ch.20
- 《Information Theory, Inference, and Learning Algorithms》MacKay
