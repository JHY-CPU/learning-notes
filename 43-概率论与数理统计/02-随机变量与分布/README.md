# 02-随机变量与分布

> 从概率空间到随机变量，掌握核心分布族及其数字特征

---

## 1. 随机变量的定义

### 1.1 基本概念

随机变量 $X$ 是定义在样本空间 $\Omega$ 上的实值函数：

$$X: \Omega \to \mathbb{R}$$

将随机试验的结果数值化，使得可以用数学工具分析随机现象。

### 1.2 分类

| 类型 | 特征 | 典型例子 |
|------|------|----------|
| **离散型** | 取值有限或可数 | 掷骰子次数、缺陷品数量 |
| **连续型** | 取值充满区间 | 温度、时间、误差 |
| **混合型** | 两者兼有 | 等待时间（含零点跳变） |

---

## 2. 离散型分布

### 2.1 伯努利分布 Bernoulli($p$)

单次试验，成功概率为 $p$。

$$P(X=k) = p^k(1-p)^{1-k}, \quad k \in \{0, 1\}$$

- $E[X] = p$，$\text{Var}(X) = p(1-p)$

### 2.2 二项分布 Binomial($n, p$)

$n$ 次独立伯努利试验中成功的次数。

$$P(X=k) = \binom{n}{k} p^k (1-p)^{n-k}, \quad k = 0, 1, \ldots, n$$

- $E[X] = np$，$\text{Var}(X) = np(1-p)$
- 当 $n$ 大、$p$ 小、$np$ 适中时，可用泊松分布近似

### 2.3 几何分布 Geometric($p$)

首次成功所需试验次数。

$$P(X=k) = (1-p)^{k-1}p, \quad k = 1, 2, \ldots$$

- $E[X] = \frac{1}{p}$，$\text{Var}(X) = \frac{1-p}{p^2}$
- **无记忆性**：$P(X > s+t | X > s) = P(X > t)$

### 2.4 超几何分布 Hypergeometric($N, M, n$)

从 $N$ 个物品（$M$ 个合格品）中不放回抽取 $n$ 个，合格品数量服从超几何分布。

$$P(X=k) = \frac{\binom{M}{k}\binom{N-M}{n-k}}{\binom{N}{n}}$$

- $E[X] = n \cdot \frac{M}{N}$
- 当 $N \to \infty$ 且 $\frac{M}{N} \to p$ 时，趋近于二项分布

### 2.5 泊松分布 Poisson($\lambda$)

单位时间/空间内随机事件发生的次数。

$$P(X=k) = \frac{\lambda^k e^{-\lambda}}{k!}, \quad k = 0, 1, 2, \ldots$$

- $E[X] = \lambda$，$\text{Var}(X) = \lambda$（期望等于方差是重要特征）
- **可加性**：$X \sim \text{Pois}(\lambda_1)$，$Y \sim \text{Pois}(\lambda_2)$ 独立，则 $X+Y \sim \text{Pois}(\lambda_1+\lambda_2)$
- **ML应用**：泊松回归、事件计数建模、泊松损失函数

---

## 3. 连续型分布

### 3.1 均匀分布 Uniform($a, b$)

$$f(x) = \frac{1}{b-a}, \quad a \leq x \leq b$$

- $E[X] = \frac{a+b}{2}$，$\text{Var}(X) = \frac{(b-a)^2}{12}$
- **ML应用**：随机数生成的基础（逆变换采样）

### 3.2 指数分布 Exponential($\lambda$)

$$f(x) = \lambda e^{-\lambda x}, \quad x \geq 0$$

- $E[X] = \frac{1}{\lambda}$，$\text{Var}(X) = \frac{1}{\lambda^2}$
- **无记忆性**：$P(X > s+t | X > s) = P(X > t)$
- 泊松过程中相邻事件间隔服从指数分布
- **ML应用**：生存分析、可靠性工程、ReLU激活函数的随机分析

### 3.3 正态分布（高斯分布）$\mathcal{N}(\mu, \sigma^2)$

$$f(x) = \frac{1}{\sqrt{2\pi}\sigma} \exp\left(-\frac{(x-\mu)^2}{2\sigma^2}\right)$$

**核心性质**：

- $E[X] = \mu$，$\text{Var}(X) = \sigma^2$
- **线性变换**：$aX+b \sim \mathcal{N}(a\mu+b,\ a^2\sigma^2)$
- **可加性**：独立正态变量之和仍为正态
- **68-95-99.7法则**：$P(|X-\mu| < k\sigma) \approx 68\%, 95\%, 99.7\%$（$k=1,2,3$）
- 中心极限定理保证了其普遍性

**标准正态分布** $\mathcal{N}(0,1)$：$\Phi(x)$ 为其CDF，$\phi(x)$ 为其PDF。

**ML应用**：权重初始化（Xavier/He初始化）、变分推断的后验近似、Diffusion Model的基础分布、Batch Normalization的目标分布。

---

## 4. PDF与CDF

### 4.1 累积分布函数（CDF）

$$F(x) = P(X \leq x)$$

性质：单调不减、右连续、$F(-\infty)=0,\ F(+\infty)=1$

### 4.2 概率密度函数（PDF）

连续型随机变量的PDF $f(x)$ 满足：

$$F(x) = \int_{-\infty}^{x} f(t)\,dt, \quad f(x) = F'(x)$$

$$P(a < X \leq b) = F(b) - F(a) = \int_a^b f(x)\,dx$$

**注意**：$f(x)$ 可以大于1，$P(X=c) = 0$（连续型单点概率为零）。

### 4.3 分位数

$p$-分位数 $x_p$ 满足 $F(x_p) = p$。

- 中位数：$x_{0.5}$
- 四分位数：$x_{0.25},\ x_{0.5},\ x_{0.75}$

---

## 5. 数学期望

### 5.1 定义

**离散型**：$E[X] = \sum_k x_k P(X = x_k)$

**连续型**：$E[X] = \int_{-\infty}^{+\infty} x f(x)\,dx$

**函数的期望**：$E[g(X)] = \sum_k g(x_k)P(X=x_k)$ 或 $\int g(x)f(x)\,dx$

### 5.2 性质

- $E[c] = c$
- $E[aX+b] = aE[X]+b$（线性性）
- $E[X+Y] = E[X]+E[Y]$（不要求独立）
- **一般情况下** $E[XY] \neq E[X] \cdot E[Y]$

### 5.3 常见分布的期望

| 分布 | 期望 |
|------|------|
| Binomial($n,p$) | $np$ |
| Poisson($\lambda$) | $\lambda$ |
| Uniform($a,b$) | $\frac{a+b}{2}$ |
| Exponential($\lambda$) | $\frac{1}{\lambda}$ |
| $\mathcal{N}(\mu,\sigma^2)$ | $\mu$ |

---

## 6. 方差与标准差

### 6.1 定义

$$\text{Var}(X) = E[(X-E[X])^2] = E[X^2] - (E[X])^2$$

标准差：$\sigma_X = \sqrt{\text{Var}(X)}$

### 6.2 性质

- $\text{Var}(c) = 0$
- $\text{Var}(aX+b) = a^2 \text{Var}(X)$
- 若 $X, Y$ 独立：$\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y)$
- 一般情况：$\text{Var}(X+Y) = \text{Var}(X) + \text{Var}(Y) + 2\text{Cov}(X,Y)$

### 6.3 常见分布的方差

| 分布 | 方差 |
|------|------|
| Bernoulli($p$) | $p(1-p)$ |
| Binomial($n,p$) | $np(1-p)$ |
| Poisson($\lambda$) | $\lambda$ |
| Uniform($a,b$) | $\frac{(b-a)^2}{12}$ |
| Exponential($\lambda$) | $\frac{1}{\lambda^2}$ |
| $\mathcal{N}(\mu,\sigma^2)$ | $\sigma^2$ |

---

## 7. 矩与矩母函数

### 7.1 原点矩与中心矩

- **$k$阶原点矩**：$\mu_k' = E[X^k]$
- **$k$阶中心矩**：$\mu_k = E[(X-E[X])^k]$

一阶原点矩即期望，二阶中心矩即方差。

### 7.2 矩母函数（MGF）

$$M_X(t) = E[e^{tX}]$$

**性质**：
- $M_X^{(n)}(0) = E[X^n]$（$n$阶矩）
- 若 $M_X(t)$ 在某区间内有限，则分布唯一确定
- **可加性**：$X, Y$ 独立则 $M_{X+Y}(t) = M_X(t) \cdot M_Y(t)$

**常见MGF**：
- $\mathcal{N}(\mu, \sigma^2)$：$M(t) = \exp(\mu t + \frac{\sigma^2 t^2}{2})$
- Poisson($\lambda$)：$M(t) = \exp(\lambda(e^t - 1))$
- Exponential($\lambda$)：$M(t) = \frac{\lambda}{\lambda - t}$

### 7.3 特征函数

$$\varphi_X(t) = E[e^{itX}]$$

始终存在（复值），是证明中心极限定理的核心工具。

---

## 8. 偏度与峰度

### 8.1 偏度（Skewness）

$$\gamma_1 = \frac{E[(X-\mu)^3]}{\sigma^3} = \frac{\mu_3}{\sigma^3}$$

- $\gamma_1 > 0$：右偏（正偏），尾部向右延伸
- $\gamma_1 < 0$：左偏（负偏），尾部向左延伸
- $\gamma_1 = 0$：对称（正态分布偏度为0）

### 8.2 峰度（Kurtosis）

$$\kappa = \frac{E[(X-\mu)^4]}{\sigma^4}$$

**超额峰度**：$\kappa - 3$（正态分布峰度为3，超额峰度为0）

- $\kappa > 3$：尖峰厚尾（比正态更极端的值更多）
- $\kappa < 3$：平峰薄尾

**ML应用**：金融风险管理中衡量极端事件概率，GAN训练中监控生成分布质量。

---

## 9. 常用分布总结表

| 分布 | 记号 | PDF/PMF | 期望 | 方差 | 关键特征 |
|------|------|---------|------|------|----------|
| Bernoulli | $B(p)$ | $p^k(1-p)^{1-k}$ | $p$ | $p(1-p)$ | 单次试验 |
| Binomial | $B(n,p)$ | $\binom{n}{k}p^k(1-p)^{n-k}$ | $np$ | $np(1-p)$ | n次独立试验 |
| Poisson | $P(\lambda)$ | $\frac{\lambda^k e^{-\lambda}}{k!}$ | $\lambda$ | $\lambda$ | 稀有事件计数 |
| Geometric | $Geo(p)$ | $(1-p)^{k-1}p$ | $1/p$ | $\frac{1-p}{p^2}$ | 首次成功 |
| Uniform | $U(a,b)$ | $\frac{1}{b-a}$ | $\frac{a+b}{2}$ | $\frac{(b-a)^2}{12}$ | 等可能 |
| Exponential | $Exp(\lambda)$ | $\lambda e^{-\lambda x}$ | $1/\lambda$ | $1/\lambda^2$ | 无记忆性 |
| Normal | $\mathcal{N}(\mu,\sigma^2)$ | $\frac{1}{\sqrt{2\pi}\sigma}e^{-\frac{(x-\mu)^2}{2\sigma^2}}$ | $\mu$ | $\sigma^2$ | CLT保证的普遍性 |

---

## 参考资料

- 《概率论与数理统计》陈希孺
- 《概率导论》Bertsekis & Tsitsiklis
- 《Pattern Recognition and Machine Learning》Bishop, Ch.2
