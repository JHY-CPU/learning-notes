# Cramer-Rao 下界与参数估计极限

## 核心概念
- **Cramer-Rao 下界 (CRLB)**：任何无偏估计量的方差有一个理论下界：
  $$\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}$$
  其中 $I(\theta)$ 是费雪信息。这给出了参数估计精度的"天花板"。
- **有效估计量**：达到 CRLB 的估计量称为有效估计量。在指数族分布中，MLE 是渐近有效的（在大样本下达到 CRLB）。
- **多参数 CRLB**：对多参数 $\boldsymbol{\theta} \in \mathbb{R}^p$，无偏估计的协方差满足：
  $$\text{Cov}(\hat{\boldsymbol{\theta}}) \succeq F(\boldsymbol{\theta})^{-1}$$
  即 $\text{Cov}(\hat{\boldsymbol{\theta}}) - F(\boldsymbol{\theta})^{-1}$ 是半正定矩阵。
- **CRLB 的渐近版本**：在大样本下，MLE 的渐近协方差矩阵为 $F(\theta)^{-1}/n$。这是 MLE 渐近正态性的直接结果。
- **紧致性**：CRLB 是否可达取决于分布族是否属于"指数族"并满足一定的正则条件。非正则情况下（如均匀分布 $U[0,\theta]$），CRLB 可能不直接适用。
- **与费雪信息的关系**：CRLB 揭示了费雪信息的本质——信息越多（$I(\theta)$ 越大），估计方差的理论下界越低，估计可以越精确。

## 数学推导
Cramer-Rao 不等式（一元）：
$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)} = \frac{1}{\mathbb{E}\left[ \left( \frac{\partial \log p(X|\theta)}{\partial \theta} \right)^2 \right]}
$$

证明概要：设 $T(X) = \hat{\theta}$ 是 $\theta$ 的无偏估计，则 $\mathbb{E}[T(X)] = \int T(x) p(x|\theta) dx = \theta$。
两边对 $\theta$ 求导：
$$
1 = \int T(x) \frac{\partial p(x|\theta)}{\partial \theta} dx = \int T(x) \frac{\partial \log p}{\partial \theta} p(x|\theta) dx = \mathbb{E}\left[T(X) \cdot \frac{\partial \log p}{\partial \theta}\right] = \text{Cov}\left(T, \frac{\partial \log p}{\partial \theta}\right)
$$

由 Cauchy-Schwarz 不等式：
$$
1 = \text{Cov}^2\left(T, \frac{\partial \log p}{\partial \theta}\right) \leq \text{Var}(T) \cdot \text{Var}\left( \frac{\partial \log p}{\partial \theta} \right) = \text{Var}(T) \cdot I(\theta)
$$

因此 $\text{Var}(T) \geq 1 / I(\theta)$。

多参数 CRLB：
$$
\text{Cov}(\hat{\boldsymbol{\theta}}) \succeq F(\boldsymbol{\theta})^{-1}
$$

## 直观理解
- **理论精度的"天花板"**：CRLB 就像说"你不能比光速更快"。无论你的估计方法多巧妙，方差不可能低于这个界。这提供了评估估计量优劣的基准。
- **"信息量"与"精度"的置换**：费雪信息 $I(\theta)$ 是数据包含的关于 $\theta$ 的信息量。信息越多（$I(\theta)$ 越大），CRLB 越低（方差越小），估计可以更精确。就像分辨率——信息越多，图像越清晰。
- **重复实验的直观**：CRLB 告诉你在 $n$ 次独立重复实验中，$\hat{\theta}$ 的标准差至少为 $\sigma / \sqrt{n}$。要精度提高 10 倍，需要 100 倍的数据。这是费雪信息相加性（$I_n(\theta) = n I_1(\theta)$）的直接结果。

## 代码示例
```python
import numpy as np
from scipy.stats import norm, bernoulli

# 1. 伯努利分布：验证 CRLB
true_p = 0.3
n_experiments = 10000
sample_sizes = [10, 50, 200, 1000]

print("Cramer-Rao 下界验证 (Bernoulli):")
for n in sample_sizes:
    # 模拟多次实验
    estimates = []
    for _ in range(n_experiments):
        samples = bernoulli.rvs(true_p, size=n)
        p_hat = np.mean(samples)  # MLE
        estimates.append(p_hat)
    
    empirical_var = np.var(estimates)
    crlb = true_p * (1 - true_p) / n  # 1/(n*I(θ)) = p(1-p)/n
    
    print(f"  n={n:4d}: 经验方差={empirical_var:.6f}, CRLB={crlb:.6f}, "
          f"达到? {empirical_var >= crlb - 1e-6}")

# 2. 高斯分布：验证 CRLB
true_mu, true_sigma = 5.0, 2.0

print(f"\nCramer-Rao 下界验证 (Gaussian):")
mu_estimates = []
sigma_estimates = []

n_samples = 100
for _ in range(10000):
    samples = norm.rvs(true_mu, true_sigma, size=n_samples)
    mu_hat = np.mean(samples)
    sigma_sq_hat = np.var(samples)  # MLE 方差
    mu_estimates.append(mu_hat)
    sigma_estimates.append(sigma_sq_hat)

emp_var_mu = np.var(mu_estimates)
emp_var_sigma = np.var(sigma_estimates)

crlb_mu = true_sigma**2 / n_samples
crlb_sigma = 2 * true_sigma**4 / n_samples

print(f"  μ CRLB: {crlb_mu:.6f}, 经验方差: {emp_var_mu:.6f}")
print(f"  σ² CRLB: {crlb_sigma:.6f}, 经验方差: {emp_var_sigma:.6f}")
print(f"  μ 达到 CRLB? {emp_var_mu >= crlb_mu - 1e-6}")
# σ² 的 MLE 是有偏的，所以可能不满足 CRLB（CRLB 只对无偏估计）
print(f"  σ² 的 MLE 有偏: {np.mean(sigma_estimates):.4f} vs 真实={true_sigma**2:.4f}")

# 3. 用无偏方差估计量验证
sigma_sq_unbiased = np.var(samples, ddof=1)  # 除以 n-1
# 对于无偏估计，应该更接近 CRLB（但仍需检查是否达到）

# 4. 不同估计量的效率比较
print(f"\n不同估计量的效率:")
n = 20
n_trials = 10000

# 估计量 1: 样本均值
mu_mle = []
# 估计量 2: 中位数
mu_median = []

for _ in range(n_trials):
    samples = norm.rvs(true_mu, true_sigma, size=n)
    mu_mle.append(np.mean(samples))
    mu_median.append(np.median(samples))

var_mle = np.var(mu_mle)
var_median = np.var(mu_median)
crlb_mu = true_sigma**2 / n

print(f"  MLE 方差: {var_mle:.6f} (效率={crlb_mu/var_mle:.2%})")
print(f"  中位数方差: {var_median:.6f} (效率={crlb_mu/var_median:.2%})")
# 中位数效率约为 64%（对于正态分布）

# 5. CRLB 与大样本渐近性质
print("\nCRLB 与样本量的关系 (Gaussian μ):")
for n in [5, 10, 50, 200, 1000]:
    crlb = true_sigma**2 / n
    se = np.sqrt(crlb)
    print(f"  n={n:4d}: CRLB(μ)={crlb:.6f}, 标准误={se:.4f}")

# 6. 多参数 CRLB（协方差矩阵）
print("\n多参数 CRLB:")
p = 3
np.random.seed(42)
# 估计多元高斯均值
data = np.random.randn(100, p) * 0.5 + np.array([1, 2, 3])

emp_cov = np.cov(data.T)
fim = np.eye(p) / (0.5**2)  # 已知 σ=0.5
crlb_cov = np.linalg.inv(fim) / 100  # 除以 n

print(f"  经验协方差:\n{np.round(emp_cov, 4)}")
print(f"  CRLB 协方差:\n{np.round(crlb_cov, 4)}")
```

## 深度学习关联
- **CRLB 与深度学习中的估计问题**：在深度学习参数估计中，由于模型的大规模和高度非凸性，CRLB 通常无法直接计算。但 CRLB 的概念仍然有价值——它提醒我们，参数的估计精度受限于数据的信息量。在小数据或高噪声场景下，即使网络容量再大，某些参数的估计方差也会很大。
- **贝叶斯 CRLB (BCRLB)**：贝叶斯版本的 CRLB 考虑了参数的先验分布，给出了贝叶斯估计量的方差下界。在贝叶斯神经网络中，BCRLB 提供了后验估计精度的理论界限，有助于理解先验选择对不确定性量化的影响。
- **Fisher 信息矩阵与模型可辨识性**：CRLB 中需要 FIM 可逆（否则下界退化为零）。如果 FIM 是奇异的，说明参数不可辨识——不同参数组合给出相同的似然。在深度学习中，参数的对称性（如神经元置换不变性）使 FIM 天然奇异，这也是深度学习估计问题的一个基本挑战。
- **CRLB 与数据效率**：CRLB 揭示了数据效率的极限——要减少一半的标准误差，需要四倍的数据。这在数据稀缺的深度学习应用（如医疗影像、药物发现）中具有重要意义，也是少样本学习、数据增强和迁移学习等研究方向的重要动机。
