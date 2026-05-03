# 35_重要性采样 (Importance Sampling) 原理

## 核心概念

- **重要性采样**：当从目标分布 $p(x)$ 采样困难时，从容易采样的提议分布 $q(x)$ 中采样，通过加权修正来估计 $\mathbb{E}_p[f(X)]$：
  $$\mathbb{E}_p[f(X)] = \mathbb{E}_q\left[f(X) \frac{p(X)}{q(X)}\right] \approx \frac{1}{n} \sum_{i=1}^n f(x_i) \frac{p(x_i)}{q(x_i)}$$
- **重要性权重**：$w(x_i) = p(x_i) / q(x_i)$，修正了从 $q$ 采样带来的偏差。权重大的样本更重要。
- **无偏性**：重要性采样估计是无偏的，$\mathbb{E}_q[w(x) f(x)] = \mathbb{E}_p[f(x)]$。但前提是 $q(x) > 0$ 时 $p(x) > 0$（支撑包含条件）。
- **方差问题**：重要性采样的方差为 $\text{Var}_q[w(x) f(x)]$。如果 $q$ 与 $p$ 差异很大，少数样本的权重会极大，导致方差剧增甚至发散。
- **自归一化重要性采样 (SNIS)**：当 $p(x)$ 仅已知未归一化的 $\tilde{p}(x)$ 时，使用归一化权重 $\tilde{w}_i = \tilde{p}(x_i)/q(x_i) / \sum_j \tilde{p}(x_j)/q(x_j)$，得到有偏但一致的估计。
- **最优提议分布**：理想情况下，$q^*(x) \propto |f(x)| p(x)$，可使方差最小。但实际上 $q^*$ 的归一化常数正是我们要求的目标。

## 数学推导

重要性采样基本等式：
$$
\mathbb{E}_p[f(X)] = \int f(x) p(x) dx = \int f(x) \frac{p(x)}{q(x)} q(x) dx = \mathbb{E}_q\left[f(X) \frac{p(X)}{q(X)}\right]
$$

蒙特卡洛估计：
$$
\hat{I}_{\text{IS}} = \frac{1}{n} \sum_{i=1}^n f(x_i) w(x_i), \quad x_i \sim q(x), \; w_i = \frac{p(x_i)}{q(x_i)}
$$

估计量的方差：
$$
\text{Var}(\hat{I}_{\text{IS}}) = \frac{1}{n} \left( \mathbb{E}_q[f(x)^2 w(x)^2] - I^2 \right)
$$

当 $q(x) = p(x)$ 时，$w(x) = 1$，方差降至蒙特卡洛基准。
当 $q(x)$ 在 $f(x)p(x)$ 大的区域分配的概率太小时，$w(x)$ 可能极大，方差失控。

有效样本量 (Effective Sample Size)：
$$
\text{ESS} = \frac{(\sum_i w_i)^2}{\sum_i w_i^2}
$$
衡量实际有效的样本数量。ESS 越小，重要性采样越不稳定。

## 直观理解

- **"借鸡生蛋"的估计**：想了解某个人群的收入分布，但难以直接抽样。可以从容易获得的群体（如在线调查）中采样，然后根据每个人与目标人群的"相似度"（权重）来修正偏差。这就是重要性采样的思想。
- **高权重 = 重要但稀有**：如果 $q$ 很少在 $p$ 概率高的区域采样，一旦偶尔采到一个该区域的样本，其重要性权重会非常大（因为 $p/q$ 大）。这少量样本会主导估计，导致高方差。
- **提议分布的选择**：好 $q$ 应该"覆盖" $p$ 的所有重要区域，并尽量集中在 $|f(x)| p(x)$ 大的地方。就像调查民意时，如果你要找年轻选民的意见，就要多在年轻人聚集的地方发放问卷。

## 代码示例

```python
import numpy as np

# 1. 重要性采样估计期望
np.random.seed(42)

# 目标分布 p: 标准正态分布 N(0,1)
# 提议分布 q: 柯西分布（厚尾，容易采样）
# 用重要性采样估计 E_p[f(x)]，其中 f(x) = x^2，真实值 = 1

true_val = 1.0  # E[N(0,1)^2] = 1

def importance_sampling(n, p_log_prob, q_sampler, q_log_prob, f):
    samples = q_sampler(n)
    log_weights = p_log_prob(samples) - q_log_prob(samples)
    weights = np.exp(log_weights - np.max(log_weights))  # 数值稳定
    weights = weights / np.sum(weights)  # 自归一化
    est = np.sum(weights * f(samples))
    ess = 1 / np.sum(weights**2)  # 有效样本量
    return est, ess

# p = N(0,1), q = Cauchy(0,1)
from scipy.stats import norm, cauchy

p = norm(0, 1)
q = cauchy(0, 1)

for n in [100, 1000, 10000]:
    est, ess = importance_sampling(
        n, p.logpdf, q.rvs, q.logpdf, lambda x: x**2
    )
    print(f"重要性采样 n={n:6d}: E[x^2]≈{est:.4f}, 有效样本量≈{ess:.1f}")

# 2. 提议分布选择的重要性
print("\n提议分布对比 (n=1000, E[x^2]):")
for q_dist, name in [(norm(0, 1), "N(0,1) (最优)"),
                      (norm(0, 2), "N(0,2) (宽)"),
                      (norm(0, 0.5), "N(0,0.5) (窄)"),
                      (cauchy(0, 1), "Cauchy (厚尾)")]:
    est, ess = importance_sampling(
        1000, p.logpdf, q_dist.rvs, q_dist.logpdf, lambda x: x**2
    )
    print(f"  {name:15s}: E[x^2]≈{est:.4f}, ESS≈{ess:.1f}")

# 3. 重要性采样在期望估计中的方差对比
n_trials = 500
n_samples = 500

# 方法1: 从 p 直接采样
direct_ests = [np.mean(np.random.randn(n_samples)**2) for _ in range(n_trials)]

# 方法2: 重要性采样（提议分布为柯西）
def is_estimate(n):
    samples = cauchy.rvs(n)
    log_w = p.logpdf(samples) - cauchy.logpdf(samples)
    w = np.exp(log_w - np.max(log_w))
    w = w / np.sum(w)
    return np.sum(w * samples**2)

is_ests = [is_estimate(n_samples) for _ in range(n_trials)]

print(f"\n方差对比 (500次重复):")
print(f"  直接采样: 均值={np.mean(direct_ests):.4f}, 方差={np.var(direct_ests):.6f}")
print(f"  重要性采样: 均值={np.mean(is_ests):.4f}, 方差={np.var(is_ests):.6f}")

# 4. 尾部概率估计
# 估计 P(X > 5) 当 X ~ N(0,1)，真实值 ≈ 2.87e-7
print(f"\n尾部概率 P(Z > 5) 估计:")
true_prob = 1 - norm.cdf(5)
print(f"  真实值: {true_prob:.2e}")

# 直接采样需要天文数字的样本量，使用重要性采样
# 提议分布: N(5, 1)（集中在尾部区域）
q_tail = norm(5, 1)
n_samples = 100000
samples = q_tail.rvs(n_samples)
log_w = p.logpdf(samples) - q_tail.logpdf(samples)
w = np.exp(log_w)
is_est = np.mean(w * (samples > 5).astype(float))
print(f"  重要性采样估计: {is_est:.2e} (n={n_samples})")
```

## 深度学习关联

- **策略梯度中的重要性采样**：在强化学习 PPO 算法中，重要性采样用于重用旧策略收集的数据：
  $$L^{\text{CLIP}}(\theta) = \mathbb{E}_t\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]$$
  其中 $r_t(\theta) = \pi_\theta(a_t|s_t) / \pi_{\theta_{\text{old}}}(a_t|s_t)$ 就是重要性权重。剪裁操作防止权重过大导致训练不稳定，正是为解决重要性采样的高方差问题。

- **off-policy 学习**：在 Q-learning 和 SAC 中，智能体使用旧策略收集的经验缓冲区来更新当前策略，属于典型的 off-policy 学习。重要性采样修正行为策略和目标策略之间的不匹配。现代算法（如 SAC）通常对重要性权重进行截断或使用双 Q 网络来控制方差。
- **变分推断中的重要性加权**：重要性加权自编码器 (IWAE) 使用多个重要性样本获得更紧的 ELBO 下界：
  $$\mathcal{L}_k = \mathbb{E}_{z_1,\dots,z_k \sim q} \left[\log \frac{1}{k} \sum_{i=1}^k \frac{p(x, z_i)}{q(z_i|x)}\right]$$
  相比 VAE 的单样本 ELBO，IWAE 的多个重要性样本提供了更准确的梯度估计，尤其在学习复杂后验分布时效果更好。

- **贝叶斯深度学习中的 MCMC**：在贝叶斯神经网络中，从后验 $p(w|D)$ 采样通常不可行。重要性采样和序贯蒙特卡洛 (SMC) 等方法通过从提议分布采样并加权来近似后验期望，用于模型平均和不确定性量化。
