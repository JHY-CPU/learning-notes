# Jensen 不等式及其在 EM 算法中的应用

## 核心概念
- **Jensen 不等式**：对于凸函数 $f$，$f(\mathbb{E}[X]) \leq \mathbb{E}[f(X)]$。对于凹函数 $g$，$g(\mathbb{E}[X]) \geq \mathbb{E}[g(X)]$。等号成立当 $X$ 是确定性变量（几乎处处相等）。
- **凸函数与凹函数**：凸函数（convex）满足 $f(\lambda x + (1-\lambda)y) \leq \lambda f(x) + (1-\lambda)f(y)$，图像是"碗形"（如 $x^2$、$e^x$）。凹函数（concave）相反，如 $\log x$、$\sqrt{x}$。
- **Jensen 不等式在信息论中**：$\log$ 是凹函数，所以 $\mathbb{E}[\log X] \leq \log \mathbb{E}[X]$。这是推导 KL 散度非负性和 VAEs 中 ELBO 的关键。
- **EM 算法 (Expectation-Maximization)**：处理含隐变量 $Z$ 的模型参数估计的迭代算法。交替执行 E 步（计算后验期望）和 M 步（最大化期望对数似然）。
- **ELBO 与 Jensen 不等式**：EM 算法使用 Jensen 不等式构造对数似然的下界（ELBO），通过最大化该下界间接最大化似然。

## 数学推导
Jensen 不等式（离散形式）：
$$
f\left(\sum_i \lambda_i x_i\right) \leq \sum_i \lambda_i f(x_i)
$$
其中 $\lambda_i \geq 0$，$\sum_i \lambda_i = 1$。

对于凹函数 $g$：
$$
g\left(\sum_i \lambda_i x_i\right) \geq \sum_i \lambda_i g(x_i)
$$

EM 算法的推导：对数似然 $\log p(X|\theta)$ 分解为：
$$
\log p(X|\theta) = \log \sum_Z p(X, Z|\theta)
$$

引入隐变量分布 $q(Z)$，应用 Jensen 不等式：
$$
\log p(X|\theta) = \log \sum_Z q(Z) \frac{p(X, Z|\theta)}{q(Z)}
\geq \sum_Z q(Z) \log \frac{p(X, Z|\theta)}{q(Z)} \quad (\text{因为 log 是凹函数})
$$

这个下界就是 ELBO（证据下界）：
$$
\mathcal{L}(q, \theta) = \sum_Z q(Z) \log \frac{p(X, Z|\theta)}{q(Z)} = \mathbb{E}_{q}[\log p(X, Z|\theta)] - \mathbb{E}_{q}[\log q(Z)]
$$

EM 算法的两步：
- **E-step**：固定 $\theta$，令 $q(Z) = p(Z|X, \theta)$，使 ELBO = 对数似然（Jensen 等号成立）
- **M-step**：固定 $q(Z)$，最大化 $\mathbb{E}_{q}[\log p(X, Z|\theta)]$ 得到新的 $\theta$

## 直观理解
- **Jensen 不等式的"弯曲"效应**：凸函数 $f$ 的曲线在弦的上方，弦上的点（对应 $\lambda x + (1-\lambda)y$ 的函数值）在曲线的下方。所以"平均的函数值" $\geq$ "函数的平均值"。
- **EM 的"爬山"类比**：EM 不是直接优化似然函数（可能很复杂），而是构造一个下界（ELBO）来逼近它。E 步让下界在当前参数处紧贴似然，M 步推高下界。每步都保证似然不下降。
- **为什么 EM 有效**：如果你看不到全局（对数似然复杂），就先构造一个局部代理（ELBO），在这个代理上优化，然后更新代理让它更准确。反复迭代，最终收敛到局部最优。

## 代码示例
```python
import numpy as np
from scipy.stats import norm

# 1. Jensen 不等式验证
# 凸函数 f(x) = x^2
x = np.array([1, 2, 3, 4])
weights = np.array([0.2, 0.3, 0.3, 0.2])

mean_x = np.average(x, weights=weights)
f_mean_x = mean_x**2  # f(E[X])
mean_f_x = np.average(x**2, weights=weights)  # E[f(X)]

print("Jensen 不等式 (凸函数 f(x)=x^2):")
print(f"  f(E[X]) = {f_mean_x:.4f}")
print(f"  E[f(X)] = {mean_f_x:.4f}")
print(f"  f(E[X]) <= E[f(X)]: {f_mean_x <= mean_f_x}")

# 凹函数 g(x) = log(x)
g_mean_x = np.log(mean_x)
mean_g_x = np.average(np.log(x), weights=weights)
print(f"\nJensen 不等式 (凹函数 log(x)):")
print(f"  g(E[X]) = {g_mean_x:.4f}")
print(f"  E[g(X)] = {mean_g_x:.4f}")
print(f"  g(E[X]) >= E[g(X)]: {g_mean_x >= mean_g_x}")

# 2. 简单 EM 算法：高斯混合模型
np.random.seed(42)

# 生成两个高斯混合数据
true_mu1, true_mu2 = -2, 3
true_sigma1, true_sigma2 = 1.0, 1.5
true_pi = 0.4  # 第一个成分的权重

n = 300
z = np.random.binomial(1, 1-true_pi, n)  # 隐变量
data = np.where(z == 0, 
                np.random.randn(n) * true_sigma1 + true_mu1,
                np.random.randn(n) * true_sigma2 + true_mu2)

# EM 算法估计参数
def em_gmm(data, n_iter=50):
    # 初始化
    mu1, mu2 = -1, 2
    sigma1, sigma2 = 1.0, 1.0
    pi = 0.5
    
    for t in range(n_iter):
        # E-step: 计算责任度 (后验概率)
        gamma1 = norm.pdf(data, mu1, sigma1) * pi
        gamma2 = norm.pdf(data, mu2, sigma2) * (1 - pi)
        sum_gamma = gamma1 + gamma2 + 1e-10
        r1 = gamma1 / sum_gamma  # P(z=1 | x)
        r2 = 1 - r1
        
        # M-step: 更新参数
        n1 = np.sum(r1)
        n2 = np.sum(r2)
        mu1 = np.sum(r1 * data) / n1
        mu2 = np.sum(r2 * data) / n2
        sigma1 = np.sqrt(np.sum(r1 * (data - mu1)**2) / n1)
        sigma2 = np.sqrt(np.sum(r2 * (data - mu2)**2) / n2)
        pi = n1 / len(data)
        
        if t == 0 or t == n_iter-1:
            ll = np.sum(np.log(sum_gamma))
            print(f"  迭代 {t+1:2d}: μ1={mu1:.3f}, μ2={mu2:.3f}, π={pi:.3f}, LL={ll:.2f}")
    
    return mu1, mu2, sigma1, sigma2, pi

print("\nEM 算法估计高斯混合模型:")
mu1_est, mu2_est, s1_est, s2_est, pi_est = em_gmm(data)
print(f"\n真实参数: μ1={true_mu1}, μ2={true_mu2}, π={true_pi}")
print(f"估计参数: μ1={mu1_est:.3f}, μ2={mu2_est:.3f}, π={pi_est:.3f}")

# 3. 验证 ELBO 单调递增
def compute_elbo(data, mu1, mu2, sigma1, sigma2, pi):
    """计算 ELBO"""
    gamma1 = norm.pdf(data, mu1, sigma1) * pi
    gamma2 = norm.pdf(data, mu2, sigma2) * (1-pi)
    sum_gamma = gamma1 + gamma2 + 1e-10
    r1 = gamma1 / sum_gamma
    r2 = 1 - r1
    
    # ELBO = E[log p(x,z)] - E[log q(z)]
    elbo1 = np.sum(r1 * (np.log(pi) + norm.logpdf(data, mu1, sigma1)))
    elbo2 = np.sum(r2 * (np.log(1-pi) + norm.logpdf(data, mu2, sigma2)))
    entropy = -np.sum(r1 * np.log(r1 + 1e-10) + r2 * np.log(r2 + 1e-10))
    
    return elbo1 + elbo2 + entropy

print(f"\n收敛时 ELBO: {compute_elbo(data, mu1_est, mu2_est, s1_est, s2_est, pi_est):.2f}")
```

## 深度学习关联
- **变分自编码器 (VAE)**：VAE 的核心是 ELBO 最大化，这正是 Jensen 不等式的直接应用：
  $$\log p(x) \geq \mathbb{E}_{q(z|x)}[\log p(x|z)] - D_{\text{KL}}(q(z|x)\|p(z))$$
  编码器 $q(z|x)$ 是 E 步的近似，解码器 $p(x|z)$ 是 M 步的生成模型。VAE 本质上是用神经网络实现的"AM 算法"（交替最大化 ELBO）。

- **EM 与 K-Means 的关系**：K-Means 是高斯混合模型 EM 算法的硬聚类极限情况（方差 $\to 0$）。E 步对应将每个点分配到最近的聚类中心（硬分配而非软分配），M 步更新聚类中心为分配点的均值。理解 EM 有助于理解各种聚类算法的联系。

- **变分推断 (VI) 与深度贝叶斯**：在贝叶斯神经网络中，后验分布 $p(w|D)$ 难以计算。变分推断使用简单分布 $q(w)$ 逼近真实后验，最小化 $D_{\text{KL}}(q\|p)$，等价于最大化 ELBO（由 Jensen 不等式保证）。这是深度贝叶斯学习方法（如 Bayes by Backprop）的理论基础。

- **扩散模型中的 ELBO**：去噪扩散概率模型 (DDPM) 的训练目标也可以从 ELBO 视角理解。前向扩散过程添加噪声，反向去噪过程学习去噪，其损失函数 $\mathbb{E}[\|\epsilon - \epsilon_\theta(z_t, t)\|^2]$ 来自 ELBO 的重新加权形式。
