# 最大似然估计 (MLE) 推导

## 核心概念
- **似然函数 (Likelihood Function)**：给定参数 $\theta$，观测到数据 $D = \{x_1, \dots, x_n\}$ 的概率（或概率密度）。对于独立同分布数据，$L(\theta|D) = \prod_{i=1}^n p(x_i|\theta)$。
- **最大似然估计 (Maximum Likelihood Estimation)**：寻找使似然函数最大的参数值 $\hat{\theta}_{\text{MLE}} = \arg\max_\theta L(\theta|D)$。MLE 选择使当前数据"最可能"出现的参数。
- **对数似然 (Log-Likelihood)**：将乘积转化为求和：$\ell(\theta) = \log L(\theta) = \sum_{i=1}^n \log p(x_i|\theta)$。对数似然在优化上更便利（单调性保护极值点）。
- **MLE 的渐近性质**：在大样本下，MLE 是一致的（收敛到真实参数）、渐近正态的（$\hat{\theta} \sim \mathcal{N}(\theta_0, I(\theta_0)^{-1})$）、渐近有效的（达到 Cramer-Rao 下界）。
- **MLE 与损失函数的关系**：MLE 等价于最小化负对数似然（NLL）。例如，高斯 MLE 等价于 MSE 损失，伯努利 MLE 等价于二元交叉熵损失。

## 数学推导
独立同分布数据的似然函数：
$$
L(\theta|D) = \prod_{i=1}^n p(x_i|\theta)
$$

对数似然：
$$
\ell(\theta) = \sum_{i=1}^n \log p(x_i|\theta)
$$

MLE 估计量：
$$
\hat{\theta}_{\text{MLE}} = \arg\max_\theta \ell(\theta) = \arg\min_\theta -\ell(\theta)
$$

求解 MLE（对对数似然求导并设为 0）：
$$
\frac{\partial \ell(\theta)}{\partial \theta} = \sum_{i=1}^n \frac{\partial \log p(x_i|\theta)}{\partial \theta} = 0
$$

### 高斯分布 MLE 推导
设 $x_i \sim \mathcal{N}(\mu, \sigma^2)$：
$$
\ell(\mu, \sigma^2) = \sum_{i=1}^n \left[ -\frac{1}{2}\log(2\pi) - \frac{1}{2}\log\sigma^2 - \frac{(x_i-\mu)^2}{2\sigma^2} \right]
$$

对 $\mu$ 求导：
$$
\frac{\partial \ell}{\partial \mu} = \sum_{i=1}^n \frac{x_i - \mu}{\sigma^2} = 0 \quad \Rightarrow \quad \hat{\mu}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i
$$

对 $\sigma^2$ 求导：
$$
\frac{\partial \ell}{\partial \sigma^2} = -\frac{n}{2\sigma^2} + \frac{1}{2\sigma^4}\sum_{i=1}^n (x_i-\mu)^2 = 0 \quad \Rightarrow \quad \hat{\sigma}^2_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n (x_i - \bar{x})^2
$$

### 伯努利分布 MLE
设 $x_i \in \{0,1\}$：
$$
\ell(p) = \sum_{i=1}^n [x_i\log p + (1-x_i)\log(1-p)]
$$
$$
\frac{d\ell}{dp} = \sum_{i=1}^n \frac{x_i}{p} - \frac{1-x_i}{1-p} = 0 \quad \Rightarrow \quad \hat{p}_{\text{MLE}} = \frac{1}{n}\sum_{i=1}^n x_i
$$

## 直观理解
- **"最合理的"解释**：假设你看到一个人 10 次投篮中了 8 次。如果猜他的真实命中率是 0.9，那看到 8/10 的概率是多少？如果是 0.7 呢？MLE 就是在问：什么参数值使得我看到的这个结果最"正常"？答案是 0.8（因为均值就是 8/10）。
- **频率学派的立场**：MLE 是频率统计的核心。它把参数视为固定但未知的常数，数据是随机的。这与贝叶斯方法将参数视为随机变量形成对比。
- **负对数似然 = 损失**：最小化负对数似然等价于最大化似然。这个"负对数似然损失"连接了统计估计和机器学习中的经验风险最小化。

## 代码示例
```python
import numpy as np
from scipy.optimize import minimize
from scipy.stats import norm

# 1. 高斯分布 MLE
np.random.seed(42)
true_mu, true_sigma = 3.0, 1.5
data = np.random.randn(1000) * true_sigma + true_mu

# 解析 MLE
mu_mle = np.mean(data)
sigma_mle = np.std(data)
print(f"真实: μ={true_mu}, σ={true_sigma}")
print(f"MLE:  μ={mu_mle:.4f}, σ={sigma_mle:.4f}")
print(f"误差:  Δμ={abs(mu_mle-true_mu):.4f}, Δσ={abs(sigma_mle-true_sigma):.4f}")

# 2. 数值优化求 MLE（负对数似然最小化）
def neg_log_likelihood(params, data):
    mu, sigma = params
    if sigma <= 0:
        return np.inf
    n = len(data)
    return n/2 * np.log(2*np.pi) + n*np.log(sigma) + np.sum((data-mu)**2)/(2*sigma**2)

result = minimize(neg_log_likelihood, x0=[0, 1], args=(data,), method='L-BFGS-B',
                  bounds=[(None, None), (1e-6, None)])
mu_num, sigma_num = result.x
print(f"\n数值优化 MLE: μ={mu_num:.4f}, σ={sigma_num:.4f}")

# 3. 伯努利分布 MLE
np.random.seed(42)
true_p = 0.35
bernoulli_data = np.random.binomial(1, true_p, size=500)
p_mle = np.mean(bernoulli_data)
print(f"\n伯努利: 真实 p={true_p}, MLE p={p_mle:.4f}")
print(f"        方差估计: {p_mle*(1-p_mle)/500:.4f}")

# 4. MLE 渐近正态性演示
def demo_mle_asymptotics(true_p=0.3, n_samples=100, n_trials=5000):
    mle_estimates = []
    for _ in range(n_trials):
        data = np.random.binomial(1, true_p, size=n_samples)
        mle_estimates.append(np.mean(data))

    mle_mean = np.mean(mle_estimates)
    mle_std = np.std(mle_estimates)
    # 理论渐近标准差: sqrt(p(1-p)/n)
    theoretical_std = np.sqrt(true_p * (1-true_p) / n_samples)
    print(f"\nMLE 渐近性 (n={n_samples}):")
    print(f"  MLE 均值: {mle_mean:.4f} (真实 p={true_p})")
    print(f"  MLE 标准差: {mle_std:.4f} (理论={theoretical_std:.4f})")

demo_mle_asymptotics(true_p=0.3, n_samples=50)
demo_mle_asymptotics(true_p=0.3, n_samples=500)

# 5. 高斯 MLE = MSE 损失
X = np.random.randn(100, 10)
true_w = np.random.randn(10)
y = X @ true_w + 0.1 * np.random.randn(100)

# MLE（高斯噪声假设）<=> 最小二乘
w_mle = np.linalg.lstsq(X, y, rcond=None)[0]
mse = np.mean((X @ w_mle - y)**2)
print(f"\n线性回归 MLE: MSE = {mse:.6f}")
print(f"权重恢复误差: {np.linalg.norm(w_mle - true_w):.4f}")
```

## 深度学习关联
- **交叉熵损失 = 伯努利/多项式 MLE**：二分类交叉熵损失 $-\sum_i [y_i \log \hat{y}_i + (1-y_i) \log (1-\hat{y}_i)]$ 正是伯努利分布的负对数似然。多分类交叉熵 $-\sum_i \sum_c y_{ic} \log \hat{y}_{ic}$ 是类别分布（多项式分布单次试验）的负对数似然。
- **MSE 损失 = 高斯 MLE**：回归任务中的均方误差 (MSE) 损失等价于高斯分布假设下的负对数似然。具体地，如果假设 $p(y|x) = \mathcal{N}(f(x), \sigma^2)$，则 MLE 给出普通最小二乘解 $f(x)^*$。
- **MLE 在深度生成模型中的应用**：GAN 的判别器训练可以看作二分类 MLE（区分真实和生成样本）；自回归模型（如 PixelCNN、GPT）每一步预测下一个 token 的分布，整体训练目标是整个序列的 MLE。
- **MLE 的局限性**：MLE 在数据有限时容易过拟合（尤其当模型容量大时）。例如，如果只有一个样本，高斯 MLE 估计的方差为 0。这也是正则化（相当于 MAP 估计或对似然施加惩罚）在深度学习中必不可少的原因。
