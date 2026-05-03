# 36_变分推断 (Variational Inference) 基础

## 核心概念

- **变分推断 (Variational Inference, VI)**：用简单分布 $q(z)$ 去近似复杂难解的后验分布 $p(z|x)$，通过最小化 $D_{\text{KL}}(q\|p)$ 或最大化 ELBO 来实现。将推断问题转化为优化问题。
- **ELBO (Evidence Lower Bound)**：
  $$\log p(x) = D_{\text{KL}}(q(z)\|p(z|x)) + \mathcal{L}(q) \geq \mathcal{L}(q)$$
  其中 $\mathcal{L}(q) = \mathbb{E}_q[\log p(x,z) - \log q(z)]$ 是证据下界。
- **平均场变分族 (Mean Field)**：假设隐变量相互独立，$q(z) = \prod_{i=1}^m q_i(z_i)$。这种分解简化了计算，但忽略了变量间的相关性。
- **坐标上升变分推断 (CAVI)**：轮流更新每个 $q_i(z_i)$：
  $$\log q_i^*(z_i) = \mathbb{E}_{j \neq i}[\log p(z_i|z_{-i}, x)] + \text{const}$$
  这是平均场 VI 的标准求解算法。
- **与 MCMC 的对比**：MCMC 通过采样逼近后验（计算密集但无偏），VI 通过优化逼近后验（更快速但可能有偏）。在大数据场景下 VI 更实用。
- **重参数化梯度 (Reparameterization Gradient)**：用可微变换 $z = g(\epsilon, \phi)$ 将随机性转移到噪声源 $\epsilon$，使 ELBO 对变分参数 $\phi$ 的梯度可以通过反向传播计算。这是深度学习中 VI 的核心技术。

## 数学推导

变分推断的目标：最小化 KL 散度
$$
q^*(z) = \arg\min_{q \in \mathcal{Q}} D_{\text{KL}}(q(z) \| p(z|x))
$$

KL 散度分解：
$$
D_{\text{KL}}(q\|p) = \mathbb{E}_q[\log q(z)] - \mathbb{E}_q[\log p(z|x)]
$$

ELBO 的两种等价形式：
$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(x, z)] - \mathbb{E}_q[\log q(z)]
$$
$$
\mathcal{L}(q) = \mathbb{E}_q[\log p(x|z)] - D_{\text{KL}}(q(z)\|p(z))
$$

CAVI 更新公式（平均场假设下）：
$$
q_i^*(z_i) \propto \exp\left(\mathbb{E}_{j \neq i}[\log p(z_i, z_{-i}, x)]\right)
$$

## 直观理解

- **"用简单形状包住复杂形状"**：后验分布是复杂形状的"云"，VI 试图用一个简单形状（如高斯）的"气球"去覆盖它。通过让气球膨胀到最像目标云的位置来近似。
- **MCMC vs VI 的比喻**：MCMC 像画家的细致描摹——花大量时间精确还原每个细节。VI 像摄影师的快速抓拍——虽然不够精确但速度很快。在大数据和深度学习场景中，"快拍"更重要。
- **为什么最小化反向 KL**：反向 KL $D_{\text{KL}}(q\|p)$ 倾向于找到 $p$ 的一个主要模式（mode-seeking），这在多模态后验中可能导致只覆盖一个峰。但这也是计算优势——反向 KL 只需要局部探索。

## 代码示例

```python
import numpy as np
from scipy.stats import norm, beta

# 1. 简单变分推断：用高斯分布近似 Beta 分布
# 目标后验: Beta(5, 5)（对称双峰形状的近似）
# 变分分布: N(mu, sigma^2)

target = beta(5, 5)

# 随机初始化变分参数
mu = 0.0
log_sigma = 0.0  # 用对数确保正性

def elbo_gaussian_approx(mu, log_sigma, n_samples=1000):
    """计算 ELBO = E_q[log p(z)] - E_q[log q(z)]"""
    sigma = np.exp(log_sigma)
    # 从 q 采样
    eps = np.random.randn(n_samples)
    z = mu + sigma * eps
    
    # log p(z) - Beta(5,5) log pdf
    log_p = target.logpdf(z)
    
    # log q(z) - N(mu, sigma) log pdf
    log_q = norm.logpdf(z, mu, sigma)
    
    return np.mean(log_p - log_q)

# 简单梯度上升
lr = 0.01
for t in range(1000):
    # 用蒙特卡洛估计梯度
    eps = np.random.randn(100)
    z = mu + np.exp(log_sigma) * eps
    
    grad_mu = np.mean(z - mu) / np.exp(log_sigma)**2  # 简化近似
    # 实际中应使用重参数化梯度，这里仅作演示
    
    # 使用数值梯度
    if t % 200 == 0:
        elbo = elbo_gaussian_approx(mu, log_sigma)
        print(f"迭代 {t:4d}: μ={mu:.4f}, σ={np.exp(log_sigma):.4f}, ELBO={elbo:.4f}")

# 2. 完整的 VI 实现：使用梯度上升优化 ELBO
def numerical_elbo_grad(mu, log_sigma, h=1e-4):
    """数值梯度"""
    elbo_base = elbo_gaussian_approx(mu, log_sigma, 200)
    elbo_mu = elbo_gaussian_approx(mu + h, log_sigma, 200)
    elbo_sigma = elbo_gaussian_approx(mu, log_sigma + h, 200)
    return (elbo_mu - elbo_base) / h, (elbo_sigma - elbo_base) / h

# 优化
mu, log_sigma = 0.3, -0.5
for t in range(500):
    g_mu, g_sigma = numerical_elbo_grad(mu, log_sigma)
    mu += 0.01 * g_mu
    log_sigma += 0.01 * g_sigma

sigma = np.exp(log_sigma)
print(f"\n最终结果: μ={mu:.4f}, σ={sigma:.4f}")
print(f"Beta(5,5) 理论均值=0.5, 标准差={np.sqrt(1/11):.4f}")

# 3. 变分后验 vs 真实后验
print("\n变分近似 vs 真实 Beta(5,5):")
print(f"  真实均值=0.5, 变分均值={mu:.4f}")
print(f"  真实标准差=0.301, 变分标准差={sigma:.4f}")

# 4. 不同先验下的后验近似
prior_mu, prior_sigma = 0, 1  # N(0,1) 先验
likelihood_mu, likelihood_sigma = 2, 0.5  # 似然
n_data = 10

# 真实后验（高斯-高斯共轭）
post_mu = (prior_mu/prior_sigma**2 + n_data*likelihood_mu/likelihood_sigma**2) / \
          (1/prior_sigma**2 + n_data/likelihood_sigma**2)
post_sigma = np.sqrt(1 / (1/prior_sigma**2 + n_data/likelihood_sigma**2))

print(f"\n共轭高斯后验: μ={post_mu:.4f}, σ={post_sigma:.4f}")
print("VI 可以近似这个后验而不需要共轭假设!")
```

## 深度学习关联

- **变分自编码器 (VAE)**：VAE 是 VI 在深度生成模型中最成功的应用。编码器 $q_\phi(z|x)$ 是变分分布，解码器 $p_\theta(x|z)$ 是生成模型。ELBO 通过重参数化技巧端到端优化：
  $$\mathcal{L}(\theta, \phi) = \mathbb{E}_{q_\phi(z|x)}[\log p_\theta(x|z)] - D_{\text{KL}}(q_\phi(z|x)\|p(z))$$
  VAE 将传统 VI 的坐标上升替换为随机梯度下降（SGD），使推断可扩展到大数据。

- **贝叶斯神经网络 (BNN)**：在 BNN 中，权重后验 $p(w|D)$ 高度复杂。VI 使用高斯分布 $q(w|\mu, \sigma)$ 近似权重后验，通过"Bayes by Backprop"（或称为梯度下降的变分学习）优化 ELBO：
  $$\mathcal{L}(\mu, \sigma) \approx \frac{1}{M} \sum_{i=1}^M \log p(y_i|x_i, w) - D_{\text{KL}}(q(w)\|p(w))$$
  这为深度模型提供了不确定性估计能力。

- **扩散模型的变分视角**：去噪扩散概率模型 (DDPM) 的训练也可以从 VI 角度理解。前向过程 $q(x_{1:T}|x_0)$ 固定，反向过程 $p_\theta(x_{0:T})$ 参数化。ELBO 分解为各时间步的去噪损失之和，简化了训练目标。
- **强化学习中的变分推断**：在基于模型的强化学习中，变分推断用于学习世界模型 $p(s_{t+1}|s_t, a_t)$ 的后验。在策略学习中，变分方法可以处理策略梯度的噪声问题，通过变分信息瓶颈 (VIB) 提高策略的鲁棒性。
