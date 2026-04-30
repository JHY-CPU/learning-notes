# 费雪信息矩阵 (Fisher Information Matrix)

## 核心概念
- **费雪信息 (Fisher Information)**：衡量观测数据 $X$ 关于未知参数 $\theta$ 的信息量。定义为对数似然函数二阶矩（或负的二阶期望）：
  $$I(\theta) = \mathbb{E}\left[\left(\frac{\partial}{\partial \theta} \log p(X|\theta)\right)^2\right] = -\mathbb{E}\left[\frac{\partial^2}{\partial \theta^2} \log p(X|\theta)\right]$$
- **费雪信息矩阵 (Fisher Information Matrix, FIM)**：多参数情况下的推广。$F(\theta) \in \mathbb{R}^{p \times p}$，其中：
  $$F_{ij}(\theta) = \mathbb{E}\left[ \frac{\partial \log p(x|\theta)}{\partial \theta_i} \frac{\partial \log p(x|\theta)}{\partial \theta_j} \right]$$
  等价于对数似然函数的梯度协方差矩阵。
- **分数函数 (Score Function)**：$s(\theta) = \nabla_\theta \log p(x|\theta)$。期望为零 $\mathbb{E}[s(\theta)] = 0$，FIM 是分数函数的协方差矩阵。
- **Fisher 信息与 Hessian**：在正则条件下，FIM 等价于负对数似然 Hessian 的期望：
  $$F(\theta) = -\mathbb{E}[\nabla^2_\theta \log p(x|\theta)]$$
- **FIM 的性质**：半正定矩阵，可逆时正定。在指数族分布中有简洁的闭式表达式：$F(\eta) = \nabla^2 A(\eta)$，其中 $A$ 是对数配分函数。
- **自然梯度 (Natural Gradient)**：使用 FIM 的逆调整梯度方向：
  $$\tilde{\nabla} L(\theta) = F(\theta)^{-1} \nabla L(\theta)$$
  自然梯度在参数空间中是"坐标无关"的，在 Riemannian 流形上沿最陡方向下降。

## 数学推导
费雪信息的定义（一元）：
$$
I(\theta) = \mathbb{E}\left[ \left( \frac{\partial}{\partial \theta} \log p(x|\theta) \right)^2 \right] = \int \left( \frac{\partial \log p}{\partial \theta} \right)^2 p(x|\theta) dx
$$

信息恒等式（假设可以交换积分和求导）：
$$
\mathbb{E}\left[ \frac{\partial}{\partial \theta} \log p(x|\theta) \right] = 0
$$

费雪信息的另一个形式（使用二阶导数）：
$$
I(\theta) = -\mathbb{E}\left[ \frac{\partial^2}{\partial \theta^2} \log p(x|\theta) \right]
$$

Cramer-Rao 下界（无偏估计量的方差下界）：
$$
\text{Var}(\hat{\theta}) \geq \frac{1}{I(\theta)}
$$

指数族分布的 FIM：
$$
F_{ij}(\eta) = \frac{\partial^2 A(\eta)}{\partial \eta_i \partial \eta_j}
$$

## 直观理解
- **"信息量"的度量**：费雪信息回答了"参数 $\theta$ 的微小变化能在多大程度上影响观测数据的分布？"如果似然函数对 $\theta$ 非常敏感（导数大），则费雪信息大，说明数据包含丰富的 $\theta$ 信息，估计会更精确。
- **似然函数的"尖峰"**：费雪信息大意味着对数似然在 $\theta$ 附近有尖锐的峰值（曲率大），说明 $\theta$ 被数据高度确定。反之，如果费雪信息小，对数似然平坦，说明数据对 $\theta$ 的信息有限。
- **几何视角**：FIM 定义了参数空间上的 Riemannian 度量。在 FIM 度量下，参数空间中"距离"表示分布之间的 KL 散度（二阶近似）：$D_{\text{KL}}(p_\theta\|p_{\theta+\delta}) \approx \frac{1}{2} \delta^T F(\theta) \delta$。

## 代码示例
```python
import numpy as np
from scipy.stats import norm, bernoulli
from scipy.special import polygamma

# 1. 伯努利分布的费雪信息
# Bernoulli(p): log L = x log p + (1-x) log(1-p)
# score: x/p - (1-x)/(1-p)
# Fisher: E[(x/p - (1-x)/(1-p))^2] = 1/(p(1-p))

p_vals = np.linspace(0.1, 0.9, 9)
for p in p_vals:
    fisher = 1 / (p * (1-p))
    print(f"Bernoulli(p={p:.1f}): Fisher Information = {fisher:.2f}")

# 2. 高斯分布的费雪信息（已知方差）
# N(μ, σ²): Fisher(μ) = 1/σ², Fisher(σ²) = 1/(2σ⁴)
sigma = 2.0
fisher_mu = 1 / sigma**2
fisher_sigma = 1 / (2 * sigma**4)
print(f"\n高斯分布 N(μ, σ={sigma}):")
print(f"  Fisher(μ) = {fisher_mu:.4f}")
print(f"  Fisher(σ²) = {fisher_sigma:.6f}")

# 3. 数值验证费雪信息
np.random.seed(42)

def empirical_fisher(data, log_lik_grad):
    """基于样本的费雪信息估计"""
    scores = log_lik_grad(data)
    return np.mean(scores**2)

# 伯努利实验
true_p = 0.3
n_samples = 100000
data = bernoulli.rvs(true_p, size=n_samples)

score_fn = lambda x: x/true_p - (1-x)/(1-true_p)
emp_fisher = empirical_fisher(data, score_fn)
theory_fisher = 1 / (true_p * (1-true_p))

print(f"\n费雪信息验证 (Bernoulli p={true_p}):")
print(f"  理论值: {theory_fisher:.4f}")
print(f"  估计值: {emp_fisher:.4f}")

# 4. 费雪信息矩阵（高斯分布）
# 二维高斯，未知 (μ₁, μ₂)
def gaussian_fim(dim, sigma=1.0):
    """独立同分布高斯的 FIM 是对角矩阵"""
    return np.eye(dim) / sigma**2

F_2d = gaussian_fim(2, sigma=0.5)
print(f"\n二维高斯 FIM (σ=0.5):")
print(F_2d)

# 5. 自然梯度 vs 标准梯度
# 简单演示：用伯努利分布的 FIM 调整梯度

def log_likelihood_bernoulli(p, data):
    """伯努利对数似然"""
    n = len(data)
    successes = np.sum(data)
    return successes * np.log(p) + (n - successes) * np.log(1-p)

def grad_log_likelihood(p, data):
    """对数似然的梯度"""
    n = len(data)
    successes = np.sum(data)
    return successes/p - (n-successes)/(1-p)

def natural_grad(p, data):
    """自然梯度 = F(p)^{-1} * 标准梯度"""
    F = 1 / (p * (1-p))  # FIM
    g = grad_log_likelihood(p, data)
    return g / F  # 自然梯度

data = bernoulli.rvs(0.5, size=100)
for p in [0.1, 0.3, 0.5, 0.7, 0.9]:
    std_g = grad_log_likelihood(p, data)
    nat_g = natural_grad(p, data)
    print(f"  p={p:.1f}: 标准梯度={std_g:.2f}, 自然梯度={nat_g:.2f}")

# 6. FIM 与 KL 散度的二阶近似
print("\nFIM 与 KL 散度:")
p0 = 0.5
F0 = 1 / (p0 * (1-p0))

for delta in [0.05, 0.1, 0.2]:
    p1 = p0 + delta
    # KL(P0 || P1) 精确值
    kl_true = p0 * np.log(p0/p1) + (1-p0) * np.log((1-p0)/(1-p1))
    # 二阶近似: 0.5 * FIM * delta^2
    kl_approx = 0.5 * F0 * delta**2
    print(f"  δ={delta:.2f}: KL精确={kl_true:.6f}, 二阶近似={kl_approx:.6f}")
```

## 深度学习关联
- **自然梯度下降 (Natural Gradient Descent)**：在深度学习中，标准梯度下降没有考虑参数空间的几何结构。自然梯度使用 FIM 的逆调整更新方向，使更新步长在 KL 散度意义上一致。尽管计算完整 FIM 的逆在大尺度上不可行（$O(p^3)$），但 K-FAC (Kronecker-Factored Approximate Curvature) 通过近似 FIM 为两个小矩阵的 Kronecker 乘积实现了可扩展的自然梯度。
- **Fisher 信息作为优化景观的曲率**：FIM 等价于（非负）Hessian 矩阵的期望。在深度网络的优化景观中，FIM 的特征值谱揭示了不同参数方向对损失的影响。大特征值方向对应"敏感"方向（小幅参数变化就能改变预测），小特征值方向对应"冗余"方向。这对剪枝（去掉小 Fisher 信息的参数）和量化（对小 Fisher 信息参数使用更低精度）有指导意义。
- **弹性权重巩固 (EWC)**：EWC 是一种持续学习方法，通过正则化防止忘记已学任务。其正则化项使用 Fisher 信息矩阵的对角线作为参数重要性权重：
  $$\mathcal{L}(\theta) = \mathcal{L}_B(\theta) + \sum_i \frac{\lambda}{2} F_i (\theta_i - \theta_{A,i}^*)^2$$
  其中 $F_i$ 是任务 A 的 Fisher 信息对角元素，衡量每个参数对任务 A 的重要性。这使网络在学习新任务时不会大幅改变重要参数。
- **Fisher 信息与模型可解释性**：在模型压缩和网络剪枝中，Fisher 信息可以识别"不重要"的参数——那些微调后对输出影响很小的参数（Fisher 信息接近零）。基于 Fisher 信息的剪枝方法（如基于 Fisher 的权重显著性评估）在大规模模型压缩中取得了优异效果。
