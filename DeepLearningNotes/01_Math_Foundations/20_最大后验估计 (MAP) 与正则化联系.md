# 最大后验估计 (MAP) 与正则化联系

## 核心概念
- **最大后验估计 (Maximum A Posteriori)**：寻找使后验概率最大的参数：$\hat{\theta}_{\text{MAP}} = \arg\max_\theta p(\theta|D) = \arg\max_\theta p(D|\theta) p(\theta)$。MAP 结合了先验知识和观测数据。
- **MAP 与 MLE 的关系**：当先验 $p(\theta)$ 是均匀分布（无信息先验）时，MAP = MLE。MAP 可以理解为 MLE 加上先验信息的"修正"。
- **先验作为正则化**：L2 正则化等价于高斯先验下的 MAP 估计。L1 正则化等价于拉普拉斯先验下的 MAP 估计。这是连接频率派和贝叶斯派的核心桥梁。
- **后验模式 (Posterior Mode)**：MAP 估计的是后验分布的"峰值"（众数），而非均值或中位数。在对称分布中三者一致，在偏态分布中不同。
- **MAP 的局限性**：MAP 是后验分布的点估计，不提供不确定性信息。相比之下，完整贝叶斯推断使用整个后验分布进行预测（通过积分），不确定性更丰富。

## 数学推导
贝叶斯定理：
$$
p(\theta|D) = \frac{p(D|\theta) p(\theta)}{p(D)} \propto p(D|\theta) p(\theta)
$$

MAP 估计（忽略归一化常数 $p(D)$）：
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \log p(\theta|D) = \arg\max_\theta [\log p(D|\theta) + \log p(\theta)]
$$

### L2 正则化 = 高斯先验 MAP
高斯先验：$p(\theta) \propto \exp(-\frac{\lambda}{2}\|\theta\|^2)$

MAP 等价于：
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \underbrace{\log p(D|\theta)}_{\text{MLE目标}} - \frac{\lambda}{2}\|\theta\|^2
$$

这正是 L2 正则化的优化目标（权重衰减）！

### L1 正则化 = 拉普拉斯先验 MAP
拉普拉斯先验：$p(\theta) \propto \exp(-\lambda \|\theta\|_1)$

MAP 等价于：
$$
\hat{\theta}_{\text{MAP}} = \arg\max_\theta \log p(D|\theta) - \lambda \|\theta\|_1
$$

这正是 L1 正则化（Lasso）的优化目标。

### 一般形式
MAP 估计 = MLE + log 先验：
$$
\hat{\theta}_{\text{MAP}} = \arg\min_\theta \underbrace{-\log p(D|\theta)}_{\text{数据拟合项}} \underbrace{- \log p(\theta)}_{\text{正则化项}}
$$

## 直观理解
- **先验 = 预先"偏见"**：先验 $p(\theta)$ 表达了在看到数据之前你对参数 $\theta$ 的信念。比如，"我认为大多数特征对预测没贡献"对应拉普拉斯先验（L1），"我认为所有特征贡献都很小"对应高斯先验（L2）。
- **MAP 作为权衡**：MAP 在"让数据开心"（似然最大）和"让先验满意"（参数不太极端）之间找平衡。数据量少时先验主导，数据量多时似然主导。
- **贝叶斯-频率派桥梁**：L2 正则化在深度学习中的成功不仅是工程实践，更是贝叶斯统计在理论上保证的——它对参数施加了"高斯先验"信念，压缩参数使其不偏离零太远。

## 代码示例
```python
import numpy as np
from scipy.optimize import minimize

# 生成数据
np.random.seed(42)
n, d = 50, 20
X = np.random.randn(n, d)
true_w = np.zeros(d)
true_w[:5] = [1.5, -1.0, 2.0, -0.5, 0.8]
y = X @ true_w + 0.2 * np.random.randn(n)

# 负对数似然 (MLE目标)
def nll(w):
    return 0.5 * np.sum((X @ w - y)**2)

# 负对数后验 (MAP目标)
def neg_log_posterior_l2(w, lam=1.0):
    return nll(w) + lam/2 * np.sum(w**2)  # NLL + 高斯先验

def neg_log_posterior_l1(w, lam=1.0):
    return nll(w) + lam * np.sum(np.abs(w))  # NLL + 拉普拉斯先验

# MLE
w_mle = np.linalg.lstsq(X, y, rcond=None)[0]

# L2 MAP (高斯先验)
w_map_l2 = minimize(neg_log_posterior_l2, x0=np.zeros(d), args=(0.5,),
                    method='L-BFGS-B').x

# L1 MAP (拉普拉斯先验)
w_map_l1 = minimize(neg_log_posterior_l1, x0=np.zeros(d), args=(0.3,),
                    method='L-BFGS-B').x

print(f"真实非零权重索引: {np.where(true_w != 0)[0]}")
print(f"MLE 非零权重数: {np.sum(np.abs(w_mle) > 0.1)}")
print(f"L2 MAP 非零权重数: {np.sum(np.abs(w_map_l2) > 0.1)}")
print(f"L1 MAP 非零权重数: {np.sum(np.abs(w_map_l1) > 0.1)}")
print(f"\nL1 MAP 得到的稀疏解索引: {np.where(np.abs(w_map_l1) > 0.05)[0]}")

# 不同先验强度的影响
for lam in [0, 0.1, 1.0, 10.0]:
    w_est = minimize(neg_log_posterior_l2, x0=np.zeros(d), args=(lam,),
                     method='L-BFGS-B').x
    w_norm = np.linalg.norm(w_est)
    train_mse = np.mean((X @ w_est - y)**2)
    print(f"\nλ={lam:.1f}: ||w||={w_norm:.4f}, 训练MSE={train_mse:.4f}")

# 小样本情况下 MAP 的优势
print("\n--- 小样本对比 ---")
n_small = 10
X_small = np.random.randn(n_small, d)
y_small = X_small @ true_w + 0.2 * np.random.randn(n_small)

w_mle_small = np.linalg.lstsq(X_small, y_small, rcond=None)[0]
w_map_small = minimize(neg_log_posterior_l2, x0=np.zeros(d), args=(0.5,),
                       method='L-BFGS-B').x

mle_error = np.linalg.norm(w_mle_small - true_w)
map_error = np.linalg.norm(w_map_small - true_w)
print(f"MLE 恢复误差: {mle_error:.4f}")
print(f"MAP 恢复误差: {map_error:.4f}")
print(f"MAP 优于 MLE: {map_error < mle_error}")
```

## 深度学习关联
- **权重衰减 (Weight Decay)**：深度学习中最常见的 L2 正则化，等价于高斯先验下的 MAP 估计。在 PyTorch 中通过 `optimizer = torch.optim.SGD(model.parameters(), lr=0.01, weight_decay=1e-4)` 实现。权重衰减不仅抑制过拟合，还使优化 landscape 更平滑。
- **贝叶斯神经网络与点估计**：标准深度训练（SGD + L2 正则化）实际上是在做 MAP 估计。更完整的贝叶斯方法（如 HMC、变分推断）计算后验分布而非点估计，可以提供不确定性估计，在医疗诊断、自动驾驶等风险敏感场景中尤为重要。
- **先验作为归纳偏置 (Inductive Bias)**：不同的先验对应不同的归纳偏置。CNN 中卷积核的局部连接和权重共享结构等价于一种强先验（图像特征是局部和平移不变的）。Transformer 中的自注意力机制对应"全局依赖"的先验。模型架构本身也是一种先验知识。
- **可学习的先验**：在元学习中，MAML 算法在元训练阶段学到的初始化参数 $\theta^*$ 可以视为一种先验——在新任务上，$\theta^*$ 附近的参数有较高概率是好的。微调 (Fine-tuning) 预训练模型也可以类似解读：预训练权重是强先验，下游任务微调是 MAP 估计。
