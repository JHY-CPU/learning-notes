# 05_SVM 对偶问题与 KKT 条件详解

## 核心概念

- **对偶问题 (Dual Problem)**：原始优化问题（Primal）在一定条件下可以转化为对偶问题求解。对偶问题的解通常更易处理，且能自然引入核技巧。
- **拉格朗日对偶**：通过拉格朗日乘子 $\alpha_i \geq 0$ 将约束融入目标函数，构造 $L(w, b, \alpha)$。先对 $w, b$ 求极小，再对 $\alpha$ 求极大。
- **弱对偶与强对偶**：对偶问题的最优值 $\leq$ 原始问题的最优值。对于 SVM（凸二次规划 + 线性约束），Slater 条件满足，强对偶成立，即两者相等。
- **KKT 条件**：强对偶成立的充要条件，包含原始可行性、对偶可行性、互补松弛和梯度为零四个方面。
- **互补松弛 (Complementary Slackness)**：$\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$，意味着要么 $\alpha_i = 0$（非支持向量），要么约束取等号（支持向量）。
- **对偶解的优势**：对偶问题的变量 $\alpha_i$ 与样本数 $m$ 相关而非特征数 $n$，且 $w$ 和决策函数完全由支持向量表示，具有稀疏性。

## 数学推导

对于软间隔 SVM 的原始问题：
$$
\min_{w,b,\xi} \frac{1}{2}\|w\|^2 + C\sum \xi_i
$$
拉格朗日函数：
$$
L(w,b,\xi,\alpha,\mu) = \frac{1}{2}\|w\|^2 + C\sum_{i=1}^m \xi_i - \sum_{i=1}^m \alpha_i[y_i(w^T x_i+b) - 1 + \xi_i] - \sum_{i=1}^m \mu_i \xi_i
$$

KKT 条件完整陈述：

- **原始可行性**：$y_i(w^T x_i + b) \geq 1 - \xi_i, \; \xi_i \geq 0$
- **对偶可行性**：$\alpha_i \geq 0, \; \mu_i \geq 0$
- **梯度为零**：$\nabla_w L = 0 \Rightarrow w = \sum \alpha_i y_i x_i$，$\partial_b L = 0 \Rightarrow \sum \alpha_i y_i = 0$，$\partial_{\xi_i} L = 0 \Rightarrow C - \alpha_i - \mu_i = 0$
- **互补松弛**：$\alpha_i [y_i(w^T x_i + b) - 1 + \xi_i] = 0$，$\mu_i \xi_i = 0$

从互补松弛可知：
- 若 $\alpha_i = 0$，则该样本不影响 $w$ 和决策函数
- 若 $0 < \alpha_i < C$，则 $\xi_i = 0$ 且 $y_i(w^T x_i + b) = 1$（恰在间隔边界上）
- 若 $\alpha_i = C$，则 $\xi_i > 0$（越过间隔边界的样本）

对偶问题的最终形式：
$$
\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j K(x_i, x_j)
$$
$$
\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum \alpha_i y_i = 0
$$
其中 $K(x_i, x_j) = x_i^T x_j$ 是线性核，可替换为任意核函数。

## 直观理解

- **拉格朗日乘子的物理类比**：拉格朗日乘子 $\alpha_i$ 可以理解为每个样本对超平面施加的"影响力"。非支持向量的影响力为零，支持向量的影响力为正。$\alpha_i$ 的上界 $C$ 限制了单个样本的最大影响力，防止异常点过度主导决策边界。
- **互补松弛的含义**：这好比"不做事就不拿钱"——如果样本没有触及间隔边界（$\alpha_i = 0$），它就不贡献任何"力"；只有当样本在边界上或越界了（约束取等号），它才对决策面有影响力。
- **从原始到对偶的价值**：原始问题在 $w$ 空间求解，维度等于特征数；对偶在 $\alpha$ 空间求解，维度等于样本数。更重要的是，对偶形式中数据以 $x_i^T x_j$ 内积出现，可替换为核函数 $K(x_i, x_j)$，轻松处理非线性问题。

## 代码示例

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification

X, y = make_classification(n_samples=50, n_features=2,
                           n_redundant=0, random_state=42)

# 训练 SVM
model = SVC(kernel='linear', C=1.0)
model.fit(X, y)

# 查看支持向量的拉格朗日乘子 (dual_coef_ = alpha_i * y_i)
print(f"对偶系数 (alpha_i * y_i):\n{model.dual_coef_}")
print(f"支持向量索引: {model.support_}")
print(f"支持向量数量: {len(model.support_)}")

# 验证 w = sum(alpha_i * y_i * x_i)
w_from_dual = (model.dual_coef_ @ model.support_vectors_).ravel()
print(f"从对偶系数计算的 w: {w_from_dual}")
print(f"模型直接给出的 w: {model.coef_.ravel()}")
```

## 深度学习关联

- **对偶思想在深度学习的应用**：虽然深度网络通常直接优化原始问题（基于梯度的端到端训练），但对偶形式的一些变体（如 Neural Tangent Kernel, NTK）揭示了无限宽度神经网络等价于核方法，建立了深度学习与 SVM 对偶理论的深刻联系。
- **支持向量与模型剪枝**：SVM 的支持向量稀疏性启发了深度学习中的模型剪枝 (Pruning) 思想——大部分参数（神经元/连接）对最终输出的贡献很小，可以被移除而不显著影响性能。
- **KKT 条件与约束优化**：深度学习中许多带约束的优化问题（如 L-BFGS 约束、对抗训练中的扰动约束）都依赖于 KKT 条件来推导最优性条件，理解 KKT 是深入理解约束优化的基础。
