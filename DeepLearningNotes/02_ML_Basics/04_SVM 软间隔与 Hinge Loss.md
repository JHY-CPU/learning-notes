# SVM 软间隔与 Hinge Loss

## 核心概念
- **软间隔 (Soft Margin)**：当数据不可完全线性可分时，允许部分样本违反间隔约束（即出现在间隔内部或被错分），但引入惩罚项控制违规程度。
- **松弛变量 $\xi$**：每个样本引入一个非负松弛变量 $\xi_i \geq 0$，表示该样本违反间隔约束的程度。若样本在正确一侧且位于间隔外，$\xi_i = 0$；否则 $\xi_i > 0$。
- **惩罚参数 $C$**：控制间隔最大化与误分类惩罚之间的权衡。$C$ 越大，对违规的容忍越低（偏向硬间隔）；$C$ 越小，允许更多违规（偏向软间隔，提高泛化）。
- **Hinge Loss**：$L_{\text{hinge}}(y, f(x)) = \max(0, 1 - y f(x))$，当 $y f(x) \geq 1$ 时损失为 0，否则线性增长。
- **等价形式**：软间隔 SVM 的原始优化问题等价于使用 Hinge Loss 加 $L_2$ 正则化的经验风险最小化：$\min_w \sum \max(0, 1 - y_i w^T x_i) + \lambda \|w\|^2$。
- **与逻辑回归对比**：Hinge Loss 在 $y f(x) > 1$ 后损失为零（只关心边界附近），而 Log Loss 永远有非零梯度（所有样本都影响模型），这解释了 SVM 的"稀疏性"。

## 数学推导
引入松弛变量后，原始问题变为：
$$
\min_{w, b, \xi} \frac{1}{2} \|w\|^2 + C \sum_{i=1}^m \xi_i
$$
$$
\text{s.t.} \quad y_i(w^T x_i + b) \geq 1 - \xi_i, \quad \xi_i \geq 0, \; i=1,\dots,m
$$

构建拉格朗日函数：
$$
L = \frac{1}{2}\|w\|^2 + C\sum \xi_i - \sum \alpha_i[y_i(w^T x_i+b) - 1 + \xi_i] - \sum \mu_i \xi_i
$$
其中 $\alpha_i \geq 0, \mu_i \geq 0$。

对 $w, b, \xi_i$ 求偏导并令为零：
$$
w = \sum \alpha_i y_i x_i, \quad \sum \alpha_i y_i = 0, \quad C - \alpha_i - \mu_i = 0
$$

代入后对偶问题变为：
$$
\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j
$$
$$
\text{s.t.} \quad 0 \leq \alpha_i \leq C, \quad \sum_{i=1}^m \alpha_i y_i = 0
$$

对比硬间隔，唯一的区别是 $\alpha_i$ 的上界从 $+\infty$ 变为 $C$。KKT 条件告诉我们：$\alpha_i = 0$ 的点不构成支持向量；$0 < \alpha_i < C$ 对应落在间隔边界上的点；$\alpha_i = C$ 对应越过间隔边界的点。

## 直观理解
- **"软"的推板**：硬间隔像两块刚性板推两类数据，板上不能有任何凸起。软间隔允许板上有些"弹性"，即少数点可以凸出到板内，但凸出越多惩罚越重。参数 $C$ 控制板的"刚度"。
- **Hinge Loss 的形状**：你希望所有样本都挂在绳子的一侧且离绳子至少一段距离。不满足条件的样本会被绳子"弹"回来，弹力大小正比于侵入距离。已经离绳子足够远的样本不再受绳子的拉力。
- **C 的直观影响**：$C$ 很大时，模型会竭尽全力把每个点都分类正确（哪怕导致决策边界非常扭曲），容易过拟合；$C$ 很小时，模型允许一些点犯错，追求更平滑的边界，泛化能力更强。

## 代码示例
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_classification
from sklearn.model_selection import cross_val_score

# 生成不完全线性可分的数据
X, y = make_classification(n_samples=200, n_features=2,
                           n_redundant=0, n_clusters_per_class=1,
                           random_state=42)

# 不同 C 值的对比
for C in [0.01, 0.1, 1.0, 10.0, 100.0]:
    model = SVC(kernel='linear', C=C)
    scores = cross_val_score(model, X, y, cv=5)
    print(f"C={C:6.2f}, 平均准确率={scores.mean():.4f}, "
          f"支持向量数={model.fit(X, y).n_support_}")

# Hinge Loss 手动计算示例
def hinge_loss(y_true, y_pred):
    return np.maximum(0, 1 - y_true * y_pred)

y_true = np.array([1, 1, -1, -1])
y_pred = np.array([0.8, 1.5, -1.2, 0.5])
print(f"Hinge Loss: {hinge_loss(y_true, y_pred)}")
```

## 深度学习关联
- **Hinge Loss 的深度变体**：深度学习中使用的 Categorical Hinge Loss（多分类 Hinge）是 SVM 思想在神经网络中的直接推广，常用于最后一层全连接的损失函数。
- **Large Margin 训练**：许多深度学习训练技术受 SVM 最大间隔思想启发，如 L-SVM (Large Margin Softmax) 和 AM-Softmax，通过在 Softmax 损失中引入角度间隔，让不同类别的特征在超球面上拉开更大的距离。
- **对抗训练中的间隔思想**：对抗训练 (Adversarial Training) 的目标之一是增大决策边界附近的最小扰动距离，这与 SVM 最大化间隔的思想一脉相承——更宽的间隔意味着更强的鲁棒性。
