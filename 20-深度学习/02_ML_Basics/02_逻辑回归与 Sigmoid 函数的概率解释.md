# 02_逻辑回归与 Sigmoid 函数的概率解释

## 核心概念

- **逻辑回归**：尽管名称含"回归"，实际是用于**二分类**的线性模型。它通过 Sigmoid 函数将线性输出映射到 $(0,1)$ 区间，表示样本属于正类的概率。
- **Sigmoid 函数**：$\sigma(z) = 1 / (1 + e^{-z})$，具有 S 形曲线，将任意实数压缩到 $(0,1)$，且 $\sigma(-z) = 1 - \sigma(z)$。
- **概率解释**：逻辑回归假设 $\ln(p/(1-p)) = w^T x$，即对数几率 (log-odds) 是特征的线性函数，这天然地将线性回归与概率建模联系起来。
- **决策边界**：当 $\sigma(w^T x) > 0.5$ 时预测为正类，等价于 $w^T x > 0$，决策边界是超平面。
- **最大似然估计**：通过最大化对数似然 $\sum [y \ln \hat{y} + (1-y) \ln(1-\hat{y})]$ 来求解参数，等价于最小化交叉熵损失。
- **特征缩放**：逻辑回归的决策边界理论上不受特征尺度影响（权重会自动缩放适应），但使用梯度下降求解时，特征缩放能显著加速收敛；使用正则化时，缩放也会影响正则化效果，因此实践中仍建议进行特征缩放。

## 数学推导

Sigmoid 函数的定义及其导数性质：
$$
\sigma(z) = \frac{1}{1 + e^{-z}}, \quad \frac{d\sigma}{dz} = \sigma(z)(1 - \sigma(z))
$$

给定参数 $w$，样本 $x$ 属于正类的概率 $P(y=1|x) = \sigma(w^T x)$，负类概率 $P(y=0|x) = 1 - \sigma(w^T x)$。写成统一形式：
$$
P(y|x) = \sigma(w^T x)^y \cdot (1 - \sigma(w^T x))^{1-y}
$$

对 $m$ 个样本取对数似然：
$$
\ell(w) = \sum_{i=1}^m \left[ y_i \ln \sigma(w^T x_i) + (1-y_i) \ln(1 - \sigma(w^T x_i)) \right]
$$

最大化 $\ell(w)$ 等价于最小化交叉熵损失 $J(w) = -\frac{1}{m} \ell(w)$。梯度为：
$$
\nabla_w J(w) = \frac{1}{m} \sum_{i=1}^m (\sigma(w^T x_i) - y_i) x_i
$$

这个梯度形式与线性回归的梯度惊人地相似——都是"残差乘以特征"，区别仅在于 $\sigma(w^T x)$ 处于 $(0,1)$ 区间。

## 直观理解

- **Sigmoid 的由来**：设想你有一组线性判据 $z = w^T x$，值越大表示越可能是正类。但直接用 $z$ 作为概率有问题：它可以取任意大或小的值。Sigmoid 函数将 $(-\infty, +\infty)$ 平滑地映射到 $(0,1)$，像是一个"软"的阶跃函数。
- **log-odds 视角**：$w^T x = \ln(p/(1-p))$，意味着每增加一个单位的 $x_j$，正类的对数几率增加 $w_j$。这个解释非常直观：权重反映特征对"胜算"的影响。
- **决策的平滑性**：不像感知机那样硬性地在 $z=0$ 处跳跃，Sigmoid 在决策边界附近提供了平滑的概率过渡，距离边界越远，置信度越高。

## 代码示例

```python
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.datasets import make_classification

# 生成二分类数据
X, y = make_classification(n_samples=200, n_features=4, random_state=42)

# 训练逻辑回归模型
model = LogisticRegression()
model.fit(X, y)

# 查看系数和预测概率
print(f"权重: {model.coef_}")
print(f"偏置: {model.intercept_}")
print(f"前3个样本的预测概率:\n{model.predict_proba(X[:3])}")
print(f"前3个样本的预测类别: {model.predict(X[:3])}")

# 手动计算概率验证
z = X[:1] @ model.coef_.T + model.intercept_
p = 1 / (1 + np.exp(-z))
print(f"手动计算概率: {p}")
```

## 深度学习关联

- **二分类输出层**：深度神经网络在做二分类时，最后一层几乎总是使用 Sigmoid 激活函数配合二元交叉熵损失 (BCE Loss)，这正是逻辑回归的直接推广。
- **多分类推广 Softmax**：逻辑回归推广到多分类即 Softmax 回归（多项逻辑回归），而 Softmax 是深度神经网络多分类输出层的标准配置。
- **概率校准**：深度学习模型（如图像分类器）的原始输出往往过于自信或不够自信。逻辑回归的 Platt Scaling 方法（用逻辑回归对模型输出重新校准）是深度学习中常用的后处理校准技术。
