# 03_SVM 间隔最大化与硬间隔推导

## 核心概念

- **支持向量机 (SVM)**：一种二分类模型，核心思想是在特征空间中寻找一个超平面，使得两类样本之间的"间隔"最大化。
- **间隔 (Margin)**：超平面到最近训练样本的距离。SVM 寻找的是最大间隔分离超平面，对噪声更鲁棒，泛化能力更强。
- **支持向量**：距离超平面最近的几个训练样本点，它们"支撑"起了间隔边界，只有这些点决定最终的超平面。
- **硬间隔**：假设数据完全线性可分，要求所有样本都正确分类且位于间隔边界之外，不允许任何违反约束的样本。
- **函数间隔与几何间隔**：函数间隔 $\hat{\gamma} = y(w^T x + b)$ 受 $w$ 缩放影响，几何间隔 $\gamma = \hat{\gamma} / \|w\|$ 才是真实距离。
- **对偶问题转化**：原始问题是凸二次规划，通过拉格朗日对偶性转化为对偶问题，可引入核技巧处理非线性。

## 数学推导

设超平面为 $w^T x + b = 0$，样本点 $(x_i, y_i)$ 满足 $y_i \in \{-1, 1\}$。

几何间隔定义为每个样本到超平面的最小距离。由于同时缩放 $w, b$ 不影响超平面，固定函数间隔为 1（即 $y_i(w^T x_i + b) \geq 1$），则问题转化为：
$$
\max_{w, b} \frac{1}{\|w\|} \quad \text{s.t.} \quad y_i(w^T x_i + b) \geq 1, \; i=1,\dots,m
$$

等价于最小化 $\frac{1}{2}\|w\|^2$：
$$
\min_{w, b} \frac{1}{2} \|w\|^2 \quad \text{s.t.} \quad y_i(w^T x_i + b) - 1 \geq 0, \; i=1,\dots,m
$$

构造拉格朗日函数 $L(w, b, \alpha) = \frac{1}{2}\|w\|^2 - \sum_{i=1}^m \alpha_i [y_i(w^T x_i + b) - 1]$，其中 $\alpha_i \geq 0$。对 $w, b$ 求偏导为零：
$$
\frac{\partial L}{\partial w} = w - \sum_{i=1}^m \alpha_i y_i x_i = 0 \implies w = \sum_{i=1}^m \alpha_i y_i x_i
$$
$$
\frac{\partial L}{\partial b} = -\sum_{i=1}^m \alpha_i y_i = 0 \implies \sum_{i=1}^m \alpha_i y_i = 0
$$

代入消去 $w, b$ 后得到对偶问题：
$$
\max_{\alpha} \sum_{i=1}^m \alpha_i - \frac{1}{2} \sum_{i=1}^m \sum_{j=1}^m \alpha_i \alpha_j y_i y_j x_i^T x_j
$$
$$
\text{s.t.} \quad \alpha_i \geq 0, \quad \sum_{i=1}^m \alpha_i y_i = 0
$$

## 直观理解

- **推离的边界**：想象两张平行板从两侧挤压两类数据点。硬间隔 SVM 将这两块板推得尽可能开，同时确保所有点都在板的外侧。板的中心就是决策超平面，板到中心的距离就是间隔。
- **只关心边界点**：在推板的过程中，只有接触板面的那几个点才真正阻止板继续移动——这些就是"支持向量"。内部点无论怎么移动，都不影响最终结果。这种"稀疏性"使得 SVM 高效且鲁棒。
- **从二维看**：在二维平面上，SVM 寻找的是两类点之间最宽的一条"街道"，街道边界上的点就是支持向量，街道的中线是决策边界。

## 代码示例

```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_blobs

# 生成线性可分的二分类数据
X, y = make_blobs(n_samples=50, centers=2, random_state=6)
y = 2 * y - 1  # 转为 {-1, 1}

# 训练硬间隔 SVM (C 取极大值近似硬间隔)
model = SVC(kernel='linear', C=1e10)
model.fit(X, y)

# 提取支持向量
print(f"支持向量索引: {model.support_}")
print(f"支持向量个数: {len(model.support_)}")
print(f"权重向量 w: {model.coef_}")
print(f"偏置 b: {model.intercept_}")

# 决策函数
print(f"样本点到超平面的距离: {model.decision_function(X[:5])}")
```

## 深度学习关联

- **最大间隔思想的推广**：深度神经网络虽然不显式最大化间隔，但近年研究发现，SGD 训练出的深度网络往往隐式地偏向于最大间隔解（尤其是在过参数化情况下），这为理解深度学习的泛化性提供了理论视角。
- **Hinge Loss 在深度学习中**：虽然深度分类器多用交叉熵损失，但 Hinge Loss 的思想（让正确类别的得分高于错误类别至少一个间隔）被广泛应用于度量学习 (Metric Learning) 和对比学习 (Contrastive Learning) 中。
- **支持向量与注意力机制**：SVM 的"稀疏支持"思想——仅依赖关键样本做决策——与 Transformer 的注意力机制有类似之处：注意力权重让模型关注输入中最相关的部分，类似于支持向量的"关键样本选取"。
