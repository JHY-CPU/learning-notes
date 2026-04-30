# 核技巧 (Kernel Trick) 与 RBF 核函数

## 核心概念
- **核技巧**：一种不需要显式计算高维映射 $\phi(x)$，仅通过核函数 $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$ 计算高维空间内积的方法，大幅降低了计算复杂度。
- **动机**：很多数据在原始空间中线性不可分，映射到高维后可能变得线性可分。但高维映射的计算和存储代价极高，核技巧规避了这一瓶颈。
- **Mercer 定理**：一个对称函数 $K(x, y)$ 是合法的核函数当且仅当对任意有限样本集，核矩阵 $K_{ij} = K(x_i, x_j)$ 是半正定的。
- **RBF 核**：径向基函数核 $K(x_i, x_j) = \exp(-\gamma \|x_i - x_j\|^2)$，是最常用的核函数之一，等价于将数据映射到无穷维空间。
- **参数 $\gamma$**：控制高斯核的"宽度"。$\gamma$ 越大，核函数随距离衰减越快，决策边界越复杂（易过拟合）；$\gamma$ 越小，决策边界越平滑。
- **常见核函数**：线性核 $x_i^T x_j$、多项式核 $(x_i^T x_j + r)^d$、Sigmoid 核 $\tanh(\gamma x_i^T x_j + r)$，以及最常用的 RBF 核。

## 数学推导
假设存在映射 $\phi: \mathcal{X} \to \mathcal{H}$，将原始空间映射到高维希尔伯特空间。SVM 对偶问题的决策函数为：
$$
f(x) = \sum_{i=1}^m \alpha_i y_i \langle \phi(x_i), \phi(x) \rangle + b
$$

核技巧的关键在于直接定义 $K(x_i, x_j) = \langle \phi(x_i), \phi(x_j) \rangle$，而不显式计算 $\phi$。

RBF 核的数学形式：
$$
K(x_i, x_j) = \exp\left(-\gamma \|x_i - x_j\|^2\right) = \exp\left(-\frac{\|x_i - x_j\|^2}{2\sigma^2}\right)
$$
其中 $\gamma = 1/(2\sigma^2)$。

RBF 核对应无穷维映射的证明思路：利用指数函数的泰勒展开：
$$
\exp(-\gamma\|x-x'\|^2) = \exp(-\gamma\|x\|^2) \exp(-\gamma\|x'\|^2) \sum_{n=0}^\infty \frac{(2\gamma x^T x')^n}{n!}
$$
展开后每一项对应于一个多项式核，因此映射维度为无穷。此外，RBF 核函数的值域为 $(0, 1]$，当两向量完全相同时为 1，正交时接近 0。

## 直观理解
- **"相似度"度量**：核函数本质上衡量两个样本之间的"相似度"。RBF 核的相似度随距离指数衰减——两个点离得越近，核函数值越大（越相似）；离得越远，值趋近于零。
- **覆盖在数据上的高斯山**：每个支持向量都像一座以自己为中心的高斯山，山的"宽度"由 $\gamma$ 控制。所有山的轮廓叠加在一起，形成了最终的决策边界。$\gamma$ 小时山很平缓，形成平滑边界；$\gamma$ 大时山很尖，边界跟着数据"崎岖不平"。
- **不用爬山只看等高线**：核技巧的精髓在于——我们知道两座山的高度乘积，却不需要真的爬上去测量。这就像看地图上的等高线距离来判断山的远近，而不需要实际攀登。

## 代码示例
```python
import numpy as np
from sklearn.svm import SVC
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split

# 生成非线性数据（同心圆）
X, y = make_circles(n_samples=300, noise=0.1, factor=0.4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3)

# 线性核 VS RBF 核
linear_svm = SVC(kernel='linear')
rbf_svm = SVC(kernel='rbf', gamma=1.0)

linear_svm.fit(X_train, y_train)
rbf_svm.fit(X_train, y_train)

print(f"线性核 准确率: {linear_svm.score(X_test, y_test):.3f}")
print(f"RBF 核 准确率: {rbf_svm.score(X_test, y_test):.3f}")

# 不同 gamma 的影响
for gamma in [0.1, 1.0, 10.0]:
    model = SVC(kernel='rbf', gamma=gamma)
    model.fit(X_train, y_train)
    print(f"gamma={gamma:.1f} 准确率: {model.score(X_test, y_test):.3f}, "
          f"支持向量数: {sum(model.n_support_)}")
```

## 深度学习关联
- **核方法 vs 神经网络**：RBF 网络 (Radial Basis Function Network) 是早期神经网络的一种，其隐含层使用 RBF 激活函数，直接对应核方法。现代深度网络虽然多以 ReLU 为主，但 RBF 网络的思想在 Gaussian Process 和贝叶斯深度学习中仍有重要应用。
- **Neural Tangent Kernel (NTK)**：NTK 理论证明，在无限宽度极限下，训练神经网络等价于使用特定核函数的核方法进行训练。这建立了深度学习和核技巧之间的理论桥梁，帮助理解深度网络的训练动态。
- **注意力机制与核函数**：Transformer 中的 self-attention 本质上可以看作一种核回归——注意力分数是 query 和 key 之间的相似度函数，类似于核函数的作用。高效注意力变体（如 Linear Attention）通过核技巧来降低注意力计算复杂度。
