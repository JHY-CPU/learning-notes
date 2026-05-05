# 07_范数 (L0, L1, L2, L-inf) 的性质与正则化作用

## 核心概念

- **范数 (Norm)**：向量 $\mathbf{x}$ 的范数 $\|\mathbf{x}\|$ 是一个衡量向量"大小"或"长度"的函数，满足非负性、齐次性和三角不等式。范数将几何直观中的"距离"概念抽象化。
- **L2 范数（欧氏范数）**：$\|\mathbf{x}\|_2 = \sqrt{\sum_{i=1}^n x_i^2}$，最常见的距离度量。在正则化中对应"权重衰减"（Weight Decay），倾向于让权重均匀地变小。
- **L1 范数**：$\|\mathbf{x}\|_1 = \sum_{i=1}^n |x_i|$，也称为曼哈顿距离或绝对值之和。在正则化中倾向于产生稀疏解（许多权重严格为零），具有特征选择的作用。
- **L0 范数**：$\|\mathbf{x}\|_0 = \sum_{i=1}^n \mathbf{1}(x_i \neq 0)$，即非零元素的个数。L0 正则化直接追求稀疏性，但优化是 NP 难的，实践中用 L1 近似。
- **L-inf 范数**：$\|\mathbf{x}\|_\infty = \max_i |x_i|$，即各分量绝对值的最大值。衡量向量的"最显著"分量。
- **范数的对偶性**：L1 的对偶范数是 L-inf，L2 的对偶范数还是 L2。对偶性在优化理论（如约束优化、支撑向量机）中很重要。

## 数学推导

各种范数的数学定义：
$$
\|\mathbf{x}\|_p = \left( \sum_{i=1}^n |x_i|^p \right)^{1/p}
$$

L0 范数（$p \to 0$ 的极限）：
$$
\|\mathbf{x}\|_0 = \lim_{p \to 0} \sum_{i=1}^n |x_i|^p = \sum_{i=1}^n \mathbf{1}(x_i \neq 0)
$$

L-inf 范数（$p \to \infty$ 的极限）：
$$
\|\mathbf{x}\|_\infty = \max_i |x_i|
$$

L2 正则化的优化问题：
$$
\min_{\mathbf{w}} \sum_{i=1}^m (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_2^2
$$
解为 $\mathbf{w} = (X^T X + \lambda I)^{-1} X^T \mathbf{y}$（岭回归）

L1 正则化的优化问题：
$$
\min_{\mathbf{w}} \sum_{i=1}^m (y_i - \mathbf{w}^T \mathbf{x}_i)^2 + \lambda \|\mathbf{w}\|_1
$$
解没有闭式表达式，通常用坐标下降法或 ISTA 求解。

## 直观理解

- **L1 vs L2 的几何解释**：考虑二维 $\mathbf{w} = (w_1, w_2)$。L1 正则化的约束区域是菱形（顶点在坐标轴上），L2 是圆形。在菱形顶点处，一个参数为零，菱形"角"使得解更容易出现在坐标轴上。这解释了为什么 L1 产生稀疏解而 L2 不产生。
- **范数球 (Norm Ball)**：$\|\mathbf{x}\|_p \leq 1$ 定义的集合称为单位球。$p=1$ 时是菱形（二维）/ 八面体（三维），$p=2$ 时是圆形/球体，$p=\infty$ 时是正方形/立方体。$p<1$ 时球体变成凹形，$p\to 0$ 时退化为坐标轴上的十字。
- **拉索回归 (Lasso) 的几何**：在最小二乘的等值线（椭圆）与 L1 菱形相切时，切点往往在菱形顶点，此时某些系数为零。这比 L2 正则化（切点在圆周上，系数一般非零）更"吝啬"。

## 代码示例

```python
import numpy as np

# 计算各种范数
x = np.array([3, -4, 2, -1])
print(f"向量 x = {x}")
print(f"L0 范数: {np.sum(x != 0)}")
print(f"L1 范数: {np.linalg.norm(x, 1)}")
print(f"L2 范数: {np.linalg.norm(x, 2)}")
print(f"L-∞ 范数: {np.linalg.norm(x, np.inf)}")

# 正则化效果演示：模拟稀疏回归
np.random.seed(42)
n, d = 100, 20
X = np.random.randn(n, d)
true_w = np.zeros(d)
true_w[:5] = [3, -2, 1.5, -1, 0.5]  # 只有前5个特征有非零权重
y = X @ true_w + 0.1 * np.random.randn(n)

# 最小二乘法 (OLS)
w_ols = np.linalg.lstsq(X, y, rcond=None)[0]
print(f"\nOLS 非零系数个数: {np.sum(np.abs(w_ols) > 0.1)}")

# L2 正则化 (Ridge)
lam = 10.0
w_ridge = np.linalg.solve(X.T @ X + lam * np.eye(d), X.T @ y)
print(f"Ridge 非零系数个数: {np.sum(np.abs(w_ridge) > 0.1)}")

# L1 正则化 (使用坐标下降的简化版本)
def soft_threshold(z, lam):
    return np.sign(z) * np.maximum(np.abs(z) - lam, 0)

def lasso_coordinate_descent(X, y, lam, max_iter=1000):
    n, d = X.shape
    w = np.zeros(d)
    residual = y.copy()
    for _ in range(max_iter):
        for j in range(d):
            residual += X[:, j] * w[j]  # 移除当前特征贡献
            z = X[:, j] @ residual / n
            w[j] = soft_threshold(z, lam / n)
            residual -= X[:, j] * w[j]  # 重新加入更新后贡献
    return w

w_lasso = lasso_coordinate_descent(X, y, lam=0.5)
print(f"Lasso 非零系数个数: {np.sum(np.abs(w_lasso) > 0.01)}")
print(f"Lasso 非零系数索引: {np.where(np.abs(w_lasso) > 0.01)[0]}")
```

## 深度学习关联

- **权重衰减 (Weight Decay)**：L2 正则化在深度学习中称为权重衰减。AdamW 优化器将权重衰减与自适应学习率解耦，避免了 L2 正则化在 Adam 中效果被削弱的问题。权重衰减促使网络学习更小的权重，从而简化模型、提高泛化能力。
- **L1 正则化与稀疏网络**：L1 正则化可用于神经网络剪枝（Pruning）。训练过程中加入 L1 正则化会使许多权重接近零，随后可以删除这些连接，获得稀疏网络以加速推理。深度压缩 (Deep Compression) 技术利用了这一思想。
- **对抗攻击与 L-inf 范数**：在对抗攻击中，常约束扰动 $\delta$ 满足 $\|\delta\|_\infty \le \epsilon$，即每个像素的修改幅度不超过 $\epsilon$。L-inf 范数可以很好地描述"人眼无法察觉"的像素级扰动，是生成对抗样本的标准约束。
- **梯度裁剪 (Gradient Clipping)**：当梯度的 L2 范数超过阈值时进行缩放，防止梯度爆炸。这等价于 $\|\nabla L\|_2 > c$ 时执行 $\nabla L \leftarrow c \cdot \nabla L / \|\nabla L\|_2$，确保每一步的参数更新量可控。
