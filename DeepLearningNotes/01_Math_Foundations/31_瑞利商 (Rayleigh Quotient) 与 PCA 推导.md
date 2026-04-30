# 瑞利商 (Rayleigh Quotient) 与 PCA 推导

## 核心概念
- **瑞利商 (Rayleigh Quotient)**：对对称矩阵 $A$ 和向量 $\mathbf{x} \neq 0$，瑞利商定义为 $R(A, \mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$。它衡量了二次型在单位向量方向上的"能量"。
- **瑞利商的极值性质**：$R(A, \mathbf{x})$ 的最小值和最大值分别等于 $A$ 的最小和最大特征值。这提供了特征值的变分刻画：
  $$\lambda_{\min} = \min_{\mathbf{x} \neq 0} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}, \quad \lambda_{\max} = \max_{\mathbf{x} \neq 0} \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}$$
- **广义瑞利商**：用于广义特征值问题 $A\mathbf{x} = \lambda B \mathbf{x}$，$R(A, B, \mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T B \mathbf{x}}$。在 Fisher 判别分析 (LDA) 中至关重要。
- **Courant-Fischer 极小极大定理**：瑞利商的第 $k$ 个极值与 $A$ 的第 $k$ 个特征值对应，提供了特征值的完整谱刻画。
- **PCA 与瑞利商**：PCA 寻找数据投影后方差最大的方向，等价于最大化瑞利商 $\frac{\mathbf{w}^T \Sigma \mathbf{w}}{\mathbf{w}^T \mathbf{w}}$，其中 $\Sigma$ 是协方差矩阵。

## 数学推导
瑞利商的定义：
$$
R(A, \mathbf{x}) = \frac{\mathbf{x}^T A \mathbf{x}}{\mathbf{x}^T \mathbf{x}}
$$

当 $\|\mathbf{x}\| = 1$ 时，$R(A, \mathbf{x}) = \mathbf{x}^T A \mathbf{x}$。

瑞利商的梯度：
$$
\nabla R(A, \mathbf{x}) = \frac{2}{\mathbf{x}^T \mathbf{x}} \left( A\mathbf{x} - R(A, \mathbf{x}) \mathbf{x} \right)
$$

设梯度为零得 $A\mathbf{x} = R(A, \mathbf{x}) \mathbf{x}$，即 $\mathbf{x}$ 是特征向量，$R(A, \mathbf{x})$ 是特征值。

PCA 的瑞利商推导：寻找最大化投影方差的 $\mathbf{w}$：
$$
\max_{\|\mathbf{w}\|=1} \mathbf{w}^T \Sigma \mathbf{w}
$$

由瑞利商性质，最大值在 $\Sigma$ 的最大特征值对应的特征向量处取得。第 $k$ 个主成分是第 $k$ 大特征值对应的特征向量（在正交于前 $k-1$ 个方向的约束下）。

## 直观理解
- **"能量集中"的度量**：瑞利商衡量了向量 $\mathbf{x}$ 在矩阵 $A$ 作用下的"拉伸倍数"。当 $\mathbf{x}$ 与 $A$ 的最大特征向量对齐时，拉伸最大（瑞利商最大）。就像用橡皮筋拉伸物体，沿某个方向最容易拉长（对应大特征值）。
- **与特征向量的关系**：瑞利商$\mathbf{x}^T A \mathbf{x} / \mathbf{x}^T \mathbf{x}$在特征向量方向取极值。这就像在椭圆上找主轴——$A$ 的特征向量指向椭圆的主轴方向，特征值是主轴半长的平方。
- **PCA 的直觉**：在 PCA 中，$\mathbf{w}^T \Sigma \mathbf{w}$ 是数据在 $\mathbf{w}$ 方向上的方差。最大化这个比值就是找"数据伸展最开"的方向。瑞利商提供了找这些方向的数学框架。

## 代码示例
```python
import numpy as np

# 1. 瑞利商的极值 = 特征值
A = np.array([[3, 1], [1, 3]])
eigvals, eigvecs = np.linalg.eigh(A)
print("矩阵 A:")
print(A)
print(f"特征值: {eigvals}")
print(f"特征向量:\n{eigvecs}")

# 计算不同方向上的瑞利商
def rayleigh_quotient(A, x):
    return (x @ A @ x) / (x @ x)

print("\n瑞利商验证:")
for i in range(2):
    v = eigvecs[:, i]
    lam = eigvals[i]
    rq = rayleigh_quotient(A, v)
    print(f"  特征向量 {i}: R(A, v)={rq:.4f}, λ={lam:.4f}, 一致? {np.isclose(rq, lam)}")

# 随机方向上的瑞利商
np.random.seed(42)
random_dir = np.random.randn(2)
random_dir = random_dir / np.linalg.norm(random_dir)
rq_random = rayleigh_quotient(A, random_dir)
print(f"\n随机方向瑞利商: {rq_random:.4f} (应在 [{eigvals[0]:.4f}, {eigvals[1]:.4f}] 之间)")

# 2. PCA 推导：最大化瑞利商
# 生成相关数据
np.random.seed(42)
n = 200
X = np.random.randn(n, 2) @ np.array([[3, 1], [0, 2]])  # 相关数据
X = X - np.mean(X, axis=0)  # 中心化
Sigma = X.T @ X / (n - 1)  # 协方差矩阵

# PCA = 协方差矩阵的特征分解
eigvals_pca, eigvecs_pca = np.linalg.eigh(Sigma)
# 最大特征值对应的特征向量 = 第一主成分
pc1 = eigvecs_pca[:, -1]
pc2 = eigvecs_pca[:, -2]

print(f"\nPCA 主成分方向:")
print(f"  第一主成分: {pc1}, 瑞利商={rayleigh_quotient(Sigma, pc1):.4f}")
print(f"  第二主成分: {pc2}, 瑞利商={rayleigh_quotient(Sigma, pc2):.4f}")

# 验证瑞利商最大化
proj_vars = [rayleigh_quotient(Sigma, np.random.randn(2)) for _ in range(1000)]
print(f"  随机方向最大瑞利商: {max(proj_vars):.4f} (理论={eigvals_pca[-1]:.4f})")
print(f"  第一主成分瑞利商: {rayleigh_quotient(Sigma, pc1):.4f}")

# 3. 广义瑞利商
B = np.array([[2, 0], [0, 1]])  # 另一个正定矩阵
def generalized_rq(A, B, x):
    return (x @ A @ x) / (x @ B @ x)

print(f"\n广义瑞利商:")
print(f"  R(A, B, e1) = {generalized_rq(A, B, np.array([1, 0])):.4f}")
print(f"  R(A, B, e2) = {generalized_rq(A, B, np.array([0, 1])):.4f}")
```

## 深度学习关联
- **主成分分析 (PCA) 预处理**：在深度学习出现之前，PCA 是高维数据降维的标准方法。当前 PCA 仍常用于数据白化预处理、可视化（降至 2D/3D）、以及分析网络表示的"有效维度"。瑞利商是 PCA 的理论核心。
- **谱聚类与瑞利商**：谱聚类算法通过最小化 RatioCut 或 NCut 将数据点划分为簇。该问题可以松弛为最小化瑞利商 $\min_{\mathbf{f}} \frac{\mathbf{f}^T L \mathbf{f}}{\mathbf{f}^T \mathbf{f}}$，其中 $L$ 是图拉普拉斯矩阵。解是 $L$ 的最小特征向量，这正是瑞利商的直接应用。
- **Fisher 判别分析 (LDA)**：线性判别分析寻找投影方向 $\mathbf{w}$ 使类间散度与类内散度的比值最大，即最大化广义瑞利商 $\frac{\mathbf{w}^T S_B \mathbf{w}}{\mathbf{w}^T S_W \mathbf{w}}$。这与神经网络中对比学习的思想高度一致——最大化不同类样本的分离度。
- **Neural Tangent Kernel (NTK)**：深度网络的 NTK 的特征值谱决定了网络的收敛速度。瑞利商被用于分析 NTK 的谱性质，即特定方向上的函数变化对参数变化的敏感度，从而理解不同频率成分的学习速度差异。
