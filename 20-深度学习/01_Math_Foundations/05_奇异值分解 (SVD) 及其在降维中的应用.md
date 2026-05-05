# 05_奇异值分解 (SVD) 及其在降维中的应用

## 核心概念

- **奇异值分解 (Singular Value Decomposition)**：任何 $m \times n$ 矩阵 $A$ 都可以分解为 $A = U\Sigma V^T$，其中 $U$ 是 $m \times m$ 正交矩阵（左奇异向量），$V$ 是 $n \times n$ 正交矩阵（右奇异向量），$\Sigma$ 是 $m \times n$ 对角矩阵（奇异值）。
- **几何意义**：SVD 将任意线性变换分解为三个基本步骤：旋转/反射（$V^T$）→ 各向异性缩放（$\Sigma$）→ 旋转/反射（$U$）。任何线性变换本质上都是对这些基本操作的复合。
- **奇异值与特征值的关系**：$A^TA$ 的特征值等于奇异值的平方（$\sigma_i^2 = \lambda_i(A^TA)$）。$A$ 的奇异值就是 $A$ 的右奇异向量的放大倍数。
- **截断 SVD (Truncated SVD)**：保留最大的 $k$ 个奇异值，得到 $A \approx U_k \Sigma_k V_k^T$。这是信息损失最小的低秩近似，在降维、去噪、压缩中广泛应用。
- **Eckart-Young 定理**：截断 SVD 是在 Frobenius 范数意义下最优的低秩近似：
  $$\min_{\text{rank}(B)=k} \|A - B\|_F = \sqrt{\sum_{i=k+1}^{\min(m,n)} \sigma_i^2}$$

## 数学推导

SVD 的完整分解形式：
$$
A_{m \times n} = U_{m \times m} \Sigma_{m \times n} V_{n \times n}^T
$$

写成向量求和形式（秩1分解）：
$$
A = \sum_{i=1}^{r} \sigma_i \mathbf{u}_i \mathbf{v}_i^T
$$

其中 $r = \text{rank}(A)$。每个 $\sigma_i \mathbf{u}_i \mathbf{v}_i^T$ 是一个秩1矩阵，$\sigma_i$ 越大，该成分在 $A$ 中越重要。

从 SVD 到 PCA 的推导：设 $X$ 是中心化后的数据矩阵（$n$ 个样本, $d$ 个特征），则协方差矩阵为：
$$
C = \frac{1}{n-1} X^T X = \frac{1}{n-1} V \Sigma^T \Sigma V^T
$$
$V$ 的列就是主成分方向，$\sigma_i^2/(n-1)$ 是对应方向的方差。

## 直观理解

- **图像压缩类比**：将一张图片视为像素矩阵。SVD 将其分解为"模式图像"的加权和。第一个奇异值 $\sigma_1$ 最大，对应最重要的模式（整体明暗分布）。保留前几个奇异值即可用很小的数据量重构出保留主要信息的图像。
- **"数据指纹"的比喻**：SVD 为矩阵提取出最本质的特征。$\Sigma$ 中的奇异值就像数据的"指纹"——即使原始数据不同，只要奇异值分布相似，其结构特征就相近。
- **矩阵的无损编码**：$A = U\Sigma V^T$ 可以理解为：先将输入向量在 $V$ 基下旋转，然后逐维度缩放（乘 $\sigma_i$），最后在 $U$ 基下旋转输出。

## 代码示例

```python
import numpy as np

# 创建一个低秩矩阵+噪声
np.random.seed(42)
m, n = 50, 30
k_true = 5
U_true = np.random.randn(m, k_true)
V_true = np.random.randn(n, k_true)
A = U_true @ V_true.T + 0.1 * np.random.randn(m, n)

# SVD 分解
U, s, Vt = np.linalg.svd(A, full_matrices=False)
print(f"奇异值: {s[:10]}...")
print(f"奇异值个数: {len(s)}")

# 截断 SVD 降维
k = 5
U_k = U[:, :k]
s_k = s[:k]
Vt_k = Vt[:k, :]
A_approx = U_k @ np.diag(s_k) @ Vt_k

# 计算近似误差
error = np.linalg.norm(A - A_approx, 'fro')
print(f"k={k} 时的 Frobenius 范数误差: {error:.4f}")

# 验证 Eckart-Young 定理
truncation_error = np.sqrt(np.sum(s[k:]**2))
print(f"理论最小误差: {truncation_error:.4f}")

# SVD 用于降维（类似 PCA）
X = A
projection = X @ Vt_k.T
print(f"降维后形状: {projection.shape}")

# 计算每个奇异值的方差解释比例
explained_var = s**2 / np.sum(s**2)
cumulative = np.cumsum(explained_var)
print(f"前5个奇异值解释方差: {cumulative[4]:.2%}")
```

## 深度学习关联

- **权重低秩分解 (Model Compression)**：深度学习中大规模权重矩阵 $W \in \mathbb{R}^{m \times n}$ 可以用 SVD 分解为 $U_k \Sigma_k V_k^T$，将参数量从 $mn$ 减少到 $k(m+n)$，实现 2-10 倍的模型压缩。在移动端部署时非常实用。
- **PCA 预处理**：在将高维数据（如图像、文本）输入神经网络前，先用 SVD/PCA 进行降维，可以减少冗余特征、加速训练、缓解过拟合。这在传统机器学习管道和某些轻量级网络中很常见。
- **神经网络的隐空间分析**：对神经网络中间层的特征矩阵做 SVD，可以分析表示空间的"有效秩"——如果奇异值快速衰减，说明表示存在大量冗余，网络可能被过度参数化。有效秩也是衡量网络泛化能力的指标之一。
- **推荐系统与协同过滤**：SVD 是协同过滤的经典方法，在矩阵分解推荐算法中将用户-物品交互矩阵分解为低秩的用户特征矩阵和物品特征矩阵。虽然现代深度推荐模型更复杂，但 SVD 仍是理解推荐系统的基础模型。
