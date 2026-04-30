# PCA 主成分分析的代数与几何视角

## 核心概念
- **主成分分析 (PCA)**：最常用的无监督线性降维方法，通过正交变换将原始特征转换为一组线性不相关的新特征（主成分），按方差大小排序。
- **最大方差视角**：第一主成分是数据投影后方差最大的方向，第二主成分在与第一主成分正交的方向中找方差最大的，以此类推。
- **最小重构误差视角**：PCA 是在所有 $k$ 维线性子空间中，使原始数据到子空间投影的重构误差（平方距离）最小的子空间。
- **协方差矩阵对角化**：对数据中心化后的协方差矩阵 $\Sigma = X^T X / m$ 做特征值分解，特征向量对应主成分方向，特征值对应该方向上的方差。
- **SVD 实现**：实际中通过对数据矩阵 $X$ 进行奇异值分解 (SVD) 来实现 PCA，比直接分解协方差矩阵更数值稳定。
- **降维与信息保留**：前 $k$ 个主成分的方差占比 $\sum_{i=1}^k \lambda_i / \sum_{i=1}^d \lambda_i$ 衡量信息保留程度，常用累计方差贡献率来选择 $k$。

## 数学推导
设数据矩阵 $X \in \mathbb{R}^{m \times d}$，已中心化（每列均值为 0）。

**协方差矩阵**：
$$
\Sigma = \frac{1}{m} X^T X
$$

对 $\Sigma$ 做特征值分解：
$$
\Sigma v_i = \lambda_i v_i, \quad \lambda_1 \geq \lambda_2 \geq \dots \geq \lambda_d \geq 0
$$

$\lambda_i$ 是第 $i$ 个特征值，$v_i$ 是对应的特征向量（主成分方向）。

**投影到 $k$ 维**：将原始数据投影到前 $k$ 个特征向量张成的子空间：
$$
Z = X W_k
$$
其中 $W_k = [v_1, v_2, \dots, v_k] \in \mathbb{R}^{d \times k}$，$Z \in \mathbb{R}^{m \times k}$ 是降维后的数据。

**重构数据**：
$$
\hat{X} = Z W_k^T
$$

重构误差：
$$
\|X - \hat{X}\|_F^2 = \sum_{i=k+1}^d \lambda_i
$$

**SVD 方法**：对 $X$ 做奇异值分解 $X = U S V^T$，其中 $U$ 的左奇异向量、$V$ 的右奇异向量、$S$ 是对角矩阵。右奇异向量 $V$ 的列即为主成分方向（与 $v_i$ 一致），奇异值 $s_i = \sqrt{m \lambda_i}$。

## 直观理解
- **几何视角：找到数据"拉得最开"的方向**：想象一群三维空间中的点，它们是扁平的盘状分布。PCA 找到的第一个方向是盘面最长的方向（方差最大），第二个方向是盘上与之垂直的第二长方向，第三个方向是厚度方向（方差很小）。丢弃第三维几乎不影响数据的结构。
- **代数视角：去掉冗余维度**：如果两个特征高度相关（比如"房屋面积"和"房间数量"），PCA 将它们组合成一个新特征（捕捉两者的共同变化），丢弃另一个特征方向（噪声或冗余）。这就是协方差矩阵对角化的意义——消除特征间的线性相关性。
- **椭球拟合**：PCA 相当于在数据上拟合一个 $d$ 维椭球，球的轴方向是特征向量方向，轴的长度正比于特征值的平方根。降维就是保留长的轴，截断短的轴。

## 代码示例
```python
import numpy as np
from sklearn.decomposition import PCA
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 使用手写数字数据集（64 维）
digits = load_digits()
X, y = digits.data, digits.target
print(f"原始数据形状: {X.shape}")

# 标准化
X_scaled = StandardScaler().fit_transform(X)

# PCA 降维到 2 维
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)
print(f"降维后数据形状: {X_reduced.shape}")

# 方差解释率
print(f"第一主成分方差比: {pca.explained_variance_ratio_[0]:.4f}")
print(f"第二主成分方差比: {pca.explained_variance_ratio_[1]:.4f}")
print(f"前2主成分累计方差: {pca.explained_variance_ratio_.sum():.4f}")

# 选择保留 90% 方差所需的维度
pca_90 = PCA(n_components=0.90)
X_90 = pca_90.fit_transform(X_scaled)
print(f"保留 90% 方差需要 {X_90.shape[1]} 个主成分")

# 重构误差检查
X_reconstructed = pca.inverse_transform(X_reduced)
reconstruction_error = np.mean((X_scaled - X_reconstructed) ** 2)
print(f"重构均方误差: {reconstruction_error:.6f}")
```

## 深度学习关联
- **Whitening 与数据预处理**：PCA Whitening（白化）是深度学习中常用的数据预处理步骤——先将数据投影到主成分方向，再除以特征值平方根，使每个维度的方差为 1 且不相关，加速网络收敛。
- **自编码器的线性版本**：线性自编码器（无激活函数、单隐含层）的最优解就是 PCA——编码器学习主成分投影，解码器学习主成分的重构。这为理解深度自编码器提供了理论起点。
- **Dropout 的 PCA 解释**：研究表明，对线性网络应用 Dropout 正则化等价于某种形式的 PCA——Dropout 噪声导致模型偏向于保留方差大的主成分方向，丢弃方差小的方向，与 PCA 的方差保留思想一致。
