# 14_K-Means 聚类与坐标下降法

## 核心概念

- **K-Means**：最经典的无监督聚类算法，将数据集划分为 $k$ 个簇，每个簇由其质心（Centroid）表示，目标是最小化所有样本到所属簇质心的距离平方和。
- **坐标下降法视角**：K-Means 的迭代过程本质上是坐标下降法——交替优化簇分配（固定质心）和质心位置（固定分配），每一步都降低目标函数值，保证收敛到局部最优。
- **E-Step (分配步骤)**：固定质心 $\mu_1, \dots, \mu_k$，将每个样本分配到离它最近的质心所在的簇：$c_i = \arg\min_k \|x_i - \mu_k\|^2$。
- **M-Step (更新步骤)**：固定簇分配，每个质心更新为其对应簇内所有样本的均值：$\mu_k = \frac{1}{|C_k|} \sum_{i \in C_k} x_i$。
- **初始质心敏感性**：K-Means 对初始质心选择非常敏感，不同的初始值可能导致完全不同的聚类结果。K-Means++ 通过概率化的初始选择缓解此问题。
- **$k$ 值的选择**：常用肘部法 (Elbow Method) 和轮廓系数 (Silhouette Score) 来确定最优的簇数量。

## 数学推导

K-Means 的目标函数——残差平方和 (SSE)：
$$
J(c, \mu) = \sum_{i=1}^m \|x_i - \mu_{c_i}\|^2 = \sum_{k=1}^K \sum_{i \in C_k} \|x_i - \mu_k\|^2
$$

**坐标下降法的两步迭代**：

**Step 1 (固定 $\mu$，优化 $c$)**：对每个 $x_i$，
$$
c_i^{(t+1)} = \arg\min_{k \in \{1,\dots,K\}} \|x_i - \mu_k^{(t)}\|^2
$$

**Step 2 (固定 $c$，优化 $\mu$)**：对每个质心，
$$
\mu_k^{(t+1)} = \frac{1}{|C_k|} \sum_{i \in C_k} x_i
$$

这是从最小化 $\sum_{i \in C_k} \|x_i - \mu_k\|^2$ 推导而来——对该式求关于 $\mu_k$ 的梯度并令为零即可得均值解。

**收敛性保证**：每一步都减小 $J(c, \mu)$ 且 $J$ 有下界 0，因此算法必然收敛（但可能收敛到局部最优而非全局最优）。时间复杂度为 $O(m \cdot K \cdot d \cdot T)$，其中 $d$ 是特征维度，$T$ 是迭代次数。

**K-Means++ 初始化**：
- 随机选第一个质心
- 对每个样本 $x_i$，计算到最近已选质心的距离 $D(x_i)$
- 以概率 $\frac{D(x_i)^2}{\sum_j D(x_j)^2}$ 选择下一个质心（距离越远越可能被选为新的质心）
- 重复直到选出 $k$ 个质心

## 直观理解

- **聚类的"引力"模型**：将质心想象为 $k$ 个"引力中心"，每个样本受到最近引力中心的吸引。每次迭代中：样本重新选择最近的引力中心（分配），引力中心移动到所有被吸引样本的中心位置（更新）。经过多次往复，系统趋于稳定。
- **为什么是均值的均值**：质心更新为簇内样本的均值是因为——对于 $n$ 个点，使 $\sum \|x_i - \mu\|^2$ 最小的 $\mu$ 就是这些点的算术平均值。这就像一根杆上挂着多个重物，平衡点就是重物的平均位置。
- **局部最优的陷阱**：想象 K-Means 像一个小球在凹凸不平的地形上滚动，它会停在最近的"山谷"（局部最优）中，但不一定是全地形的最低点（全局最优）。多次随机初始化可以增加找到更好解的概率。

## 代码示例

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import silhouette_score

# 生成聚类数据
X, y_true = make_blobs(n_samples=300, centers=4,
                       cluster_std=0.60, random_state=42)

# K-Means 聚类
kmeans = KMeans(n_clusters=4, init='k-means++', n_init=10,
                max_iter=300, random_state=42)
y_pred = kmeans.fit_predict(X)

print(f"质心位置:\n{kmeans.cluster_centers_}")
print(f"SSE (惯性): {kmeans.inertia_:.2f}")
print(f"轮廓系数: {silhouette_score(X, y_pred):.4f}")
print(f"迭代次数: {kmeans.n_iter_}")

# 手写 K-Means 的一个迭代
def kmeans_one_iter(X, centroids):
    # E-step: 分配
    distances = np.linalg.norm(X[:, None] - centroids[None], axis=2)
    labels = np.argmin(distances, axis=1)
    # M-step: 更新
    new_centroids = np.array([X[labels == k].mean(axis=0)
                              for k in range(len(centroids))])
    return new_centroids, labels
```

## 深度学习关联

- **深度嵌入聚类 (Deep Embedded Clustering)**：K-Means 与深度特征提取结合，在自编码器或预训练网络提取的深度特征上进行 K-Means 聚类，大幅提升高维数据（如图像）的聚类效果。DEC (Deep Embedded Clustering) 网络联合优化特征提取和聚类分配。
- **向量量化 (VQ-VAE)**：K-Means 的思想直接体现在 VQ-VAE (Vector Quantized VAE) 中——将编码器的连续输出映射到最近的嵌入向量（代码本），对应 K-Means 的最近质心分配。这是生成模型（如图像、音频生成）中的核心技术。
- **注意力机制的聚类视角**：Transformer 的 Multi-Head Attention 可以看作一种软性聚类——Query 根据与 Key 的相似度聚拢信息（加权平均 Value），类似于 K-Means 的软分配版本，其中注意力权重是软性的簇隶属度。
