# 46_谱聚类与图切分 (Graph Cut)

## 核心概念

- **谱聚类 (Spectral Clustering)**：利用图拉普拉斯矩阵的谱（特征向量）进行聚类的方法。核心思想：将数据点看作图节点，边权重表示相似度，然后切图使不同簇间的连接尽可能少、簇内连接尽可能多。
- **图切分 (Graph Cut)**：将图 $G = (V, E)$ 划分为 $k$ 个不相交子集 $A_1, \dots, A_k$，目标是最小化子集之间的边权重和。直接优化图切分是 NP-难问题，谱聚类通过松弛求解。
- **RatioCut**：最小化 $\sum_{i=1}^k \frac{\text{cut}(A_i, \bar{A}_i)}{|A_i|}$，考虑了簇大小的均衡性。其中 $\text{cut}(A, B) = \sum_{i \in A, j \in B} w_{ij}$。
- **NCut (Normalized Cut)**：最小化 $\sum_{i=1}^k \frac{\text{cut}(A_i, \bar{A}_i)}{\text{vol}(A_i)}$，其中 $\text{vol}(A) = \sum_{i \in A} D_{ii}$。NCut 考虑簇的"体积"（度之和）。
- **谱聚类算法步骤**：
  1. 构建相似度图（如 kNN 图或全连接高斯核图）
  2. 计算拉普拉斯矩阵 $L$ 或归一化拉普拉斯 $L_{\text{sym}}$
  3. 计算前 $k$ 个最小特征值对应的特征向量
  4. 将特征向量作为新特征，使用 K-Means 聚类
- **与 K-Means 的关系**：谱聚类可以捕捉非凸形状的簇，而 K-Means 只能处理凸簇。谱聚类先通过拉普拉斯特征映射将数据变换到"更易聚类"的空间，然后在该空间用 K-Means。

## 数学推导

对于二分问题，MinCut 目标：
$$
\min_{A \subset V} \text{cut}(A, \bar{A}) = \min_{A} \sum_{i \in A, j \in \bar{A}} w_{ij}
$$

引入指示向量 $\mathbf{f} \in \{-1, 1\}^n$，$f_i = 1$ 若 $i \in A$，否则 $f_i = -1$。可以证明：
$$
\text{cut}(A, \bar{A}) = \frac{1}{4} \mathbf{f}^T L \mathbf{f}
$$

RatioCut 的松弛：将约束放宽为 $\mathbf{f}^T \mathbf{1} = 0, \|\mathbf{f}\|^2 = n$，得：
$$
\min_{\mathbf{f} \in \mathbb{R}^n} \mathbf{f}^T L \mathbf{f} \quad \text{s.t.} \quad \mathbf{f} \perp \mathbf{1}, \|\mathbf{f}\| = \sqrt{n}
$$

由瑞利商性质，解为 $L$ 的第二小特征值 $\lambda_2$ 对应的特征向量（Fiedler 向量）。

对于 $k$ 个簇，解为前 $k$ 个最小特征值对应的特征向量构成的矩阵 $U \in \mathbb{R}^{n \times k}$，然后对 $U$ 的行进行 K-Means 聚类。

## 直观理解

- **"切蛋糕"的艺术**：谱聚类就像切蛋糕——从蛋糕上较薄弱的"颈"处切开（这些位置边的权重和最小）。拉普拉斯矩阵的特征向量自动识别了这些薄弱位置。
- **为什么用特征向量**：Fiedler 向量的每个元素对应一个节点的"坐标"。如果两个节点在 Fiedler 向量上的值相近，它们很可能属于同一簇。将多个特征向量堆叠起来，就给每个节点赋予了一个多维"谱坐标"，在这个空间中 K-Means 很容易找到簇。
- **拉普拉斯特征映射**：谱聚类本质上先用 $L$ 的特征向量将节点映射到 $\mathbb{R}^k$ 空间。在这个新空间中，原始数据点的全局结构被保持，而局部噪声被抑制。

## 代码示例

```python
import numpy as np
from sklearn.cluster import KMeans
from sklearn.datasets import make_moons, make_circles
from sklearn.neighbors import kneighbors_graph
from scipy.sparse.linalg import eigsh

# 1. 谱聚类手动实现
def spectral_clustering(X, n_clusters=2, sigma=1.0, n_neighbors=10):
    n = len(X)
    # 构建相似度图（kNN + 高斯核）
    A = kneighbors_graph(X, n_neighbors, mode='distance')
    A = A.toarray()
    # 距离转相似度
    A = np.exp(-A**2 / (2*sigma**2))
    A[A < 1e-5] = 0
    
    # 确保对称
    A = (A + A.T) / 2
    
    # 拉普拉斯矩阵
    D = np.diag(np.sum(A, axis=1))
    L = D - A
    
    # 求前 n_clusters 个最小特征向量
    eigvals, eigvecs = eigsh(L, k=n_clusters, which='SM')
    
    # 对特征向量行进行 K-Means
    X_spectral = eigvecs / np.linalg.norm(eigvecs, axis=1, keepdims=True)
    labels = KMeans(n_clusters=n_clusters, n_init=10, random_state=42).fit_predict(X_spectral)
    return labels

# 2. 在非凸数据上测试
np.random.seed(42)
X_moons, y_moons = make_moons(n_samples=200, noise=0.05, random_state=42)
X_circles, y_circles = make_circles(n_samples=200, noise=0.03, factor=0.5, random_state=42)

# 应用谱聚类
labels_moons = spectral_clustering(X_moons, n_clusters=2)
labels_circles = spectral_clustering(X_circles, n_clusters=2)

# 计算与真实标签的一致性
from sklearn.metrics import adjusted_rand_score
ari_moons = adjusted_rand_score(y_moons, labels_moons)
ari_circles = adjusted_rand_score(y_circles, labels_circles)

print("谱聚类在非凸数据上的表现:")
print(f"  双月数据 ARI: {ari_moons:.4f}")
print(f"  双环数据 ARI: {ari_circles:.4f}")

# 3. 与 K-Means 对比
from sklearn.cluster import KMeans
km_moons = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_moons)
km_circles = KMeans(n_clusters=2, n_init=10, random_state=42).fit_predict(X_circles)

ari_km_moons = adjusted_rand_score(y_moons, km_moons)
ari_km_circles = adjusted_rand_score(y_circles, km_circles)
print(f"\n对比 K-Means:")
print(f"  双月 ARI: K-Means={ari_km_moons:.4f}, Spectral={ari_moons:.4f}")
print(f"  双环 ARI: K-Means={ari_km_circles:.4f}, Spectral={ari_circles:.4f}")

# 4. 聚类结果分析
print(f"\n谱聚类双月标签分布: {np.bincount(labels_moons)}")
print(f"真实标签分布: {np.bincount(y_moons)}")

# 5. 参数敏感性
print("\n参数敏感性分析 (n_neighbors):")
for nn in [3, 5, 10, 20]:
    labels = spectral_clustering(X_moons, n_clusters=2, n_neighbors=nn)
    ari = adjusted_rand_score(y_moons, labels)
    print(f"  n_neighbors={nn:2d}: ARI={ari:.4f}")

# 6. 谱聚类特征向量的信息
n = 20
# 简单路径图：0-1-2-...-19
A_path = np.zeros((n, n))
for i in range(n-1):
    A_path[i, i+1] = A_path[i+1, i] = 1

L_path = np.diag(np.sum(A_path, axis=1)) - A_path
eigvals, eigvecs = np.linalg.eigh(L_path)
# Fiedler 向量
fiedler = eigvecs[:, 1]
print(f"\n路径图的 Fiedler 向量 (λ₂={eigvals[1]:.4f}):")
print(f"  前5个值: {np.round(fiedler[:5], 4)}")
print(f"  后5个值: {np.round(fiedler[-5:], 4)}")
# 通过正负可以二分路径图
print(f"  二分点: {np.sum(fiedler < 0)} 节点在左侧, {np.sum(fiedler >= 0)} 在右侧")
```

## 深度学习关联

- **谱聚类与图神经网络**：谱聚类为图神经网络提供了理论基础。GCN 的传播规则 $\tilde{D}^{-1/2}\tilde{A}\tilde{D}^{-1/2}$ 正是归一化拉普拉斯矩阵的变体。谱聚类中的特征向量对应 GCN 中"平滑"的低频图信号。多个 GCN 层的堆叠相当于对节点特征进行迭代拉普拉斯平滑，最终使同一簇内节点特征趋于一致。
- **自监督学习中的聚类**：DeepCluster 和 SwAV 等自监督方法交替进行特征提取和聚类分配，其中聚类步骤可以看作可学习的"软谱聚类"。这些方法通过将图像特征聚类来生成伪标签，然后用伪标签训练特征提取器，形成闭环。
- **社区检测与社交网络**：谱聚类广泛应用于社交网络中的社区检测——自动发现用户群组、兴趣圈子。在推荐系统中，用户-物品二分图的谱聚类可以同时发现用户群组和物品类别（协同聚类），提高推荐质量。
- **谱聚类与注意力机制**：Transformer 的自注意力矩阵可以看作动态构建的全连接图，注意力分数是边权重。谱聚类可以分析注意力头的"社区结构"——不同注意力头是否关注不同的"令牌群组"，这有助于解释 Transformer 的内部工作机制。
