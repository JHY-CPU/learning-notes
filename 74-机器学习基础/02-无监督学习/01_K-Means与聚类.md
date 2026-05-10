# K-Means与聚类 - 机器学习基础


# K-Means与聚类算法


机器学习基础 - 无监督学习 | 数据分组与模式发现


## 目录


1. [K-Means 算法](#kmeans)
2. [K-Means++ 初始化](#kmeans-pp)
3. [DBSCAN 密度聚类](#dbscan)
4. [层次聚类](#hierarchical)
5. [聚类评估指标](#evaluation)
6. [肘部法则与轮廓系数](#elbow)
7. [代码实现](#code)


## 1. K-Means 算法


K-Means 是最经典的聚类算法，通过迭代地将数据点分配到最近的簇中心并更新中心来实现聚类。


### 1.1 算法流程


1. 随机选择 K 个点作为初始簇中心
2. **分配步骤**
   ：将每个数据点分配到距离最近的簇中心
3. **更新步骤**
   ：重新计算每个簇的中心（均值）
4. 重复步骤 2-3 直到簇中心不再变化或达到最大迭代次数


### 1.2 目标函数


K-Means 最小化簇内平方和（Within-Cluster Sum of Squares, WCSS）：


$$
J = Σₖ₌₁ᴷ Σ_{xᵢ∈Cₖ} ||xᵢ - μₖ||²
$$


其中 μₖ 是第 k 个簇的中心，Cₖ 是第 k 个簇的数据点集合。


### 1.3 收敛性


- K-Means 保证收敛（每步 WCSS 不增），但可能收敛到
   **局部最优**
- 结果依赖于初始中心的选择
- 实践中多次运行取 WCSS 最小的结果


### 1.4 复杂度


- 时间复杂度：O(n·K·d·T)，n 为样本数，d 为维度，T 为迭代次数
- 空间复杂度：O(n·d + K·d)


### 1.5 K-Means 的局限


- 需要预先指定 K 值
- 只能发现球形（凸）簇
- 对初始中心敏感
- 对异常值敏感
- 假设各簇大小和密度相近


## 2. K-Means++ 初始化


K-Means++ 通过智能的初始化策略，显著改善了 K-Means 的收敛质量和速度。


### 2.1 初始化步骤


1. 随机选择第一个簇中心 μ₁
2. 对每个数据点 xᵢ，计算到最近已选中心的距离 D(xᵢ)
3. 按概率 P(xᵢ) = D(xᵢ)² / Σⱼ D(xⱼ)² 选择下一个中心
4. 重复步骤 2-3 直到选出 K 个中心


### 2.2 优势


- 初始中心尽可能分散，避免聚集在一起
- 理论上保证 O(log K) 的近似比
- sklearn 中
   `init='k-means++'`
   是默认选项


## 3. DBSCAN 密度聚类


DBSCAN（Density-Based Spatial Clustering of Applications with Noise）是一种基于密度的聚类算法，能发现任意形状的簇并自动识别噪声点。


### 3.1 核心概念


- **ε-邻域**
   ：以点 p 为中心、半径 ε 内的所有点
- **核心点**
   ：ε-邻域内至少有 MinPts 个点
- **边界点**
   ：在某个核心点的 ε-邻域内，但自身不是核心点
- **噪声点**
   ：既不是核心点也不是边界点


### 3.2 算法流程


1. 对每个未访问的点，检查其 ε-邻域
2. 如果是核心点，创建新簇，将邻域内所有点加入簇
3. 对新加入的核心点，递归扩展其邻域内的点
4. 如果不是核心点，标记为噪声（后续可能被其他簇吸收为边界点）


### 3.3 参数选择


- **ε**
   ：可通过 k-距离图（k = MinPts - 1）的"肘部"确定
- **MinPts**
   ：经验法则取 ≥ d + 1（d 为维度），通常取 5-10


### 3.4 DBSCAN vs K-Means


| 特性 | K-Means | DBSCAN |
| --- | --- | --- |
| 簇形状 | 球形 | 任意形状 |
| 需要指定 K | 是 | 否 |
| 噪声处理 | 无 | 自动识别噪声 |
| 参数 | K | ε, MinPts |
| 适用场景 | 球形簇、大样本 | 任意形状、有噪声 |


## 4. 层次聚类


层次聚类通过逐层合并或分裂簇来构建聚类的层次结构（树状图/Dendrogram）。


### 4.1 凝聚式（自底向上）


1. 将每个数据点视为一个单独的簇
2. 找到距离最近的两个簇，合并它们
3. 重复步骤 2 直到只剩一个簇或达到目标簇数


### 4.2 链接方式 (Linkage)


| 方式 | 定义 | 特点 |
| --- | --- | --- |
| Single | 两簇间最近点的距离 | 可能产生链式簇 |
| Complete | 两簇间最远点的距离 | 簇更紧凑 |
| Average | 两簇间所有点对的平均距离 | 折中方案 |
| Ward | 合并后 WCSS 的增加量 | 类似 K-Means，最常用 |


### 4.3 分裂式（自顶向下）


从所有数据在一个簇开始，每次选择一个簇进行分裂。计算量通常更大，实践中较少使用。


### 4.4 复杂度


- 时间复杂度：O(n³)（朴素实现），O(n²logn)（优化实现）
- 不适合大规模数据集（n > 10000）


## 5. 聚类评估指标


### 5.1 内部评估（无标签）


| 指标 | 公式思想 | 最优 |
| --- | --- | --- |
| **轮廓系数** | (b-a)/max(a,b)，a=簇内距离，b=最近簇距离 | 接近 1 |
| **Calinski-Harabasz** | 簇间方差/簇内方差 | 越大越好 |
| **Davies-Bouldin** | 簇内散布/簇间分离 | 越小越好 |
| **WCSS** | 簇内平方和 | 越小越好 |


### 5.2 外部评估（有标签）


| 指标 | 说明 | 取值范围 |
| --- | --- | --- |
| **ARI** (Adjusted Rand Index) | 调整后的兰德指数 | [-1, 1] |
| **NMI** (Normalized Mutual Information) | 归一化互信息 | [0, 1] |
| **FMI** (Fowlkes-Mallows Index) | 精确率和召回率的几何平均 | [0, 1] |


## 6. 肘部法则与轮廓系数


### 6.1 肘部法则 (Elbow Method)


对不同的 K 值计算 WCSS，绘制 K-WCSS 曲线。曲线的"肘部"（下降速度明显变缓的点）即为最佳 K 值。


- 优点：简单直观
- 缺点：肘部不明显时难以判断


### 6.2 轮廓系数法


对每个样本 i，计算：


$$
a(i) = 簇内所有点到 i 的平均距离
                b(i) = i 到最近的其他簇中所有点的平均距离
                s(i) = (b(i) - a(i)) / max(a(i), b(i))
$$


所有样本的轮廓系数均值即为聚类的总轮廓系数。选择使轮廓系数最大的 K 值。


- s(i) ≈ 1：聚类效果好
- s(i) ≈ 0：在两簇边界
- s(i) ≈ -1：可能分错了簇


## 7. Python 代码实现


### 7.1 K-Means 与肘部法则


```
from sklearn.cluster import KMeans, DBSCAN, AgglomerativeClustering
from sklearn.metrics import silhouette_score, calinski_harabasz_score
import matplotlib.pyplot as plt

# 肘部法则
wcss = []
sil_scores = []
K_range = range(2, 11)
for k in K_range:
    kmeans = KMeans(n_clusters=k, init='k-means++', n_init=10, random_state=42)
    kmeans.fit(X)
    wcss.append(kmeans.inertia_)
    sil_scores.append(silhouette_score(X, kmeans.labels_))

# 选择最优 K（轮廓系数最大）
best_k = list(K_range)[np.argmax(sil_scores)]
print(f"最优簇数: {best_k}, 轮廓系数: {max(sil_scores):.4f}")

# 最终聚类
kmeans = KMeans(n_clusters=best_k, init='k-means++', n_init=10)
labels = kmeans.fit_predict(X)
print(f"CH 指标: {calinski_harabasz_score(X, labels):.2f}")
```


### 7.2 DBSCAN


```
# DBSCAN
dbscan = DBSCAN(eps=0.5, min_samples=5)
labels = dbscan.fit_predict(X)

n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise = list(labels).count(-1)
print(f"簇数: {n_clusters}, 噪声点: {n_noise}")

# 使用 k-距离图确定 eps
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X)
distances, _ = nn.kneighbors(X)
k_distances = np.sort(distances[:, -1])  # 第 k 近邻距离排序
```


### 7.3 层次聚类


```
from scipy.cluster.hierarchy import dendrogram, linkage

# 绘制树状图
Z = linkage(X, method='ward')
dendrogram(Z, truncate_mode='lastp', p=12)

# 凝聚聚类
agg = AgglomerativeClustering(n_clusters=3, linkage='ward')
labels = agg.fit_predict(X)
```


## 8. 总结


1. **K-Means**
   简单高效，适合球形簇，需要预设 K，对初始化敏感
2. **K-Means++**
   改进初始化，使结果更稳定
3. **DBSCAN**
   能发现任意形状的簇和噪声，不需要指定 K，但对参数 ε 敏感
4. **层次聚类**
   提供聚类的层次结构，适合探索性分析，但 O(n³) 复杂度限制了扩展性
5. 选择 K 值可用
   **肘部法则**
   和
   **轮廓系数**


> **Tip:** **实践建议：**
> 先用 K-Means 快速尝试；簇形状不规则时用 DBSCAN；需要层次结构时用凝聚聚类。特征标准化对所有聚类算法都很重要。


机器学习基础笔记 - 无监督学习 - K-Means与聚类


内容涵盖：K-Means/K-Means++、DBSCAN密度聚类、层次聚类(Ward链接)、轮廓系数、肘部法则


<!-- Converted from: 01_K-Means与聚类.html -->
