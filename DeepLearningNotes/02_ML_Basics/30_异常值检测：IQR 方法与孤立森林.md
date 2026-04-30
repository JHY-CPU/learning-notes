# 异常值检测：IQR 方法与孤立森林

## 核心概念
- **异常值 (Outlier)**：显著偏离其他数据点的观测值，可能由测量误差、数据录入错误或真实的罕见事件产生。
- **IQR 方法**：基于四分位距的统计方法，定义 $Q1 - 1.5 \times IQR$ 和 $Q3 + 1.5 \times IQR$ 为正常区间，超出为异常。简单高效，假设数据近似对称分布。
- **孤立森林 (Isolation Forest)**：一种基于集成学习的无监督异常检测算法，利用异常点"少而不同"的特性——异常更容易被孤立，因此所需分裂次数更少。
- **异常得分**：孤立森林中，样本的异常得分定义为 $\text{Score}(x) = 2^{-\frac{E(h(x))}{c(n)}}$，其中 $h(x)$ 是样本在树中的路径长度，$c(n)$ 是平均路径长度的归一化常数。得分接近 1 为异常，接近 0.5 为正常。
- **局部异常因子 (LOF)**：基于局部密度的异常检测方法，比较样本点邻域的密度与其邻居的密度。如果样本的密度远低于邻居的密度，则为异常。
- **高维挑战**：在高维空间中，所有点之间的距离都趋于相似（维度灾难），基于距离的异常检测方法失效。孤立森林对高维数据相对更鲁棒。

## 数学推导
**IQR 方法**：
$$
IQR = Q3 - Q1
$$
异常区间（Tukey's fences）：
$$
[Q1 - 1.5 \times IQR, \; Q3 + 1.5 \times IQR]
$$
对于正态分布，此区间约包含 99.3% 的数据。

**孤立森林**：
建树过程：随机选一个特征 $f$，在特征 $f$ 的最小值和最大值之间随机选一个切分值 $p$，将数据分成左右子树。递归直到所有样本被孤立或达到最大深度。

路径长度 $h(x)$：样本 $x$ 从根节点到叶节点经过的边数。

平均路径长度的归一化常数（BST 理论）：
$$
c(n) = 2H(n-1) - \frac{2(n-1)}{n}
$$
其中 $H(i) \approx \ln(i) + 0.577$ 是调和数。

异常得分：
$$
s(x, n) = 2^{-\frac{E(h(x))}{c(n)}}
$$
- $s \to 1$：异常
- $s \approx 0.5$：正常
- $s \to 0$：密集的正常点

**LOF (局部异常因子)**：
$$
LOF_k(A) = \frac{\frac{1}{k} \sum_{B \in N_k(A)} lrd_k(B)}{lrd_k(A)}
$$
其中 $lrd_k(A)$ 是点 $A$ 的局部可达密度。$LOF > 1$ 表示点 $A$ 的密度低于其邻居（异常），$LOF \approx 1$ 表示正常。

## 直观理解
- **IQR 的"箱线图"视角**：箱线图中，箱体覆盖了中间 50% 的数据（Q1 到 Q3），须延伸到 $1.5 \times IQR$ 的位置。超出须的数据点以圆点标出——这些就是 IQR 方法识别的异常值。这个方法假设大部分数据集中在中间，远离中心的是异常。
- **孤立森林的"捉迷藏"比喻**：在森林中孤立的"异常者"很容易被找到（很快被隔离），但在人群中的"正常人"很难被隔离。孤立森林就是随机在特征空间中"切一刀"，看多少次能把一个点单独切出来——异常点很快就被孤立（路径短），正常点需要很多次分割（路径长）。
- **为什么孤立森林用路径长度**：想象在二维平面上，一个孤立的点可以在 2-3 次随机切分后就被单独分离；但一群密集的点可能需要 10 多次才能把其中一个单独分离。路径长度天然地反映了"孤立难度"。

## 代码示例
```python
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split

# 生成含异常的数据
X, y = make_classification(n_samples=300, n_features=5,
                           n_informative=3, n_repeated=0,
                           contamination=0.05, random_state=42)

# 孤立森林
iso_forest = IsolationForest(contamination=0.05, random_state=42)
y_pred_iso = iso_forest.fit_predict(X)  # -1 = 异常, 1 = 正常
anomaly_scores = iso_forest.score_samples(X)
n_anomalies = sum(y_pred_iso == -1)
print(f"孤立森林检测到 {n_anomalies} 个异常")

# LOF
lof = LocalOutlierFactor(contamination=0.05, n_neighbors=20)
y_pred_lof = lof.fit_predict(X)
print(f"LOF 检测到 {sum(y_pred_lof == -1)} 个异常")

# IQR 方法（单变量示例）
data_1d = X[:, 0]
Q1, Q3 = np.percentile(data_1d, [25, 75])
IQR = Q3 - Q1
lower, upper = Q1 - 1.5*IQR, Q3 + 1.5*IQR
outliers_1d = np.where((data_1d < lower) | (data_1d > upper))[0]
print(f"IQR 方法检测到 {len(outliers_1d)} 个异常 (特征0)")
```

## 深度学习关联
- **自编码器用于异常检测**：深度自编码器是深度学习中最常用的异常检测方法——训练自编码器重构正常数据，异常数据的重构误差会显著高于正常数据。这与孤立森林基于"异常点更容易被分离"的思想类似，只是用重构误差代替了路径长度。
- **时序异常检测的深度方法**：在时间序列异常检测中，LSTM-Autoencoder 和 Transformer-based 方法（如 Anomaly Transformer）结合了序列建模和重构误差，是更先进的异常检测方案，尤其适用于复杂的时间依赖模式。
- **对比学习与异常检测**：SimCLR、MoCo 等对比学习方法也被应用于异常检测——正常样本在特征空间中会形成紧密的聚类，异常样本与所有聚类的距离都较远。这与孤立森林中"异常点孤立"的思想在特征空间层面一致。
