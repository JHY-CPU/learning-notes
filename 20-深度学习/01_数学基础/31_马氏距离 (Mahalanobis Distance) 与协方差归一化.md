# 32_马氏距离 (Mahalanobis Distance) 与协方差归一化

## 核心概念

- **马氏距离 (Mahalanobis Distance)**：考虑数据协方差结构的距离度量，$D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}$。它消除了特征之间的相关性影响并进行了尺度归一化。
- **与欧氏距离的关系**：当 $\Sigma = I$（各向同性）时，马氏距离退化为欧氏距离。当 $\Sigma$ 是对角矩阵时，马氏距离是标准化后的欧氏距离。
- **协方差归一化**：马氏距离通过 $\Sigma^{-1}$ 对数据进行"白化"——去除相关性和缩放差异。变换后的数据 $\mathbf{z} = \Sigma^{-1/2}(\mathbf{x} - \boldsymbol{\mu})$ 满足 $\text{Cov}(\mathbf{z}) = I$。
- **马氏距离的性质**：
  - 平移不变性：与数据的绝对位置无关
  - 尺度不变性：对不同尺度的特征自动加权
  - 相关性不变性：消除了特征间相关性的影响
- **异常检测中的应用**：马氏距离衡量样本偏离分布中心的程度，可用于多元异常检测。距离超过阈值（如卡方分布分位数）的样本被视为异常。

## 数学推导

马氏距离定义：
$$
D_M(\mathbf{x}) = \sqrt{(\mathbf{x} - \boldsymbol{\mu})^T \Sigma^{-1} (\mathbf{x} - \boldsymbol{\mu})}
$$

其中 $\boldsymbol{\mu} = \mathbb{E}[\mathbf{X}]$ 是均值向量，$\Sigma = \text{Cov}(\mathbf{X})$ 是协方差矩阵。

两点之间的马氏距离：
$$
D_M(\mathbf{x}, \mathbf{y}) = \sqrt{(\mathbf{x} - \mathbf{y})^T \Sigma^{-1} (\mathbf{x} - \mathbf{y})}
$$

Cholesky 分解实现白化：$\Sigma = LL^T$，则 $\mathbf{z} = L^{-1}(\mathbf{x} - \boldsymbol{\mu})$，$\text{Cov}(\mathbf{z}) = I$，且：
$$
D_M(\mathbf{x}) = \|\mathbf{z}\|_2 = \sqrt{\mathbf{z}^T \mathbf{z}}
$$

马氏距离的平方服从卡方分布（当 $\mathbf{x}$ 服从多元高斯时）：
$$
D_M^2(\mathbf{x}) \sim \chi^2_d
$$

## 直观理解

- **椭球距离**：欧氏距离用圆形（球体）等距面，马氏距离用椭球等距面。椭球的形状由 $\Sigma$ 的特征向量和特征值决定——沿数据方差大的方向距离"缩短"，方差小的方向距离"拉长"。
- **"标准化"的直觉**：想象测量身高（厘米）和体重（公斤）的差异。欧氏距离中，身高的 1 厘米差异和体重的 1 公斤差异权重相同，这显然不合理。马氏距离会自动进行"标准化"——身高的方差大，所以 1 厘米的"权重"较小。
- **流形上的距离**：马氏距离可以看作数据流形上的局部距离近似。当数据分布呈椭球状时，马氏距离是在"拉伸"后的空间中沿直线测量距离。

## 代码示例

```python
import numpy as np

# 1. 马氏距离 vs 欧氏距离
np.random.seed(42)

# 生成相关数据：x1 和 x2 高度相关
mean = np.array([0, 0])
cov = np.array([[4, 3], [3, 4]])  # 强正相关
data = np.random.multivariate_normal(mean, cov, 500)

# 两个"异常"点
point_a = np.array([2, 2])  # 沿相关方向（合理位置）
point_b = np.array([3, -3])  # 垂直于相关方向（不合理位置）

# 计算距离
def mahalanobis(x, mean, cov):
    diff = x - mean
    return np.sqrt(diff @ np.linalg.inv(cov) @ diff)

euclidean_a = np.linalg.norm(point_a - mean)
euclidean_b = np.linalg.norm(point_b - mean)
mahalanobis_a = mahalanobis(point_a, mean, cov)
mahalanobis_b = mahalanobis(point_b, mean, cov)

print("马氏距离 vs 欧氏距离:")
print(f"  点 A (2,2) - 沿相关方向:")
print(f"    欧氏距离: {euclidean_a:.4f}")
print(f"    马氏距离: {mahalanobis_a:.4f}")
print(f"  点 B (3,-3) - 垂直相关方向:")
print(f"    欧氏距离: {euclidean_b:.4f}")
print(f"    马氏距离: {mahalanobis_b:.4f}")
print(f"  马氏距离 A vs B: 点A更合理(高概率), 点B是异常(低概率)")

# 2. 白化变换
L = np.linalg.cholesky(cov)
L_inv = np.linalg.inv(L)

# 白化数据
whitened = (data - mean) @ L_inv.T
print(f"\n白化后协方差矩阵:")
print(np.cov(whitened, rowvar=False))  # 应接近单位矩阵

# 3. 使用马氏距离做异常检测
np.random.seed(42)
n, d = 200, 5
mean_normal = np.random.randn(d)
cov_normal = np.random.randn(d, d)
cov_normal = cov_normal.T @ cov_normal + np.eye(d) * 0.1

normal_data = np.random.multivariate_normal(mean_normal, cov_normal, n)
# 加入异常
outliers = np.random.uniform(-10, 10, (10, d))
all_data = np.vstack([normal_data, outliers])

# 计算每个样本的马氏距离
empirical_mean = np.mean(all_data, axis=0)
empirical_cov = np.cov(all_data, rowvar=False)
inv_cov = np.linalg.inv(empirical_cov)

distances = []
for i in range(len(all_data)):
    diff = all_data[i] - empirical_mean
    dist = np.sqrt(diff @ inv_cov @ diff)
    distances.append(dist)

distances = np.array(distances)
# 卡方分布的 99% 分位数 (df=5)
threshold = np.sqrt(20.515)  # chi2.ppf(0.99, df=5)
detected_outliers = np.where(distances > threshold)[0]

print(f"\n异常检测 (99% 置信水平):")
print(f"  总样本: {len(all_data)}, 注入异常: {10}")
print(f"  检测为异常: {len(detected_outliers)}")
print(f"  其中真实异常: {np.sum(detected_outliers >= n)}")

# 4. 马氏距离在分类中的应用
# 简单二次判别分析 (QDA)
class1_mean = np.array([1, 1])
class2_mean = np.array([4, 4])
class1_cov = np.array([[1, 0.5], [0.5, 2]])
class2_cov = np.array([[2, -0.5], [-0.5, 1]])

test_point = np.array([3, 2])
d1 = mahalanobis(test_point, class1_mean, class1_cov)
d2 = mahalanobis(test_point, class2_mean, class2_cov)
print(f"\n分类: 点到类别1的马氏距离={d1:.4f}, 到类别2={d2:.4f}")
print(f"  分类为类别 {'1' if d1 < d2 else '2'}")
```

## 深度学习关联

- **白化与 BatchNorm**：马氏距离的核心操作——协方差归一化——与批量归一化 (BN) 的目标一致。BN 对每个特征维度独立标准化（相当于 $\Sigma$ 是对角矩阵），而全白化（ZCA）同时对特征做去相关处理。在实践中，完全的白化计算成本高，BN 提供了实用的近似。
- **度量学习 (Metric Learning)**：深度度量学习通过神经网络学习嵌入空间，使马氏距离（或欧氏距离）在嵌入空间中反映样本的语义相似度。Triplet Loss 和 Contrastive Loss 都基于这一思想：同类样本距离近，异类样本距离远。
- **高斯判别分析 (GDA) 与分类**：GDA/QDA 假设每类数据服从多元高斯分布，分类决策基于马氏距离。神经网络分类器可以看作对类别后验的柔性建模，其中最后一层等价于计算输入与类别中心的某种距离。
- **自编码器与异常检测**：在高维异常检测中，自编码器的重建误差常作为异常分数。基于马氏距离的方法（如计算隐空间中点到分布中心的距离）可以结合深度特征和统计方法，在工业异常检测（如 MVTec AD）中取得了良好效果。
