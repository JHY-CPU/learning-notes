# 15_GMM 高斯混合模型与 EM 算法迭代步骤

## 核心概念

- **高斯混合模型 (GMM)**：用 $K$ 个高斯分布的加权和来拟合数据的概率分布，每个高斯分量对应一个潜在的"簇"。适用于数据由多个子群体混合生成的情形。
- **软聚类**：与 K-Means 的硬分配不同，GMM 给出每个样本属于每个簇的**概率**（后验概率），称为"责任度" (Responsibility)。
- **EM 算法**：Expectation-Maximization 算法，是含有隐变量（样本来自哪个高斯分量）的极大似然估计的标准方法。
- **E 步 (期望步)**：基于当前参数估计，计算每个样本属于每个高斯分量的后验概率（责任度 $\gamma_{ik}$）。
- **M 步 (最大化步)**：基于责任度，加权更新每个高斯分量的均值、协方差和混合系数，最大化完整数据的对数似然期望。
- **与 K-Means 的关系**：当 GMM 的协方差矩阵固定为单位矩阵的倍数且 $\gamma_{ik}$ 退化为 0/1 硬分配时，GMM 退化为 K-Means。因此 K-Means 是 GMM 的一种特例。

## 数学推导

GMM 的概率密度函数：
$$
p(x|\theta) = \sum_{k=1}^K \pi_k \mathcal{N}(x|\mu_k, \Sigma_k)
$$
其中 $\sum_{k=1}^K \pi_k = 1, \pi_k \geq 0$，$\theta = \{\pi_k, \mu_k, \Sigma_k\}_{k=1}^K$。

对数似然函数（包含隐变量 $z_{ik}$ 表示样本 $i$ 来自分量 $k$）：
$$
\ell(\theta) = \sum_{i=1}^m \ln \sum_{k=1}^K \pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)
$$

直接最大化很困难（对数内求和），EM 算法通过迭代求解。

**E 步**：计算责任度（后验概率）：
$$
\gamma_{ik} = p(z_{ik}=1|x_i, \theta) = \frac{\pi_k \mathcal{N}(x_i|\mu_k, \Sigma_k)}{\sum_{j=1}^K \pi_j \mathcal{N}(x_i|\mu_j, \Sigma_j)}
$$

**M 步**：更新参数（$N_k = \sum_{i=1}^m \gamma_{ik}$ 为第 $k$ 个分量的有效样本数）：
$$
\mu_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^m \gamma_{ik} x_i
$$
$$
\Sigma_k^{\text{new}} = \frac{1}{N_k} \sum_{i=1}^m \gamma_{ik} (x_i - \mu_k^{\text{new}})(x_i - \mu_k^{\text{new}})^T
$$
$$
\pi_k^{\text{new}} = \frac{N_k}{m}
$$

**E-M 迭代的收敛**：EM 算法保证每一步都使对数似然 $\ell(\theta)$ 单调不减（或严格递增），最终收敛到局部最优。收敛判据通常是对数似然变化小于阈值。

## 直观理解

- **"剥洋葱"聚类**：GMM 不像 K-Means 那样硬性地说"这个样本属于簇 A"，而是说"这个样本有 70% 的可能属于簇 A，20% 属于簇 B，10% 属于簇 C"。这种软性分配在簇重叠严重时更合理。
- **EM 的"先猜后修正"**：EM 算法像在玩一个"先猜后证"的游戏——先猜测各高斯分量的参数（瞎猜），然后评估每个样本来自哪个分量的概率（E 步），再用这些概率反过来修正参数估计（M 步）。循环往复，猜测越来越准。
- **为什么 EM 有效**："你中有我"的工作方式——不知道参数就无法确定样本归属（隐变量），不知道样本归属就无法精确估计参数。EM 通过交替进行，让两个未知量逐步明朗化，最终同时得到合理的参数和归属。

## 代码示例

```python
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split

# 生成数据（3 个重叠的高斯簇）
X, y = make_blobs(n_samples=500, centers=3, cluster_std=1.5,
                  random_state=42)

# GMM 拟合
gmm = GaussianMixture(n_components=3, covariance_type='full',
                      random_state=42, max_iter=200)
gmm.fit(X)

# 查看参数
print(f"混合系数: {gmm.weights_}")
print(f"均值:\n{gmm.means_}")
print(f"对数似然: {gmm.score(X):.2f}")

# 软聚类 - 每个样本属于各簇的概率
probs = gmm.predict_proba(X[:5])
print(f"前5个样本的簇隶属概率:\n{probs}")

# BIC/AIC 选最优分量数
bic_scores = []
for k in range(1, 8):
    gmm_k = GaussianMixture(n_components=k, random_state=42)
    gmm_k.fit(X)
    bic_scores.append(gmm_k.bic(X))
print(f"不同 K 的 BIC: {bic_scores}")
```

## 深度学习关联

- **Mixture Density Networks (MDN)**：MDN 是 GMM 与深度神经网络的结合——用一个神经网络来输出 GMM 的参数（$\pi_k, \mu_k, \Sigma_k$），使网络不仅能预测值，还能预测完整的概率分布。广泛应用于语音合成 (WaveNet) 和不确定性估计。
- **VAE 与重参数化技巧**：变分自编码器 (VAE) 的核心假设——隐变量服从高斯分布——与 GMM 的高斯分量假设一脉相承。VAE 的 ELBO 推导和重参数化技巧 (Reparameterization Trick) 与 EM 算法中的 E 步有密切联系。
- **Soft Clustering 在 NLP 中**：GMM 的软聚类思想被广泛应用于神经主题模型 (Neural Topic Models) 和文本聚类中，每个文档被视为多个主题的混合，类似于 GMM 中每个样本属于多个簇的概率表示。
