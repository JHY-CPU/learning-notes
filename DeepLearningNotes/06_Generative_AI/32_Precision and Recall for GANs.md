# 32_Precision and Recall for GANs

## 核心概念
- **Precision (精确率)/Recall (召回率)**：从真实分布和生成分布中分别采样，用最近邻方法计算两者之间的双向覆盖度，独立评估生成质量（Precision）和多样性（Recall）。
- **Precision (质量)**：生成样本中有多少落在真实分布流形的范围内。高 Precision 意味着生成的样本看起来像真的。
- **Recall (多样性/召回率)**：真实样本中有多少能被生成分布流形覆盖。高 Recall 意味着生成器覆盖了真实分布的所有模式。
- **与 FID 的区别**：FID 是一个混杂指标——它同时受质量和多样性影响，无法单独判断一个模型是"质量差"还是"多样性差"。Precision 和 Recall 将两者解耦。
- **流形估计**：方法将特征空间中每个点的 $k$ 个最近邻作为该点所在局部流形的近似。对于每个生成点，判断它是否在真实流形内；对每个真实点，判断它是否在生成流形内。
- **算法步骤**：(1) 提取真实和生成图像的特征，(2) 对每个点找到其在同类中的 $k$ 个最近邻，(3) 用这些邻域估计流形，(4) 计算双向覆盖。

## 数学推导

**流形近似**：

设 $\Phi_r = \{\phi_r^{(1)}, ..., \phi_r^{(N_r)}\}$ 是真实图像的特征集合，$\Phi_g = \{\phi_g^{(1)}, ..., \phi_g^{(N_g)}\}$ 是生成图像的特征集合。

对于特征空间中的点 $\phi$，定义其到同类集合的 $k$ 近邻距离（$k=3$ 通常选择 3）：

$$
d_k(\phi, \Phi) = \|\phi - \text{NN}_k(\phi, \Phi)\|_2
$$

其中 $\text{NN}_k(\phi, \Phi)$ 是 $\phi$ 在 $\Phi$ 中的第 $k$ 个最近邻。

**流形二元分类**：

- 如果 $\phi_g^{(i)}$ 到真实集合的 $k$ 近邻距离小于其到真实集合的 $k$ 近邻距离的某个阈值，则 $\phi_g^{(i)}$ 被认为在真实流形内——这决定了 Precision。
- 如果 $\phi_r^{(j)}$ 到生成集合的 $k$ 近邻距离类似，它就在生成流形内——这决定了 Recall。

**Precision 的定义**：

$$
\text{Precision}(\Phi_r, \Phi_g) = \frac{|\{\phi_g \in \Phi_g : \phi_g \in \text{manifold}(\Phi_r)\}|}{|\Phi_g|}
$$

**Recall 的定义**：

$$
\text{Recall}(\Phi_r, \Phi_g) = \frac{|\{\phi_r \in \Phi_r : \phi_r \in \text{manifold}(\Phi_g)\}|}{|\Phi_r|}
$$

**改进方法：密度和覆盖 (Density & Coverage)**：

Density 和 Coverage 是 Precision/Recall 的改进，对 $k$ 值的选择不那么敏感：

- Density：衡量生成分布"填充"真实流形的程度（可大于 1，允许多个生成点落在同一真实邻域内）
- Coverage：衡量真实流形被生成分布覆盖的比例（0-1 之间）

## 直观理解
- **Precision & Recall = 射击评估的两个维度**：想象你在打靶——Precision 是"每次击中的环数"（质量），Recall 是"覆盖了靶面的多少区域"（多样性）。一个只打 10 环中心的枪法 Precision 高但 Recall 低（只覆盖了靶心），一个乱打覆盖整个靶面的枪法 Recall 高但 Precision 低。
- **为什么需要解耦这两个指标**：在 GAN 训练中，有时你改善了质量但牺牲了多样性（模式崩溃加剧），或者反过来。FID 可能不变——但你知道自己做了 trade-off。PR 让你看到这种取舍。
- **$k$ 值的影响**：$k=1$ 时流形估计不稳定（每个点只靠自己）；$k=3$ 较常用；$k$ 太大时流形估计过于平滑，丢失精细结构。选 $k$ 重要但不太敏感。
- **流形假设的局限性**：方法假设特征空间中的数据分布在一个低维流形上——在某个点周围，它的 $k$ 个最近邻可以近似代表该点附近的局部流形方向。

## 代码示例

```python
import torch
import numpy as np
from sklearn.neighbors import NearestNeighbors

def compute_pr(real_features, gen_features, k=3):
    """
    计算 Precision and Recall for GANs
    
    参数:
        real_features: [N_real, D] 真实图像特征
        gen_features: [N_gen, D] 生成图像特征
        k: 最近邻数量
    返回:
        precision, recall
    """
    # 拟合最近邻模型
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_features)
    nn_gen = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(gen_features)
    
    # 计算每个点到其自身分布的 k 近邻距离
    dist_real_to_real, _ = nn_real.kneighbors(real_features)  # [N_real, k]
    dist_gen_to_gen, _ = nn_gen.kneighbors(gen_features)  # [N_gen, k]
    
    # 计算阈值：第 k 近邻距离的最大值（或取 50% 分位数）
    # 原始论文使用球形流形半径
    radius_real = np.max(dist_real_to_real[:, -1])
    radius_gen = np.max(dist_gen_to_gen[:, -1])
    
    # 计算 Precision：生成点是否在真实流形内
    dist_gen_to_real, _ = nn_real.kneighbors(gen_features, n_neighbors=k)
    in_real_manifold = dist_gen_to_real[:, -1] <= radius_real
    precision = np.mean(in_real_manifold)
    
    # 计算 Recall：真实点是否在生成流形内
    dist_real_to_gen, _ = nn_gen.kneighbors(real_features, n_neighbors=k)
    in_gen_manifold = dist_real_to_gen[:, -1] <= radius_gen
    recall = np.mean(in_gen_manifold)
    
    return precision, recall

def compute_density_coverage(real_features, gen_features, k=5):
    """
    计算 Density 和 Coverage（改进版 PR）
    
    Density: 生成点在真实流形中的"密度"，可大于 1
    Coverage: 真实流形被覆盖的比例，[0, 1]
    """
    nn_real = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(real_features)
    nn_gen = NearestNeighbors(n_neighbors=k, metric='euclidean').fit(gen_features)
    
    # 找到每个真实点的 k 近邻真实点，计算半径
    dist_real_to_real, _ = nn_real.kneighbors(real_features, n_neighbors=k)
    radii_real = dist_real_to_real[:, -1]  # [N_real] 每个真实点的流形半径
    
    # Density: 对每个生成点，计算它落在多少个真实点的流形内
    dist_gen_to_real, indices_gen = nn_real.kneighbors(gen_features, n_neighbors=len(real_features))
    
    density = 0
    for i, (dist, _) in enumerate(zip(dist_gen_to_real, indices_gen)):
        # 对每个生成点，计算它在半径内的真实点的数量
        in_radius = dist[:, None] <= radii_real[None, :]
        density += np.sum(in_radius) / k
    
    density = density / len(gen_features)
    
    # Coverage: 对每个真实点，检查是否有至少一个生成点在它的流形内
    coverage = 0
    for i in range(len(real_features)):
        real_point_radius = radii_real[i]
        dist_to_gen, _ = nn_gen.kneighbors(real_features[i:i+1])
        if dist_to_gen[0, 0] <= real_point_radius:
            coverage += 1
    
    coverage = coverage / len(real_features)
    
    return density, coverage

# 模拟分析
print("=== Precision & Recall 分析 ===")
print()

np.random.seed(42)
N = 1000
D = 2048  # Inception 特征维度

# 场景 1: 高质量、高多样
real_feat = np.random.randn(N, D)
gen_feat_good = real_feat + np.random.randn(N, D) * 0.1  # 略微偏差

pr_1, rec_1 = compute_pr(real_feat, gen_feat_good)
den_1, cov_1 = compute_density_coverage(real_feat, gen_feat_good)
print(f"场景 1 (高质量/高多样): P={pr_1:.3f}, R={rec_1:.3f}, D={den_1:.3f}, C={cov_1:.3f}")

# 场景 2: 高质量、低多样（模式崩溃——全都一样）
gen_feat_collapse = np.random.randn(1, D).repeat(N, axis=0) + np.random.randn(N, D) * 0.01
pr_2, rec_2 = compute_pr(real_feat, gen_feat_collapse)
den_2, cov_2 = compute_density_coverage(real_feat, gen_feat_collapse)
print(f"场景 2 (模式崩溃):      P={pr_2:.3f}, R={rec_2:.3f}, D={den_2:.3f}, C={cov_2:.3f}")

# 场景 3: 低质量、高多样（全都生成噪声）
gen_feat_noise = np.random.randn(N, D) * 10
pr_3, rec_3 = compute_pr(real_feat, gen_feat_noise)
den_3, cov_3 = compute_density_coverage(real_feat, gen_feat_noise)
print(f"场景 3 (纯噪声):        P={pr_3:.3f}, R={rec_3:.3f}, D={den_3:.3f}, C={cov_3:.3f}")

print()
print("=== 解读 ===")
print("FID 是综合指标，无法区分 quality vs diversity")
print("Precision: 高 = 生成图像看起来真实")
print("Recall:    高 = 生成图像覆盖了各种模式")
print("Density:   与 Precision 相似但对异常值更鲁棒")
print("Coverage:  与 Recall 相似但不受局部密度影响")
```

## 深度学习关联
- **PR 在 GAN 训练中的应用**：在训练过程中同时监控 PR 可以更早检测模式崩溃——当 Precision 正常但 Recall 骤降时，就知道生成器正在丢失模式。
- **StyleGAN-XL 中的 PR 应用**：StyleGAN-XL 在训练中明确优化 PR，通过调整截断技巧（Truncation Trick）在 Precision 和 Recall 之间做权衡。
- **改善指标 (Kynkäänniemi et al.)**：Kynkäänniemi 等人改进了原始 PR 方法，使用更精确的流形估计（Manifold Approximation）和基于距离的二元分类。
- **生成模型的"PR 曲线"**：类似于分类中的 PR 曲线，通过改变采样参数（如 CFG 强度、截断阈值）可以绘制生成模型的 PR 曲线——一种更全面的模型评估方式。
