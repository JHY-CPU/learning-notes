# 18_t-SNE 降维与高维空间相似度保持

## 核心概念

- **t-SNE**：t-Distributed Stochastic Neighbor Embedding，一种非线性降维可视化方法，旨在保持高维空间中数据点之间的相似度关系映射到低维。
- **核心思想**：在高维空间用高斯分布定义点之间的相似度（条件概率 $p_{j|i}$），在低维空间用 t 分布定义相似度 $q_{j|i}$，最小化两个概率分布之间的 KL 散度。
- **SNE vs t-SNE**：SNE 使用对称高斯分布；t-SNE 在低维空间使用**学生 t 分布**（自由度为 1，即 Cauchy 分布），解决"拥挤问题" (Crowding Problem)。
- **困惑度 (Perplexity)**：控制高斯核的带宽，可理解为每个点有效的近邻数量。通常取 5-50。困惑度越大，算法越关注全局结构。
- **KL 散度损失**：$KL(P\|Q) = \sum_i \sum_j p_{ij} \log(p_{ij}/q_{ij})$，对高维空间中近邻关系给予更高权重（保留局部结构优先于全局结构）。
- **t 分布的长尾特性**：t 分布的尾巴比高斯分布更厚，使得低维空间中相距较远的点之间的相似度不至于过小，允许聚类在低维空间中更分散，避免簇间挤压。

## 数学推导

高维空间中点 $i$ 和 $j$ 的相似度条件概率：
$$
p_{j|i} = \frac{\exp(-\|x_i - x_j\|^2 / 2\sigma_i^2)}{\sum_{k \neq l} \exp(-\|x_k - x_l\|^2 / 2\sigma_i^2)}
$$

对称化 SNE 的联合概率（取对称平均避免不对称性）：
$$
p_{ij} = \frac{p_{j|i} + p_{i|j}}{2m}
$$

低维空间中使用 Student t 分布（自由度 1）定义相似度：
$$
q_{ij} = \frac{(1 + \|y_i - y_j\|^2)^{-1}}{\sum_{k \neq l} (1 + \|y_k - y_l\|^2)^{-1}}
$$

KL 散度损失函数：
$$
C = KL(P\|Q) = \sum_{i \neq j} p_{ij} \log \frac{p_{ij}}{q_{ij}}
$$

梯度为：
$$
\frac{\partial C}{\partial y_i} = 4 \sum_j (p_{ij} - q_{ij})(y_i - y_j)(1 + \|y_i - y_j\|^2)^{-1}
$$

梯度下降的物理意义：$y_i$ 受到所有其他点的"合力"作用——$p_{ij} > q_{ij}$ 时相互吸引，$p_{ij} < q_{ij}$ 时相互排斥。

**困惑度 (Perplexity) 与 $\sigma_i$**：
$$
Perp(P_i) = 2^{H(P_i)} = 2^{-\sum_j p_{j|i} \log_2 p_{j|i}}
$$
通过二分搜索选择 $\sigma_i$ 使特定样本的困惑度等于预设值。

## 直观理解

- **"保持朋友圈"的地图绘制**：t-SNE 的目标类似于绘制一张城市地图——在高维空间中，A 和 B 是好朋友（相似度高），B 和 C 也是好朋友，但 A 和 C 一般。t-SNE 要在低维地图上尽量保持这些"朋友圈关系"。
- **t 分布为什么更好**：t 分布的长尾允许低维空间中距离较远的点之间仍有非零的相似度。这好比在拥挤的地铁上——如果人与人之间用高斯分布的排斥力，大家挤在一起动弹不得（拥挤问题）；用 t 分布的排斥力，远处的人也有一定的推开力，大家自然地分散开来，形成清晰的簇。
- **局部 vs 全局**：KL 散度的非对称性 $(P\|Q)$ 意味着：高维空间中离得近的点（$p_{ij}$ 大），在低维空间必须离得近（否则惩罚很大）；但高维空间中离得远的点（$p_{ij}$ 小），在低维空间的放置位置可以随意（惩罚很小）。因此 t-SNE 优先保留局部结构。

## 代码示例

```python
import numpy as np
from sklearn.manifold import TSNE
from sklearn.datasets import load_digits
from sklearn.preprocessing import StandardScaler

# 加载手写数字数据
digits = load_digits()
X, y = digits.data, digits.target
print(f"数据形状: {X.shape}")

# t-SNE 降维到 2D（t-SNE 通常不需要标准化，但这里先标准化）
X_scaled = StandardScaler().fit_transform(X)

tsne = TSNE(n_components=2, perplexity=30, learning_rate=200,
            n_iter=1000, random_state=42)
X_tsne = tsne.fit_transform(X_scaled)

print(f"降维后形状: {X_tsne.shape}")
print(f"KL 散度: {tsne.kl_divergence_:.4f}")

# 不同困惑度的效果对比
for perp in [5, 30, 50]:
    tsne_p = TSNE(n_components=2, perplexity=perp,
                  n_iter=500, random_state=42)
    X_p = tsne_p.fit_transform(X_scaled)
    print(f"perplexity={perp}, KL={tsne_p.kl_divergence_:.4f}")
```

## 深度学习关联

- **可视化深度特征**：t-SNE 是深度学习中**最常用的特征可视化工具**——将神经网络最后一层或中间层的特征向量用 t-SNE 投影到 2D，观察类别是否形成清晰聚类，从而判断特征提取质量。
- **对比学习的隐式 t-SNE**：SimCLR、MoCo 等对比学习方法的本质与 t-SNE 高度相似——它们都将同一图像的不同增强视为正对（接近），不同图像视为负对（推开），目标也是在特征空间中形成良好的聚类结构。
- **UMAP——t-SNE 的深度继承者**：UMAP (Uniform Manifold Approximation and Projection) 是目前更先进的降维可视化方法，其数学框架（基于黎曼几何和代数拓扑）相比 t-SNE 更理论化，但核心"保持邻居关系"的思想源于 t-SNE。
