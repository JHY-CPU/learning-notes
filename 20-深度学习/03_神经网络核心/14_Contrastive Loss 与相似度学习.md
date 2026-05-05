# 14_Contrastive Loss 与相似度学习

## 核心概念

- **Contrastive Loss 定义**：对比损失（Contrastive Loss）是一种度量学习损失函数，用于学习嵌入空间中样本之间的相似关系。它通过拉近正样本对（相似样本）的距离、推开负样本对（不相似样本）的距离来训练网络。
- **能量视角**：Contrastive Loss 基于能量模型（Energy-Based Model）的思想——为相似的样本对分配低能量（距离小），为不相似的样本对分配高能量（距离大）。损失函数直接在能量空间中进行优化。
- **Margin 的概念**：负样本对的损失只在一个预定义的边界（margin）内起作用——当负样本距离已经大于 margin 时，不再产生损失。这防止了网络无休止地将所有负样本推远，集中精力处理那些"困难"的样本对。
- **与分类损失的本质区别**：分类损失（如交叉熵）学习的是类别的决策边界，而 Contrastive Loss 学习的是样本之间的相似性度量。后者不关心具体类别是什么，只关心哪些样本应该在一起、哪些应该分开。

## 数学推导

**Contrastive Loss 定义**：

对一对样本 $(x_i, x_j)$，通过编码器 $f$ 得到嵌入向量 $h_i = f(x_i)$，$h_j = f(x_j)$。定义它们的欧氏距离：

$$
d_{ij} = \|h_i - h_j\|_2
$$

Contrastive Loss:

$$
L(x_i, x_j, y) = \frac{1}{2}(1-y)d_{ij}^2 + \frac{1}{2}y \cdot \max(0, m - d_{ij})^2
$$

其中 $y = 0$ 表示正样本对（相似），$y = 1$ 表示负样本对（不相似）。$m > 0$ 是边界（margin）。

**正样本对的梯度**（$y = 0$）：

$$
L_{\text{pos}} = \frac{1}{2}d_{ij}^2
$$

$$
\frac{\partial L_{\text{pos}}}{\partial h_i} = h_i - h_j
$$

**负样本对的梯度**（$y = 1$，且 $d_{ij} < m$ 时）：

$$
L_{\text{neg}} = \frac{1}{2}(m - d_{ij})^2
$$

$$
\frac{\partial L_{\text{neg}}}{\partial h_i} = -(m - d_{ij}) \cdot \frac{h_i - h_j}{d_{ij}} = (d_{ij} - m) \cdot \frac{h_i - h_j}{d_{ij}}
$$

当 $d_{ij} \geq m$ 时，$L_{\text{neg}} = 0$，梯度为 0。

**梯度分析**：
- 正样本对：梯度指向使嵌入更接近的方向
- 负样本对（$d_{ij} < m$）：梯度指向使嵌入更远离的方向，力度与 $(m - d_{ij})$ 成正比
- 负样本对（$d_{ij} \geq m$）：不产生梯度，已经足够远了

## 直观理解

Contrastive Loss 可以想象为一个"橡皮筋系统"：正样本对之间用橡皮筋连接，拉近它们；负样本对之间用弹簧连接，推开它们。但弹簧有一个"松弛长度"（margin）——当负样本距离超过这个长度后，弹簧松开，不再施加力。

Margin 参数是关键：它定义了"足够远"的概念。太小的 margin 会让负样本挤在一起（没有区分度）；太大的 margin 会让网络过于努力地推开所有负样本，可能导致嵌入空间的扭曲。

对比学习的一个重要直觉是：**好的嵌入空间应该使同类样本聚集、异类样本分离**。Contrastive Loss 直接优化这个目标，而不是像分类损失那样间接地通过决策边界来实现。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class ContrastiveLoss(nn.Module):
    """对比损失实现"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, h1, h2, y):
        # h1, h2: 嵌入向量 (batch_size, embed_dim)
        # y: 0 表示正样本对，1 表示负样本对
        distances = F.pairwise_distance(h1, h2)  # 欧氏距离
        pos_loss = 0.5 * (1 - y) * distances ** 2
        neg_loss = 0.5 * y * torch.clamp(self.margin - distances, min=0) ** 2
        return (pos_loss + neg_loss).mean()

# 简单的孪生网络
class SiameseNetwork(nn.Module):
    def __init__(self, input_dim=28*28, embed_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x):
        return F.normalize(self.encoder(x), dim=1)  # L2 归一化

# 演示
torch.manual_seed(42)
model = SiameseNetwork()
criterion = ContrastiveLoss(margin=2.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 生成正负样本对
def generate_pairs(n_pairs=100):
    # 假设有 5 个类别
    X = torch.randn(n_pairs, 2, 28*28)  # (pairs, 2, dim)
    labels = torch.randint(0, 5, (n_pairs, 2))
    y = (labels[:, 0] != labels[:, 1]).float()  # 0: 正样本, 1: 负样本
    return X[:, 0], X[:, 1], y

x1, x2, y = generate_pairs(200)

print("训练对比损失:")
for epoch in range(100):
    h1, h2 = model(x1), model(x2)
    loss = criterion(h1, h2, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 20 == 0:
        with torch.no_grad():
            pos_dist = F.pairwise_distance(h1[y == 0], h2[y == 0]).mean()
            neg_dist = F.pairwise_distance(h1[y == 1], h2[y == 1]).mean()
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}, "
              f"正样本距离: {pos_dist:.4f}, 负样本距离: {neg_dist:.4f}")

# 验证学习到的嵌入
with torch.no_grad():
    h1_test, h2_test = model(x1[:5]), model(x2[:5])
    print(f"\n正样本对嵌入距离: {F.pairwise_distance(h1_test[y[:5]==0], h2_test[y[:5]==0])}")
    print(f"负样本对嵌入距离: {F.pairwise_distance(h1_test[y[:5]==1], h2_test[y[:5]==1])}")
```

## 深度学习关联

- **人脸识别和验证**：Contrastive Loss 最早被成功应用于人脸验证（Face Verification），如 DeepFace、FaceNet 等系统。通过学习判别性的嵌入空间，可以验证两张人脸是否属于同一个人，即使从未见过该人的训练样本。
- **对比学习的复兴**：Contrastive Loss 的原理是现代自监督对比学习（SimCLR、MoCo、BYOL）的基础。这些方法通过构造正样本对（同一图像的不同增强）和负样本对（不同图像的增强），在没有标签的情况下学习强大的视觉表示。
- **与 Triplet Loss 的关系**：Contrastive Loss 处理成对的样本，而 Triplet Loss 处理三元组（锚点、正样本、负样本）。Triplet Loss 可以看作 Contrastive Loss 的扩展，它显式地优化"锚点与正样本的距离小于锚点与负样本的距离"，学习目标更直接。
