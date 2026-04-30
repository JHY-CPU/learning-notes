# 15 Triplet Loss 与锚点选择策略

## 核心概念

- **Triplet Loss 定义**：Triplet Loss 使用三元组（锚点 $a$、正样本 $p$、负样本 $n$）进行训练。目标是将锚点与正样本在嵌入空间中拉近，同时将锚点与负样本推远。损失函数定义为 $L = \max(d(a,p) - d(a,n) + \text{margin}, 0)$。

- **相对相似性**：与 Contrastive Loss 只考虑绝对距离不同，Triplet Loss 优化的是相对距离——它关心的是"正样本是否比负样本更接近锚点"。这种相对约束使得嵌入空间更具判别性。

- **锚点选择策略**：锚点的选择对训练至关重要。随机选择的三元组中，大多数已经是"简单"样本（$d(a,p) + \text{margin} < d(a,n)$），不会产生有效梯度。因此需要精心选择"困难"三元组。

- **困难样本挖掘**：有三种策略——Easy（损失为 0，无梯度）、Semi-hard（$d(a,p) < d(a,n) < d(a,p) + \text{margin}$）、Hard（$d(a,n) < d(a,p)$，即负样本比正样本还近）。FaceNet 使用 semi-hard 挖掘取得了最佳效果。

## 数学推导

**Triplet Loss 定义**：

$$
L = \sum_{i=1}^{N} \max\left( \|f(x_i^a) - f(x_i^p)\|_2^2 - \|f(x_i^a) - f(x_i^n)\|_2^2 + \alpha, 0 \right)
$$

其中 $f$ 是嵌入函数，$\alpha$ 是 margin。

写成更紧凑的形式：

$$
L = \max(D_{ap} - D_{an} + \alpha, 0)
$$

其中 $D_{ap} = \|f(x^a) - f(x^p)\|_2^2$，$D_{an} = \|f(x^a) - f(x^n)\|_2^2$。

**梯度推导**（当 $D_{ap} - D_{an} + \alpha > 0$ 时）：

对锚点 $f(x^a)$：

$$
\frac{\partial L}{\partial f(x^a)} = 2(f(x^a) - f(x^p)) - 2(f(x^a) - f(x^n)) = 2(f(x^n) - f(x^p))
$$

对正样本 $f(x^p)$：

$$
\frac{\partial L}{\partial f(x^p)} = -2(f(x^a) - f(x^p)) = 2(f(x^p) - f(x^a))
$$

对负样本 $f(x^n)$：

$$
\frac{\partial L}{\partial f(x^n)} = 2(f(x^a) - f(x^n))
$$

**三类三元组的分析**：

设 $D_{ap} = \|a - p\|_2^2$，$D_{an} = \|a - n\|_2^2$：

1. **Easy triplets**：$D_{an} \geq D_{ap} + \alpha$，损失为 0，不参与训练。
2. **Semi-hard triplets**：$D_{ap} < D_{an} < D_{ap} + \alpha$，负样本不比正样本近，但也不够远。
3. **Hard triplets**：$D_{an} \leq D_{ap}$，负样本比正样本离锚点更近，是训练的重点。

## 直观理解

Triplet Loss 可以理解为"排序学习"——它要求嵌入空间中"正样本排在负样本前面"，具体来说就是正样本的距离至少比负样本近一个 margin。这类似于搜索引擎中相关结果应该排在无关结果前面的思想。

选择策略的类比：
- **随机选择**：就像随机从全校学生中挑选学生进行排名对比，大部分比较结果显而易见（大学生 vs 小学生），没有学习价值。
- **困难样本挖掘**：就像挑选水平相近的学生进行竞争——势均力敌的对比才能促进进步。
- **Semi-hard 挖掘**：选择"略有挑战但不过分"的样本——太难（Hard）的样本可能导致训练不稳定，太简单的样本没有学习效果。

Margin 参数 $\alpha$ 决定了嵌入空间的"安全距离"——即正样本和负样本应该被分开的最小距离。$\alpha$ 越大，类间的间隔越大，但训练也越困难。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class TripletLoss(nn.Module):
    """Triplet Loss with semi-hard mining"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative):
        d_ap = (anchor - positive).pow(2).sum(1)  # 锚点-正样本距离
        d_an = (anchor - negative).pow(2).sum(1)  # 锚点-负样本距离
        losses = torch.clamp(d_ap - d_an + self.margin, min=0)
        # 只对损失大于 0 的三元组求平均
        return losses.mean()

class TripletMiningLoss(nn.Module):
    """在线困难样本挖掘的 Triplet Loss"""
    def __init__(self, margin=1.0):
        super().__init__()
        self.margin = margin

    def forward(self, embeddings, labels):
        # embeddings: (batch_size, embed_dim)
        # labels: (batch_size,)
        batch_size = embeddings.size(0)

        # 计算所有样本对的距离矩阵
        dist_matrix = torch.cdist(embeddings, embeddings)  # (B, B)

        # 找每个锚点的最远正样本和最近负样本（batch-hard mining）
        triplets = []
        for i in range(batch_size):
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=labels.device) != i)
            neg_mask = labels != labels[i]

            if pos_mask.sum() == 0 or neg_mask.sum() == 0:
                continue

            # 最远的正样本（hardest positive）
            hardest_pos = dist_matrix[i][pos_mask].max()
            # 最近的负样本（hardest negative）
            hardest_neg = dist_matrix[i][neg_mask].min()
            triplets.append(hardest_pos - hardest_neg + self.margin)

        if not triplets:
            return torch.tensor(0.0, requires_grad=True)

        losses = torch.stack(triplets)
        return torch.clamp(losses, min=0).mean()

# 演示
torch.manual_seed(42)
embed_dim = 64
model = nn.Sequential(
    nn.Linear(10, embed_dim),
)
criterion = TripletLoss(margin=1.0)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# 生成数据：4 个类别
batch_size = 16
x = torch.randn(batch_size, 10)
labels = torch.randint(0, 4, (batch_size,))

# 随机选择三元组
def get_random_triplets(x, labels):
    batch = x.size(0)
    a_idx, p_idx, n_idx = [], [], []
    for i in range(batch):
        pos = torch.where(labels == labels[i])[0]
        pos = pos[pos != i]
        neg = torch.where(labels != labels[i])[0]
        if len(pos) > 0 and len(neg) > 0:
            a_idx.append(i)
            p_idx.append(pos[torch.randint(0, len(pos), (1,))].item())
            n_idx.append(neg[torch.randint(0, len(neg), (1,))].item())
    return x[a_idx], x[p_idx], x[n_idx]

print("训练 Triplet Loss:")
for epoch in range(100):
    anchor, positive, negative = get_random_triplets(x, labels)
    emb_a, emb_p, emb_n = model(anchor), model(positive), model(negative)
    loss = criterion(emb_a, emb_p, emb_n)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 20 == 0:
        print(f"Epoch {epoch:3d}, Loss: {loss.item():.4f}")
```

## 深度学习关联

- **FaceNet 人脸识别**：Triplet Loss 最著名的应用是 FaceNet。通过在大规模人脸数据集上训练 Triplet Loss，FaceNet 学习到了一个嵌入空间，其中相同身份的人脸距离近、不同身份的人脸距离远，实现了高精度的人脸验证和识别。

- **度量学习的核心损失**：Triplet Loss 是度量学习（Metric Learning）领域的基础损失函数。在细粒度分类（如车型识别、鸟类识别）、行人重识别（Person ReID）等任务中，Triplet Loss 通常比分类损失表现更好。

- **对比学习的演进**：现代对比学习方法（如 SimCLR）在思想上是 Triplet Loss 的扩展。SimCLR 使用 NT-Xent 损失（Normalized Temperature-Scaled Cross-Entropy），本质上是将 Triplet Loss 推广到了多负样本场景——每个锚点与一个正样本和大量负样本进行对比。
