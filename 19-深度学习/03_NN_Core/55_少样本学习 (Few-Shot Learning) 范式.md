# 55_少样本学习 (Few-Shot Learning) 范式

## 核心概念

- **少样本学习定义**：少样本学习（Few-Shot Learning, FSL）旨在从极少量标注样本（通常每类 1-5 个）中学习识别新类别。典型的设置是 N-way K-shot 分类：有 $N$ 个新类别，每类只有 $K$ 个标注样本。
- **支撑集与查询集**：每个 episode（训练回合）包含一个支撑集（Support Set）$S$ 和一个查询集（Query Set）$Q$。支撑集提供少量标注样本，模型根据支撑集学习后对查询集进行分类。训练和测试都使用这种"episode"模式。
- **三种主要范式**：
  1. **度量学习**：学习嵌入空间，使同类样本距离近、异类样本距离远
  2. **元学习**：学习快速适应新任务（如 MAML）
  3. **数据增强**：通过数据合成扩充少量样本
- **N-way K-shot 设置**：N 是类别数，K 是每类支撑样本数。常见的设置是 5-way 1-shot（5 个类别，每类 1 个样本）和 5-way 5-shot（每类 5 个样本）。1-shot 是最具挑战性的场景。

## 数学推导

**度量学习方法——原型网络（Prototypical Networks）**：

对每个类别 $c$，计算原型向量（支撑集嵌入的均值）：

$$
\mathbf{p}_c = \frac{1}{|S_c|} \sum_{(\mathbf{x}_i, y_i) \in S_c} f_\theta(\mathbf{x}_i)
$$

对查询样本 $\mathbf{x}_q$，计算其与每个原型的距离，使用 Softmax 输出类别概率：

$$
p(y = c | \mathbf{x}_q) = \frac{\exp(-d(f_\theta(\mathbf{x}_q), \mathbf{p}_c))}{\sum_{c'} \exp(-d(f_\theta(\mathbf{x}_q), \mathbf{p}_{c'}))}
$$

其中 $d$ 通常是欧氏距离或余弦距离。

损失函数：查询集的交叉熵损失：

$$
L = -\frac{1}{|Q|} \sum_{(\mathbf{x}_q, y_q) \in Q} \log p(y = y_q | \mathbf{x}_q)
$$

**匹配网络（Matching Networks）**：

使用注意力机制，对每个查询样本，计算支撑集样本的加权投票：

$$
\hat{y}_q = \sum_{(\mathbf{x}_i, y_i) \in S} a(\mathbf{x}_q, \mathbf{x}_i) \cdot y_i
$$

其中注意力权重 $a(\cdot, \cdot)$ 基于嵌入的余弦相似度：

$$
a(\mathbf{x}_q, \mathbf{x}_i) = \frac{\exp(c(f_\theta(\mathbf{x}_q), g_\phi(\mathbf{x}_i)))}{\sum_j \exp(c(f_\theta(\mathbf{x}_q), g_\phi(\mathbf{x}_j)))}
$$

**关系网络（Relation Networks）**：

不是使用固定的距离度量，而是学习一个"关系模块"来比较：

$$
r_{ij} = g_\phi([f_\theta(\mathbf{x}_i), f_\theta(\mathbf{x}_j)])
$$

其中 $[\cdot]$ 表示拼接，$g_\phi$ 是一个小网络，输出关系得分。

## 直观理解

少样本学习可以类比为"看一眼就记住"——只给几张照片（支撑集），就要求能认出同类的其他照片（查询集）。人类擅长这种"少样本学习"，但传统的机器学习需要大量数据。

支撑集类似于"通缉令"——警察局有通缉犯的少量照片（支撑样本），警察需要在人群中找出他们（查询样本）。好的嵌入空间就像一份好的"面部特征描述"——即使只有一张照片，也能准确识别。

不同的少样本学习范式的类比：

- **度量学习**（原型网络）：学习一个"人脸识别系统"——计算出每类通缉犯的平均长相（原型），然后对新面孔计算与这些平均长相的相似度。
- **元学习**（MAML）：学习"如何快速记住新面孔"——大脑具有快速适应新面孔的能力，元学习训练这种"快速适应能力"。
- **数据增强**：给通缉犯照片加上各种滤镜和变换（数据增强），相当于在想象中补充了更多的角度和光照条件下的照片。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 原型网络 (Prototypical Networks) 实现
class ProtoNet(nn.Module):
    """少样本学习的原型网络"""
    def __init__(self, input_dim=64, hidden_dim=64, output_dim=64):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x):
        return self.encoder(x)

    def compute_prototypes(self, support_x, support_y, n_way):
        """计算每个类别的原型"""
        prototypes = []
        for c in range(n_way):
            class_mask = (support_y == c)
            class_embeddings = support_x[class_mask]
            prototype = class_embeddings.mean(0)
            prototypes.append(prototype)
        return torch.stack(prototypes)

    def predict(self, query_x, prototypes):
        """预测查询样本的类别"""
        # 计算距离
        dists = torch.cdist(query_x, prototypes)  # (n_query, n_way)
        # 距离转换为概率（使用负距离+Softmax）
        logits = -dists
        return F.softmax(logits, dim=1)

# 生成少样本分类任务
def sample_few_shot_task(n_way=5, k_shot=1, n_query=5, input_dim=64):
    """生成 N-way K-shot 分类任务"""
    # 随机生成类别中心
    centers = torch.randn(n_way, input_dim) * 2
    # 每类生成支撑样本和查询样本
    support_x, support_y = [], []
    query_x, query_y = [], []

    for c in range(n_way):
        # 支撑样本
        s_x = centers[c:c+1] + torch.randn(k_shot, input_dim) * 0.3
        s_y = torch.full((k_shot,), c)
        support_x.append(s_x)
        support_y.append(s_y)

        # 查询样本
        q_x = centers[c:c+1] + torch.randn(n_query, input_dim) * 0.3
        q_y = torch.full((n_query,), c)
        query_x.append(q_x)
        query_y.append(q_y)

    support_x = torch.cat(support_x)
    support_y = torch.cat(support_y)
    query_x = torch.cat(query_x)
    query_y = torch.cat(query_y)

    return support_x, support_y, query_x, query_y

# 训练原型网络
torch.manual_seed(42)

model = ProtoNet(64, 128, 64)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
n_way, k_shot, n_query = 5, 1, 5

print(f"训练原型网络 ({n_way}-way {k_shot}-shot):")
for episode in range(200):
    support_x, support_y, query_x, query_y = sample_few_shot_task(
        n_way, k_shot, n_query, 64)

    # 编码
    support_emb = model(support_x)
    query_emb = model(query_x)

    # 计算原型
    prototypes = model.compute_prototypes(support_emb, support_y, n_way)

    # 预测
    query_probs = model.predict(query_emb, prototypes)
    loss = F.cross_entropy(query_probs, query_y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if episode % 40 == 0:
        acc = (query_probs.argmax(1) == query_y).float().mean()
        print(f"  Episode {episode:3d}, Loss: {loss.item():.4f}, Acc: {acc.item():.4f}")

# 测试
print(f"\n测试 ({n_way}-way {k_shot}-shot):")
test_accs = []
for _ in range(50):
    support_x, support_y, query_x, query_y = sample_few_shot_task(
        n_way, k_shot, n_query, 64)
    support_emb = model(support_x)
    query_emb = model(query_x)
    prototypes = model.compute_prototypes(support_emb, support_y, n_way)
    query_probs = model.predict(query_emb, prototypes)
    acc = (query_probs.argmax(1) == query_y).float().mean()
    test_accs.append(acc.item())

print(f"  平均准确率: {np.mean(test_accs):.4f} ± {np.std(test_accs):.4f}")

# 不同 shot 数量对比
print("\n不同 K-shot 设置对比:")
for k in [1, 3, 5]:
    accs = []
    for _ in range(30):
        sx, sy, qx, qy = sample_few_shot_task(5, k, 5, 64)
        se, qe = model(sx), model(qx)
        proto = model.compute_prototypes(se, sy, 5)
        qp = model.predict(qe, proto)
        acc = (qp.argmax(1) == qy).float().mean()
        accs.append(acc.item())
    print(f"  {k}-shot: {np.mean(accs):.4f}")
```

## 深度学习关联

- **少样本图像分类**：少样本学习在图像分类领域最经典。miniImageNet、CUB-200、Omniglot 是标准的少样本学习基准。前沿方法（如 CrossTransformers、DeepEMD）在 5-way 1-shot 设置上已达到 70-80% 的准确率。
- **细粒度分类**：在细粒度分类（如不同品种的鸟类、车型识别）中，少样本学习特别有价值——因为标注细粒度类别的成本很高，每类通常只有少量标注样本。
- **大规模语言模型的少样本能力**：GPT-3 展示了大语言模型惊人的少样本学习能力。通过 in-context learning（在提示中提供几个示例），GPT-3 可以在不更新参数的情况下完成新任务。这与传统少样本学习（需要元训练或微调）有本质不同，引发了少样本学习研究的新方向。
