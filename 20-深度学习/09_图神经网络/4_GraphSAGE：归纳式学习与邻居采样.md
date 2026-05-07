# 4_GraphSAGE：归纳式学习与邻居采样

## 1. 动机：从转导式到归纳式

GCN 的核心问题：需要**完整的邻接矩阵**，新节点加入必须重新训练。

| 学习范式 | 代表模型 | 说明 |
|----------|----------|------|
| 转导式 (Transductive) | GCN | 训练时需要完整图结构，不能处理新节点 |
| 归纳式 (Inductive) | GraphSAGE | 学习聚合函数，可泛化到未见过的节点/图 |

GraphSAGE（Hamilton et al., 2017）的核心思想：**学习如何聚合邻居信息**，而非学习每个节点的嵌入。

## 2. GraphSAGE 算法

### 2.1 前向传播

对于节点 $v$，第 $k$ 层：

$$\mathbf{h}_{\mathcal{N}(v)}^k = \text{AGGREGATE}_k\left(\{\mathbf{h}_u^{k-1}, \forall u \in \mathcal{N}(v)\}\right)$$

$$\mathbf{h}_v^k = \sigma\left(W^k \cdot [\mathbf{h}_v^{k-1} \| \mathbf{h}_{\mathcal{N}(v)}^k]\right)$$

最后做 $L_2$ 归一化：$\mathbf{h}_v^k \leftarrow \mathbf{h}_v^k / \|\mathbf{h}_v^k\|_2$

**关键**：将自身特征与邻居聚合特征**拼接**，然后线性变换 + 激活。

### 2.2 邻居采样

为处理大规模图，GraphSAGE 固定每个节点采样 $S$ 个邻居：

```
第 k 层：采样 S_k 个邻居
第 k-1 层：采样 S_{k-1} 个邻居
...
总邻居数：S_k × S_{k-1} × ... × S_1
```

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader

# ========== 从零实现 ==========
class GraphSAGEFromScratch(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        # 拼接后维度翻倍：[自身 || 邻居]
        self.W1 = nn.Linear(2 * in_dim, hidden_dim)
        self.W2 = nn.Linear(2 * hidden_dim, out_dim)

    def aggregate(self, x, neighbors):
        """均值聚合"""
        # neighbors: list of neighbor indices per node
        agg = torch.zeros_like(x)
        for i, nbrs in enumerate(neighbors):
            if len(nbrs) > 0:
                agg[i] = x[nbrs].mean(dim=0)
        return agg

    def forward(self, x, neighbors):
        # Layer 1
        agg1 = self.aggregate(x, neighbors)
        h1 = F.relu(self.W1(torch.cat([x, agg1], dim=-1)))
        h1 = F.normalize(h1, p=2, dim=-1)

        # Layer 2
        agg2 = self.aggregate(h1, neighbors)
        h2 = self.W2(torch.cat([h1, agg2], dim=-1))
        return h2
```

## 3. 聚合器类型

### 3.1 均值聚合器（Mean Aggregator）

$$\text{AGG}_{\text{mean}} = \text{MEAN}\left(\{\mathbf{h}_u, \forall u \in \mathcal{N}(v)\}\right)$$

与 GCN 的归一化邻接矩阵等价。

```python
# 均值聚合
h_agg = x[neighbors].mean(dim=1)  # [N, hidden_dim]
```

### 3.2 LSTM 聚合器

将邻居的嵌入输入 LSTM，但邻居无序——**先随机打乱**：

$$\text{AGG}_{\text{LSTM}} = \text{LSTM}\left([\mathbf{h}_{u_1}, \dots, \mathbf{h}_{u_S}]\right)$$

```python
class LSTMAggregator(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.lstm = nn.LSTM(hidden_dim, hidden_dim, batch_first=True)

    def forward(self, neighbor_feats):
        # neighbor_feats: [N, S, hidden_dim]（邻居已随机打乱）
        _, (h_n, _) = self.lstm(neighbor_feats)
        return h_n.squeeze(0)  # [N, hidden_dim]
```

### 3.3 池化聚合器

先对每个邻居做 MLP，再最大池化：

$$\text{AGG}_{\text{pool}} = \max\left(\{\sigma(W_{\text{pool}} \mathbf{h}_u + b), \forall u \in \mathcal{N}(v)\}\right)$$

```python
class PoolAggregator(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU()
        )

    def forward(self, neighbor_feats):
        # neighbor_feats: [N, S, in_dim]
        h = self.mlp(neighbor_feats)  # [N, S, hidden_dim]
        return h.max(dim=1).values    # [N, hidden_dim]
```

## 4. Mini-Batch 训练

### 4.1 邻居采样算法

```
输入：节点集 B, 图 G=(V,E), 采样数 {S_1, ..., S_K}
输出：节点 B 的嵌入

for k = K to 1:
    for v in B:
        N_k(v) = 随机采样 S_k 个 v 的邻居
    B = B ∪ N_k(v)  # 将邻居加入计算图
```

```python
from torch_geometric.loader import NeighborLoader

# 每层采样 5 个邻居，2 层
loader = NeighborLoader(
    data,
    num_neighbors=[5, 5],  # 每层采样数
    batch_size=256,
    shuffle=True
)

for batch in loader:
    # batch 包含采样子图
    out = model(batch.x, batch.edge_index)
    loss = criterion(out[batch.train_mask], batch.y[batch.train_mask])
    loss.backward()
```

### 4.2 与全图训练对比

| 方面 | 全图训练 | Mini-Batch |
|------|----------|------------|
| 内存 | 需要加载整张图 | 只加载子图 |
| 速度 | 小图快 | 大图更快 |
| 批归一化 | 全图统计 | 子图统计 |
| 适用场景 | 小图 | 大规模图 (百万节点) |

## 5. 使用 PyG 实现

```python
from torch_geometric.nn import SAGEConv
from torch_geometric.datasets import Reddit
import torch.nn.functional as F

class GraphSAGEModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = SAGEConv(in_dim, hidden_dim)
        self.conv2 = SAGEConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Reddit 数据集（大规模图）
dataset = Reddit(root='/tmp/Reddit')
data = dataset[0]
print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")

model = GraphSAGEModel(dataset.num_node_features, 128, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# NeighborLoader 实现 mini-batch
loader = NeighborLoader(data, num_neighbors=[25, 10], batch_size=1024)

for epoch in range(10):
    model.train()
    total_loss = 0
    for batch in loader:
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        loss = F.nll_loss(out[:batch.batch_size], batch.y[:batch.batch_size])
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    print(f"Epoch {epoch}: Loss={total_loss:.4f}")
```

## 6. GraphSAGE vs GCN 对比

| 特性 | GCN | GraphSAGE |
|------|-----|-----------|
| 学习范式 | 转导式 | 归纳式 |
| 聚合方式 | 固定归一化 | 可学习（均值/LSTM/池化） |
| 拼接自身 | 隐式（自环） | 显式拼接 $[h_v \| h_N]$ |
| 大规模图 | 需要全图 | 邻居采样支持 |
| 归一化 | 内置 | $L_2$ 归一化 |
| 训练方式 | 全图 | 支持 mini-batch |

## 7. 小结

- GraphSAGE 是归纳式 GNN，学习聚合函数而非节点嵌入
- 三种聚合器：均值（最简单）、LSTM（序列建模）、池化（自适应）
- 邻居采样使 mini-batch 训练成为可能，适用于大规模图
- 核心公式：$h_v^k = \sigma(W^k [h_v^{k-1} \| \text{AGG}(\{h_u^{k-1}\})])$
