# 9_GNN 过平滑问题 (Over-smoothing)

## 1. 什么是过平滑

随着 GNN 层数增加，不同节点的表示趋于相同——这就是**过平滑（Over-smoothing）**。

与 CNN 不同，深层 GNN 的性能反而**退化**：

| 层数 | GCN 在 Cora 上的测试准确率 |
|------|---------------------------|
| 2 层 | ~81% |
| 4 层 | ~76% |
| 8 层 | ~60%（接近随机） |
| 16 层 | ~30% |

### 1.1 数学直觉

经过 $K$ 层消息传递后，节点 $v$ 的感受野包含所有 $K$-hop 邻居。在连通图上，当 $K$ 足够大时，每个节点的表示混合了全图所有节点的信息：

$$h_v^{(K)} \approx \frac{1}{|V|} \sum_{u \in V} h_u^{(0)}, \quad \forall v$$

所有节点趋同，丧失区分性。

### 1.2 数学分析：随机游走的平稳分布

GCN 的聚合等价于归一化邻接矩阵的幂次：

$$H^{(K)} = (\hat{D}^{-1/2}\hat{A}\hat{D}^{-1/2})^K H^{(0)} W^{(1)} \cdots W^{(K)}$$

当 $K \rightarrow \infty$，$\hat{A}_{\text{norm}}^K$ 收敛到平稳分布，所有行趋于相同。

## 2. 残差连接 (Residual Connection)

借鉴 ResNet 的思想，将浅层特征直接加到深层：

$$h_v^{(l+1)} = \text{GNN}^{(l)}(h_v^{(l)}, \mathcal{N}(v)) + h_v^{(l)}$$

或者使用跳跃连接到初始特征：

$$h_v^{(L)} = \text{GNN}^{(L)}(\cdots) + \alpha \cdot h_v^{(0)}$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv

class ResGCN(nn.Module):
    """带残差连接的深层 GCN"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=8):
        super().__init__()
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))
        for conv, bn in zip(self.convs, self.bns):
            h_new = F.relu(bn(conv(h, edge_index)))
            h = h_new + h  # 残差连接
        return F.log_softmax(self.output_proj(h), dim=1)
```

## 3. DropEdge

Rong et al. (2020) 提出在每轮训练中随机删除一定比例的边：

$$\hat{A}' = \text{DropEdge}(\hat{A}, p)$$

**效果**：
- 减缓过平滑：降低每层的信息混合程度
- 数据增强：不同的边子集提供正则化
- 类似 Dropout 的效果

```python
from torch_geometric.utils import dropout_edge

class DropEdgeGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=8, drop_rate=0.3):
        super().__init__()
        self.drop_rate = drop_rate
        self.input_proj = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = F.relu(self.input_proj(x))

        # 训练时随机丢弃边
        if self.training:
            edge_index, _ = dropout_edge(edge_index, p=self.drop_rate)

        for conv in self.convs:
            h = F.relu(conv(h, edge_index)) + h
        return self.output_proj(h)
```

## 4. JK-Net (Jumping Knowledge Network)

Xu et al. (2018) 提出聚合**所有层**的表示，让模型自适应选择合适的感受野：

$$h_v = \text{AGG}\left(h_v^{(1)}, h_v^{(2)}, \dots, h_v^{(L)}\right)$$

聚合方式：
- **Concat**：拼接所有层
- **Max**：逐元素取最大
- **LSTM**：用 LSTM 加权
- **Attention**：注意力加权

```python
from torch_geometric.nn import GCNConv

class JKNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=6, mode='cat'):
        super().__init__()
        self.mode = mode
        self.convs = nn.ModuleList()
        self.bns = nn.ModuleList()
        self.convs.append(GCNConv(in_dim, hidden_dim))
        self.bns.append(nn.BatchNorm1d(hidden_dim))
        for _ in range(num_layers - 1):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
            self.bns.append(nn.BatchNorm1d(hidden_dim))

        if mode == 'cat':
            self.lin = nn.Linear(hidden_dim * num_layers, out_dim)
        else:
            self.lin = nn.Linear(hidden_dim, out_dim)

        # Attention 模式
        if mode == 'attn':
            self.attn = nn.Linear(hidden_dim, 1)

    def forward(self, x, edge_index):
        layer_outputs = []
        h = x
        for conv, bn in zip(self.convs, self.bns):
            h = F.relu(bn(conv(h, edge_index)))
            layer_outputs.append(h)

        if self.mode == 'cat':
            h_jump = torch.cat(layer_outputs, dim=-1)
        elif self.mode == 'max':
            h_jump = torch.stack(layer_outputs, dim=0).max(dim=0).values
        elif self.mode == 'attn':
            scores = torch.stack([self.attn(h) for h in layer_outputs], dim=0)
            weights = F.softmax(scores, dim=0)
            stacked = torch.stack(layer_outputs, dim=0)
            h_jump = (weights * stacked).sum(dim=0)

        return self.lin(h_jump)
```

## 5. 其他深层 GNN 方法

### 5.1 PairNorm

Zhao & Akoglu (2020)：对每层输出做归一化，防止表示塌缩：

$$\tilde{h}_i = s \cdot \frac{h_i - \bar{h}}{\sqrt{\sum_j \|h_j - \bar{h}\|^2}}$$

```python
class PairNorm(nn.Module):
    def __init__(self, scale=1.0):
        super().__init__()
        self.scale = scale

    def forward(self, x):
        mean = x.mean(dim=0, keepdim=True)
        x = x - mean
        row_norm = x.norm(dim=1, keepdim=True)
        x = self.scale * x / (row_norm.mean() + 1e-8)
        return x
```

### 5.2 NodeDrop (DropNode)

类似于 Dropout，但直接丢弃整个节点（将其特征置零），强制模型不过度依赖某些节点。

### 5.3 常见策略对比

| 方法 | 核心思想 | 效果 | 额外参数 |
|------|----------|------|----------|
| 残差连接 | 跳跃连接保留浅层信息 | 好 | 无 |
| DropEdge | 随机丢弃边 | 好 | 无 |
| JK-Net | 聚合所有层表示 | 很好 | 少量 |
| PairNorm | 节点表示归一化 | 中等 | 无 |
| 深度初始 (DeeperGCN) | 残差 + 归一化 + 激活设计 | 很好 | 少量 |

## 6. DeeperGCN 的实践经验

Li et al. (2020) 总结了训练深层 GNN 的最佳实践：

```python
from torch_geometric.nn import GENConv, LayerNorm

class DeeperGCN(nn.Module):
    """56 层 GNN 也能训练"""
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=28):
        super().__init__()
        self.node_encoder = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()
        for _ in range(num_layers):
            self.convs.append(GENConv(hidden_dim, hidden_dim, aggr='softmax'))
            self.norms.append(LayerNorm(hidden_dim))
        self.out_lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        h = self.node_encoder(x)
        for conv, norm in zip(self.convs, self.norms):
            h = h + F.relu(norm(conv(h, edge_index)))  # pre-norm 残差
        return self.out_lin(h)
```

**关键要素**：
1. **Pre-norm 残差**：先归一化再激活（比 post-norm 更稳定）
2. **Softmax 聚合**：自适应加权邻居
3. **LayerNorm**：比 BatchNorm 更适合图数据

## 7. 小结

- 过平滑：深层 GNN 中所有节点表示趋于相同，性能急剧下降
- 残差连接是最基本的解决方案，保留浅层信息
- DropEdge 通过随机丢弃边进行正则化，有效缓解过平滑
- JK-Net 聚合所有层表示，自适应选择感受野
- 训练深层 GNN 需要综合运用多种技术：残差 + 归一化 + 适当的聚合
