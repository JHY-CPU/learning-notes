# 6_图池化方法 (Graph Pooling)

## 1. 图池化的目的

在 CNN 中，池化层逐步减小空间分辨率，提取高层语义。类似地，图池化将节点表示聚合成图级别的表示，或生成粗化的图。

**两种用途**：
1. **读出（Readout）**：生成图级别表示，用于图分类/回归
2. **层次池化**：逐步粗化图结构，保留层次信息

## 2. 全局池化方法

最简单的方式：对所有节点做对称聚合。

### 2.1 全局平均/最大/求和池化

$$h_G = \frac{1}{|V|} \sum_{v \in V} h_v \quad \text{(mean)}$$

$$h_G = \max_{v \in V} h_v \quad \text{(max)}$$

$$h_G = \sum_{v \in V} h_v \quad \text{(sum)}$$

```python
import torch
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool

# x: [N, d] 节点特征
# batch: [N] 每个节点属于哪个图
h_mean = global_mean_pool(x, batch)  # [B, d]
h_max = global_max_pool(x, batch)    # [B, d]
h_sum = global_add_pool(x, batch)    # [B, d]

# 拼接多种池化
h_graph = torch.cat([h_mean, h_max, h_sum], dim=-1)  # [B, 3d]
```

### 2.2 Set2Set 注意力读出

Meringer et al. 提出的基于注意力的读出：

$$q_t = \text{LSTM}(q_{t-1}^*, r_{t-1})$$

$$e_{v,t} = \text{attn}(q_t, h_v)$$

$$r_t = \sum_v \text{softmax}(e_{v,t}) \cdot h_v$$

$$q_t^* = [q_t \| r_t]$$

```python
from torch_geometric.nn import Set2Set

set2set = Set2Set(in_channels=d, processing_steps=3)
h_graph = set2set(x, batch)  # [B, 2d]
```

## 3. 层次池化方法

### 3.1 TopKPool（自注意力池化）

节点打分，选择 top-k 个节点：

$$y = \frac{X p}{\|p\|}, \quad \text{idx} = \text{topk}(y, k)$$

$$X' = X_{\text{idx}} \odot \tanh(y_{\text{idx}})$$

```python
from torch_geometric.nn import TopKPooling

class TopKModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, ratio=0.5):
        super().__init__()
        self.pool1 = TopKPooling(hidden_dim, ratio=ratio)
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.conv3 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = TopKPooling(hidden_dim, ratio=ratio)
        self.pool3 = TopKPooling(hidden_dim, ratio=ratio)
        self.lin = nn.Linear(2 * hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)

        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)

        x = F.relu(self.conv3(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)

        x = torch.cat([global_mean_pool(x, batch),
                       global_max_pool(x, batch)], dim=-1)
        return self.lin(x)
```

### 3.2 SAGPool（自注意力图池化）

Lee et al. (2019)：用 GAT 计算注意力分数进行池化：

$$Z = \text{GAT}(X, A), \quad \text{idx} = \text{topk}(Z, k)$$

$$A' = A_{\text{idx}, \text{idx}}, \quad X' = X_{\text{idx}} \odot \sigma(Z_{\text{idx}})$$

```python
from torch_geometric.nn import SAGPooling

class SAGPoolModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.pool1 = SAGPooling(hidden_dim, ratio=0.8)
        self.conv2 = GCNConv(hidden_dim, hidden_dim)
        self.pool2 = SAGPooling(hidden_dim, ratio=0.8)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x = F.relu(self.conv2(x, edge_index))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x = global_mean_pool(x, batch)
        return self.lin(x)
```

### 3.3 DiffPool（可微分池化）

Ying et al. (2018)：**学习**图的粗化方式，而非硬选择。

为每一层学习一个软分配矩阵 $S^{(l)}$：

$$S^{(l)} = \text{softmax}\left(\text{GNN}_{l,\text{embed}}(A^{(l)}, H^{(l)})\right)$$

$$X^{(l+1)} = S^{(l)T} H^{(l)}, \quad A^{(l+1)} = S^{(l)T} A^{(l)} S^{(l)}$$

```python
from torch_geometric.nn import DenseGCNConv, dense_diff_pool

class DiffPoolModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_nodes_pool):
        super().__init__()
        self.conv1 = DenseGCNConv(in_dim, hidden_dim)
        self.conv2 = DenseGCNConv(hidden_dim, hidden_dim)
        # 分配矩阵的 GNN
        self.pool1 = DenseGCNConv(in_dim, num_nodes_pool)
        self.lin = nn.Linear(hidden_dim, out_dim)

    def forward(self, x, adj, mask=None):
        # 第一层：计算嵌入和分配矩阵
        s = F.softmax(self.pool1(x, adj, mask), dim=-1)
        x = F.relu(self.conv1(x, adj, mask))

        # DiffPool：粗化图
        x, adj, l1, e1 = dense_diff_pool(x, adj, s, mask)

        # 第二层
        x = F.relu(self.conv2(x, adj))
        x = x.mean(dim=1)
        return self.lin(x)
```

## 4. 池化方法对比

| 方法 | 类型 | 可学习 | 参数量 | 层次信息 |
|------|------|--------|--------|----------|
| Global Mean/Max/Sum | 全局读出 | 否 | 0 | 无 |
| Set2Set | 注意力读出 | 是 | 中等 | 无 |
| TopKPool | 层次，硬选择 | 是 | 少 | 有 |
| SAGPool | 层次，自注意力 | 是 | 少 | 有 |
| DiffPool | 层次，软分配 | 是 | 大 | 有 |

## 5. 池化损失辅助训练

DiffPool 使用额外损失约束分配矩阵：

$$\mathcal{L}_{\text{pool}} = \|A^{(l)} - S^{(l)} S^{(l)T}\|_F + \frac{1}{N} \sum_i H(S_i)$$

- **链路预测损失**：鼓励同簇节点相连
- **熵损失**：鼓励分配矩阵更确定（非均匀分布）

## 6. 小结

- 全局池化简单有效，适合大多数图分类任务
- 层次池化保留图的层次结构，信息更丰富
- DiffPool 学习软分配，表达力最强但计算开销大
- TopKPool 和 SAGPool 是高效的层次池化方法
- 实践中，全局池化（mean+max 拼接）是很好的起点
