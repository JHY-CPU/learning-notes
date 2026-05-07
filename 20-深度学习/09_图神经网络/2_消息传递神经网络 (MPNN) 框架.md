# 2_消息传递神经网络 (MPNN) 框架

## 1. MPNN 统一框架

Gilmer et al. (2017) 提出的**消息传递神经网络（Message Passing Neural Network）**为各类 GNN 提供了统一视角。整个过程分为三步：

$$\mathbf{m}_v^{(l)} = \text{MSG}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \mathbf{h}_u^{(l-1)}, \mathbf{e}_{uv}\right), \quad u \in \mathcal{N}(v)$$

$$\mathbf{h}_v^{(l)} = \text{UPDATE}^{(l)}\left(\mathbf{h}_v^{(l-1)}, \text{AGG}^{(l)}\left(\{\mathbf{m}_v^{(l)} : u \in \mathcal{N}(v)\}\right)\right)$$

### 三步范式

| 步骤 | 名称 | 作用 | 要求 |
|------|------|------|------|
| Message | 消息计算 | 为每条边计算消息 | $m_{uv} = \text{MSG}(h_v, h_u, e_{uv})$ |
| Aggregate | 消息聚合 | 将邻居消息汇总 | 排列不变，处理变长集合 |
| Update | 节点更新 | 结合自身和聚合信息 | 产生新的节点表示 |

```python
import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MPNNLayer(MessagePassing):
    """通用 MPNN 层"""
    def __init__(self, in_dim, out_dim, edge_dim=None):
        super().__init__(aggr='add')  # 聚合方式：add/mean/max
        self.msg_nn = nn.Sequential(
            nn.Linear(2 * in_dim + (edge_dim or 0), out_dim),
            nn.ReLU()
        )
        self.update_nn = nn.Sequential(
            nn.Linear(in_dim + out_dim, out_dim),
            nn.ReLU()
        )

    def forward(self, x, edge_index, edge_attr=None):
        # x: [N, in_dim], edge_index: [2, M]
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)

    def message(self, x_i, x_j, edge_attr=None):
        # x_i: 目标节点特征, x_j: 源节点特征
        # 构建消息 m_{ij} = MSG(x_i, x_j, e_{ij})
        if edge_attr is not None:
            return self.msg_nn(torch.cat([x_i, x_j, edge_attr], dim=-1))
        return self.msg_nn(torch.cat([x_i, x_j], dim=-1))

    def update(self, aggr_out, x):
        # aggr_out: 聚合后的消息, x: 原始节点特征
        return self.update_nn(torch.cat([x, aggr_out], dim=-1))
```

## 2. 消息函数（Message Function）

消息函数定义每条边上传递什么信息。

### 常见消息函数

```python
# 1. 简单加法消息：m_{ij} = h_j
def msg_add(x_i, x_j, edge_attr=None):
    return x_j

# 2. 拼接消息：m_{ij} = MLP([h_i || h_j || e_{ij}])
def msg_concat(x_i, x_j, edge_attr=None):
    return MLP(torch.cat([x_i, x_j, edge_attr], dim=-1))

# 3. 边权重消息：m_{ij} = w_{ij} · W · h_j
def msg_weighted(x_i, x_j, edge_attr):
    # edge_attr: [M, 1] 权重
    return edge_attr * (W @ x_j)

# 4. 距离消息：m_{ij} = MLP(h_j) · exp(-||pos_i - pos_j||^2 / sigma^2)
def msg_distance(x_i, x_j, pos_i, pos_j):
    dist = torch.norm(pos_i - pos_j, dim=-1, keepdim=True)
    return MLP(x_j) * torch.exp(-dist**2 / sigma**2)
```

## 3. 聚合函数（Aggregate Function）

聚合函数必须是**排列不变**的，因为邻居没有固定顺序。

| 聚合函数 | 公式 | 优点 | 缺点 |
|----------|------|------|------|
| 求和 (sum) | $\sum_{u \in \mathcal{N}(v)} m_{u}$ | 保留度信息 | 受节点度影响大 |
| 均值 (mean) | $\frac{1}{|\mathcal{N}(v)|} \sum m_{u}$ | 度归一化 | 忽略度差异 |
| 最大值 (max) | $\max_{u \in \mathcal{N}(v)} m_{u}$ | 捕捉最显著特征 | 丢失分布信息 |
| 注意力 | $\sum \alpha_{vu} m_{u}$ | 自适应权重 | 计算量大 |

```python
# 聚合函数的排列不变性证明
# 无论邻居以什么顺序排列，结果相同
neighbors = torch.tensor([[1, 0, 2], [0, 2, 1], [2, 1, 0]])
for perm in neighbors:
    result = x[perm].sum(dim=0)  # sum 保证排列不变
    print(result)  # 所有排列的求和结果相同
```

## 4. 更新函数（Update Function）

```python
# 1. GRU 更新（GGNN 风格）
# h_v = GRU(h_v, m_v)
class GRUUpdate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gru = nn.GRUCell(hidden_dim, hidden_dim)

    def forward(self, h, m):
        return self.gru(m, h)

# 2. 残差更新
# h_v = h_v + MLP(m_v)
class ResidualUpdate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )

    def forward(self, h, m):
        return h + self.mlp(m)

# 3. 门控更新
# h_v = σ(W_g [h_v || m_v]) ⊙ tanh(W_h [h_v || m_v])
class GatedUpdate(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.gate = nn.Linear(2 * hidden_dim, hidden_dim)
        self.transform = nn.Linear(2 * hidden_dim, hidden_dim)

    def forward(self, h, m):
        cat = torch.cat([h, m], dim=-1)
        g = torch.sigmoid(self.gate(cat))
        h_new = torch.tanh(self.transform(cat))
        return g * h_new + (1 - g) * h
```

## 5. 完整 MPNN 实现

```python
class MPNN(nn.Module):
    """完整的 MPNN 模型"""
    def __init__(self, node_dim, edge_dim, hidden_dim, num_layers, num_classes):
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden_dim)

        self.layers = nn.ModuleList([
            MPNNLayer(hidden_dim, hidden_dim, edge_dim)
            for _ in range(num_layers)
        ])

        self.readout = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_classes)
        )

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        x = self.input_proj(x)

        for layer in self.layers:
            x = layer(x, edge_index, edge_attr)
            x = F.relu(x)

        # 图级别读出：全局平均池化
        if batch is not None:
            from torch_geometric.nn import global_mean_pool
            x = global_mean_pool(x, batch)

        return self.readout(x)
```

## 6. 经典 GNN 在 MPNN 框架下的映射

| 模型 | Message | Aggregate | Update |
|------|---------|-----------|--------|
| GCN | $\frac{1}{\sqrt{d_i d_j}} W h_j$ | Sum | 无额外更新 |
| GraphSAGE | $W \cdot [h_v \| h_u]$ | Mean/Max/LSTM | 无额外更新 |
| GAT | $\alpha_{vu} W h_v$ | Sum | 无额外更新 |
| GGNN | $A_{uv} W h_u$ | Sum | GRU |
| MPNN | $f(h_v, h_u, e_{uv})$ | Sum | $g(h_v, m_v)$ |
| GIN | $h_u$ | Sum | MLP$( (1+\epsilon)h_v + \sum h_u )$ |

## 7. 过平滑与 MPNN 的表达力限制

当 MPNN 层数过多时，所有节点的表示趋于相同（过平滑），这是因为多次聚合后，每个节点的表示覆盖了越来越大的邻域，最终接近全图平均。

**MPNN 的表达力上限**：MPNN 最多与 1-WL 测试等价，存在无法区分的图（如正则图）。

```python
# 过平滑示例
x_layers = [x]
for layer in gnn_layers:
    x = layer(x, edge_index)
    x_layers.append(x)

# 计算节点表示的方差（衡量多样性）
for i, h in enumerate(x_layers):
    print(f"Layer {i}: var = {h.var(dim=0).mean():.4f}")
# 输出：var 逐层减小，趋近于 0
```

## 8. 小结

- MPNN 将 GNN 统一为 Message-Aggregate-Update 三步范式
- 聚合函数必须排列不变，常用 sum/mean/max/attention
- 不同 GNN 模型本质上是消息函数、聚合函数、更新函数的不同选择
- MPNN 的表达力受限于 1-WL 测试，深层存在过平滑问题
