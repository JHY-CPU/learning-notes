# 5_GAT 图注意力网络

## 1. 动机：自适应邻居权重

GCN 使用固定的归一化系数 $1/\sqrt{d_i d_j}$，GraphSAGE 使用均值聚合——它们都**平等对待所有邻居**。但实际中，不同邻居的重要性应该不同。

GAT（Velickovic et al., 2018）引入注意力机制，让模型**自适应学习邻居权重**。

## 2. 注意力系数计算

### 2.1 基本公式

对于节点 $v$ 和其邻居 $u \in \mathcal{N}(v)$，注意力系数：

$$e_{vu} = \text{LeakyReLU}\left(\mathbf{a}^T [W h_v \| W h_u]\right)$$

其中：
- $W \in \mathbb{R}^{d' \times d}$ 是共享线性变换
- $\mathbf{a} \in \mathbb{R}^{2d'}$ 是注意力向量
- $\|$ 表示拼接

### 2.2 归一化（Softmax）

$$\alpha_{vu} = \frac{\exp(e_{vu})}{\sum_{k \in \mathcal{N}(v)} \exp(e_{vk})}$$

### 2.3 聚合

$$h_v' = \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu} W h_u\right)$$

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """从零实现 GAT 层"""
    def __init__(self, in_dim, out_dim, n_heads=1, dropout=0.6, alpha=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_dim = out_dim
        self.dropout = dropout

        # 线性变换 W ∈ R^{n_heads × out_dim × in_dim}
        self.W = nn.Linear(in_dim, out_dim * n_heads, bias=False)

        # 注意力向量 a ∈ R^{2 * out_dim}
        self.a = nn.Parameter(torch.zeros(n_heads, 2 * out_dim))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)

        self.leaky_relu = nn.LeakyReLU(alpha)

    def forward(self, x, edge_index):
        N = x.size(0)
        # 线性变换：[N, n_heads, out_dim]
        h = self.W(x).view(N, self.n_heads, self.out_dim)

        src, dst = edge_index  # 源节点和目标节点

        # 计算注意力 e_{ij}
        # h[dst]: [M, n_heads, out_dim], h[src]: [M, n_heads, out_dim]
        e_src = (h[src] * self.a[:, :self.out_dim]).sum(-1)  # [M, n_heads]
        e_dst = (h[dst] * self.a[:, self.out_dim:]).sum(-1)  # [M, n_heads]
        e = self.leaky_relu(e_src + e_dst)  # [M, n_heads]

        # Softmax 归一化（按目标节点分组）
        # 使用 scatter 实现
        e_max = self._scatter_max(e, dst, N)
        e = torch.exp(e - e_max[dst])
        e_sum = self._scatter_sum(e, dst, N)
        alpha = e / (e_sum[dst] + 1e-8)  # [M, n_heads]

        # 聚合
        alpha = F.dropout(alpha, self.dropout, training=self.training)
        # h[src]: [M, n_heads, out_dim]，按 dst 聚合
        out = torch.zeros(N, self.n_heads, self.out_dim, device=x.device)
        out.scatter_add_(0, dst.unsqueeze(-1).unsqueeze(-1).expand_as(h[src]),
                         alpha.unsqueeze(-1) * h[src])

        return out.mean(dim=1)  # 多头取平均

    def _scatter_max(self, src, index, dim_size):
        out = torch.full((dim_size, src.size(1)), float('-inf'), device=src.device)
        out.scatter_reduce_(0, index.unsqueeze(-1).expand_as(src), src, reduce='amax')
        return out

    def _scatter_sum(self, src, index, dim_size):
        out = torch.zeros(dim_size, src.size(1), device=src.device)
        out.scatter_add_(0, index.unsqueeze(-1).expand_as(src), src)
        return out
```

## 3. 多头注意力

与 Transformer 类似，GAT 使用多头注意力增加表达力：

$$h_v' = \Big\|_{k=1}^{K} \sigma\left(\sum_{u \in \mathcal{N}(v)} \alpha_{vu}^k W^k h_u\right)$$

- **中间层**：多头**拼接**（concat），输出维度 $= K \times d'$
- **输出层**：多头**平均**（average），输出维度 $= d'$

```python
# PyG 实现
from torch_geometric.nn import GATConv

class GATModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, heads=8):
        super().__init__()
        # 中间层：多头拼接
        self.conv1 = GATConv(in_dim, hidden_dim, heads=heads, dropout=0.6)
        # 输出层：多头平均
        self.conv2 = GATConv(hidden_dim * heads, out_dim, heads=1,
                            concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)
```

## 4. GAT 与 Transformer 的关系

| 特性 | GAT | Transformer Self-Attention |
|------|-----|---------------------------|
| 输入 | 图节点 | 序列 token |
| 注意力范围 | 仅邻居 $\mathcal{N}(v)$ | 全部位置 |
| 位置编码 | 无（图无序） | 有（序列位置） |
| 计算复杂度 | $O(\|E\| d)$ | $O(N^2 d)$ |
| 注意力计算 | LeakyReLU 激活 | 缩放点积 |

$$\text{GAT: } e_{vu} = \text{LeakyReLU}(\mathbf{a}^T [W h_v \| W h_u])$$

$$\text{Transformer: } e_{vu} = \frac{(W_Q h_v)^T (W_K h_u)}{\sqrt{d_k}}$$

GAT 可以看作**只在邻居上做注意力的 Transformer**。

## 5. 注意力系数的可视化

```python
import matplotlib.pyplot as plt
import networkx as nx

def visualize_attention(model, data):
    model.eval()
    with torch.no_grad():
        x, edge_index = data.x, data.edge_index
        # 获取注意力权重
        _, attention_weights = model.conv1(x, edge_index, return_attention_weights=True)
        edge_index, alpha = attention_weights

    G = nx.Graph()
    for i in range(edge_index.size(1)):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        G.add_edge(src, dst, weight=alpha[i, 0].item())

    pos = nx.spring_layout(G)
    edges = G.edges(data=True)
    weights = [e[2]['weight'] * 3 for e in edges]

    nx.draw(G, pos, with_labels=True, width=weights, edge_color=weights,
            edge_cmap=plt.cm.Blues)
    plt.title("GAT Attention Weights")
    plt.show()
```

## 6. GAT 的优势与局限

| 优势 | 局限 |
|------|------|
| 自适应权重，更灵活 | 计算量大于 GCN |
| 多头注意力增加稳定性 | 注意力不一定是"解释" |
| 适用于异质图 | 仍受限于 1-WL 表达力 |
| 可处理不同度数的节点 | 过平滑问题依然存在 |

## 7. GATv2 改进

Brody et al. (2022) 指出 GAT 的注意力是**静态的**（先算 $W h$ 再拼接），GATv2 改为**动态注意力**：

$$e_{vu} = \mathbf{a}^T \text{LeakyReLU}\left(W_1 h_v \| W_2 h_u\right)$$

将 LeakyReLU 移到注意力向量之前，使得注意力是输入依赖的。

```python
# PyG 支持 GATv2
from torch_geometric.nn import GATv2Conv

class GATv2Layer(nn.Module):
    def __init__(self, in_dim, out_dim, heads=4):
        super().__init__()
        self.conv = GATv2Conv(in_dim, out_dim, heads=heads)
```

## 8. 小结

- GAT 引入自注意力机制，为每条边学习不同权重
- 注意力系数：$\alpha_{vu} = \text{softmax}_u(\text{LeakyReLU}(\mathbf{a}^T [W h_v \| W h_u]))$
- 多头注意力：中间层拼接，输出层平均
- GAT 是图上的 Transformer，但只在邻居范围内计算注意力
- GATv2 改进为动态注意力，消除 GAT 的静态限制
