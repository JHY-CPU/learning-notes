# 13_图 Transformer

## 1. 从 GAT 到图 Transformer

GAT 限制注意力在局部邻居范围内，而 Transformer 的全局注意力可以捕获**长距离依赖**。挑战在于：图没有自然的序列顺序，且全局注意力的复杂度 $O(N^2)$ 对大图不可行。

**图 Transformer 的核心问题**：
1. 如何编码图的结构信息？
2. 如何降低注意力计算复杂度？

## 2. Graphormer

Ying et al. (2021)：微软提出的图 Transformer，在分子属性预测上取得突破。

### 2.1 结构编码

Graphormer 引入三种结构编码：

**中心性编码（Centrality Encoding）**：编码节点的度数信息：

$$h_v^{(0)} = x_v + z_{\deg^-(v)} + z_{\deg^+(v)}$$

其中 $z$ 是可学习的嵌入向量。

**空间编码（Spatial Encoding）**：编码节点间的最短路径距离：

$$b_{vu} = \phi(d(v, u))$$

其中 $d(v, u)$ 是 $v$ 和 $u$ 之间的最短路径长度，$\phi$ 是可学习的嵌入。

注意力矩阵中加入空间编码：

$$A_{vu} = \frac{q_v^T k_u}{\sqrt{d}} + b_{vu}$$

**边编码（Edge Encoding）**：对边特征的编码。

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GraphormerLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, max_path=5):
        super().__init__()
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
        )
        self.norm1 = nn.LayerNorm(embed_dim)
        self.norm2 = nn.LayerNorm(embed_dim)

        # 空间编码
        self.spatial_enc = nn.Embedding(max_path + 1, num_heads)

    def forward(self, x, dist):
        # x: [B, N, d], dist: [B, N, N] 最短路径距离矩阵
        # 空间编码：作为注意力偏置
        spatial_bias = self.spatial_enc(dist.clamp(0, 50))  # [B, N, N, n_heads]
        spatial_bias = spatial_bias.permute(0, 3, 1, 2)  # [B, n_heads, N, N]

        # MultiheadAttention 不直接支持 bias，需要修改
        # 简化版：直接加到注意力分数上
        attn_out, _ = self.attn(x, x, x)
        x = self.norm1(x + attn_out)
        x = self.norm2(x + self.ffn(x))
        return x
```

### 2.2 完整的 Graphormer

```python
class Graphormer(nn.Module):
    def __init__(self, node_feat_dim, embed_dim, num_heads, num_layers, num_classes):
        super().__init__()
        self.node_enc = nn.Linear(node_feat_dim, embed_dim)
        self.centrality_enc = nn.Embedding(20, embed_dim)  # 度数编码
        self.layers = nn.ModuleList([
            GraphormerLayer(embed_dim, num_heads) for _ in range(num_layers)
        ])
        self.graph_token = nn.Parameter(torch.randn(1, 1, embed_dim))
        self.output_proj = nn.Linear(embed_dim, num_classes)

    def forward(self, x, edge_index, dist):
        B = x.size(0)
        # 节点编码 + 中心性编码
        h = self.node_enc(x)
        deg = edge_index[1].bincount(minlength=x.size(0)).clamp(0, 19)
        h = h + self.centrality_enc(deg)

        # 添加图级别 token
        graph_token = self.graph_token.expand(B, -1, -1)
        h = torch.cat([graph_token, h], dim=1)  # [B, N+1, d]

        for layer in self.layers:
            h = layer(h, dist)

        # 取图 token 的输出
        return self.output_proj(h[:, 0, :])
```

## 3. 注意力复杂度问题

全图注意力的复杂度为 $O(N^2 d)$，对于大图不可接受。

**解决方案**：

| 方法 | 复杂度 | 策略 |
|------|--------|------|
| 全局注意力 | $O(N^2)$ | 基准 |
| 稀疏注意力 | $O(N \sqrt{N})$ | 只在邻居/路径上计算 |
| 线性注意力 | $O(N)$ | 核函数近似 |
| 随机注意力 | $O(N)$ | 随机采样注意力对 |

## 4. GPS（General Powerful Scalable）

Rampasek et al. (2022)：结合 MPNN 和全局注意力的混合架构。

$$H^{(l+1)} = \text{FFN}\left(\alpha \cdot \text{MPNN}(H^{(l)}) + (1-\alpha) \cdot \text{Transformer}(H^{(l)})\right)$$

```python
from torch_geometric.nn import GPSConv, GCNConv
from torch_geometric.nn import TransformerConv

class GPSModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_layers=4, heads=4):
        super().__init__()
        self.node_enc = nn.Linear(in_dim, hidden_dim)
        self.convs = nn.ModuleList()
        for _ in range(num_layers):
            # MPNN 子层：使用 GCN
            mpnn = GCNConv(hidden_dim, hidden_dim)
            self.convs.append(GPSConv(hidden_dim, conv=mpnn, heads=heads))

        self.output_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, data):
        x = self.node_enc(data.x)
        for conv in self.convs:
            x = conv(x, data.edge_index)
        return self.output_proj(x)
```

**GPS 的关键洞察**：MPNN 捕获局部结构，Transformer 捕获全局依赖，二者互补。

## 5. 结构编码方法总结

| 编码类型 | 方法 | 说明 |
|----------|------|------|
| 位置编码 (PE) | 随机游走 PE, Laplacian PE | 编码节点在图中的位置 |
| 空间编码 | 最短路径距离 | 编码节点间结构距离 |
| 中心性编码 | 度数嵌入 | 编码节点的重要性 |
| 边编码 | 边特征嵌入 | 编码边的属性 |

```python
# 拉普拉斯位置编码（LPE）
import torch
from torch_geometric.utils import get_laplacian

def laplacian_pe(edge_index, num_nodes, k=8):
    """计算 k 维拉普拉斯位置编码"""
    L_edge_index, L_edge_weight = get_laplacian(
        edge_index, normalization='sym', num_nodes=num_nodes
    )
    # 构建稀疏矩阵并做特征分解
    L = torch.sparse_coo_tensor(L_edge_index, L_edge_weight, (num_nodes, num_nodes))
    L_dense = L.to_dense()
    eigenvalues, eigenvectors = torch.linalg.eigh(L_dense)
    # 取最小的 k 个非零特征值对应的特征向量
    return eigenvectors[:, 1:k+1]  # 跳过特征值为 0 的
```

## 6. 实用建议

| 场景 | 推荐方法 |
|------|----------|
| 小图 (<100 节点) | Graphormer（全局注意力可行） |
| 中等图 (100-1000) | GPS（MPNN + 局部注意力） |
| 大图 (>1000) | 先用 MPNN，或线性注意力 |
| 分子属性预测 | Graphormer, GPS |
| 节点分类 | MPNN + PE 通常足够 |

## 7. 小结

- 图 Transformer 将全局注意力引入图结构数据
- Graphormer 通过空间编码、中心性编码融入图结构
- 注意力复杂度是关键瓶颈：稀疏/线性注意力可降低计算量
- GPS 混合 MPNN 和 Transformer，兼顾局部和全局信息
- 结构编码（位置编码、最短路径距离）是图 Transformer 成功的关键
