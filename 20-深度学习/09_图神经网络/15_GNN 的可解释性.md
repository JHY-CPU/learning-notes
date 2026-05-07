# 15_GNN 的可解释性

## 1. GNN 可解释性的动机

GNN 的黑盒性质使其在高风险场景中难以信任：
- 药物发现：为什么模型认为这个分子有毒？
- 社交网络：哪些邻居影响了用户的购买决策？
- 金融风控：哪些交易关系导致了欺诈检测？

## 2. GNN 可解释性的分类

| 类型 | 解释什么 | 示例方法 |
|------|----------|----------|
| 事后解释 (Post-hoc) | 解释已训练好的模型 | GNNExplainer, PGExplainer |
| 内在可解释 | 模型本身就可解释 | Attention weights, 解释性 GNN |
| 实例级解释 | 为什么预测节点 $v$ 为类 $c$ | GNNExplainer |
| 模型级解释 | 模型学到的通用模式 | SubgraphX |

## 3. GNNExplainer

Ying et al. (2019)：通过学习一个掩码来识别重要的子图和节点特征。

### 3.1 核心思想

对于节点 $v$ 的预测 $y_v$，GNNExplainer 找到一个子图 $G_S \subseteq G$ 和特征子集，使得：

$$\max_{S} \text{MI}(Y, G_S) = H(Y) - H(Y | G_S)$$

等价于最小化 $G_S$ 对预测的不确定性。

### 3.2 软掩码优化

GNNExplainer 为每条边学习一个可学习的软掩码：

$$\hat{A} = A \odot \sigma(M), \quad M_{ij} \in \mathbb{R}$$

$$\min_M \mathcal{L}_{\text{CE}}(f(\hat{A}, X), y) + \lambda_1 \|M\|_1 + \lambda_2 H(M)$$

- $\mathcal{L}_{\text{CE}}$：预测保持不变（解释应该支持原预测）
- $\|M\|_1$：边掩码稀疏（解释应该简洁）
- $H(M)$：熵最小化（掩码趋向 0/1）

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class GNNExplainer(nn.Module):
    def __init__(self, model, num_edges, num_node_features, epochs=100, lr=0.01):
        super().__init__()
        self.model = model
        self.model.eval()
        self.epochs = epochs
        self.lr = lr

        # 边掩码
        self.edge_mask = nn.Parameter(torch.ones(num_edges) * 0.5)
        # 特征掩码
        self.feature_mask = nn.Parameter(torch.ones(num_node_features) * 0.5)

    def explain(self, data, node_idx=None):
        """为节点或整个图生成解释"""
        optimizer = torch.optim.Adam([self.edge_mask, self.feature_mask], lr=self.lr)

        for epoch in range(self.epochs):
            optimizer.zero_grad()

            # 软掩码邻接矩阵
            mask_sigmoid = torch.sigmoid(self.edge_mask)
            modified_edge_weight = data.edge_attr * mask_sigmoid if data.edge_attr is not None else mask_sigmoid

            # 软掩码特征
            feat_mask = torch.sigmoid(self.feature_mask)
            modified_x = data.x * feat_mask

            # 前向传播（使用修改后的图）
            out = self.model(modified_x, data.edge_index)

            # 损失：保持预测 + 稀疏 + 熵
            pred_loss = F.cross_entropy(out[node_idx:node_idx+1], data.y[node_idx:node_idx+1])
            sparsity_loss = 0.01 * mask_sigmoid.sum()
            entropy_loss = 0.001 * (-(mask_sigmoid * torch.log(mask_sigmoid + 1e-8) +
                                      (1 - mask_sigmoid) * torch.log(1 - mask_sigmoid + 1e-8))).mean()

            loss = pred_loss + sparsity_loss + entropy_loss
            loss.backward()
            optimizer.step()

        # 返回重要的边
        with torch.no_grad():
            important_edges = torch.sigmoid(self.edge_mask) > 0.5
        return important_edges
```

### 3.3 PyG 中的 GNNExplainer

```python
from torch_geometric.explain import Explainer, GNNExplainer

model = ...  # 预训练的 GNN 模型

explainer = Explainer(
    model=model,
    algorithm=GNNExplainer(epochs=200),
    explanation_type='model',
    node_mask_type='attributes',
    edge_mask_type='object',
    model_config=dict(
        mode='multiclass_classification',
        task_level='node',
        return_type='log_probs',
    ),
)

# 解释节点 10 的预测
explanation = explainer(data.x, data.edge_index, index=10)

# 查看重要边
print(f"边掩码: {explanation.edge_mask}")  # [num_edges]
print(f"节点掩码: {explanation.node_mask}")  # [num_nodes]

# 可视化
explanation.visualize_graph(path='explanation.png')
```

## 4. PGExplainer

Luo et al. (2020)：参数化的解释器，用一个网络预测边的重要性。

```python
class PGExplainer(nn.Module):
    """参数化图解释器"""
    def __init__(self, gnn_model, hidden_dim, temperature=1.0):
        super().__init__()
        self.gnn = gnn_model
        self.edge_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )
        self.temperature = temperature

    def get_edge_mask(self, z, edge_index):
        """预测每条边的重要性"""
        src, dst = edge_index
        edge_repr = torch.cat([z[src], z[dst]], dim=-1)
        logits = self.edge_predictor(edge_repr).squeeze(-1)

        # Gumbel-Softmax 采样
        if self.training:
            mask = torch.sigmoid(logits / self.temperature)
        else:
            mask = (logits > 0).float()
        return mask

    def forward(self, data, edge_mask):
        # 使用掩码后的边权重
        out = self.gnn(data.x, data.edge_index)
        return out
```

## 5. 注意力可视化

对于 GAT 类模型，注意力权重本身提供了一定的可解释性。

```python
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx

def visualize_gnn_attention(model, data, node_idx, layer_idx=0):
    """可视化 GAT 的注意力权重"""
    model.eval()
    with torch.no_grad():
        # 获取指定层的注意力权重
        x = data.x
        for i, conv in enumerate(model.convs):
            x, (edge_index, alpha) = conv(x, data.edge_index, return_attention_weights=True)
            x = F.relu(x)
            if i == layer_idx:
                break

    # 筛选与目标节点相关的边
    src, dst = edge_index
    mask = (dst == node_idx) | (src == node_idx)
    relevant_edges = edge_index[:, mask]
    relevant_alpha = alpha[mask]

    # 可视化
    G = nx.Graph()
    for i in range(relevant_edges.size(1)):
        s, d = relevant_edges[0, i].item(), relevant_edges[1, i].item()
        G.add_edge(s, d, weight=relevant_alpha[i, 0].item())

    pos = nx.spring_layout(G)
    weights = [G[u][v]['weight'] * 5 for u, v in G.edges()]
    nx.draw(G, pos, with_labels=True, width=weights,
            node_color=['red' if n == node_idx else 'lightblue' for n in G.nodes()])
    plt.title(f"Attention around node {node_idx}")
    plt.show()
```

## 6. 子图解释方法

### 6.1 SubgraphX

Yuan et al. (2021)：使用蒙特卡洛树搜索找到重要的子图。

```python
class SubgraphX:
    """基于子图搜索的解释"""
    def __init__(self, model, num_hops=3, num_rollouts=20):
        self.model = model
        self.num_hops = num_hops
        self.num_rollouts = num_rollouts

    def explain(self, data, node_idx):
        """找到解释预测的最小连通子图"""
        # 提取 k-hop 子图
        from torch_geometric.utils import k_hop_subgraph
        subset, sub_edge_index, mapping, edge_mask = k_hop_subgraph(
            node_idx, self.num_hops, data.edge_index
        )

        best_subgraph = None
        best_score = 0

        for _ in range(self.num_rollouts):
            # 随机删除一些边
            keep_mask = torch.rand(sub_edge_index.size(1)) > 0.3
            if keep_mask.sum() < 2:
                continue

            new_edge_index = sub_edge_index[:, keep_mask]

            # 评估删除后的预测变化
            score = self._evaluate(data.x[subset], new_edge_index, node_idx)
            if score > best_score:
                best_score = score
                best_subgraph = (subset, new_edge_index)

        return best_subgraph

    def _evaluate(self, x, edge_index, node_idx):
        """评估子图的预测保真度"""
        with torch.no_grad():
            out = self.model(x, edge_index)
            return out[node_idx].max().item()
```

## 7. 可解释性方法对比

| 方法 | 类型 | 速度 | 解释粒度 | 适用 |
|------|------|------|----------|------|
| GNNExplainer | 事后 | 慢 | 边+特征 | 任意 GNN |
| PGExplainer | 事后，参数化 | 快（推理时） | 边 | 任意 GNN |
| Attention | 内在 | 快 | 边 | GAT |
| SubgraphX | 事后 | 慢 | 子图 | 任意 GNN |
| GradCAM | 事后 | 快 | 节点 | 任意 GNN |

## 8. 小结

- GNN 可解释性对于高风险应用至关重要
- GNNExplainer 通过学习边和特征的软掩码生成解释
- PGExplainer 训练参数化解释器，推理速度更快
- 注意力权重提供天然的可解释性，但不一定反映真实的特征重要性
- 子图级解释（SubgraphX）能提供结构更完整的解释
