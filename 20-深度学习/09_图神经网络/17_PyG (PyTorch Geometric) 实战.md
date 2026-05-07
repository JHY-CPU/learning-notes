# 17_PyG (PyTorch Geometric) 实战

## 1. PyG 简介

PyTorch Geometric (PyG) 是基于 PyTorch 的图深度学习库，提供：
- 图数据结构（`Data`, `Batch`）
- 经典 GNN 层（GCN, GAT, SAGE 等）
- 常用数据集（Cora, Planetoid, TU 等）
- 高效的消息传递实现

```bash
pip install torch-geometric
```

## 2. Data 对象

`Data` 是 PyG 中图的基本数据结构。

```python
import torch
from torch_geometric.data import Data

# 定义一张图
# edge_index: [2, num_edges]，COO 格式
edge_index = torch.tensor([
    [0, 1, 1, 2, 2, 3, 3, 0],  # 源节点
    [1, 0, 2, 1, 3, 2, 0, 3],  # 目标节点
], dtype=torch.long)

# 节点特征
x = torch.randn(4, 16)  # 4 个节点，16 维特征

# 节点标签
y = torch.tensor([0, 1, 0, 1])

# 边属性（可选）
edge_attr = torch.randn(8, 4)  # 8 条边，4 维边特征

data = Data(x=x, edge_index=edge_index, y=y, edge_attr=edge_attr)

print(data)
# Data(x=[4, 16], edge_index=[2, 8], y=[4], edge_attr=[8, 4])
print(f"节点数: {data.num_nodes}")
print(f"边数: {data.num_edges}")
print(f"特征维度: {data.num_node_features}")
print(f"是否无向: {data.is_undirected()}")
```

### 2.1 Data 的常用属性和方法

```python
# 设备迁移
data = data.to('cuda')

# 深拷贝
data_clone = data.clone()

# 连续性检查（某些操作需要）
data = data.contiguous()

# 是否自环
print(data.has_self_loops())

# 是否已排序
print(data.is_coalesced())

# 转为 NetworkX 图
import networkx as nx
G = data.to_networkx()
```

## 3. Dataset：自定义数据集

### 3.1 内存数据集

```python
from torch_geometric.data import InMemoryDataset

class MyDataset(InMemoryDataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = torch.load(self.processed_paths[0])

    @property
    def raw_file_names(self):
        return ['data.csv']

    @property
    def processed_file_names(self):
        return ['data.pt']

    def download(self):
        # 下载原始数据
        pass

    def process(self):
        # 处理原始数据为 Data 对象
        data_list = []
        # ... 从原始数据构建图 ...
        for i in range(100):
            x = torch.randn(20, 16)
            edge_index = torch.randint(0, 20, (2, 60))
            y = torch.tensor([i % 2])
            data_list.append(Data(x=x, edge_index=edge_index, y=y))

        # collate 为单个大图 + slices
        data, slices = self.collate(data_list)
        torch.save((data, slices), self.processed_paths[0])

# 使用
dataset = MyDataset(root='/tmp/my_data')
```

### 3.2 大型数据集（On-disk）

```python
from torch_geometric.data import Dataset

class LargeDataset(Dataset):
    def __init__(self, root, transform=None, pre_transform=None):
        super().__init__(root, transform, pre_transform)

    @property
    def raw_file_names(self):
        return [...]

    @property
    def processed_file_names(self):
        return [f'data_{i}.pt' for i in range(1000)]

    def process(self):
        # 逐个处理并保存
        for i, raw_path in enumerate(self.raw_paths):
            data = self._process_single(raw_path)
            torch.save(data, self.processed_paths[i])

    def len(self):
        return len(self.processed_file_names)

    def get(self, idx):
        return torch.load(self.processed_paths[idx])
```

## 4. DataLoader 与批处理

PyG 使用 `Batch` 对象将多张图拼接为一张大图（不共享边）。

```python
from torch_geometric.loader import DataLoader

dataset = ...  # 图分类数据集
loader = DataLoader(dataset, batch_size=32, shuffle=True)

for batch in loader:
    print(batch)
    # Batch(x=[total_nodes, d], edge_index=[2, total_edges], batch=[total_nodes], ...)
    print(f"包含 {batch.num_graphs} 张图")
    print(f"batch 向量: {batch.batch}")  # 每个节点属于哪张图

    # 全局池化
    from torch_geometric.nn import global_mean_pool
    h_graph = global_mean_pool(h_nodes, batch.batch)  # [B, d]
```

### 4.1 NeighborLoader（GraphSAGE 邻居采样）

```python
from torch_geometric.loader import NeighborLoader

# 大图节点分类
data = ...  # 单张大图

train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],  # 每层采样的邻居数
    batch_size=1024,
    input_nodes=data.train_mask,
)

for batch in train_loader:
    # batch 是子图：只包含采样的节点和边
    out = model(batch.x, batch.edge_index)
    # 只计算中心节点的损失
    loss = F.cross_entropy(out[:batch.batch_size], batch.y[:batch.batch_size])
```

## 5. MessagePassing 基类

PyG 的 `MessagePassing` 是所有消息传递层的基类。

```python
from torch_geometric.nn import MessagePassing
from torch_geometric.utils import add_self_loops, degree

class MyGCNConv(MessagePassing):
    """基于 MessagePassing 的 GCN 实现"""
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='add')  # 聚合方式：add, mean, max
        self.lin = nn.Linear(in_dim, out_dim, bias=False)
        self.bias = nn.Parameter(torch.zeros(out_dim))

    def forward(self, x, edge_index):
        # 添加自环
        edge_index, _ = add_self_looms(edge_index, num_nodes=x.size(0))
        # 线性变换
        x = self.lin(x)
        # 计算归一化系数
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        # 消息传递
        out = self.propagate(edge_index, x=x, norm=norm)
        return out + self.bias

    def message(self, x_j, norm):
        """x_j 是源节点的特征（自动通过 edge_index 选择）"""
        return norm.view(-1, 1) * x_j
```

### 5.1 MessagePassing 的三个核心方法

```python
class CustomConv(MessagePassing):
    def __init__(self):
        super().__init__(aggr='mean')

    def forward(self, x, edge_index):
        # propagate 自动调用 message -> aggregate -> update
        return self.propagate(edge_index, x=x)

    def message(self, x_j, x_i, edge_index):
        """
        定义每条边上的消息
        x_j: 源节点特征 [E, d]
        x_i: 目标节点特征 [E, d]
        返回: 消息向量 [E, d']
        """
        return x_j  # 最简单：传递源节点特征

    def aggregate(self, inputs, index):
        """
        聚合邻居消息
        inputs: [E, d'] 消息
        index: [E] 目标节点索引
        默认由 aggr 参数控制
        """
        return super().aggregate(inputs, index)

    def update(self, aggr_out):
        """
        聚合后的更新
        aggr_out: [N, d']
        """
        return aggr_out
```

### 5.2 自定义消息传递层示例

```python
class EdgeConv(MessagePassing):
    """边卷积：DGCNN"""
    def __init__(self, in_dim, out_dim):
        super().__init__(aggr='max')
        self.mlp = nn.Sequential(
            nn.Linear(in_dim * 2, out_dim),
            nn.BatchNorm1d(out_dim),
            nn.ReLU(),
            nn.Linear(out_dim, out_dim),
        )

    def forward(self, x, edge_index):
        return self.propagate(edge_index, x=x)

    def message(self, x_i, x_j):
        # 拼接源和目标节点特征，过 MLP
        tmp = torch.cat([x_i, x_j - x_i], dim=-1)
        return self.mlp(tmp)
```

## 6. 常用 GNN 层速查

```python
from torch_geometric.nn import (
    GCNConv,           # 图卷积
    SAGEConv,          # GraphSAGE
    GATConv,           # 图注意力
    GATv2Conv,         # GATv2 动态注意力
    GINConv,           # 图同构网络
    GINEConv,          # 带边特征的 GIN
    TransformerConv,   # 图 Transformer
    SplineConv,        # 样条卷积
    RGCNConv,          # 关系 GCN
    TAGConv,           # Topology Adaptive GCN
    SGConv,            # 简化 GCN
    APPNP,             # 个性化 PageRank
)

# 使用示例
conv = GCNConv(64, 128)
x_out = conv(x, edge_index)

# 带边特征
conv = GINEConv(nn.Sequential(nn.Linear(64, 128), nn.ReLU()), edge_dim=16)
x_out = conv(x, edge_index, edge_attr)
```

## 7. 常用工具函数

```python
from torch_geometric.utils import (
    add_self_loops,        # 添加自环
    remove_self_loops,     # 移除自环
    to_undirected,         # 转为无向图
    degree,                # 计算度数
    dropout_edge,          # 随机丢弃边
    negative_sampling,     # 负采样
    k_hop_subgraph,        # 提取 k-hop 子图
    subgraph,              # 提取子图
    to_dense_adj,          # 转为稠密邻接矩阵
    to_dense_batch,        # 转为稠密批次
)

from torch_geometric.transforms import (
    NormalizeFeatures,     # 特征归一化
    AddSelfLoops,          # 添加自环
    ToUndirected,          # 转为无向图
    Compose,               # 组合变换
    KNNGraph,              # 构建 kNN 图
    RadiusGraph,           # 构建半径图
)

from torch_geometric.nn import (
    global_mean_pool,      # 全局平均池化
    global_max_pool,       # 全局最大池化
    global_add_pool,       # 全局求和池化
    TopKPooling,           # TopK 池化
    SAGPooling,            # 自注意力池化
)
```

## 8. 训练模板

```python
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_mean_pool
from torch_geometric.datasets import Planetoid
from torch_geometric.loader import DataLoader

# 节点分类模板
class NodeClassifier(torch.nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = GCNConv(in_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, out_dim)

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        return F.log_softmax(self.conv2(x, edge_index), dim=1)

# 加载数据
dataset = Planetoid(root='/tmp/Cora', name='Cora')
data = dataset[0]

# 训练
model = NodeClassifier(dataset.num_node_features, 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        pred = out.argmax(dim=1)
        acc = (pred[data.test_mask] == data.y[data.test_mask]).float().mean()
        print(f'Epoch {epoch}: Loss={loss:.4f}, Test Acc={acc:.4f}')
```

## 9. 小结

- PyG 的核心数据结构是 `Data`（单图）和 `Batch`（多图批处理）
- `InMemoryDataset` 适合小数据集，`Dataset` 适合大数据集
- `MessagePassing` 基类通过 `message`、`aggregate`、`update` 实现消息传递
- `NeighborLoader` 支持大规模图的邻居采样训练
- PyG 提供丰富的 GNN 层、工具函数和预置数据集
