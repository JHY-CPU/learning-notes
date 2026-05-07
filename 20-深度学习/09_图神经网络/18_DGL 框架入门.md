# 18_DGL 框架入门

## 1. DGL 简介

Deep Graph Library (DGL) 是另一个主流的图深度学习框架，特点：
- **原生异构图支持**：多节点/边类型
- **大规模图训练**：支持分布式和 mini-batch
- **自动批处理**：消息传递自动向量化
- **与主流深度学习框架兼容**：PyTorch、TensorFlow、MXNet

```bash
pip install dgl
```

## 2. DGL 基础

### 2.1 创建图

```python
import dgl
import torch

# 从边列表创建
src = [0, 1, 1, 2, 2, 3, 3, 0]
dst = [1, 0, 2, 1, 3, 2, 0, 3]
g = dgl.graph((src, dst))

print(g)
# Graph(num_nodes=4, num_edges=8,
#       ndata_schemes={}
#       edata_schemes={})

# 添加节点特征
g.ndata['h'] = torch.randn(4, 16)
# 添加边特征
g.edata['w'] = torch.randn(8, 4)

# 转为无向图
g = dgl.to_bidirected(g)

# 添加自环
g = dgl.add_self_loop(g)
```

### 2.2 图的常用操作

```python
# 节点数和边数
print(g.num_nodes(), g.num_edges())

# 获取度数
in_deg = g.in_degrees()   # 入度
out_deg = g.out_degrees()  # 出度

# 子图提取
subg = dgl.node_subgraph(g, [0, 1, 2])  # 保留节点 0, 1, 2
subg = dgl.edge_subgraph(g, [0, 1, 2])  # 保留边 0, 1, 2

# 查找边
src, dst = g.edges()
eid = g.edge_ids(0, 1)  # 找节点 0->1 的边

# 设备迁移
g = g.to('cuda')
```

## 3. 消息传递

DGL 的消息传递分为三个阶段：

```python
import dgl.function as fn

# 定义消息函数和聚合函数
g.ndata['h'] = torch.randn(4, 16)

# 使用内置函数（高效）
g.send_and_recv(
    g.edges(),                   # 所有边
    fn.copy_u('h', 'm'),         # 消息：复制源节点特征到 'm'
    fn.mean('m', 'h_new'),       # 聚合：对 'm' 取均值到 'h_new'
)

# 一步完成所有边的消息传递
g.update_all(
    fn.copy_u('h', 'm'),         # send
    fn.mean('m', 'h_new'),       # recv (reduce)
)
```

### 3.1 自定义消息函数

```python
def msg_func(edges):
    """自定义消息函数"""
    return {'m': edges.src['h'] * edges.data['w']}

def reduce_func(nodes):
    """自定义聚合函数"""
    return {'h_new': torch.sum(nodes.mailbox['m'], dim=1)}

g.update_all(msg_func, reduce_func)
```

### 3.2 用户定义函数（UDF）vs 内置函数

```python
# 内置函数（推荐，自动向量化，更快）
g.update_all(fn.copy_u('h', 'm'), fn.sum('m', 'h_new'))

# 用户定义函数（灵活但较慢）
def message(edges):
    return {'m': edges.src['h']}

g.update_all(message, fn.sum('m', 'h_new'))
```

## 4. DGL 中构建 GNN 模型

### 4.1 使用 dgl.nn 模块

```python
import dgl.nn as dglnn
import torch.nn as nn
import torch.nn.functional as F

class GCN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, out_dim)

    def forward(self, g, x):
        h = F.relu(self.conv1(g, x))
        h = self.conv2(g, h)
        return h

class GAT(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, num_heads=4):
        super().__init__()
        self.conv1 = dglnn.GATConv(in_dim, hidden_dim, num_heads)
        self.conv2 = dglnn.GATConv(hidden_dim * num_heads, out_dim, 1)

    def forward(self, g, x):
        h = self.conv1(g, x).flatten(1)
        h = F.elu(h)
        h = self.conv2(g, h).mean(1)
        return h
```

### 4.2 使用 nn.Module 封装

```python
class SAGEConv(nn.Module):
    """手动实现 GraphSAGE"""
    def __init__(self, in_dim, out_dim):
        super().__init__()
        self.W = nn.Linear(in_dim * 2, out_dim)

    def forward(self, g, h):
        with g.local_scope():
            g.ndata['h'] = h
            g.update_all(fn.copy_u('h', 'm'), fn.mean('m', 'h_neigh'))
            h_neigh = g.ndata['h_neigh']
            return self.W(torch.cat([h, h_neigh], dim=-1))
```

## 5. 异构图支持

DGL 原生支持异构图，这是其核心优势之一。

### 5.1 创建异构图

```python
import dgl

# 定义异构图
graph_data = {
    ('user', 'follows', 'user'): (torch.tensor([0, 1]), torch.tensor([1, 2])),
    ('user', 'plays', 'game'): (torch.tensor([0, 0, 1]), torch.tensor([0, 1, 0])),
    ('game', 'played_by', 'user'): (torch.tensor([0, 1, 0]), torch.tensor([0, 0, 1])),
}
g = dgl.heterograph(graph_data)

print(g)
# Graph(num_nodes={'user': 3, 'game': 2},
#       num_edges={('user', 'follows', 'user'): 2, ...})

# 添加特征
g.nodes['user'].data['h'] = torch.randn(3, 16)
g.nodes['game'].data['h'] = torch.randn(2, 8)
```

### 5.2 异构图上的消息传递

```python
# 按边类型更新
def message_func(edges):
    return {'m': edges.src['h']}

def reduce_func(nodes):
    return {'h_new': torch.mean(nodes.mailbox['m'], dim=1)}

# 对每种边类型分别更新
g.multi_update_all(
    {
        ('user', 'follows', 'user'): (fn.copy_u('h', 'm'), fn.mean('m', 'h_new')),
        ('user', 'plays', 'game'): (fn.copy_u('h', 'm'), fn.mean('m', 'h_new')),
    },
    'sum',  # 跨类型聚合
)
```

### 5.3 异构 GNN 模型

```python
class HeteroGNN(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, etypes):
        super().__init__()
        # 每种边类型一个卷积层
        self.conv1 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(in_dim, hidden_dim)
            for etype in etypes
        })
        self.conv2 = dglnn.HeteroGraphConv({
            etype: dglnn.GraphConv(hidden_dim, out_dim)
            for etype in etypes
        })

    def forward(self, g, h_dict):
        h_dict = self.conv1(g, h_dict)
        h_dict = {k: F.relu(v) for k, v in h_dict.items()}
        h_dict = self.conv2(g, h_dict)
        return h_dict
```

## 6. 大规模图训练

### 6.1 Mini-batch 训练（邻居采样）

```python
import dgl
from dgl.dataloading import DataLoader, NeighborSampler

# 采样器：每层采样 5 个邻居
sampler = NeighborSampler([5, 5])

# 训练节点
train_nids = torch.where(data.train_mask)[0]

dataloader = DataLoader(
    g, train_nids, sampler,
    batch_size=1024,
    shuffle=True,
    drop_last=False,
)

for input_nodes, output_nodes, blocks in dataloader:
    # blocks: 采样得到的子图序列（每层一个 block）
    # input_nodes: 子图中所有涉及的节点
    # output_nodes: 目标节点

    x = g.ndata['feat'][input_nodes]
    h = x
    for i, block in enumerate(blocks):
        block = block.to(device)
        h = model.convs[i](block, h)
        h = F.relu(h)

    loss = F.cross_entropy(h, labels[output_nodes])
```

### 6.2 分布式训练

DGL 支持将大图分布到多机多卡上：

```python
# 使用 DistGraph 进行分布式训练
import dgl
from dgl.distributed import DistGraph, DistDataLoader

g = DistGraph('graph_name', part_config='data/graph.json')
```

## 7. DGL vs PyG 对比

| 特性 | DGL | PyG |
|------|-----|-----|
| 异构图支持 | 原生支持，非常方便 | 支持，但稍显复杂 |
| 消息传递 | `update_all` + 函数式 | `MessagePassing` 基类 |
| 大规模图 | 原生支持分布式 | NeighborLoader |
| API 风格 | 函数式 | 面向对象 |
| 学习曲线 | 中等 | 较低 |
| 社区/文档 | 活跃 | 更活跃 |

### 选择建议

- **研究实验、小中型图**：PyG 更方便（API 简洁，预置模型多）
- **异构图、工业级大图**：DGL 更强（原生异构图、分布式支持）
- 两者都很好，选择你更熟悉的一个

## 8. DGL 训练完整示例

```python
import dgl
import dgl.nn as dglnn
import torch
import torch.nn as nn
import torch.nn.functional as F

# 加载 Cora
from dgl.data import CoraGraphDataset
dataset = CoraGraphDataset()
g = dataset[0]

class GCNModel(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.conv1 = dglnn.GraphConv(in_dim, hidden_dim)
        self.conv2 = dglnn.GraphConv(hidden_dim, out_dim)

    def forward(self, g, x):
        h = F.relu(self.conv1(g, x))
        return self.conv2(g, h)

model = GCNModel(g.ndata['feat'].shape[1], 16, dataset.num_classes)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-2, weight_decay=5e-4)

for epoch in range(200):
    model.train()
    logits = model(g, g.ndata['feat'])
    loss = F.cross_entropy(logits[g.ndata['train_mask']],
                          g.ndata['labels'][g.ndata['train_mask']])
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 50 == 0:
        model.eval()
        pred = logits.argmax(dim=1)
        acc = (pred[g.ndata['test_mask']] == g.ndata['labels'][g.ndata['test_mask']]).float().mean()
        print(f'Epoch {epoch}: Loss={loss:.4f}, Test Acc={acc:.4f}')
```

## 9. 小结

- DGL 使用 `update_all` + 函数式 API 进行消息传递
- 原生异构图支持是 DGL 的核心优势
- `HeteroGraphConv` 简化了异构图上的模型构建
- 支持分布式和 mini-batch 训练，适合工业级大规模图
- DGL 与 PyG 各有所长，根据需求选择
