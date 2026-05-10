# GraphSAGE与采样


## 1. 归纳式学习 vs 直推式学习


GraphSAGE（Graph SAmple and aggreGatE）由 Hamilton 等人在 2017 年提出，核心贡献是将GNN从直推式（transductive）扩展到归纳式（inductive）学习。


### 两种学习范式对比


| 特性 | 直推式学习 | 归纳式学习 |
| --- | --- | --- |
| 代表模型 | GCN | GraphSAGE |
| 训练时看到的图 | 完整图（含测试节点） | 仅训练子图 |
| 新节点处理 | 需要重新训练 | 直接推理 |
| 可扩展性 | 受限（需全图矩阵） | 好（支持mini-batch） |
| 适用场景 | 静态图、小规模 | 动态图、大规模 |
| 训练方式 | 全图矩阵乘法 | 采样+聚合 |


> **Note:** **关键区别：**
> GCN学习的是每个节点的嵌入，GraphSAGE学习的是
> **聚合函数**
> 。聚合函数是与节点无关的，因此可以推广到训练时未见过的新节点。


## 2. GraphSAGE 的邻居采样策略


### 固定大小采样


GraphSAGE 对每个节点固定采样 K 个邻居（不足则有放回采样），控制计算量。


> **Example:** #### 采样示例
>
>
> 假设 K=2，2层采样：
>
>
> - 目标节点 v 有5个邻居 → 随机采样2个
> - 每个被采样的邻居又有各自的邻居 → 再各采样2个
> - 2层采样后，计算图包含 1 + 2 + 4 = 7 个节点


### 前向传播算法


$$
Algorithm: GraphSAGE Forward
                Input: 图 G=(V,E)，节点特征 {xv ∀v∈V}，层数 K，权重 Wk
                hv0 = xv, ∀v ∈ V
                for k = 1, ..., K do:
                  for v ∈ V do:
                    N(v) ← Sample(Sk, N(v))  // 采样Sk个邻居
                    hN(v)k ← AGGREGATEk({huk-1, ∀u∈N(v)})
                    hvk ← σ(Wk · [hvk-1 || hN(v)k])
                    hvk ← hvk / ||hvk||2  // L2归一化
                return {hvK, ∀v∈V}
$$


## 3. 聚合函数


GraphSAGE 提出了三种聚合函数，GCN使用的均值聚合是其中之一。


| 聚合函数 | 公式 | 特点 |
| --- | --- | --- |
| Mean（均值） | h~N(v)~ = mean({h~u~^k-1^, ∀u∈N(v)}) | 简单高效，类似GCN |
| LSTM | h~N(v)~ = LSTM([h~u~^k-1^, ∀u∈π(N(v))]) | 表达力强，但非排列不变（需随机排列） |
| Pooling | h~N(v)~ = max({σ(W~pool~h~u~^k-1^ + b), ∀u∈N(v)}) | 捕获最显著特征 |


> **Note:** **聚合函数选择：**
> 在原论文实验中，三种聚合函数的表现差异不大，Mean 聚合通常作为默认选择，因为最简单且效果稳定。


### 拼接 vs 直接聚合


GraphSAGE 将自身表示和邻居表示拼接后线性变换，这比GCN的直接求和更灵活：


$$
GraphSAGE: hvk = σ(Wk · [hvk-1 || hN(v)k])
                GCN: hvk = σ(Σu∈N(v)∪{v} cuv W huk-1)
$$


## 4. Mini-batch 训练


GraphSAGE 支持 mini-batch 训练，每次只处理一批目标节点及其采样的邻居子图，而非整个图。


### 训练流程


1. 从图中采样一批目标节点 B
2. 对 B 中每个节点，递归采样 K 层邻居，构建计算图
3. 在计算图上前向传播，得到目标节点的表示
4. 计算损失（如交叉熵），反向传播更新参数


### 计算图 vs 全图


| 方式 | 内存占用 | 计算图 | 适用规模 |
| --- | --- | --- | --- |
| 全图训练（GCN） | 高（存储所有节点特征） | 整张图 | 百万节点以内 |
| Mini-batch（GraphSAGE） | 低（只存储子图） | 采样子图 | 数十亿节点 |


> **Important:** **注意：**
> 训练时邻居数量较小（如[25,10]），推理时可以使用更多邻居甚至全部邻居以获得更好的结果。


## 5. 适合大规模图的原因


- **内存友好**
   ：每次只加载子图到内存，不需要完整图
- **并行化**
   ：不同batch之间完全独立，可并行处理
- **在线推理**
   ：新节点加入图后无需重新训练模型
- **参数共享**
   ：所有节点共享同一套聚合函数参数


### 性能基准


| 数据集 | 节点数 | 边数 | GraphSAGE准确率 | GCN准确率 |
| --- | --- | --- | --- | --- |
| Reddit | 232K | 114M | 95.3% | 91.5%（OOM困难） |
| PPI | 56K | 818K | 61.2% | 59.5% |


> **Note:** **Reddit 数据集：**
> Reddit论坛的帖子图，节点是帖子，边连接同一用户评论的帖子。由于图很大（23万节点），全图训练方式（如GCN）非常困难，GraphSAGE的采样策略在此场景下优势明显。


## 6. PyTorch Geometric 代码示例


```
import torch
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv
from torch_geometric.loader import NeighborLoader
from torch_geometric.datasets import Reddit
import time

# ========== GraphSAGE 模型 ==========
class GraphSAGE(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels, num_layers=2):
        super().__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(SAGEConv(in_channels, hidden_channels))
        for _ in range(num_layers - 2):
            self.convs.append(SAGEConv(hidden_channels, hidden_channels))
        self.convs.append(SAGEConv(hidden_channels, out_channels))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index)
            if i < len(self.convs) - 1:
                x = F.relu(x)
                x = F.dropout(x, p=0.5, training=self.training)
        return x

# ========== 邻居采样加载器 ==========
dataset = Reddit(root='/tmp/Reddit')
data = dataset[0]

# 训练时采样：每个节点采样[25, 10]个邻居（2层）
train_loader = NeighborLoader(
    data,
    num_neighbors=[25, 10],   # 每层采样数
    batch_size=1024,           # batch大小
    input_nodes=data.train_mask,
    shuffle=True
)

# 验证/测试时采样更多邻居
test_loader = NeighborLoader(
    data,
    num_neighbors=[-1, -1],   # -1 表示使用全部邻居
    batch_size=2048,
    input_nodes=data.test_mask,
    shuffle=False
)

# ========== 训练 ==========
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = GraphSAGE(
    in_channels=dataset.num_features,
    hidden_channels=256,
    out_channels=dataset.num_classes
).to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

def train():
    model.train()
    total_loss = 0
    for batch in train_loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        out = model(batch.x, batch.edge_index)
        # 只计算目标节点（batch中的前batch_size个）的损失
        loss = F.cross_entropy(
            out[:batch.batch_size],
            batch.y[:batch.batch_size]
        )
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(train_loader)

@torch.no_grad()
def test():
    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        batch = batch.to(device)
        out = model(batch.x, batch.edge_index)
        pred = out[:batch.batch_size].argmax(dim=-1)
        correct += (pred == batch.y[:batch.batch_size]).sum().item()
        total += batch.batch_size
    return correct / total

for epoch in range(1, 11):
    t0 = time.time()
    loss = train()
    acc = test()
    print(f'Epoch: {epoch:02d}, Loss: {loss:.4f}, '
          f'Test Acc: {acc:.4f}, Time: {time.time()-t0:.1f}s')
```


## 总结


- GraphSAGE通过学习聚合函数而非节点嵌入实现归纳式学习
- 邻居采样控制每层参与计算的邻居数量，支持mini-batch训练
- 三种聚合函数：Mean（默认）、LSTM、Pooling
- 拼接自身和邻居表示的方式比GCN的直接求和更灵活
- PyG的NeighborLoader提供高效的采样数据加载
- 特别适合大规模图和动态图场景


<!-- Converted from: 01_GraphSAGE与采样.html -->
