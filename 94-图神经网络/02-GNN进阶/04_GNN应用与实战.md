# GNN应用与实战


## 1. 社交网络分析


社交网络天然具有图结构——用户是节点，关注/好友关系是边。GNN可以捕获用户之间的复杂关系模式。


### 典型任务


| 任务 | 描述 | GNN方法 |
| --- | --- | --- |
| 节点分类 | 预测用户兴趣、人口属性 | GCN, GAT |
| 链接预测 | 推荐可能认识的人 | Node2Vec + MLP, SEAL |
| 社区发现 | 发现兴趣群组 | 图聚类 + GNN |
| 信息传播预测 | 预测信息扩散范围 | 时序GNN |
| 虚假账号检测 | 识别机器人账号 | 异配GNN, GIN |


> **Example:** #### 微信社交网络分析示例
>
>
> 节点特征：用户画像（年龄、地区、活跃度）
>
>                 边特征：互动频率、关系类型（好友/群聊）
>
>                 任务：基于社交图预测用户流失风险


## 2. 推荐系统中的 GNN


推荐系统可以建模为用户-物品二部图，GNN通过消息传递捕获高阶协同过滤信号。


### 图推荐模型


- **PinSage**
   ：Pinterest的工业级GNN推荐，对数十亿节点的图使用GraphSAGE采样
- **LightGCN**
   ：去掉特征变换和非线性激活，只保留邻域聚合，简洁高效
- **NGCF**
   ：神经图协同过滤，显式建模高阶协同信号
- **SR-GNN**
   ：将用户会话建模为图，用于序列推荐


$$
LightGCN:
                eu(k+1) = Σi∈Nu (1/√|Nu||Ni|) ei(k)
                eu = (1/K+1) Σk=0K eu(k)  // 各层平均
$$


> **Note:** **工业实践：**
> PinSage 在 Pinterest 的生产环境中处理超过30亿节点和180亿边的商品图，通过随机游走采样和MapReduce并行化实现高效训练。


## 3. 药物发现与分子图


分子天然可以用图表示——原子是节点，化学键是边。GNN在分子性质预测和药物发现中展现了强大能力。


### 分子图建模


- **节点特征**
   ：原子类型、电荷、杂化类型、芳香性
- **边特征**
   ：键类型（单键/双键/三键）、共轭性
- **图级任务**
   ：溶解度预测、毒性预测、活性预测


### 代表性模型


| 模型 | 特点 | 应用 |
| --- | --- | --- |
| MPNN | 统一的消息传递框架 | 量子化学性质预测 |
| AttentiveFP | 注意力机制+图池化 | 药物可解释性 |
| SchNet | 连续距离滤波器 | 3D分子建模 |
| 3D-MGN | 等变GNN | 分子动力学模拟 |
| Uni-Mol | 统一分子表示 | 多任务分子预测 |


## 4. 交通预测：时空图


交通预测需要同时建模空间依赖（路网拓扑）和时间依赖（历史流量），形成时空图（Spatial-Temporal Graph）。


### 时空图建模


- **空间维度**
   ：路网拓扑结构，节点是路段/传感器，边表示连接关系
- **时间维度**
   ：每个时间步的图快照序列
- **预测目标**
   ：未来时刻各路段的交通速度/流量


### 代表性模型


- **STGCN**
   ：时空图卷积网络，用GCN建模空间依赖，用1D卷积/门控机制建模时间依赖
- **DCRNN**
   ：扩散卷积循环神经网络，用图上的扩散过程建模空间依赖
- **GMAN**
   ：图多注意力网络，使用时空注意力机制
- **PDFormer**
   ：将Transformer应用于时空图


$$
STGCN 时空块:
                H(l+1) = ReLU(Θs *g (σt(H(l) Wt + b)))
                其中 σt 是时间门控卷积，*g 是图卷积
$$


## 5. 程序分析与代码理解


程序代码可以用多种图结构表示：抽象语法树（AST）、控制流图（CFG）、数据流图（DFG）等。GNN可以学习代码的语义表示。


### 代码的图表示


| 图类型 | 节点 | 边 | 捕获的信息 |
| --- | --- | --- | --- |
| AST | 语法节点 | 父子关系 | 代码语法结构 |
| CFG | 基本块 | 控制转移 | 程序执行流程 |
| DFG | 变量/语句 | 数据依赖 | 变量使用关系 |
| CPG | 混合 | 混合 | 综合多种信息 |


### 应用场景


- **代码漏洞检测**
   ：检测缓冲区溢出、SQL注入等安全漏洞
- **代码克隆检测**
   ：发现相似的代码片段
- **变量名预测**
   ：根据上下文推荐变量名
- **程序修复**
   ：自动修复代码缺陷


## 6. 完整实战：用 PyG 做 Cora 节点分类


```
import torch
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, GATConv
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
import time

# ========== 1. 加载数据 ==========
dataset = Planetoid(root='/tmp/Cora', name='Cora',
                    transform=NormalizeFeatures())
data = dataset[0]
print(f"数据集: Cora")
print(f"节点数: {data.num_nodes}, 边数: {data.num_edges}")
print(f"特征维度: {data.num_node_features}, 类别数: {dataset.num_classes}")
print(f"训练/验证/测试: {data.train_mask.sum()}/{data.val_mask.sum()}/{data.test_mask.sum()}")

# ========== 2. 定义模型 ==========
class GCN(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = GCNConv(dataset.num_features, 16)
        self.conv2 = GCNConv(16, dataset.num_classes)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = self.conv1(x, edge_index)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class GATModel(torch.nn.Module):
    def __init__(self, heads=8):
        super().__init__()
        self.conv1 = GATConv(dataset.num_features, 8, heads=heads)
        self.conv2 = GATConv(8 * heads, dataset.num_classes, heads=1,
                             concat=False)
        self.dropout = 0.6

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# ========== 3. 训练与评估 ==========
def train_and_evaluate(model_class, model_name, epochs=200):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = model_class().to(device)
    data_device = data.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)

    best_val_acc = 0
    best_test_acc = 0
    t0 = time.time()

    for epoch in range(1, epochs + 1):
        model.train()
        optimizer.zero_grad()
        out = model(data_device)
        loss = F.nll_loss(out[data_device.train_mask], data_device.y[data_device.train_mask])
        loss.backward()
        optimizer.step()

        # 评估
        model.eval()
        pred = model(data_device).argmax(dim=1)
        val_correct = (pred[data_device.val_mask] == data_device.y[data_device.val_mask]).sum()
        val_acc = int(val_correct) / int(data_device.val_mask.sum())
        test_correct = (pred[data_device.test_mask] == data_device.y[data_device.test_mask]).sum()
        test_acc = int(test_correct) / int(data_device.test_mask.sum())

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_test_acc = test_acc

        if epoch % 50 == 0:
            print(f'[{model_name}] Epoch {epoch:03d}, Loss: {loss.item():.4f}, '
                  f'Val: {val_acc:.4f}, Test: {test_acc:.4f}')

    elapsed = time.time() - t0
    print(f'\n[{model_name}] Best Val Acc: {best_val_acc:.4f}, '
          f'Best Test Acc: {best_test_acc:.4f}, Time: {elapsed:.1f}s')
    return best_test_acc

# 运行对比实验
print("=" * 60)
gcn_acc = train_and_evaluate(GCN, "GCN")
print("=" * 60)
gat_acc = train_and_evaluate(GATModel, "GAT")
print("=" * 60)
print(f"\n对比结果: GCN={gcn_acc:.4f}, GAT={gat_acc:.4f}")
```


## 7. GNN 的前沿方向


### 异构图神经网络（Heterogeneous GNN）


处理包含多种节点类型和边类型的知识图谱场景，代表模型：HAN、HGT、SimpleHGN。


### 动态图神经网络（Dynamic/Temporal GNN）


处理随时间演变的图结构，如社交网络中不断新增的用户和关系，代表模型：TGAT、TGN、DyGFormer。


### 超图神经网络（Hypergraph Neural Network）


超边可以连接任意数量的节点，建模高阶关系，如共同作者关系，代表模型：HGNN、HNHN。


### 几何图神经网络（Equivariant GNN）


对旋转、平移等几何变换保持等变性，用于3D分子建模，代表模型：SchNet、DimeNet、EGNN。


### 图大模型（Graph Foundation Model）


将大语言模型与图结构结合，探索图上的预训练和迁移学习，代表工作：GraphGPT、LLM4GNN。


| 前沿方向 | 核心挑战 | 典型应用 | 代表模型 |
| --- | --- | --- | --- |
| 异构图 | 多种节点/边类型 | 知识图谱、学术网络 | HAN, HGT |
| 动态图 | 时序演化 | 交易网络、社交网络 | TGAT, TGN |
| 超图 | 高阶关系 | 多人协作、蛋白质 | HGNN, HNHN |
| 几何GNN | 等变性 | 分子动力学 | EGNN, PaiNN |
| 图大模型 | 图-文本融合 | 多模态推理 | GraphGPT |


> **Important:** **学习路线建议：**
>
> 1. 掌握基础GNN（GCN、GAT、GraphSAGE）
> 2. 理解消息传递框架和过平滑问题
> 3. 学习图级别任务（池化、图分类）
> 4. 深入一个应用领域（分子/推荐/知识图谱）
> 5. 跟进前沿方向（异构图/动态图/几何GNN）


## 总结


- GNN在社交网络、推荐系统、药物发现、交通预测、程序分析等领域有广泛应用
- 每个领域需要根据数据特性设计图结构和特征
- Cora节点分类是GNN的经典入门实战任务
- 异构图、动态图、超图、几何GNN是当前研究热点
- 图与大语言模型的结合是新兴前沿方向


<!-- Converted from: 04_GNN应用与实战.html -->
