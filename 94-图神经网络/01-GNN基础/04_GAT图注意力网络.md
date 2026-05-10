# GAT图注意力网络


## 1. 注意力机制在图上的应用


GAT（Graph Attention Network）由 Velickovic 等人在 2018 年提出，将注意力机制引入图神经网络，让每个节点自适应地学习邻居的重要性权重。


### 动机


- GCN 对所有邻居使用相同的归一化权重，无法区分不同邻居的重要性
- 在实际场景中，不同邻居对目标节点的影响程度是不同的
- 注意力机制可以自动学习权重，不需要额外的先验知识


### GAT 的核心优势


| 特性 | 说明 |
| --- | --- |
| 自适应权重 | 通过注意力机制动态学习邻居权重 |
| 不需要拉普拉斯矩阵 | 不需要特征分解或矩阵求逆 |
| 可处理归纳任务 | 注意力函数可迁移到未见过的图 |
| 隐式多关系建模 | 多头注意力可捕获不同关系模式 |


## 2. 注意力系数计算


### 步骤1：计算注意力分数


$$
eij = LeakyReLU(aT [W hi || W hj])
$$


其中：


- **W**
   ∈ R
   ^F'×F^
   ：共享的线性变换矩阵
- **h~i~, h~j~**
   ：节点 i 和 j 的特征向量
- **||**
   ：拼接操作
- **a**
   ∈ R
   ^2F'^
   ：注意力向量（可学习参数）
- **LeakyReLU**
   ：带泄漏的ReLU，负半轴斜率为0.2


### 步骤2：Softmax 归一化


$$
αij = softmaxj(eij) = exp(eij) / Σk∈N(i) exp(eik)
$$


只对节点 i 的一阶邻居 N(i) 做归一化。


### 步骤3：加权聚合


$$
hi' = σ(Σj∈N(i) αij W hj)
$$


> **Example:** #### 注意力分数计算示例
>
>
> 假设节点 i 有3个邻居 j1, j2, j3：
>
>
> 1. 计算 e
>    ~i,j1~
>    , e
>    ~i,j2~
>    , e
>    ~i,j3~
> 2. Softmax 得到 α
>    ~i,j1~
>    =0.5, α
>    ~i,j2~
>    =0.3, α
>    ~i,j3~
>    =0.2
> 3. 聚合：h
>    ~i~
>    ' = σ(0.5Wh
>    ~j1~
>    + 0.3Wh
>    ~j2~
>    + 0.2Wh
>    ~j3~
>    )


## 3. 多头注意力（Multi-head Attention）


类似Transformer，GAT也使用多头注意力来增强模型的表达能力。每个注意力头独立学习不同的注意力模式。


### 两种多头合并方式


#### 方式1：拼接（Concatenation）—— 中间层


$$
hi' = ||k=1K σ(Σj∈N(i) αijk Wk hj)
$$


输出维度 = K × F'，每个头输出 F' 维，拼接后为 KF' 维。


#### 方式2：平均（Averaging）—— 最终层


$$
hi' = σ((1/K) Σk=1K Σj∈N(i) αijk Wk hj)
$$


输出维度 = F'，所有头取平均后通过激活函数。


| 多头方式 | 适用场景 | 输出维度 | 特点 |
| --- | --- | --- | --- |
| 拼接 | 中间层 | K × F' | 保留每个头的独立信息 |
| 平均 | 最终输出层 | F' | 平滑多个头的输出 |


> **Note:** **实践经验：**
> GAT论文中使用 K=8 个注意力头，中间层使用拼接，最终层使用平均。每个头的隐藏维度 F'=8 或 16。


## 4. GAT vs GCN 对比


| 特性 | GCN | GAT |
| --- | --- | --- |
| 权重计算 | 固定（基于度归一化） | 动态（基于注意力） |
| 权重来源 | √(d~i~·d~j~) | 学习得到的注意力系数 |
| 计算复杂度 | O(\|E\|·F') | O(\|E\|·F' + \|V\|·F'^2^) |
| 参数量 | O(F·F') | O(F·F' + 2F') |
| 注意力头 | 无 | 支持多头注意力 |
| 图结构信息 | 显式使用度信息 | 隐式学习 |
| 可迁移性 | 转导式 | 归纳式（注意力函数可迁移） |


### 计算复杂度分析


- GAT 的注意力计算需要对每条边计算 e
   ~ij~
   ：O(|E|·F')
- 线性变换 Wx 对每个节点计算：O(|V|·F·F')
- 对于稀疏图 |E| << |V|
   ^2^
   ，GAT 的复杂度与 GCN 同阶
- 多头注意力使计算量增加 K 倍，但每个头更小


> **Important:** **选择建议：**
>
> - 当邻居重要性不均匀时，优先使用 GAT
> - 当图结构规则且邻居重要性相似时，GCN 足够且更快
> - GAT 在异配图（heterophilic graph）上通常表现更好


## 5. PyTorch 代码实现 GAT 层


```
import torch
import torch.nn as nn
import torch.nn.functional as F

class GATLayer(nn.Module):
    """手写GAT注意力层"""
    def __init__(self, in_features, out_features, n_heads=4, dropout=0.6, alpha=0.2):
        super().__init__()
        self.n_heads = n_heads
        self.out_features = out_features
        self.dropout = dropout

        # 线性变换
        self.W = nn.Linear(in_features, out_features * n_heads, bias=False)
        # 注意力向量 a = [a_left || a_right]
        self.a_left = nn.Parameter(torch.zeros(n_heads, out_features))
        self.a_right = nn.Parameter(torch.zeros(n_heads, out_features))
        self.leaky_relu = nn.LeakyReLU(alpha)

        nn.init.xavier_uniform_(self.W.weight)
        nn.init.xavier_uniform_(self.a_left)
        nn.init.xavier_uniform_(self.a_right)

    def forward(self, h, edge_index):
        """
        h: [N, in_features]
        edge_index: [2, E]
        """
        N = h.size(0)
        E = edge_index.size(1)

        # 线性变换: [N, n_heads, out_features]
        Wh = self.W(h).view(N, self.n_heads, self.out_features)

        src, dst = edge_index[0], edge_index[1]

        # 计算注意力分数
        # e_ij = LeakyReLU(a_left * Wh_i + a_right * Wh_j)
        e_left = (Wh[dst] * self.a_left).sum(dim=-1)   # [E, n_heads]
        e_right = (Wh[src] * self.a_right).sum(dim=-1)  # [E, n_heads]
        e = self.leaky_relu(e_left + e_right)           # [E, n_heads]

        # Softmax归一化（按目标节点分组）
        attention = torch.zeros(N, self.n_heads, device=h.device)
        attention.index_add_(0, dst, torch.exp(e))
        alpha = torch.exp(e) / (attention[dst] + 1e-8)  # [E, n_heads]
        alpha = F.dropout(alpha, self.dropout, training=self.training)

        # 加权聚合
        out = torch.zeros(N, self.n_heads, self.out_features, device=h.device)
        weighted = Wh[src] * alpha.unsqueeze(-1)  # [E, n_heads, out_features]
        out.index_add_(0, dst, weighted)

        # 拼接多头
        out = out.view(N, self.n_heads * self.out_features)
        return F.elu(out)


# ========== 使用PyG的GAT ==========
from torch_geometric.nn import GATConv

class GAT(torch.nn.Module):
    def __init__(self, num_features, hidden_dim, num_classes, heads=8):
        super().__init__()
        self.conv1 = GATConv(num_features, hidden_dim, heads=heads, dropout=0.6)
        # 第二层使用平均，输出维度 = hidden_dim
        self.conv2 = GATConv(hidden_dim * heads, num_classes, heads=1,
                             concat=False, dropout=0.6)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.6, training=self.training)
        x = F.elu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.6, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# 测试
model = GAT(num_features=1433, hidden_dim=8, num_classes=7, heads=8)
print(f"GAT模型参数量: {sum(p.numel() for p in model.parameters())}")
```


## 6. GAT 的变体与扩展


| 变体 | 改进点 | 核心思想 |
| --- | --- | --- |
| GATv2 | 注意力的动态性 | 将注意力向量放在LeakyReLU之后，使注意力真正动态 |
| SuperGAT | 自监督注意力 | 利用图的自监督信号改进注意力学习 |
| Graph Transformer | 全局注意力 | 引入位置编码，对所有节点对计算注意力 |
| Exphormer | 高效全局注意力 | 使用expander图减少注意力复杂度 |


> **Note:** **GATv2 的改进：**
> 原始GAT的注意力权重在学习后是静态的（与查询节点无关），GATv2通过改变计算顺序使注意力动态化：
>
>
> 原始GAT: e
> ~ij~
> = LeakyReLU(a
> ^T^
> [Wh
> ~i~
> ||Wh
> ~j~
> ])
>
>
> GATv2: e
> ~ij~
> = a
> ^T^
> LeakyReLU(W[h
> ~i~
> ||h
> ~j~
> ])


## 总结


- GAT通过注意力机制为邻居节点分配自适应权重，解决了GCN固定权重的问题
- 注意力系数 α
   ~ij~
   = softmax(LeakyReLU(a
   ^T^
   [Wh
   ~i~
   ||Wh
   ~j~
   ]))
- 多头注意力提供更丰富的表示：中间层拼接，最终层平均
- GAT比GCN更灵活但计算开销略大
- GATv2等改进版本解决了原始GAT注意力静态化的问题


<!-- Converted from: 04_GAT图注意力网络.html -->
