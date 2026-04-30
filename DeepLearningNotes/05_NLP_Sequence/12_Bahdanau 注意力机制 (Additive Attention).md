# 12_Bahdanau 注意力机制 (Additive Attention)

## 核心概念
- **注意力机制的核心思想**：让解码器在生成每个词时，动态地关注编码器输出的不同位置，而非依赖单一的上下文向量。每个目标词有自己的"注意力焦点"。
- **Bahdanau Attention (Additive Attention)**：由 Bahdanau 等人于 2014 年提出，是最早应用于 Seq2Seq 的注意力机制。使用一个小型前馈网络计算注意力分数。
- **对齐分数 (Alignment Score)**：$e_{ij} = v_a^\top \tanh(W_a[s_{i-1}; h_j])$，衡量解码器状态 $s_{i-1}$ 与编码器状态 $h_j$ 的匹配度。名称"additive"来源于将两个向量的变换结果相加。
- **注意力权重**：通过对齐分数做 softmax 归一化得到 $\alpha_{ij}$，表示生成第 $i$ 个目标词时应该关注第 $j$ 个源词的程度。
- **上下文向量**：对所有编码器状态按注意力权重加权求和：$c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j$，作为解码器第 $i$ 步的额外输入。
- **与 Seq2Seq 的整合**：注意力上下文向量 $c_i$ 与当前解码器状态 $s_i$ 拼接，共同预测目标词。与后续纯注意力架构的区别是它仍然使用 RNN 作为主干。

## 数学推导
Bahdanau 注意力计算过程：

$$
e_{ij} = v_a^\top \tanh(W_a [s_{i-1}; h_j]), \quad j = 1, \ldots, T_x
$$

$$
\alpha_{ij} = \frac{\exp(e_{ij})}{\sum_{k=1}^{T_x} \exp(e_{ik})}
$$

$$
c_i = \sum_{j=1}^{T_x} \alpha_{ij} h_j
$$

其中 $W_a \in \mathbb{R}^{d_a \times 2d_h}$（或将 $s_{i-1}$ 和 $h_j$ 分别变换后相加：$W_1 s_{i-1} + W_2 h_j$），$v_a \in \mathbb{R}^{d_a}$ 是注意力参数。

解码器状态更新变为：
$$
s_i = \tanh(W_{ss} s_{i-1} + W_{sy} y_{i-1} + W_{sc} c_i)
$$

$$
P(y_i | y_{<i}, c_i) = \text{softmax}(W_{hy} [s_i; c_i])
$$

## 直观理解
- **Bahdanau Attention 像"翻阅参考书"**：当你写一句话时，不再只靠记忆，而是随时回头看原文。比如翻译"我吃了苹果"到英文时，生成"apple"时会重点关注"苹果"这个词的位置。
- **Additive 名称的由来**：$W_1 s_{i-1} + W_2 h_j$ 先将两个向量分别线性变换到同一空间再相加，就像"把两个不同维度的问题转换到同一个坐标系中比较"。加性注意力可以捕捉到 Query 和 Key 之间的复杂非线性关系。
- **注意力可视化**：如果把注意力权重 $\alpha_{ij}$ 画成热力图，你会看到一个"对角线带状图案"——源语言和目标语言的对应词大致按顺序对齐，但允许一定的偏移（后文先译等）。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class BahdanauAttention(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.W_a = nn.Linear(hidden_size * 2, hidden_size)  # 拼接后变换
        self.v_a = nn.Linear(hidden_size, 1)

    def forward(self, query, values):
        # query: (batch, hidden) — 解码器当前状态 s_{i-1}
        # values: (batch, seq_len, hidden) — 编码器所有隐藏状态 H
        batch_size, seq_len, _ = values.shape

        # 扩展 query 到每个时间步
        query = query.unsqueeze(1).repeat(1, seq_len, 1)  # (batch, seq, hidden)

        # 计算对齐分数
        energy = torch.tanh(self.W_a(torch.cat([query, values], dim=2)))
        # energy: (batch, seq, hidden)
        e = self.v_a(energy).squeeze(2)                    # (batch, seq)

        # 注意力权重
        alpha = F.softmax(e, dim=1)                        # (batch, seq)

        # 上下文向量
        context = torch.bmm(alpha.unsqueeze(1), values)   # (batch, 1, hidden)
        return context.squeeze(1), alpha                   # (batch, hidden), (batch, seq)

# 使用示例
attn = BahdanauAttention(hidden_size=256)
query = torch.randn(4, 256)          # 解码器状态
values = torch.randn(4, 20, 256)     # 编码器所有隐藏状态
context, alpha = attn(query, values)
print("上下文向量:", context.shape)   # (4, 256)
print("注意力权重:", alpha.shape)     # (4, 20)
```

## 深度学习关联
- **注意力机制的开端**：Bahdanau 注意力是"将注意力引入深度学习"的标志性工作，从此注意力成为 NLP 中最核心的组件之一。
- **与 Transformer 的演进关系**：Additive Attention 在 Transformer 中被 Scaled Dot-Product Attention 取代，因为后者计算效率更高（可用矩阵乘法优化），但本质都是计算 Query 和 Key 的匹配度。
- **注意力可解释性**：Bahdanau 注意力的权重可视化提供了模型决策的可解释性——你可以直观地看到模型翻译每个词时"看"了原文的哪些位置，这在机器翻译质量分析和模型调试中仍然有用。
