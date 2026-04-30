# 51_依存句法分析 (Dependency Parsing)

## 核心概念
- **依存句法分析 (Dependency Parsing)**：分析句子中词与词之间的依存关系。每个词（依赖词）与它的"父词"（head）之间存在一条有向弧，标有依存关系类型（如主谓关系、动宾关系等）。
- **依存关系树**：句子形成一个以谓语动词为根的树形结构。每个词有且只有一个父节点（根节点的父节点为空），无环。这棵树描述了句子的句法结构。
- **Universal Dependencies (UD)**：统一的依存关系标注体系，定义了约 37 种依存关系类型（nsubj、obj、amod、nmod 等），跨语言适用。
- **Transition-based Parsing**：基于状态转移的贪心解析方法。使用栈（Stack）和缓冲区（Buffer）维护解析状态，通过一系列转移动作（Shift、Left-Arc、Right-Arc）逐步构建依存树。
- **Graph-based Parsing**：基于图的解析方法。计算所有可能的依存弧的得分，然后使用最大生成树算法（如 Eisner 算法 / Chu-Liu-Edmonds）寻找得分最高的依存树。
- **深度学习方法**：使用 Bi-LSTM 或 Transformer 编码句子，为每对可能的（head, dependent）计算依存弧存在概率。基于图的深层双仿射注意力（Deep Biaffine Attention）是当前 SOTA 方法。
- **评估指标**：UAS (Unlabeled Attachment Score)—不考虑关系类型时头词预测正确的比例；LAS (Labeled Attachment Score)—考虑关系类型时头词和关系都正确的比例。

## 数学推导
**基于图的解析**：计算每个可能的依存弧 $(h, m)$ 的得分 $s(h, m)$。

双仿射注意力（Dozat & Manning, 2017）：
$$
s(h, m) = r_h^\top U r_m + W_h r_h + W_m r_m + b
$$

其中 $r_h, r_m \in \mathbb{R}^d$ 是头词和依赖词的 Bi-LSTM/Transformer 表示。

**最大生成树（MST）**：在完全有向图中找到以根为根节点的最大生成树：
$$
T^* = \arg\max_{T \in \mathcal{T}} \sum_{(h,m) \in T} s(h, m)
$$

可以使用 Chu-Liu-Edmonds 算法在 $O(n^2)$ 时间内精确求解。

**Transition-based 解析的损失**：
使用交叉熵损失，预测每一步的正确动作（SHIFT、LEFT-ARC、RIGHT-ARC）：
$$
\mathcal{L} = -\sum_{t} \log P(\text{action}_t | \text{state}_t)
$$

## 直观理解
- **依存句法分析像"句子用词关系图"**：句子"猫追老鼠"中，"追"是根（核心），"猫"是"追"的主语（nsubj），"老鼠"是"追"的宾语（obj）。依存句法就是找出"谁依赖谁"的关系网络。
- **Transition-based 像"贪心拼图"**：从左到右扫描句子，每次决定是将当前词压栈还是与栈顶词建立关系。就像拼拼图——每次只决定当前小块应该和已有的哪块连接。速度快但可能因早期错误导致后续全部错误。
- **Graph-based 像"全局优化"**：计算所有可能的连接得分，然后全局选择得分最高的组合。就像拍照时把所有可能的姿势都考虑一遍，然后选最好的。精度高但计算成本也高。
- **双仿射注意力**：它不仅仅问"两个词是否可能相关"，还问"如果一个词是头，另一个是依赖词，它们以这种特定关系相关的概率是多少"。这就像在确定"朋友关系"时，同时确定"谁对谁友善"。

## 代码示例
```python
import torch
import torch.nn as nn

class BiaffineDependencyParser(nn.Module):
    """简化的双仿射依存句法分析器"""
    def __init__(self, d_model, n_relations):
        super().__init__()
        # 头词和依赖词的 MLP
        self.head_mlp = nn.Linear(d_model, d_model)
        self.dep_mlp = nn.Linear(d_model, d_model)
        # 双仿射变换（弧得分）
        self.U_arc = nn.Linear(d_model, d_model, bias=False)
        # 双仿射变换（关系得分）
        self.U_rel = nn.Parameter(torch.randn(n_relations, d_model, d_model) * 0.01)

    def forward(self, hidden_states):
        # hidden_states: (batch, seq, d_model)
        heads = self.head_mlp(hidden_states)
        deps = self.dep_mlp(hidden_states)

        # 弧得分 (batch, seq, seq)
        arc_scores = torch.bmm(heads, self.U_arc(deps).transpose(1, 2))

        # 关系得分 (batch, seq, seq, n_relations)
        rel_scores = torch.einsum('bhi,rij,bhj->bhij', heads, self.U_rel, deps)

        return arc_scores, rel_scores

# 使用示例
batch, seq, d_model = 2, 10, 512
n_relations = 47  # UD 的依存关系数量

parser = BiaffineDependencyParser(d_model, n_relations)
hidden = torch.randn(batch, seq, d_model)
arc_scores, rel_scores = parser(hidden)

print(f"弧得分矩阵: {arc_scores.shape}")    # (2, 10, 10)
print(f"关系得分张量: {rel_scores.shape}")  # (2, 10, 10, 47)

# 依存句法可视化示例
sentence = "猫 追 老鼠"
# 解析结果: 追->猫 (nsubj), 追->老鼠 (obj)
dependency_tree = {
    "追": {"head": None, "dep": "root"},
    "猫": {"head": "追", "dep": "nsubj"},
    "老鼠": {"head": "追", "dep": "obj"},
}
print(f"\n依存句法树:")
for word, info in dependency_tree.items():
    print(f"  {word} <- {info['head']} ({info['dep']})")
```

## 深度学习关联
- **预训练模型提升解析精度**：使用 BERT 等预训练语言模型的编码表示替代 Bi-LSTM，可以将 LAS 提升 3-5 个百分点。当前 SOTA 依存句法分析器几乎都基于预训练模型。
- **句法信息的隐式学习**：虽然现代 NER 和关系抽取直接使用预训练模型而不显式使用句法分析，但研究表明 Transformer 的多头注意力隐式编码了句法结构信息——低层头学习表层结构，高层头学习语义关系。
- **多语言解析的一致性**：Universal Dependencies 推动了跨语言依存句法分析的发展，mBERT 等跨语言模型的出现使得一个模型可以在多种语言上进行解析。
