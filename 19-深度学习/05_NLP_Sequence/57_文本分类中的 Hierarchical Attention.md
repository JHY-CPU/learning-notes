# 57_文本分类中的 Hierarchical Attention

## 核心概念

- **层级注意力网络 (Hierarchical Attention Network, HAN)**：由 Yang et al. (2016) 提出，针对文档级文本分类任务。利用文档的天然层次结构（词 -> 句子 -> 文档），在两个层级分别应用注意力机制。
- **词级编码 (Word Encoder)**：使用 Bi-GRU 编码每个句子中的词，然后应用词级注意力（Word Attention）——衡量句子的每个词对理解该句子的重要性。
- **句级编码 (Sentence Encoder)**：将词级注意力输出的句子表示输入另一层 Bi-GRU，然后应用句级注意力（Sentence Attention）——衡量每个句子对整个文档分类的重要性。
- **双层级注意力**：词级注意力让模型关注句子中的关键词语，句级注意力让模型关注文档中的关键句子。两个注意力权重都具有可解释性。
- **上下文向量 (Context Vector)**：与文档内容无关的、可训练的"查询向量"，用于计算词/句子与分类任务的相关性。通过上下文向量与隐藏状态的点积计算注意力分数。
- **层次结构利用**：HAN 利用了文档的自然层次结构，即信息从词级聚合到句级，再从句级聚合到文档级。对比直接将文档展平为词序列的方法，HAN 更符合人类的阅读方式。
- **可解释性**：词级和句级的注意力权重可以可视化，显示模型"认为哪些词和句子对分类最重要"。

## 数学推导

**词级编码与注意力**：
$$
h_{it} = \text{Bi-GRU}(w_{it}), \quad t \in [1, T_i]
$$

$$
u_{it} = \tanh(W_w h_{it} + b_w)
$$

$$
\alpha_{it} = \frac{\exp(u_{it}^\top u_w)}{\sum_t \exp(u_{it}^\top u_w)}
$$

$$
s_i = \sum_t \alpha_{it} h_{it}
$$

其中 $w_{it}$ 是第 $i$ 个句子的第 $t$ 个词，$s_i$ 是词级注意力输出的句子表示，$u_w$ 是词级的上下文向量。

**句级编码与注意力**：
$$
h_i = \text{Bi-GRU}(s_i), \quad i \in [1, L]
$$

$$
u_i = \tanh(W_s h_i + b_s)
$$

$$
\alpha_i = \frac{\exp(u_i^\top u_s)}{\sum_i \exp(u_i^\top u_s)}
$$

$$
v = \sum_i \alpha_i h_i
$$

其中 $v$ 是最终的文档表示，$u_s$ 是句级的上下文向量。

**文档分类**：
$$
P(y|v) = \text{softmax}(W_c v + b_c)
$$

## 直观理解

- **HAN 像"多级摘要"**：读一篇文章时，先理解每个词如何构成句子（词级注意力），再理解每个句子如何构成文章（句级注意力）。最后综合整个文档的信息做出判断。
- **词级注意力的作用**：在"这部电影太精彩了！"中，词级注意力可能会高度关注"精彩"和"太"（表达强烈的正面评价），忽略"这"、"部"等虚词——模型学会了抓住情感关键词。
- **句级注意力的作用**：一篇影评可能有 200 句话，但句级注意力会挑选出"我认为这是今年最好的电影"这样的关键句，而忽略"我周日去看的"这样的背景信息句。
- **上下文向量的作用**：上下文向量 $u_w$ 是"任务相关的查询"——对于情感分类任务，它会自动学会关注情感词；对于主题分类，它会关注关键词。这就像在不同场景下问不同的问题。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class HierarchicalAttentionNetwork(nn.Module):
    """层级注意力网络实现"""
    def __init__(self, vocab_size, embed_size, hidden_size, num_classes):
        super().__init__()
        # 词嵌入
        self.embed = nn.Embedding(vocab_size, embed_size)
        # 词级 GRU + 注意力
        self.word_gru = nn.GRU(embed_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.word_attn = nn.Linear(hidden_size, hidden_size)
        self.word_context = nn.Parameter(torch.randn(hidden_size))
        # 句级 GRU + 注意力
        self.sent_gru = nn.GRU(hidden_size, hidden_size // 2, bidirectional=True, batch_first=True)
        self.sent_attn = nn.Linear(hidden_size, hidden_size)
        self.sent_context = nn.Parameter(torch.randn(hidden_size))
        # 分类器
        self.classifier = nn.Linear(hidden_size, num_classes)

    def _attention(self, h, attn_layer, context):
        # h: (batch, seq, hidden)
        u = torch.tanh(attn_layer(h))          # (batch, seq, hidden)
        scores = torch.matmul(u, context)      # (batch, seq)
        alpha = F.softmax(scores, dim=1)       # (batch, seq)
        out = torch.bmm(alpha.unsqueeze(1), h).squeeze(1)  # (batch, hidden)
        return out, alpha

    def forward(self, x):
        # x: (batch, n_sentences, n_words)
        batch, n_sents, n_words = x.shape

        # 词级处理
        x = x.view(batch * n_sents, n_words)  # (batch*n_sent, n_words)
        emb = self.embed(x)                   # (batch*n_sent, n_words, embed)
        word_out, _ = self.word_gru(emb)      # (batch*n_sent, n_words, hidden)
        sent_vecs, word_weights = self._attention(word_out, self.word_attn, self.word_context)
        sent_vecs = sent_vecs.view(batch, n_sents, -1)  # (batch, n_sents, hidden)

        # 句级处理
        sent_out, _ = self.sent_gru(sent_vecs)  # (batch, n_sents, hidden)
        doc_vec, sent_weights = self._attention(sent_out, self.sent_attn, self.sent_context)

        # 分类
        logits = self.classifier(doc_vec)
        return logits, word_weights, sent_weights

# 使用示例
vocab_size, embed_size, hidden_size, num_classes = 50000, 200, 256, 5
model = HierarchicalAttentionNetwork(vocab_size, embed_size, hidden_size, num_classes)

# 模拟输入: (batch=2, 10句话, 每句20词)
x = torch.randint(0, vocab_size, (2, 10, 20))
logits, word_weights, sent_weights = model(x)

print(f"分类 logits: {logits.shape}")        # (2, 5)
print(f"词级注意力: {word_weights.shape}")   # (20, 20) flatten后的
print(f"句级注意力: {sent_weights.shape}")   # (2, 10)
```

## 深度学习关联

- **层级结构的继承**：HAN 的层级结构设计在 Transformer 时代有新的诠释——多层 Transformer 本身也有层级特征（低层学语法、中层学语义、高层学任务特定特征）。后来的一些工作将 HAN 的层级思想引入了 Transformer 文档分类器中。
- **注意力可解释性的典范**：HAN 是"注意力权重即可解释性"的代表工作之一。通过展示哪些词和句子获得了高注意力权重，模型提供了可解释的分类依据。
- **与 Transformer 的对比**：HAN 使用双层 GRU 处理长文档（可处理数千词），而标准 Transformer 的自注意力 $O(n^2)$ 复杂度对长文档不友好。Hierarchical Transformer（如 Longformer）采用滑动窗口 + 全局注意力的方式也利用了类似的分层思想。
