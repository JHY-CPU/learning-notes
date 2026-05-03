# 10_Bi-directional RNN 双向建模

## 核心概念

- **双向 RNN (Bi-RNN)**：由两个独立的 RNN 组成——前向 RNN 从左到右读取序列，后向 RNN 从右到左读取。每个时间步的最终输出是两个方向的隐藏状态拼接。
- **上下文信息利用**：单向 RNN 只能看到当前时间步之前的信息，而双向 RNN 在每个时间步都可以利用整个序列的上下文。对 NLP 任务至关重要——理解一个词通常需要看它前后两侧的词。
- **前向编码**：$\overrightarrow{h}_t = \text{RNN}_{\text{fwd}}(x_t, \overrightarrow{h}_{t-1})$，按时间正向传播，捕捉从开头到当前位置的信息。
- **后向编码**：$\overleftarrow{h}_t = \text{RNN}_{\text{bwd}}(x_t, \overleftarrow{h}_{t+1})$，按时间逆向传播，捕捉从末尾到当前位置的信息。
- **输出拼接**：$h_t = [\overrightarrow{h}_t; \overleftarrow{h}_t]$ 将两个方向的隐藏状态沿特征维度拼接，维度加倍。
- **局限**：双向模型在序列生成任务（如机器翻译、文本生成）中不能直接使用，因为生成时未来信息不可见——除非使用编码器-解码器框架或掩码机制。

## 数学推导

双向 RNN 在每个时间步 $t$ 的计算：

$$
\overrightarrow{h}_t = \tanh(W_{\overrightarrow{h}\overrightarrow{h}} \overrightarrow{h}_{t-1} + W_{x\overrightarrow{h}} x_t + b_{\overrightarrow{h}})
$$

$$
\overleftarrow{h}_t = \tanh(W_{\overleftarrow{h}\overleftarrow{h}} \overleftarrow{h}_{t+1} + W_{x\overleftarrow{h}} x_t + b_{\overleftarrow{h}})
$$

$$
h_t = \overrightarrow{h}_t \oplus \overleftarrow{h}_t \quad \text{(拼接)}
$$

$$
\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

其中 $\overrightarrow{h}_t, \overleftarrow{h}_t \in \mathbb{R}^{d_h}$，拼接后 $h_t \in \mathbb{R}^{2d_h}$。两个方向的 RNN 参数完全独立，各有自己的权重矩阵。总参数量是单向 RNN 的两倍（不算输出层）。

**梯度传播差异**：前向 RNN 的梯度从右向左传播（常规 BPTT），后向 RNN 的梯度从左向右传播。

## 直观理解

- **双向 RNN 像上下文理解**：读中文句子"他打开了___"，仅看前面不知道空格是什么。再看后面"……银行"，结合上下文可知可能是"账户"而非"门"。双向 RNN 在每个位置上都获得了这种"前后文"视角。
- **前向和后向就像两种阅读习惯**：前向 RNN 像正常阅读（从左到右），后向 RNN 像倒着读（从右到左）。两个"读者"各看各的，最后总结出对每个词最全面的理解。
- **不完全适合生成任务**：就像你不能在写文章时"看到未来的句子"一样，生成任务中未来信息不可用。但阅读理解（作为编码器）时，双向信息是完全合理的。

## 代码示例

```python
import torch
import torch.nn as nn

class BiRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1, num_classes=2):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.birnn = nn.RNN(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True   # 启用双向
        )
        # 双向时输入维度为 2 * hidden_size
        self.fc = nn.Linear(2 * hidden_size, num_classes)

    def forward(self, x):
        # x: (batch, seq_len)
        emb = self.embedding(x)
        # out: (batch, seq_len, 2 * hidden_size)
        out, h_n = self.birnn(emb)

        # 取最后时间步的输出（双向已包含上下文信息）
        # 也可以使用全部序列做池化
        last_out = out[:, -1, :]
        logits = self.fc(last_out)
        return logits

# 使用示例
model = BiRNN(vocab_size=10000, embed_size=128, hidden_size=256, num_classes=2)
x = torch.randint(0, 10000, (4, 20))      # (batch, seq_len)
logits = model(x)
print("分类输出:", logits.shape)           # (4, 2)

# 查看双向隐藏状态维度
birnn = nn.RNN(128, 256, batch_first=True, bidirectional=True)
dummy = torch.randn(2, 10, 128)
out, h_n = birnn(dummy)
print("双向输出维度:", out.shape)          # (2, 10, 512)
print("双向隐藏状态维度:", h_n.shape)      # (2, 2, 256)  [num_layers*2, batch, hidden]
```

## 深度学习关联

- **BERT 的双向基础**：BERT 的"双向"概念直接继承自 Bi-RNN——通过掩码语言模型 (MLM) 同时利用左右上下文信息。区别在于 BERT 使用 Transformer 而非 RNN，因此真正做到了双向自注意力。
- **编码器中的标配**：在 NER、文本分类、序列标注等非生成式任务中，Bi-LSTM/Bi-GRU + CRF 长期作为标准配置，是预训练模型普及前最强大的序列编码器。
- **编码器-解码器中的应用**：在 Seq2Seq 框架中，编码器通常使用双向 RNN 充分理解输入序列，而解码器使用单向 RNN 逐步生成输出。这一范式在 Transformer 中演变为编码器（双向注意力）和解码器（因果注意力）的双向概念分离。
