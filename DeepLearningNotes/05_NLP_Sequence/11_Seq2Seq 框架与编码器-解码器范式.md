# 11_Seq2Seq 框架与编码器-解码器范式

## 核心概念
- **编码器-解码器 (Encoder-Decoder) 架构**：由两个 RNN 组成的序列到序列 (Seq2Seq) 框架。编码器将输入序列编码为固定长度的上下文向量（context vector），解码器基于此向量生成目标序列。
- **上下文向量瓶颈**：编码器的最终隐藏状态 $h_T$（或最后几步的某种聚合）作为唯一的信息通道传递给解码器。当输入序列很长时，单一向量难以编码全部信息——这是后期注意力机制要解决的问题。
- **解码器的条件生成**：解码器在每个时间步基于上下文向量 $c$、上一时刻输出 $y_{t-1}$ 和隐藏状态 $s_{t-1}$ 生成当前词 $y_t$。训练时使用"教师强制"(Teacher Forcing)。
- **Teacher Forcing**：训练时解码器的输入使用真实的目标序列而非模型自己的预测，加速收敛。但会导致"暴露偏差"(Exposure Bias)——训练和推理时的输入分布不一致。
- **EOS/SOS Token**：使用特殊标记指定序列边界。解码器以 <SOS> 开始生成，遇到 <EOS> 停止。通常还会加入 <PAD> 对齐批次内不同长度的序列。
- **原始 Seq2Seq 应用**：最早由 Sutskever 等人 (2014) 和 Cho 等人 (2014) 提出，主要用于机器翻译。编码器读取源语言句子，解码器生成目标语言句子。

## 数学推导
编码器（以 RNN 为例）：
$$
h_t = \tanh(W_{hh}^{(enc)} h_{t-1} + W_{xh}^{(enc)} x_t + b_h)
$$

$$
c = h_T \quad \text{(或 Bi-RNN 的拼接)}
$$

解码器：
$$
s_t = \tanh(W_{hh}^{(dec)} s_{t-1} + W_{ys}^{(dec)} y_{t-1} + W_{cs} c + b_s)
$$

$$
P(y_t | y_{<t}, c) = \text{softmax}(W_{hy}^{(dec)} s_t + b_y)
$$

总损失（交叉熵）：
$$
\mathcal{L} = -\frac{1}{T_y}\sum_{t=1}^{T_y} \log P(y_t | y_{<t}, c)
$$

## 直观理解
- **编码器-解码器像翻译**：编码器就像读一本英文书，理解后将内容压缩成"摘要笔记"（上下文向量），解码器基于这个"笔记"用法语复述出来。
- **上下文向量瓶颈像记忆有限**：想象你读一篇长文章然后必须凭记忆转述——开头和中间的内容很容易忘记。这就是上下文向量作为信息瓶颈的局限，也是注意力机制要解决的问题。
- **Teacher Forcing 像有导师指导**：训练时每一步都有导师告诉你"正确答案是什么"，而不是让你自己猜。这样学得快，但考试（推理）时没有导师就容易出问题。

## 代码示例
```python
import torch
import torch.nn as nn

class Encoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)

    def forward(self, x):
        emb = self.embedding(x)           # (batch, seq, embed)
        _, hidden = self.rnn(emb)         # hidden: (1, batch, hidden)
        return hidden

class Decoder(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.GRU(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden):
        emb = self.embedding(x)           # (batch, seq, embed)
        output, hidden = self.rnn(emb, hidden)
        logits = self.fc(output)          # (batch, seq, vocab)
        return logits, hidden

class Seq2Seq(nn.Module):
    def __init__(self, encoder, decoder):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, src, tgt):
        hidden = self.encoder(src)
        output, _ = self.decoder(tgt, hidden)
        return output

# 示例
enc = Encoder(vocab_size=10000, embed_size=128, hidden_size=256)
dec = Decoder(vocab_size=10000, embed_size=128, hidden_size=256)
model = Seq2Seq(enc, dec)

src = torch.randint(0, 10000, (4, 10))   # (batch, src_seq)
tgt = torch.randint(0, 10000, (4, 12))   # (batch, tgt_seq)
output = model(src, tgt)
print("Seq2Seq 输出:", output.shape)      # (4, 12, 10000)
```

## 深度学习关联
- **注意力机制的前奏**：Seq2Seq 的上下文向量瓶颈是注意力机制被提出的直接动机——Bahdanau 等人发现让解码器"回看"编码器的所有隐藏状态而非仅最后一个，能大幅提升长序列翻译质量。
- **Transformer 的编码器-解码器**：Transformer 继承了这一范式，但将 RNN 替换为自注意力和交叉注意力层，并解决了并行计算问题。
- **预训练时代的演变**：现代预训练模型（T5、BART）仍然使用编码器-解码器架构，但编码器和解码器都由 Transformer 层堆叠而成，并在大规模语料上预训练后微调。
