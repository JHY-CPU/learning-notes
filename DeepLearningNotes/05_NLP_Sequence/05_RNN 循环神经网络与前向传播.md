# 05_RNN 循环神经网络与前向传播

## 核心概念
- **时间步概念**：RNN 在时间轴上展开，每个时间步 $t$ 对应序列中的一个元素。隐藏状态 $h_t$ 携带了截止到当前步的历史信息。
- **共享参数**：RNN 在所有时间步使用相同的权重矩阵 $W_{hh}$、$W_{xh}$ 和 $W_{hy}$，使模型可以处理任意长度的序列，同时大幅减少参数量。
- **隐状态更新公式**：当前隐藏状态由上一时刻隐藏状态和当前输入共同决定：$h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$。
- **前向传播流程**：按时间顺序从 $t=1$ 到 $t=T$ 依次计算 $h_t$ 和输出 $y_t$（或 $\hat{y}_t$），每一步的输出可以独立计算损失。
- **多输出模式**：RNN 支持多种输入输出模式——一对一（标准分类）、一对多（图像描述）、多对一（情感分析）、多对多（机器翻译）。
- **词嵌入输入**：实际应用中，$x_t$ 通常是词嵌入向量而非 one-hot 向量，通过嵌入层从离散 token 映射到稠密向量。

## 数学推导
RNN 前向传播公式：

$$
h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)
$$

$$
\hat{y}_t = \text{softmax}(W_{hy} h_t + b_y)
$$

其中 $h_t \in \mathbb{R}^d$ 是 $d$ 维隐藏状态，$x_t \in \mathbb{R}^{d_{\text{in}}}$ 是输入向量，$W_{hh} \in \mathbb{R}^{d\times d}$，$W_{xh} \in \mathbb{R}^{d\times d_{\text{in}}}$，$W_{hy} \in \mathbb{R}^{|V|\times d}$。

损失函数通常为交叉熵（对每一步取平均）：
$$
\mathcal{L} = \frac{1}{T}\sum_{t=1}^{T} -\log P(y_t | x_1,\ldots,x_t) = \frac{1}{T}\sum_{t=1}^{T} -\log \hat{y}_{t}[y_t]
$$

## 直观理解
- **RNN 像带记忆的阅读**：你在读小说时，当前的理解（隐藏状态 $h_t$）是你之前读过的内容（$h_{t-1}$）和当前句子（$x_t$）的综合结果。读得越多，记忆积累越丰富。
- **参数共享就像同一个大脑**：无论你读《红楼梦》第 1 章还是第 120 章，使用的都是同一个大脑——同一套参数在所有时间步上共享。
- **展开图**：将 RNN 按时间展开后，它变成了一个深层的"前馈网络"，层数等于序列长度。这就是为什么长序列会导致梯度问题。

## 代码示例
```python
import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers=1):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(
            input_size=embed_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        # x: (batch_size, seq_len)
        emb = self.embedding(x)                     # (batch, seq, embed)
        out, h_n = self.rnn(emb, hidden)            # out: (batch, seq, hidden)
        logits = self.fc(out)                       # (batch, seq, vocab)
        return logits, h_n

# 使用示例
model = SimpleRNN(vocab_size=10000, embed_size=128, hidden_size=256)
x = torch.randint(0, 10000, (2, 10))               # batch=2, seq_len=10
logits, h_n = model(x)
print("输出形状:", logits.shape)                    # (2, 10, 10000)
print("最终隐藏状态:", h_n.shape)                    # (1, 2, 256)
```

## 深度学习关联
- **Transformer 的前身**：RNN 是处理序列数据的奠基性架构，其在解决长程依赖上的不足直接催生了 LSTM、GRU 等门控机制，最终推动了 Transformer 的诞生。
- **序列建模基础**：几乎所有现代序列模型（包括 RNN-T、LAS 等语音识别模型）都建立在 RNN 前向传播的基本框架之上。
- **梯度问题的历史经验**：RNN 训练中发现的梯度消失/爆炸问题为深度学习中的梯度传播理论提供了重要案例，也促进了残差连接、LayerNorm 等技术的广泛应用。
