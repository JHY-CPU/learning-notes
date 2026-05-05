# 06_RNN 反向传播与 BPTT 算法推导

## 核心概念

- **BPTT (Backpropagation Through Time)**：RNN 的标准训练算法，本质是将 RNN 按时间展开为深层前馈网络后应用标准反向传播。误差信号从最后时间步逐层反向传播到最初时间步。
- **时间展开**：将 RNN 在时间维度上展开为 $T$ 层的前馈网络，每层共享相同权重。这使得标准的链式法则可以应用于循环结构。
- **梯度累积**：损失对参数的梯度是所有时间步的梯度之和：$\frac{\partial \mathcal{L}}{\partial W} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W}$。因为每个时间步的输出都依赖于同一参数矩阵。
- **梯度消失与爆炸**：BPTT 中梯度包含连乘项 $\prod_{k=1}^{t} \frac{\partial h_k}{\partial h_{k-1}}$，其中包含多个 Jacobian 矩阵相乘。若最大特征值 < 1，梯度指数级衰减（消失）；若 > 1，指数级增长（爆炸）。
- **Truncated BPTT**：为减少计算开销，将长序列截断为固定长度子序列进行 BPTT，限制反向传播的时间步数。在实践中广泛使用。
- **梯度裁剪 (Gradient Clipping)**：将梯度范数限制在一个阈值内，防止梯度爆炸。是训练 RNN 的关键技巧。

## 数学推导

考虑简单 RNN 在第 $t$ 步的隐藏状态 $h_t = \tanh(W_{hh}h_{t-1} + W_{xh}x_t + b_h)$，总损失为 $\mathcal{L} = \sum_{t=1}^{T} \mathcal{L}_t$。

对 $W_{hh}$ 的梯度需要沿时间反向传播：
$$
\frac{\partial \mathcal{L}}{\partial W_{hh}} = \sum_{t=1}^{T} \frac{\partial \mathcal{L}_t}{\partial W_{hh}} = \sum_{t=1}^{T} \sum_{k=1}^{t} \frac{\partial \mathcal{L}_t}{\partial h_t} \frac{\partial h_t}{\partial h_k} \frac{\partial h_k}{\partial W_{hh}}
$$

其中关键的链式部分：
$$
\frac{\partial h_t}{\partial h_k} = \prod_{i=k+1}^{t} \frac{\partial h_i}{\partial h_{i-1}} = \prod_{i=k+1}^{t} \text{diag}(\tanh'(W_{hh}h_{i-1} + W_{xh}x_i)) \cdot W_{hh}
$$

令 $\gamma = \|W_{hh}^\top \text{diag}(\tanh'(\cdot))\|$，则当 $t-k$ 大时：
- $\gamma < 1$：$\frac{\partial h_t}{\partial h_k} \to 0$（梯度消失）
- $\gamma > 1$：$\frac{\partial h_t}{\partial h_k} \to \infty$（梯度爆炸）

## 直观理解

- **BPTT 像多米诺骨牌反向追溯**：想象一排多米诺骨牌（时间步），你要找出推倒第 N 张牌的力来自于哪张牌。BPTT 就是反过来逐张追查，越往前推的牌反而影响越小（梯度消失）。
- **梯度消失像长距离消息衰减**：在传话游戏中，从第一个人传到第十个人时消息已严重失真。RNN 中距离遥远的词对当前预测的影响也以指数级衰减。
- **梯度爆炸像情绪失控**：微小的不满在传播中被放大，最终导致爆发。类似地，小梯度在连乘中被指数放大，导致参数更新过大、训练不稳定。

## 代码示例

```python
import torch
import torch.nn as nn

# 使用 Truncated BPTT 训练的简单语言模型
class RNNLM(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.rnn = nn.RNN(embed_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, vocab_size)

    def forward(self, x, hidden=None):
        emb = self.embed(x)
        out, hidden = self.rnn(emb, hidden)
        logits = self.fc(out)
        return logits, hidden

model = RNNLM(vocab_size=10000, embed_size=128, hidden_size=256)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

# 模拟训练循环 —— Truncated BPTT
seq_len = 35
for epoch in range(3):
    hidden = None
    for batch in range(100):  # 假设有 100 个 batch
        x = torch.randint(0, 10000, (4, seq_len))   # (batch, seq_len)
        y = torch.randint(0, 10000, (4, seq_len))

        logits, hidden = model(x, hidden)
        # 分离隐藏状态计算图（Truncated BPTT 的关键）
        hidden = hidden.detach()

        loss = nn.CrossEntropyLoss()(logits.view(-1, 10000), y.view(-1))

        optimizer.zero_grad()
        loss.backward()
        # 梯度裁剪——防止梯度爆炸
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}")
```

## 深度学习关联

- **梯度消失的理论基石**：BPTT 中发现的梯度消失问题直接推动了 LSTM 和 GRU 的发明——通过门控机制提供梯度高速公路。
- **Transformer 的革新**：BPTT 的串行计算限制（必须按时间依次计算）是 RNN 训练效率低下的根源，Transformer 的并行自注意力彻底解决了这一问题。
- **残差连接与归一化**：BPTT 中连乘效应的教训启示了现代深度网络中使用残差连接（ResNet）和层归一化（LayerNorm）来保证梯度健康流动。
