# 41 梯度裁剪 (Gradient Clipping) 防止爆炸

## 核心概念

- **梯度裁剪的定义**：梯度裁剪（Gradient Clipping）是一种防止梯度爆炸的技术。当梯度的范数超过预设阈值时，将其缩放到阈值范围内。最常见的是按范数裁剪：如果 $\|g\| > c$，则 $g \leftarrow c \cdot g/\|g\|$。

- **梯度爆炸的原因**：在深度网络（特别是 RNN）中，梯度在反向传播时经过多层的链式乘积。如果矩阵的谱范数大于 1，梯度呈指数增长，导致参数更新过大，训练发散。

- **裁剪阈值 $c$**：阈值 $c$ 是超参数，通常设置为 1.0、5.0 或 10.0。太大起不到防止爆炸的作用，太小会扭曲梯度方向影响训练。实践中常用 1.0 或 5.0 作为初始值。

- **按值裁剪 vs 按范数裁剪**：按值裁剪（clip by value）将每个梯度元素限制在 $[-c, c]$ 内；按范数裁剪（clip by norm）保持梯度方向不变，整体缩放。按范数裁剪更常用，因为它保留了梯度方向。

## 数学推导

**按范数裁剪**：

$$
g_{\text{clipped}} = \begin{cases}
g & \text{if } \|g\|_2 \leq c \\
\frac{c}{\|g\|_2} \cdot g & \text{if } \|g\|_2 > c
\end{cases}
$$

其中 $c$ 是裁剪阈值，$\|g\|_2$ 是梯度的 L2 范数。

这种裁剪保持了梯度方向不变（当范数超过阈值时，所有分量等比例缩放），只改变步长大小。

**按值裁剪**：

$$
(g_{\text{clipped}})_i = \begin{cases}
c & \text{if } g_i > c \\
-c & \text{if } g_i < -c \\
g_i & \text{otherwise}
\end{cases}
$$

按值裁剪改变了梯度方向，但保证了每个梯度元素的绝对大小不超过 $c$。

**全局范数裁剪 vs 逐层裁剪**：

全局范数裁剪（最常用）：计算所有参数梯度的整体范数：

$$
\|g\|_2 = \sqrt{\sum_{l} \|g^{(l)}\|_2^2}
$$

逐层裁剪：每层独立裁剪，可能破坏不同层之间的梯度比例。

**裁剪对梯度统计的影响**：

裁剪后梯度的期望值有偏。对于大梯度，$E[g_{\text{clip}}] < E[g]$。这意味着裁剪引入了偏差（bias），但避免了方差爆炸。

## 直观理解

梯度裁剪就像"给发动机安装限速器"——当梯度过大（发动机转速过高）时，限速器（裁剪）自动降低速度，防止发动机损坏（训练发散）。裁剪不会改变行驶方向（梯度方向），只控制速度（步长）。

在 RNN 训练中，梯度爆炸通常发生在处理长序列时。想象一下滚雪球——雪球（梯度）从山顶滚下，越滚越大，最终可能变成雪崩（梯度爆炸）。梯度裁剪就像在雪球的路径上设置了一系列"减速带"，防止它变得失控。

裁剪阈值 $c$ 的选择是一个权衡：$c$ 太小，梯度被过度限制，训练变慢（类似于开车一直限速 20km/h）；$c$ 太大，起不到防止爆炸的作用（类似于限速 200km/h，但车子只能跑 100km/h）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 梯度裁剪演示
torch.manual_seed(42)

# 模拟一个 RNN 训练场景（梯度容易爆炸）
class SimpleRNN(nn.Module):
    def __init__(self, input_size=10, hidden_size=64, num_layers=3):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.rnn(x)
        return self.fc(out[:, -1, :])

rnn = SimpleRNN()
optimizer = torch.optim.SGD(rnn.parameters(), lr=0.1)

# 生成较长序列
x = torch.randn(4, 50, 10)  # batch=4, seq_len=50
y = torch.randn(4, 1)

# 无梯度裁剪
pred = rnn(x)
loss = ((pred - y) ** 2).mean()
loss.backward()

# 检查梯度范数
total_norm = 0
for p in rnn.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm += param_norm.item() ** 2
total_norm = total_norm ** 0.5
print(f"裁剪前梯度范数: {total_norm:.4f}")

# 应用梯度裁剪
c = 1.0
torch.nn.utils.clip_grad_norm_(rnn.parameters(), c)

total_norm_clipped = 0
for p in rnn.parameters():
    if p.grad is not None:
        param_norm = p.grad.data.norm(2)
        total_norm_clipped += param_norm.item() ** 2
total_norm_clipped = total_norm_clipped ** 0.5
print(f"裁剪后梯度范数: {total_norm_clipped:.4f} (阈值={c})")

# 手动实现梯度裁剪
def clip_grad_norm_manual(parameters, max_norm):
    """手动实现按范数梯度裁剪"""
    total_norm = 0
    for p in parameters:
        if p.grad is not None:
            total_norm += p.grad.data.norm(2).item() ** 2
    total_norm = total_norm ** 0.5

    clip_coef = max_norm / (total_norm + 1e-6)
    if clip_coef < 1:
        for p in parameters:
            if p.grad is not None:
                p.grad.data.mul_(clip_coef)
    return total_norm

# 验证手动实现与 PyTorch 内置的一致性
rnn2 = SimpleRNN()
pred2 = rnn2(x)
loss2 = ((pred2 - y) ** 2).mean()
loss2.backward()

norm_before = clip_grad_norm_manual(rnn2.parameters(), 1.0)
print(f"手动裁剪后范数: {norm_before:.4f} (裁剪为 1.0)")

# 梯度裁剪对训练的影响
print("\n梯度裁剪对训练的影响:")
def train_rnn_with_clipping(clip=None, epochs=50):
    torch.manual_seed(42)
    model = SimpleRNN()
    opt = torch.optim.SGD(model.parameters(), lr=0.1)
    criterion = nn.MSELoss()

    losses = []
    for epoch in range(epochs):
        x = torch.randn(4, 50, 10)
        y = torch.randn(4, 1)
        pred = model(x)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()

        if clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)

        opt.step()
        losses.append(loss.item())

    return losses[-1], max(losses)

no_clip_final, no_clip_max = train_rnn_with_clipping(None)
clip_final, clip_max = train_rnn_with_clipping(1.0)

print(f"  无裁剪: final_loss={no_clip_final:.4f}, max_loss={no_clip_max:.4f}")
print(f"  有裁剪: final_loss={clip_final:.4f}, max_loss={clip_max:.4f}")

# 按值裁剪
print("\n按值裁剪演示:")
params = [torch.randn(3, 3, requires_grad=True)]
loss = params[0].sum()
loss.backward()
print(f"裁剪前: grad={params[0].grad}")
torch.nn.utils.clip_grad_value_(params, clip_value=0.5)
print(f"按值裁剪: grad={params[0].grad}")
```

## 深度学习关联

- **RNN/LSTM 训练的标准配置**：梯度裁剪是 RNN 和 LSTM 训练的标准技术。由于 RNN 的时间展开特性，梯度爆炸问题特别严重。几乎所有 RNN 训练代码都包含梯度裁剪步骤，阈值通常设为 1.0-5.0。

- **Transformer 训练中的使用**：在大规模 Transformer 训练中，梯度裁剪同样重要。GPT、BERT 等模型在预训练时都使用梯度裁剪，通常将全局梯度范数裁剪到 1.0。这防止了训练早期可能出现的梯度尖峰。

- **对抗梯度爆炸的多种策略**：除了梯度裁剪，其他防止梯度爆炸的策略包括：更小的学习率、更好的参数初始化（正交初始化）、梯度归一化、Layer Normalization 等。这些方法可以组合使用，但梯度裁剪是最简单有效的"最后一道防线"。
