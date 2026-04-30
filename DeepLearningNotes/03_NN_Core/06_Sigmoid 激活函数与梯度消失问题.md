# 06 Sigmoid 激活函数与梯度消失问题

## 核心概念

- **Sigmoid 函数定义**：Sigmoid 函数 $\sigma(x) = 1/(1 + e^{-x})$ 将任意实数输入映射到 $(0,1)$ 区间，具有平滑的 S 形曲线。它最早被广泛用作神经网络的激活函数，因为其输出可以解释为概率值。

- **梯度饱和现象**：Sigmoid 函数的两端（$x \to +\infty$ 或 $x \to -\infty$）导数趋近于 0。当神经元的输入落在这个饱和区域时，梯度几乎为零，导致该神经元的权重几乎无法更新。

- **梯度消失问题**：在深层网络中，梯度需要通过多个 Sigmoid 层反向传播。每经过一层，梯度都要乘以 Sigmoid 的导数（最大值为 0.25）。经过多层后，梯度呈指数级衰减至接近零，使得深层网络的参数几乎无法得到有效训练。

- **非零中心输出**：Sigmoid 的输出始终为正（0 到 1），这会导致后续层的输入全部为正，进而导致权重梯度全部为正或全部为负，使得优化路径呈"之"字形，收敛速度变慢。

## 数学推导

Sigmoid 函数及其导数：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\sigma'(x) = \sigma(x)(1 - \sigma(x)) = \frac{e^{-x}}{(1 + e^{-x})^2}
$$

导数的重要性质：$\sigma'(x)$ 的最大值在 $x = 0$ 处取得，值为 $\sigma'(0) = 0.25$。当 $|x| > 5$ 时，$\sigma'(x) \approx 0$。

**梯度消失的数学分析**：

考虑一个 $L$ 层网络，每层使用 Sigmoid 激活。反向传播时，第 $1$ 层（最底层）的梯度：

$$
\frac{\partial L}{\partial W_1} = \frac{\partial L}{\partial a_L} \cdot \prod_{l=2}^{L} \left( W_l^T \odot \sigma'(a_{l-1}) \right) \cdot \frac{\partial a_1}{\partial W_1}
$$

由于每个 $\sigma'(a_{l-1})$ 的最大值为 0.25，当 $L$ 增大时，乘积项 $\prod \sigma'(a_{l-1})$ 以 $0.25^L$ 的速度指数级衰减。即使初始化使激活值集中在 0 附近，经过深层堆叠后梯度仍然会急剧减小。

具体来说，对于 $L = 10$ 层网络，梯度幅值最多为初始的 $0.25^{10} \approx 9.5 \times 10^{-7}$，几乎可以忽略不计。

## 直观理解

Sigmoid 的梯度饱和可以想象成一个"传送带"：在传送带中间（$x \approx 0$），物品移动灵敏，稍微施力就能移动很大距离（梯度大）；到了传送带两端，无论怎么施力，物品几乎不动（梯度接近 0）。如果网络的许多神经元都处于饱和区，相当于整个"学习传送带"卡住了。

梯度消失问题就像"打电话时的信号衰减"：每经过一个中继站（一层），信号强度就衰减一次。如果中继站太多，传到终点的信号几乎为零，无法传达有效信息。

非零中心问题可以用一个类比理解：如果所有人的反馈都只有"好"和"很好"两种（没有"不好"），那么你很难做出精细的调整。Sigmoid 只输出正值的特性使得网络更新方向受限，难以灵活调整。

## 代码示例

```python
import torch
import matplotlib.pyplot as plt

# Sigmoid 函数及其导数
x = torch.linspace(-10, 10, 100)
sigmoid = torch.sigmoid(x)
sigmoid_deriv = sigmoid * (1 - sigmoid)  # 导数解析式

print(f"Sigmoid(0) = {torch.sigmoid(torch.tensor(0.0)):.4f}")
print(f"Sigmoid'(0) = {sigmoid_deriv[50]:.4f} (最大值)")
print(f"Sigmoid'(5) = {sigmoid_deriv[75]:.6f}")
print(f"Sigmoid'(10) = {sigmoid_deriv[-1]:.8f}")

# 演示梯度消失：10 层 Sigmoid 网络的梯度衰减
layers = 10
grad_scale = 1.0
for i in range(layers):
    grad_scale *= 0.25  # 每层最多乘 0.25
    print(f"经过 {i+1} 层后最大梯度比例: {grad_scale:.2e}")

# 实际训练演示
torch.manual_seed(42)
net = torch.nn.Sequential(
    torch.nn.Linear(10, 64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64, 64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64, 64),
    torch.nn.Sigmoid(),
    torch.nn.Linear(64, 1)
)

x_data = torch.randn(32, 10)
y_data = torch.randn(32, 1)

loss_fn = torch.nn.MSELoss()
optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

for epoch in range(5):
    pred = net(x_data)
    loss = loss_fn(pred, y_data)
    optimizer.zero_grad()
    loss.backward()
    # 检查梯度范数
    total_norm = 0
    for p in net.parameters():
        if p.grad is not None:
            total_norm += p.grad.norm().item() ** 2
    total_norm = total_norm ** 0.5
    print(f"Epoch {epoch}, Loss: {loss.item():.4f}, Grad Norm: {total_norm:.6f}")
    optimizer.step()
```

## 深度学习关联

- **梯度消失催生了 ReLU 系列**：Sigmoid 的梯度消失问题是 ReLU 激活函数被广泛采用的主要原因。ReLU 在正半轴导数为 1，不会压缩梯度，使得深层网络的训练成为可能。

- **Batch Normalization 缓解梯度消失**：Batch Normalization 通过将每层输入标准化为零均值单位方差，将激活值控制在 Sigmoid 的非饱和区域（$x \approx 0$），有效缓解了梯度消失问题。

- **Sigmoid 在现代架构中的保留**：尽管 Sigmoid 不再作为隐藏层激活函数，它在 RNN 的门控机制（LSTM、GRU 的遗忘门、输入门、输出门）中仍然不可或缺，因为门控需要输出在 0 到 1 之间。
