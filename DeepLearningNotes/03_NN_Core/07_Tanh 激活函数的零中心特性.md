# 07 Tanh 激活函数的零中心特性

## 核心概念

- **Tanh 函数定义**：Tanh（双曲正切）函数 $\tanh(x) = (e^x - e^{-x})/(e^x + e^{-x})$ 将输入映射到 $(-1, 1)$ 区间。它是零中心的（zero-centered），这是相比 Sigmoid 的核心优势。

- **零中心输出的重要性**：由于 Tanh 的输出均值为 0，后续层的输入也是零中心的。这避免了梯度更新时的"之字形"路径，使得优化更加高效。零中心特性意味着权重梯度可以有正有负，网络可以更灵活地调整方向。

- **梯度饱和依然存在**：Tanh 的导数 $\tanh'(x) = 1 - \tanh^2(x)$ 在两端趋于 0，因此仍然存在梯度饱和问题。不过 Tanh 在 $x=0$ 处的导数为 1，比 Sigmoid 的 0.25 更大，梯度消失程度稍轻。

- **与 Sigmoid 的关系**：Tanh 是 Sigmoid 的缩放平移版本：$\tanh(x) = 2\sigma(2x) - 1$。Tanh 的梯度范围是 $[0, 1]$（最大值为 1），而 Sigmoid 的梯度范围是 $[0, 0.25]$，因此 Tanh 的梯度消失问题相对较轻。

## 数学推导

**Tanh 函数及其导数**：

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = 1 - \frac{2}{e^{2x} + 1}
$$

$$
\tanh'(x) = 1 - \tanh^2(x) = \frac{4e^{2x}}{(e^{2x} + 1)^2}
$$

**与 Sigmoid 的关系**：

$$
\sigma(x) = \frac{1}{1 + e^{-x}}
$$

$$
\tanh(x) = \frac{e^x - e^{-x}}{e^x + e^{-x}} = \frac{1 - e^{-2x}}{1 + e^{-2x}} = 2\sigma(2x) - 1
$$

**均值为零的证明**：

假设 $x$ 在 $[-a, a]$ 上均匀分布，由于 $\tanh(x)$ 是奇函数（$\tanh(-x) = -\tanh(x)$）：

$$
\mathbb{E}[\tanh(x)] = \frac{1}{2a} \int_{-a}^{a} \tanh(x) dx = 0
$$

这对任何对称分布 $p(x) = p(-x)$ 都成立。因此，如果输入是零中心的，Tanh 的输出也是零中心的。

**梯度流比较**：

对于深层网络，第 $l$ 层的梯度 $\delta_l$ 与第 $l+1$ 层的梯度 $\delta_{l+1}$ 的关系为：

$$
\delta_l = (W_{l+1}^T \delta_{l+1}) \odot f'(z_l)
$$

对于 Sigmoid：$f'(z_l) \in [0, 0.25]$，梯度压缩因子 $\le 0.25$
对于 Tanh：$f'(z_l) \in [0, 1]$，梯度压缩因子 $\le 1.0$

因此 Tanh 的梯度流强度是 Sigmoid 的 4 倍，梯度消失问题相对较轻但仍然存在。

## 直观理解

Tanh 的零中心特性可以类比为"对中有零的仪表盘"：指针在正中表示零，向左为负向右为正。这样的仪表盘可以直观地显示正负方向。如果仪表盘只能显示非负值（如 Sigmoid），你需要额外的心理换算来判断相对方向。

从优化角度看，零中心输出就像"在所有方向都可以自由移动"的粒子，而非零中心输出像是"只能向正方向移动"的受限粒子，后者的运动轨迹必然是曲折的。

Tanh 的梯度最大值为 1（在 $x=0$ 处），意味着在零点附近，梯度可以无损地通过。但远离零点时，梯度仍然会饱和——这类似于"在陡坡上灵敏度很高，在平地上就不动了"。

## 代码示例

```python
import torch

# Tanh 函数及其导数的可视化
x = torch.linspace(-5, 5, 100)
tanh = torch.tanh(x)
tanh_deriv = 1 - tanh ** 2

print(f"tanh(0) = {torch.tanh(torch.tensor(0.0)):.4f}")
print(f"tanh'(0) = {(1 - torch.tanh(torch.tensor(0.0))**2):.4f} (最大值)")
print(f"tanh'(2) = {(1 - torch.tanh(torch.tensor(2.0))**2):.4f}")
print(f"tanh'(5) = {(1 - torch.tanh(torch.tensor(5.0))**2):.8f}")

# 证明零中心特性
test_input = torch.randn(10000)  # 零均值的输入
tanh_output = torch.tanh(test_input)
print(f"\n输入均值: {test_input.mean():.4f}")
print(f"Tanh 输出均值: {tanh_output.mean():.4f} (应接近 0)")

# 对比 Sigmoid 和 Tanh 的梯度流
sigmoid = torch.nn.Sigmoid()
sigmoid_out = sigmoid(test_input)
sigmoid_deriv = sigmoid_out * (1 - sigmoid_out)
print(f"\nSigmoid 输出均值: {sigmoid_out.mean():.4f} (非零中心)")
print(f"Sigmoid 梯度均值: {sigmoid_deriv.mean():.4f}")
print(f"Tanh 梯度均值: {tanh_deriv.mean():.4f}")

# 简单分类演示
def train_with_activation(activation_fn, name):
    torch.manual_seed(42)
    model = torch.nn.Sequential(
        torch.nn.Linear(2, 32),
        activation_fn,
        torch.nn.Linear(32, 32),
        activation_fn,
        torch.nn.Linear(32, 1),
        torch.nn.Sigmoid()
    )
    # 生成环形数据
    t = torch.linspace(0, 2*torch.pi, 200)
    r1, r2 = 1.0, 2.0
    x1 = torch.stack([r1*torch.cos(t), r1*torch.sin(t)], 1)
    x2 = torch.stack([r2*torch.cos(t), r2*torch.sin(t)], 1)
    X = torch.cat([x1, x2]) + torch.randn(400, 2) * 0.1
    y = torch.cat([torch.zeros(200), torch.ones(200)]).view(-1, 1)

    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    loss_fn = torch.nn.BCELoss()
    for epoch in range(1000):
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
    return loss.item()

print(f"\nSigmoid 网络最终损失: {train_with_activation(torch.nn.Sigmoid(), 'Sigmoid'):.4f}")
print(f"Tanh 网络最终损失: {train_with_activation(torch.nn.Tanh(), 'Tanh'):.4f}")
```

## 深度学习关联

- **LSTM/GRU 中的主要激活**：Tanh 是 LSTM 和 GRU 中细胞状态更新的主要激活函数。在这些循环架构中，Tanh 负责生成候选细胞状态，其零中心特性有助于稳定循环网络中的梯度流。

- **编码器-解码器中的输出层**：Tanh 通常用作生成模型的输出激活函数，当输出需要限制在 $(-1, 1)$ 范围时（如图像生成中的像素值、GAN 的生成器输出），Tanh 是常见选择。

- **被 ReLU 系列替代的趋势**：在现代深层卷积网络和 Transformer 中，Tanh 作为隐层激活函数已被 ReLU/GELU 等函数取代。但在需要零中心输出且梯度饱和不是主要问题的浅层网络中，Tanh 仍是不错的选择。
