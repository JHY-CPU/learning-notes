# 24 Xavier 初始化与方差保持推导

## 核心概念

- **Xavier 初始化的动机**：Xavier 初始化（也称为 Glorot 初始化）旨在解决深度网络中的梯度消失/爆炸问题。核心思想是让每一层的输入和输出的方差保持一致，使得信号在前向和反向传播时都能保持合适的尺度。

- **均匀分布形式**：$W \sim \mathcal{U}[-\sqrt{6/(n_{in} + n_{out})}, \sqrt{6/(n_{in} + n_{out})}]$，其中 $n_{in}$ 和 $n_{out}$ 分别是该层的输入和输出维度。

- **正态分布形式**：$W \sim \mathcal{N}(0, 2/(n_{in} + n_{out}))$。

- **方差保持条件**：Xavier 推导出，要保持前向传播的方差不变，需要 $\text{Var}(W) = 1/n_{in}$；要保持反向传播的方差不变，需要 $\text{Var}(W) = 1/n_{out}$。Xavier 初始化取两者的调和平均 $2/(n_{in} + n_{out})$。

## 数学推导

**前向传播的方差分析**：

考虑第 $l$ 层的线性变换：

$$
z^{(l)} = W^{(l)} a^{(l-1)} + b^{(l)}
$$

其中 $a^{(l-1)} = f(z^{(l-1)})$ 是前一层的激活值。

假设激活函数是恒等映射（$f(x) = x$），即不考虑激活函数的影响。同时假设 $W_{ij}$ 独立同分布，$a_j$ 独立同分布，且 $W$ 和 $a$ 相互独立。

$$
\text{Var}(z_i^{(l)}) = \text{Var}\left(\sum_{j=1}^{n_{in}} W_{ij}^{(l)} a_j^{(l-1)}\right) = \sum_{j=1}^{n_{in}} \text{Var}(W_{ij}^{(l)} a_j^{(l-1)})
$$

由于 $W$ 和 $a$ 独立，且 $\mathbb{E}[W] = 0$：

$$
\text{Var}(W_{ij} a_j) = \mathbb{E}[W^2]\mathbb{E}[a^2] - \mathbb{E}[W]^2\mathbb{E}[a]^2 = \text{Var}(W) \cdot \mathbb{E}[a^2]
$$

如果 $\mathbb{E}[a^2] = \text{Var}(a)$（即 $\mathbb{E}[a] = 0$），则：

$$
\text{Var}(z_i^{(l)}) = n_{in} \cdot \text{Var}(W^{(l)}) \cdot \text{Var}(a^{(l-1)})
$$

要维持方差不变，即 $\text{Var}(z^{(l)}) = \text{Var}(a^{(l-1)})$，需要：

$$
n_{in} \cdot \text{Var}(W^{(l)}) = 1 \Rightarrow \text{Var}(W^{(l)}) = \frac{1}{n_{in}}
$$

**反向传播的方差分析**：

类似地，从反向传播角度：

$$
\frac{\partial L}{\partial a^{(l-1)}} = (W^{(l)})^T \frac{\partial L}{\partial z^{(l)}}
$$

$$
\text{Var}\left(\frac{\partial L}{\partial a_j^{(l-1)}}\right) = n_{out} \cdot \text{Var}(W^{(l)}) \cdot \text{Var}\left(\frac{\partial L}{\partial z_i^{(l)}}\right)
$$

要维持梯度方差不变：

$$
\text{Var}(W^{(l)}) = \frac{1}{n_{out}}
$$

**Xavier 的折中方案**：

前向需要 $\text{Var}(W) = 1/n_{in}$，反向需要 $\text{Var}(W) = 1/n_{out}$。取两者的调和平均：

$$
\text{Var}(W) = \frac{2}{n_{in} + n_{out}}
$$

对应的均匀分布为：

$$
W \sim \mathcal{U}\left[-\sqrt{\frac{6}{n_{in} + n_{out}}}, \sqrt{\frac{6}{n_{in} + n_{out}}}\right]
$$

因为均匀分布 $\mathcal{U}[-a, a]$ 的方差为 $a^2/3$，所以 $a^2/3 = 2/(n_{in} + n_{out}) \Rightarrow a = \sqrt{6/(n_{in} + n_{out})}$。

## 直观理解

Xavier 初始化可以想象为"给每层的信息铺设一条合适宽度的管道"：如果管道太细（权重太小），信号在通过时会衰减；如果管道太粗（权重太大），信号会放大。Xavier 初始化确保每层的"管道"宽度正好能让信号（前向的激活值和反向的梯度）以恒定的强度通过整个网络。

从信息流的角度看，一个深度网络有 $L$ 层。如果每层将信号的方差缩放 $\alpha$ 倍，经过 $L$ 层后信号方差变为 $\alpha^L$ 倍。当 $\alpha < 1$ 时信号指数衰减（梯度消失），$\alpha > 1$ 时指数增长（梯度爆炸）。Xavier 的目的就是让 $\alpha \approx 1$。

需要注意的是，Xavier 假设激活函数在 0 附近是线性的（恒等映射）。这对于 Tanh 是合理的（在 0 附近近似线性），但对于 ReLU 则不适用（因为 ReLU 在负半轴输出为 0，破坏了零均值的假设）。

## 代码示例

```python
import torch
import torch.nn as nn

# Xavier 初始化在 PyTorch 中的使用
def xavier_init(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.zeros_(m.bias)

# 验证方差保持特性
torch.manual_seed(42)

# 创建一个深层网络并检查各层输出的方差
class DeepNetwork(nn.Module):
    def __init__(self, n_layers=10, hidden_dim=256):
        super().__init__()
        layers = []
        layers.append(nn.Linear(128, hidden_dim))
        for _ in range(n_layers):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(nn.Tanh())
        layers.append(nn.Linear(hidden_dim, 10))
        self.net = nn.Sequential(*layers)

# 使用 Xavier 初始化
net_xavier = DeepNetwork(10, 256)
net_xavier.apply(xavier_init)

# 使用默认初始化（均匀分布）
net_default = DeepNetwork(10, 256)

# 测试前向传播的方差
x = torch.randn(1000, 128)

with torch.no_grad():
    # 检查各层输出的方差
    activations = x
    for i, layer in enumerate(net_xavier.net):
        activations = layer(activations)
        if isinstance(layer, nn.Tanh):
            var = activations.var().item()

print("Xavier 初始化后的层输出方差:")
print(f"  Tanh 层方差: {var:.6f}")

# PyTorch 内置的 Xavier 初始化
linear = nn.Linear(100, 200)
nn.init.xavier_uniform_(linear.weight)
print(f"\nXavier 均匀分布权重: mean={linear.weight.mean():.6f}, std={linear.weight.std():.6f}")

nn.init.xavier_normal_(linear.weight)
print(f"Xavier 正态分布权重: mean={linear.weight.mean():.6f}, std={linear.weight.std():.6f}")

# 不同初始化对训练的影响
def test_init(init_fn, name):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(50, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 128), nn.Tanh(),
        nn.Linear(128, 1)
    )
    model.apply(init_fn)
    opt = torch.optim.SGD(model.parameters(), lr=0.01)

    X = torch.randn(200, 50)
    y = torch.randn(200, 1)

    losses = []
    for epoch in range(200):
        pred = model(X)
        loss = ((pred - y) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
        losses.append(loss.item())

    return losses[-1]

print("\n不同初始化训练结果:")
uniform_init = lambda m: (isinstance(m, nn.Linear) and 
    nn.init.uniform_(m.weight, -0.5, 0.5) is None)
xavier_init_fn = lambda m: (isinstance(m, nn.Linear) and 
    nn.init.xavier_uniform_(m.weight) is None)

print(f"  Uniform(-0.5, 0.5): {test_init(uniform_init, 'uniform'):.6f}")
print(f"  Xavier Uniform: {test_init(xavier_init_fn, 'xavier'):.6f}")
```

## 深度学习关联

- **激活函数的适配性**：Xavier 初始化假设激活函数在 0 附近近似线性和对称（零中心），因此最适合 Tanh 和 Sigmoid 等函数。对于 ReLU，Xavier 不再适用（因为 ReLU 的输出非负，破坏了零均值假设），需要使用 He 初始化。

- **现代框架中的默认设置**：PyTorch 的 `nn.Linear` 默认使用 Kaiming/He 均匀初始化（`$\mathcal{U}[-\sqrt{1/\text{fan_in}}, \sqrt{1/\text{fan_in}}]$`），这是考虑到 ReLU 的普适性。但在使用 Tanh 时，手动切换到 Xavier 初始化通常表现更好。

- **XL 初始化**：在 Transformer 中，为了稳定深层模型的训练，使用了 Xavier 初始化的改进版本——将残差分支的初始化缩放为 $1/\sqrt{2L}$（其中 $L$ 是层数），确保深度叠加后信号的方差不会增长。这体现了 Xavier 方差保持思想的延伸应用。
