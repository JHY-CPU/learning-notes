# 25_He 初始化与 ReLU 的适配性

## 核心概念

- **He 初始化（Kaiming 初始化）的动机**：He 初始化专门为 ReLU 及其变体设计。Xavier 初始化假设激活函数在零点附近是线性的且零对称，这对 ReLU 不适用（ReLU 将负值全部置零，输出非负，破坏了方差分析的假设）。
- **ReLU 的方差修正**：ReLU 将约一半的神经元输出设为 0，导致方差减半。He 初始化通过将权重方差加倍来补偿这一损失：$\text{Var}(W) = 2/n_{in}$，比 Xavier 的 $1/n_{in}$ 大了一倍。
- **均匀分布形式**：$W \sim \mathcal{U}[-\sqrt{6/n_{in}}, \sqrt{6/n_{in}}]$
- **正态分布形式**：$W \sim \mathcal{N}(0, 2/n_{in})$
- **PReLU 的推广**：对于 PReLU（参数化 ReLU），He 初始化进一步推广为 $\text{Var}(W) = 2/((1+a^2)n_{in})$，其中 $a$ 是 PReLU 负半轴的斜率参数。

## 数学推导

**ReLU 激活的方差分析**：

假设 $W_{ij} \sim \mathcal{N}(0, \sigma^2)$，输入 $x_j$ 独立同分布且均值为 0。

前向传播：

$$
y_i = \sum_{j=1}^{n_{in}} W_{ij} x_j, \quad a_i = \max(0, y_i)
$$

$\text{Var}(y_i) = n_{in} \cdot \sigma^2 \cdot \text{Var}(x)$

关键是计算 $\text{Var}(a_i)$。由于 ReLU 的 $a_i = \max(0, y_i)$，且 $y_i$ 是对称分布（均值为 0）：

$$
\mathbb{E}[a_i^2] = \mathbb{E}[\max(0, y_i)^2] = \frac{1}{2}\mathbb{E}[y_i^2] = \frac{1}{2}\text{Var}(y_i)
$$

注意这里 $\mathbb{E}[a_i] \neq 0$，但 $\mathbb{E}[a_i^2] = \frac{1}{2}\text{Var}(y_i)$。

因此前向传播的方差关系为：

$$
\text{Var}(a_i) = \mathbb{E}[a_i^2] - \mathbb{E}[a_i]^2 = \frac{1}{2}\text{Var}(y_i) - \left(\frac{\text{Var}(y_i)}{2\pi}\right)
$$

为简化，论文中使用 $\mathbb{E}[a^2] = \frac{1}{2}\text{Var}(y)$ 作为近似。

所以：

$$
\text{Var}(y_i^{(l)}) = n_{in} \cdot \sigma^2 \cdot \frac{1}{2} \text{Var}(y^{(l-1)})
$$

要维持方差不变 $\text{Var}(y^{(l)}) = \text{Var}(y^{(l-1)})$：

$$
\frac{1}{2} n_{in} \cdot \sigma^2 = 1 \Rightarrow \sigma^2 = \frac{2}{n_{in}}
$$

**反向传播的分析类似**：

$$
\text{Var}\left(\frac{\partial L}{\partial x}\right) = \frac{1}{2} n_{out} \cdot \text{Var}(W) \cdot \text{Var}\left(\frac{\partial L}{\partial y}\right)
$$

需要 $\text{Var}(W) = 2/n_{out}$（反向视角）。

**He 初始化的统一形式**：

前向使用 $2/n_{in}$，反向可使用 $2/n_{out}$。实践中常用 $\text{Var}(W) = 2/n_{in}$，因为前向方差保持是最重要的。

## 直观理解

He 初始化相当于"给 ReLU 网络装一个信号放大器"：因为 ReLU 切掉了一半的信号（负值置零），导致了信号功率减半。He 初始化通过将初始权重的方差加倍来预补偿这一损失，确保信号在经过 ReLU 后仍然保持合适的尺度。

简单来说：Xavier 假设所有神经元都工作，He 知道只有一半在工作（ReLU 砍掉了另一半），所以把权重放大 $\sqrt{2}$ 倍来补偿。

这个 $\sqrt{2}$ 因子在实际中非常重要。如果使用 Xavier 初始化训练 ReLU 网络，深层网络的信号会逐渐衰减；如果使用更大尺度的初始化（如 He），信号可以稳定地流过深层网络。

## 代码示例

```python
import torch
import torch.nn as nn

# PyTorch 中的 He (Kaiming) 初始化
def kaiming_init(m):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, mode='fan_in', nonlinearity='relu')
        nn.init.zeros_(m.bias)

# 比较 Xavier 和 He 初始化在 ReLU 网络中的表现
torch.manual_seed(42)

class DeepReLUNet(nn.Module):
    def __init__(self, depth=10, width=256):
        super().__init__()
        layers = [nn.Linear(128, width), nn.ReLU()]
        for _ in range(depth):
            layers.append(nn.Linear(width, width))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(width, 10))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)

# 测试不同初始化
x = torch.randn(500, 128)

for init_name, init_fn in [
    ("Xavier", lambda m: (isinstance(m, nn.Linear) and nn.init.xavier_uniform_(m.weight))),
    ("He", lambda m: (isinstance(m, nn.Linear) and nn.init.kaiming_uniform_(m.weight, nonlinearity='relu'))),
]:
    model = DeepReLUNet(10, 256)
    model.apply(init_fn)

    with torch.no_grad():
        # 计算经过 5 层 ReLU 后的激活值统计
        h = x
        for i, layer in enumerate(model.net):
            h = layer(h)
            if isinstance(layer, nn.ReLU):
                dead_ratio = (h == 0).float().mean()
                if i == 3:  # 第 2 个 ReLU 后
                    print(f"{init_name}: 激活均值={h.mean():.4f}, "
                          f"激活 std={h.std():.4f}, 死亡比例={dead_ratio:.4f}")

# 训练对比
def train_with_init(init_fn, name):
    torch.manual_seed(42)
    model = DeepReLUNet(8, 256)
    model.apply(init_fn)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.CrossEntropyLoss()

    X = torch.randn(500, 128)
    y = torch.randint(0, 10, (500,))

    for epoch in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

print("\n不同初始化训练损失:")
xavier_loss = train_with_init(
    lambda m: (isinstance(m, nn.Linear) and nn.init.xavier_uniform_(m.weight)), 
    'Xavier')
print(f"  Xavier: {xavier_loss:.4f}")
he_loss = train_with_init(
    lambda m: (isinstance(m, nn.Linear) and nn.init.kaiming_uniform_(m.weight, nonlinearity='relu')), 
    'He')
print(f"  He: {he_loss:.4f}")

# He 初始化的不同 mode
print("\nHe 初始化 fan_in vs fan_out:")
linear = nn.Linear(100, 200)
nn.init.kaiming_uniform_(linear.weight, mode='fan_in', nonlinearity='relu')
print(f"  fan_in: std={linear.weight.std():.4f}")

linear2 = nn.Linear(100, 200)
nn.init.kaiming_uniform_(linear2.weight, mode='fan_out', nonlinearity='relu')
print(f"  fan_out: std={linear2.weight.std():.4f}")
```

## 深度学习关联

- **现代 CNN 的标准初始化**：He 初始化是 ResNet、DenseNet、MobileNet 等现代卷积网络的标准初始化方法。这些网络大量使用 ReLU 及其变体，He 初始化为它们提供了稳定的训练起点。
- **PReLU 的适配**：He 初始化的原始论文同时提出了 PReLU 激活函数，并推导了 PReLU 对应的初始化方差。当 PReLU 的负斜率 $a$ 不为 0 时，初始化方差为 $2/((1+a^2)n_{in})$，这体现了理论分析的完备性。
- **PyTorch 的默认选择**：在 PyTorch 中，`nn.Linear` 和 `nn.Conv2d` 的默认初始化就是 Kaiming 均匀初始化（`mode='fan_in'`，`nonlinearity='relu'`）。这意味着即使不显式调用初始化函数，PyTorch 已经为 ReLU 网络提供了适配的初始化。
