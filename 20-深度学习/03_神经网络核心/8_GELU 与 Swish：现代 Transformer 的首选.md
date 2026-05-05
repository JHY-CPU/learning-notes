# 09_GELU 与 Swish：现代 Transformer 的首选

## 核心概念

- **GELU 定义**：GELU（Gaussian Error Linear Unit，高斯误差线性单元）将输入乘以自身的累积分布函数（CDF）：$\text{GELU}(x) = x \cdot \Phi(x)$，其中 $\Phi(x)$ 是标准正态分布的 CDF。它融合了 ReLU 的线性特性和随机正则化的思想。
- **Swish/SiLU 定义**：Swish（也叫 SiLU，Sigmoid Linear Unit）定义为 $\text{Swish}(x) = x \cdot \sigma(x) = x / (1 + e^{-x})$。它由 Google 研究者通过自动化搜索发现，实际上与 SiLU 等价。Swish 是 GELU 的一个良好近似。
- **平滑且非单调**：与 ReLU 不同，GELU 和 Swish 是平滑函数（处处可导），且在 $x \approx -1$ 附近有微小的负值区域（非单调）。这种非单调性为网络引入了额外的非线性表达能力。
- **现代 Transformer 的首选**：GELU 是 BERT、GPT、ViT 等几乎所有 Transformer 模型的标准激活函数。Swish 在 EfficientNet 等卷积网络中也被广泛使用。它们相比 ReLU 带来了约 0.5-1% 的性能提升。

## 数学推导

**GELU 的精确定义**：

$$
\text{GELU}(x) = x \cdot \Phi(x) = x \cdot \frac{1}{2}\left[1 + \text{erf}\left(\frac{x}{\sqrt{2}}\right)\right]
$$

其中 $\Phi(x)$ 是标准正态分布的 CDF，$\text{erf}(\cdot)$ 是误差函数。

**GELU 的近似计算**（实际实现中使用，避免 erf 计算）：

$$
\text{GELU}(x) \approx 0.5x\left(1 + \tanh\left[\sqrt{\frac{2}{\pi}}(x + 0.044715x^3)\right]\right)
$$

**GELU 的导数**：

$$
\frac{d}{dx}\text{GELU}(x) = \Phi(x) + x \cdot \phi(x) = \Phi(x) + x \cdot \frac{1}{\sqrt{2\pi}}e^{-x^2/2}
$$

其中 $\phi(x)$ 是标准正态分布的 PDF。

**Swish 的定义和导数**：

$$
\text{Swish}(x) = x \cdot \sigma(x) = \frac{x}{1 + e^{-x}}
$$

$$
\frac{d}{dx}\text{Swish}(x) = \sigma(x) + x \cdot \sigma(x)(1 - \sigma(x)) = \frac{1 + e^{-x} + x e^{-x}}{(1 + e^{-x})^2}
$$

**GELU 和 Swish 的比较**：

$$|\text{GELU}(x) - \text{Swish}(x)| < 0.02 \quad \text{对大多数 } x$$

所以 GELU 和 Swish 数值上非常接近，可以互换使用。两者是独立发现的相似函数，而非一方近似另一方。

**随机正则化视角**：GELU 可以看作是对输入的随机门控：$\text{GELU}(x) = x \cdot \mathbb{I}(m > 0)$ 的期望形式，其中 $m \sim \mathcal{N}(x, 1)$。这种随机门控视角将 Dropout 和激活函数统一了起来。

## 直观理解

GELU 和 Swish 的形状介于 ReLU 和线性函数之间。在正半轴，它们近似线性（像 ReLU）；在负半轴，它们有一个平滑的过渡区域。关键的差异在于负半轴：ReLU 在负半轴完全截止，而 GELU/Swish 允许微小的负值通过，这有点像"在关闭阀门时留了一个小缝隙"。

非单调性（在 $x \approx -1$ 处函数值略低于零）是一个有趣的性质。这意味着很小的负输入会产生负输出，而更负的输入输出会向零回升。这有点像"逆反心理"——轻微的批评会让人沮丧（负输出），但强烈的批评反而让人麻木（输出回升到零）。

从正则化角度看，GELU 可以理解为"自适应 Dropout"：每个神经元根据其输入值自动决定是否被 Dropout——当输入很小时（$x \approx 0$），有约 50% 的概率被抑制；当输入很大时，几乎总是被保留。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# GELU 的精确和近似实现
def gelu_exact(x):
    """精确 GELU"""
    return x * torch.erfc(-x / torch.sqrt(torch.tensor(2.0))) / 2.0

def gelu_approx(x):
    """近似 GELU (tanh 近似)"""
    return 0.5 * x * (1 + torch.tanh(
        torch.sqrt(torch.tensor(2.0 / torch.pi)) * (x + 0.044715 * x ** 3)
    ))

def swish(x):
    """Swish / SiLU"""
    return x * torch.sigmoid(x)

# 比较 GELU 和 Swish
x = torch.linspace(-5, 5, 100)
gelu_exact_val = gelu_exact(x)
gelu_approx_val = gelu_approx(x)
swish_val = swish(x)

max_diff = torch.abs(gelu_exact_val - swish_val).max()
print(f"GELU 与 Swish 最大差异: {max_diff:.6f}")
print(f"GELU 精确与近似最大差异: {torch.abs(gelu_exact_val - gelu_approx_val).max():.6f}")

# --- 在简单 MLP 中比较 ReLU vs GELU ---
class MLPClassifier(nn.Module):
    def __init__(self, activation="relu"):
        super().__init__()
        act = {
            "relu": nn.ReLU(),
            "gelu": nn.GELU(),
            "silu": nn.SiLU(),  # PyTorch 内置 SiLU (= Swish)
        }[activation]
        self.net = nn.Sequential(
            nn.Linear(100, 256),
            act,
            nn.Linear(256, 128),
            act,
            nn.Linear(128, 2),
        )

    def forward(self, x):
        return self.net(x)

# 生成合成数据
torch.manual_seed(42)
X = torch.randn(1000, 100)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1.0).long()

def train_activation(act_name):
    model = MLPClassifier(act_name)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.CrossEntropyLoss()

    for epoch in range(500):
        pred = model(X)
        loss = loss_fn(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (model(X).argmax(1) == y).float().mean()
    return acc.item()

print(f"\nReLU 准确率: {train_activation('relu'):.4f}")
print(f"GELU 准确率: {train_activation('gelu'):.4f}")
print(f"SiLU 准确率: {train_activation('silu'):.4f}")
```

## 深度学习关联

- **Transformer 的标准配置**：BERT、GPT 系列、ViT、CLIP 等几乎所有 Transformer 模型在 FFN（前馈网络）层中使用 GELU 激活函数。研究表明，GELU 比 ReLU 在 Transformer 上通常有 0.5%-1% 的性能提升。PyTorch 的 `nn.GELU()` 和 `nn.SiLU()` 已内置支持。
- **EfficientNet 的成功应用**：Swish 在 EfficientNet 系列中被发现是最优激活函数。通过神经架构搜索（NAS），研究者发现 Swish 配合特定的网络结构可以在同等计算量下达到更高的准确率。
- **自门控机制的统一视角**：GELU/Swish 都属于"自门控"（self-gating）激活函数族，即输出是输入乘以输入的门控值。这种形式 $\text{act}(x) = x \cdot g(x)$ 统一了多种激活函数：Swish（$g = \sigma$）、GELU（$g = \Phi$）、甚至 GLU 变体（$g$ 来自另一个线性层的输出）。
