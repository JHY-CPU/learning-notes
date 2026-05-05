# 02_多层感知机 (MLP) 的万能近似定理

## 核心概念

- **万能近似定理**：万能近似定理（Universal Approximation Theorem）指出，一个包含至少一个隐层的前馈神经网络，只要隐层神经元数量足够多，并且使用连续的非线性激活函数（如 Sigmoid、ReLU），就可以以任意精度逼近任何连续函数。
- **深度与宽度的权衡**：万能近似定理只保证了存在性，没有说明如何找到这样的网络。理论上，单隐层网络可以逼近任何函数，但可能需要指数级多的神经元；而深层的网络可以用更少的参数实现同样的表示能力。
- **非线性激活的必要性**：如果 MLP 只使用线性激活函数，那么无论堆叠多少层，整个网络都等价于一个线性变换。非线性激活函数（ReLU、Sigmoid、Tanh 等）是 MLP 获得强大表示能力的关键——它们使网络可以学习复杂的非线性映射。
- **Cybenko 定理**：Cybenko（1989）证明了如果激活函数是连续的 sigmoidal 函数（如 Sigmoid），那么单隐层网络可以在紧集上以任意精度逼近任意连续函数。这是万能近似定理最早的严格证明之一。

## 数学推导

考虑一个具有 $n$ 维输入 $x \in \mathbb{R}^n$、一个隐层（$m$ 个神经元）和标量输出的 MLP：

$$
f(x) = \sum_{j=1}^{m} v_j \cdot \sigma(w_j^T x + b_j) + c
$$

其中 $\sigma$ 是非线性激活函数，$w_j \in \mathbb{R}^n$ 和 $b_j$ 是输入到隐层的权重和偏置，$v_j$ 和 $c$ 是隐层到输出的权重和偏置。

万能近似定理的形式化表述（Cybenko, 1989）：

设 $\sigma$ 是连续 sigmoidal 函数（即 $\lim_{t \to -\infty} \sigma(t) = 0$，$\lim_{t \to +\infty} \sigma(t) = 1$）。那么对任意连续函数 $g: [0,1]^n \to \mathbb{R}$ 和任意 $\epsilon > 0$，存在 $m \in \mathbb{N}$ 和参数 $v_j, w_j, b_j$，使得：

$$
\sup_{x \in [0,1]^n} |f(x) - g(x)| < \epsilon
$$

对于 ReLU 激活函数，万能近似定理同样成立。ReLU $\sigma(x) = \max(0, x)$ 虽然不是 sigmoidal 函数，但通过分段线性组合，ReLU 网络同样可以逼近任意连续函数。

证明的关键思路：sigmoidal 函数的组合可以构造"凸包函数"（bump functions），而这些凸包函数的线性组合可以近似任意连续函数。这类似于傅里叶级数中用正弦波逼近任意周期函数的思想。

## 直观理解

万能近似定理可以类比为"乐高定理"——只要你有足够多的基本积木（神经元），就可以搭建出任意形状的物体（函数）。每个隐层神经元相当于一个基本"砖块"，通过调整它们的位置（权重）和高度（输出权重），可以拼出任意复杂的函数形状。

对于 ReLU 网络尤其直观：ReLU 产生分段线性函数，每个隐层神经元贡献一个"折点"。当神经元数量足够多时，折点密度足够大，就可以用分段线性函数以任意精度逼近任何连续曲线。

另一个类比是傅里叶级数：任何周期函数都可以用足够多的正弦和余弦波叠加来近似。在神经网络中，"波"换成了激活函数的非线性响应，但核心思想是相同的。

## 代码示例

```python
import torch
import torch.nn as nn
import numpy as np

# 演示万能近似：用 MLP 学习一个非线性函数 f(x) = sin(x) + 0.5*x
torch.manual_seed(42)

# 生成训练数据
x = torch.linspace(-2*np.pi, 2*np.pi, 1000).reshape(-1, 1)
y = torch.sin(x) + 0.5 * x  # 目标函数

class MLP(nn.Module):
    """单隐层 MLP，验证万能近似定理"""
    def __init__(self, hidden_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(1, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )

    def forward(self, x):
        return self.net(x)

model = MLP(hidden_size=32)
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
loss_fn = nn.MSELoss()

# 训练
for epoch in range(3000):
    pred = model(x)
    loss = loss_fn(pred, y)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    if epoch % 500 == 0:
        print(f"Epoch {epoch:4d}, Loss: {loss.item():.6f}")

# 验证
with torch.no_grad():
    test_x = torch.tensor([[1.5], [3.0], [-1.0]])
    test_y = model(test_x)
    true_y = torch.sin(test_x) + 0.5 * test_x
    for i in range(3):
        print(f"x={test_x[i].item():.1f}, pred={test_y[i].item():.4f}, true={true_y[i].item():.4f}")
```

## 深度学习关联

- **现代神经网络的架构基础**：所有现代深度学习架构（CNN、RNN、Transformer）都可以看作是 MLP 的扩展或变体。卷积层本质上是带有权重共享和局部连接的 MLP，而 Transformer 中的 FFN（前馈网络）层就是标准的 MLP。
- **深度优于宽度的经验**：虽然单隐层 MLP 理论上可以逼近任何函数，但实践证明，深层网络（更多层）比宽层网络（更多神经元）在参数效率和学习效果上更优。深层网络能够学习层次化的特征表示，从低级到高级逐步抽象。
- **残差连接解决退化问题**：当 MLP 变得非常深时，会出现梯度消失或梯度爆炸问题，以及网络退化问题（更深并不总能带来更好性能）。残差连接（ResNet）通过引入跳跃连接，使得训练深层 MLP 成为可能，推动了网络深度从几十层迈向数百层甚至上千层。
