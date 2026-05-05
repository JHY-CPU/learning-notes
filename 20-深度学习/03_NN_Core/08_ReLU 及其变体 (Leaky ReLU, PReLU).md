# 08_ReLU 及其变体 (Leaky ReLU, PReLU)

## 核心概念

- **ReLU 定义**：ReLU（Rectified Linear Unit，线性整流单元）定义为 $\text{ReLU}(x) = \max(0, x)$。它的正半轴导数为 1，负半轴导数为 0，计算极其简单且能有效缓解梯度消失问题。
- **稀疏激活特性**：ReLU 的负半轴输出为 0，意味着在任何时刻大约一半的神经元处于"未激活"状态。这种稀疏性有助于网络去相关，减轻过拟合，同时计算效率极高。
- **Dying ReLU 问题**：ReLU 在负半轴梯度为 0，这意味着一旦某个神经元的输出落入负区域，它的梯度为零，权重将永远无法更新——神经元"死亡"了。如果大量神经元同时死亡，网络的容量会显著降低。
- **Leaky ReLU 和 PReLU 的改进**：Leaky ReLU 在负半轴引入一个小的斜率 $\alpha$（通常为 0.01），使得负区域的梯度不为零，缓解了 Dying ReLU 问题。PReLU（Parametric ReLU）将 $\alpha$ 作为可学习参数，让网络自己决定负半轴的斜率。

## 数学推导

**ReLU 系列函数的数学定义**：

ReLU:
$$
f(x) = \max(0, x) = \begin{cases}
x & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

$$
f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
0 & \text{if } x \leq 0
\end{cases}
$$

Leaky ReLU:
$$
f(x) = \begin{cases}
x & \text{if } x > 0 \\
\alpha x & \text{if } x \leq 0
\end{cases}
$$

$$
f'(x) = \begin{cases}
1 & \text{if } x > 0 \\
\alpha & \text{if } x \leq 0
\end{cases}
$$

其中 $\alpha$ 通常取 0.01。当 $\alpha = 0$ 时退化为 ReLU。

PReLU（Parametric ReLU）：
$$
f(x_i) = \begin{cases}
x_i & \text{if } x_i > 0 \\
a_i x_i & \text{if } x_i \leq 0
\end{cases}
$$

其中 $a_i$ 是可学习的参数，每个通道可以有不同的斜率。$a_i$ 的梯度为：

$$
\frac{\partial L}{\partial a_i} = \sum_{x_i \leq 0} \frac{\partial L}{\partial f(x_i)} \cdot x_i
$$

**梯度消失的改善分析**：

传统 Sigmoid 网络，梯度经过 $L$ 层后的缩放因子：

$$
\prod_{l=1}^{L} \sigma'(z_l) \leq (0.25)^L \quad \text{（指数级衰减）}
$$

ReLU 网络，梯度经过 $L$ 层后的缩放因子：

$$
\prod_{l=1}^{L} \mathbb{I}(z_l > 0) \quad \text{（要么 0 要么 1）}
$$

由于梯度不经过任何压缩（当 $x > 0$ 时导数为 1），深层网络的梯度可以无损地反向传播，这是 ResNet 等超深网络能够训练的基石之一。

## 直观理解

ReLU 可以类比为"单向阀门"：正方向的电流可以通过且无阻碍，负方向则完全截止。这赋予了网络"选择性通过"的能力——有用的信号正向传播，无用的信号被阻断。

稀疏激活就像"一个大型团队的会议，不是所有人都需要发言"。只有与当前任务最相关的神经元被激活（输出 > 0），其他神经元保持沉默（输出 = 0）。这种稀疏性不仅节省计算量，还使网络更容易解释（我们可以看哪些神经元对特定输入有响应）。

Dying ReLU 问题可以理解为"永久沉默的神经元"：如果一个神经元因为不好的权重初始化或过大的学习率而进入了负区域，它将永远无法恢复——就像一台关掉就再也无法打开的开关。

Leaky ReLU 的改进很直观：给"阀门"加一个微小泄漏，让负方向的信号也能缓慢通过，这样神经元即使偶尔进入负区域也有机会恢复。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 各种 ReLU 变体
x = torch.linspace(-5, 5, 20)

relu = F.relu(x)
leaky_relu = F.leaky_relu(x, negative_slope=0.1)  # Leaky ReLU, alpha=0.1
prelu = nn.PReLU(num_parameters=1, init=0.25)     # PReLU, 可学习参数

print(f"x:                  {x}")
print(f"ReLU:               {relu}")
print(f"Leaky ReLU(0.1):    {leaky_relu}")
print(f"PReLU(init=0.25):   {prelu(x)}")
print(f"PReLU 的 alpha 参数: {prelu.weight.item():.4f}")

# Dying ReLU 演示
torch.manual_seed(42)
dying_layer = nn.Linear(100, 100)
dying_input = -torch.ones(10, 100) * 2  # 所有输入为负

with torch.no_grad():
    output = dying_layer(dying_input)
    relu_output = F.relu(output)
    dead_fraction = (relu_output == 0).float().mean()
    print(f"\nReLU 死亡神经元比例: {dead_fraction:.2%}")

    leaky_output = F.leaky_relu(output, 0.01)
    dead_fraction_leaky = (leaky_output == 0).float().mean()
    print(f"Leaky ReLU '死亡'比例: {dead_fraction_leaky:.2%}")

# 带 PReLU 的简单 CNN 训练示例
class SimpleNet(nn.Module):
    def __init__(self, use_prelu=False):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.act1 = nn.PReLU(256) if use_prelu else nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.act2 = nn.PReLU(128) if use_prelu else nn.ReLU()
        self.fc3 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.act1(self.fc1(x))
        x = self.act2(self.fc2(x))
        return self.fc3(x)

print("\nReLU vs PReLU 参数对比:")
relu_net = SimpleNet(False)
prelu_net = SimpleNet(True)
print(f"ReLU 网络参数: {sum(p.numel() for p in relu_net.parameters())}")
print(f"PReLU 网络参数: {sum(p.numel() for p in prelu_net.parameters())} (+384 个可学习斜率)")
```

## 深度学习关联

- **ReLU 是深度学习的基石**：ReLU 的引入是深度学习复兴的关键转折点。它的非饱和梯度特性使得训练超过 5 层的深度网络成为可能，直接推动了从 AlexNet（2012）到 ResNet（2015）再到当今千层 Transformer 的发展。
- **Transformer 中的 ReLU 变体**：虽然现代 Transformer 更多使用 GELU 或 Swish，但 ReLU 系列仍然广泛存在。BERT 等早期 Transformer 模型在 FFN 层使用 GELU，而更轻量的模型（如 MobileNet）仍然使用 ReLU6（输出限制在 0~6 的 ReLU 变体）。
- **量化友好的特性**：ReLU 的计算仅涉及比较和取最大值操作，不需要浮点指数运算，非常适合在移动端、嵌入式设备和 FPGA 上部署。ReLU 的输出范围明确，对低精度量化（INT8）更加友好。
