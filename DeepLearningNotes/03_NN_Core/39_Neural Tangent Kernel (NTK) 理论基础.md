# 39 Neural Tangent Kernel (NTK) 理论基础

## 核心概念

- **NTK 定义**：神经正切核（Neural Tangent Kernel, NTK）是描述无限宽神经网络在梯度下降训练过程中的核函数。它定义为网络输出对参数的梯度之间的内积：$\Theta(x, x') = \nabla_\theta f(x)^T \nabla_\theta f(x')$。

- **无限宽极限**：当网络的宽度（隐层神经元数量）趋于无穷大时，NTK 在训练过程中保持不变（即不随时间变化）。这使得无限宽神经网络的训练动态可以用一个线性模型完全描述。

- **线性化训练动态**：在 NTK 理论中，神经网络的训练动态简化为一个线性模型：$f_t(x) = f_0(x) - \eta \sum_i \Theta(x, x_i) \cdot \partial L/\partial f_{t-1}(x_i)$。这意味着神经网络的训练等价于使用 NTK 的核回归。

- **与神经网络训练的联系**：NTK 理论表明，在无限宽极限下，神经网络的行为由 NTK 完全决定。这建立了深度学习与核方法之间的桥梁，解释了为什么深度神经网络即使在随机初始化后通过梯度下降也能很好地学习。

## 数学推导

**NTK 的基本公式**：

考虑神经网络 $f(x; \theta)$，其中 $\theta$ 是参数。在梯度下降下，参数的更新为：

$$
\theta_{t+1} = \theta_t - \eta \nabla_\theta L(\theta_t)
$$

输出函数 $f_t(x) = f(x; \theta_t)$ 的变化：

$$
f_{t+1}(x) - f_t(x) \approx \nabla_\theta f_t(x)^T (\theta_{t+1} - \theta_t) = -\eta \nabla_\theta f_t(x)^T \nabla_\theta L(\theta_t)
$$

损失梯度 $\nabla_\theta L(\theta_t) = \sum_i \nabla_\theta f_t(x_i) \cdot \frac{\partial L}{\partial f_t(x_i)}$（基于训练样本 $(x_i, y_i)$）。

代入得到：

$$
f_{t+1}(x) - f_t(x) = -\eta \sum_{i} \underbrace{\nabla_\theta f_t(x)^T \nabla_\theta f_t(x_i)}_{\Theta_t(x, x_i)} \cdot \frac{\partial L}{\partial f_t(x_i)}
$$

其中 $\Theta_t(x, x') = \nabla_\theta f_t(x)^T \nabla_\theta f_t(x')$ 是时间 $t$ 的 NTK。

**无限宽极限下的核不变性**：

关键定理（Jacot et al., 2018）：在网络宽度 $\to \infty$ 的极限下，NTK 在训练过程中保持不变：

$$
\Theta_t(x, x') \to \Theta_\infty(x, x') \quad \text{当宽度} \to \infty
$$

这意味着 $f_t(x)$ 的动态变为一个线性系统：

$$
\frac{df_t(x)}{dt} = -\eta \sum_i \Theta_\infty(x, x_i) \cdot \frac{\partial L}{\partial f_t(x_i)}
$$

对于 MSE 损失 $L = \frac{1}{2}\sum_i (f(x_i) - y_i)^2$，训练动态可以精确求解：

$$
f_t(x) = f_0(x) + \sum_{i,j} \Theta_\infty(x, x_i) (K^{-1}(I - e^{-\eta K t}))_{ij} (y_j - f_0(x_j))
$$

其中 $K_{ij} = \Theta_\infty(x_i, x_j)$ 是训练数据上的 NTK Gram 矩阵。

当 $t \to \infty$，预测收敛到核回归解：

$$
f_\infty(x) = \Theta_\infty(x, X)^T K^{-1} y
$$

**NTK 的计算**：

对于两层全连接网络 $f(x) = \frac{1}{\sqrt{m}} \sum_{j=1}^{m} a_j \sigma(w_j^T x)$，其 NTK 为：

$$
\Theta(x, x') = x^T x' \cdot \mathbb{E}_{w \sim \mathcal{N}(0,I)}[\sigma'(w^T x)\sigma'(w^T x')] + \mathbb{E}_{w \sim \mathcal{N}(0,I)}[\sigma(w^T x)\sigma(w^T x')]
$$

## 直观理解

NTK 理论最核心的洞察是：无限宽神经网络的训练过程等效于一个简单的核方法（Kernel Method）。这就像发现了一个"隐藏的桥梁"，连接了深度学习（复杂、非凸、神秘）和核方法（简单、凸、理论完备）。

为什么宽度很重要？当网络很宽时，每个参数的改变对输出的影响很小（因为参数众多，每个参数的贡献被"稀释"了）。这导致参数在训练过程中移动很少，网络工作在初始参数的"线性区域"内。在这个区域中，NTK 保持不变，系统变为线性。

NTK 的核 $\Theta(x, x')$ 衡量了样本 $x$ 和 $x'$ 在"参数空间中的关联程度"——即改变参数对两个样本预测的影响的相关性。当 $\Theta(x, x')$ 很大时，训练 $x$ 也会显著影响 $x'$ 的预测。

**有限宽度的偏离**：在实际中，网络的宽度是有限的，NTK 在训练过程中会发生变化。窄网络可以学习到更丰富的特征（"特征学习"阶段），而宽网络的训练更像核方法。这个"特征学习 vs 核回归"的权衡是 NTK 理论的重要研究方向。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 计算有限宽度神经网络的 NTK
def compute_ntk(model, x1, x2=None):
    """计算神经网络在数据点上的 NTK"""
    if x2 is None:
        x2 = x1

    # 对 x1 的每个样本，计算输出对参数的梯度
    # 简化版本：适用于小批量
    n1, n2 = x1.size(0), x2.size(0)

    # 累积梯度矩阵
    grads1 = []
    for i in range(n1):
        model.zero_grad()
        out = model(x1[i:i+1])
        out.backward(torch.ones_like(out))
        grad = torch.cat([p.grad.view(-1).detach() for p in model.parameters()])
        grads1.append(grad)
    grads1 = torch.stack(grads1)  # (n1, num_params)

    grads2 = []
    for i in range(n2):
        model.zero_grad()
        out = model(x2[i:i+1])
        out.backward(torch.ones_like(out))
        grad = torch.cat([p.grad.view(-1).detach() for p in model.parameters()])
        grads2.append(grad)
    grads2 = torch.stack(grads2)  # (n2, num_params)

    # NTK = grads1 @ grads2^T
    ntk = grads1 @ grads2.T
    return ntk

# 一个小型 MLP
class SmallMLP(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.fc1 = nn.Linear(10, width)
        self.fc2 = nn.Linear(width, width)
        self.fc3 = nn.Linear(width, 1)

    def forward(self, x):
        h = F.relu(self.fc1(x))
        h = F.relu(self.fc2(h))
        return self.fc3(h)

torch.manual_seed(42)

# 演示 NTK 的计算
model_small = SmallMLP(64)
x_data = torch.randn(20, 10)

ntk_matrix = compute_ntk(model_small, x_data[:5])
print(f"NTK 矩阵 shape: {ntk_matrix.shape}")
print(f"NTK 矩阵 (前 5 个样本):\n{ntk_matrix}")

# 观察 NTK 在训练过程中的变化
print("\nNTK 在训练过程中的变化:")
model = SmallMLP(64)
x_train = torch.randn(50, 10)
y_train = torch.sin(x_train.sum(1, keepdim=True))
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

ntk_before = compute_ntk(model, x_train[:10])
ntk_diff = []

for epoch in range(50):
    pred = model(x_train)
    loss = ((pred - y_train) ** 2).mean()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        ntk_current = compute_ntk(model, x_train[:10])
        diff = (ntk_current - ntk_before).norm().item()
        ntk_diff.append(diff)
        print(f"  Epoch {epoch:2d}, NTK 变化量: {diff:.4f}")

# 观察宽度对 NTK 稳定性的影响
print("\n宽度对 NTK 稳定性的影响:")
for width in [16, 64, 256]:
    model_wide = SmallMLP(width)

    # 计算两次（不同初始化）的 NTK 差异？
    # 这里比较训练前后的 NTK 变化
    x_test = torch.randn(10, 10)
    ntk_init = compute_ntk(model_wide, x_test)

    # 训练几步
    opt = torch.optim.SGD(model_wide.parameters(), lr=0.01)
    for _ in range(20):
        pred = model_wide(x_test)
        loss = pred.sum()
        opt.zero_grad()
        loss.backward()
        opt.step()

    ntk_final = compute_ntk(model_wide, x_test)
    change = (ntk_final - ntk_init).norm().item() / ntk_init.norm().item()
    print(f"  width={width:3d}: NTK 相对变化 = {change:.4f}")
```

## 深度学习关联

- **理论深度学习的基石**：NTK 是理论深度学习中最重要的工作之一。它提供了分析神经网络训练动态的数学框架，解释了为什么梯度下降在非凸优化中如此有效。NTK 理论获得了 2020 年 NeurIPS 最佳论文奖。

- **与高斯过程的联系**：在初始化时，无限宽神经网络等价于高斯过程（NNGP, Neural Network Gaussian Process）。NTK 将这一联系从初始化扩展到了整个训练过程，建立了"初始化时的 GP"和"训练时的线性化模型"之间的完整理论。

- **实际应用的局限性**：NTK 理论主要适用于无限宽网络，而实际使用的网络宽度有限。有限宽度网络可以进行真正的"特征学习"，即网络在学习过程中改变其表示方式。特征学习对实际深度学习的成功至关重要，但 NTK 理论无法完全描述这一过程。后续工作（如 Mean Field Theory、Feature Learning Theory）尝试弥合这一差距。
