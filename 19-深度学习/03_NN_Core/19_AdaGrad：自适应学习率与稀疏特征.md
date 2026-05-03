# 19_AdaGrad：自适应学习率与稀疏特征

## 核心概念

- **AdaGrad 的核心思想**：AdaGrad（Adaptive Gradient）为每个参数自适应地调整学习率。频繁更新的参数学习率衰减更快（因为累积梯度大），而稀疏更新的参数（如低频特征）学习率衰减更慢，从而获得更大的更新步长。
- **稀疏特征适配**：在自然语言处理等场景中，某些特征（如稀有词）只出现在少量样本中，导致其对应的梯度稀疏且幅值小。AdaGrad 对这类参数保持较高的学习率，使其能从不多的出现机会中充分学习。
- **梯度平方累积**：AdaGrad 累积所有历史梯度的平方和 $G_{t,ii} = \sum_{\tau=1}^t g_{\tau,i}^2$，然后使用 $\eta / \sqrt{G_t + \epsilon}$ 作为自适应学习率。分母中的 $\sqrt{G_t}$ 起到了梯度归一化的作用。
- **学习率单调递减**：由于 $G_t$ 始终单调递增（不断累加平方梯度），AdaGrad 的学习率严格单调递减。这意味着在非凸优化中，AdaGrad 可能在到达最优点之前学习率就已经衰减到接近零，导致无法继续学习。

## 数学推导

**AdaGrad 更新规则**：

设 $g_t = \nabla L(\theta_t)$ 为第 $t$ 步的梯度。

参数 $\theta_i$ 在第 $t$ 步的梯度平方累积：

$$
G_{t,ii} = \sum_{\tau=1}^{t} g_{\tau,i}^2
$$

其中 $G_t \in \mathbb{R}^{d \times d}$ 是对角矩阵，第 $i$ 个对角元素是参数 $\theta_i$ 的历史梯度平方和。

参数更新：

$$
\theta_{t+1,i} = \theta_{t,i} - \frac{\eta}{\sqrt{G_{t,ii} + \epsilon}} \cdot g_{t,i}
$$

向量化形式：

$$
\theta_{t+1} = \theta_t - \eta \cdot (G_t + \epsilon I)^{-1/2} \odot g_t
$$

简单实现形式：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\sqrt{\sum_{\tau=1}^t g_\tau^2 + \epsilon}} \cdot g_t
$$

**梯度影响分析**：

假设参数 $\theta_1$ 梯度大且频繁（频繁特征），$\theta_2$ 梯度小且稀疏（稀有特征）：

经过 $T$ 步：

$$
G_{T,11} = \sum g_{t,1}^2 \gg G_{T,22} = \sum g_{t,2}^2
$$

有效学习率：

$$
\eta_{1}^{\text{eff}} = \frac{\eta}{\sqrt{G_{T,11}}} \ll \eta_{2}^{\text{eff}} = \frac{\eta}{\sqrt{G_{T,22}}}
$$

稀疏特征的学习率远大于频繁特征的学习率。

**RMS 视角**：

AdaGrad 也可以理解为使用梯度的 RMS（Root Mean Square）归一化：

$$
\theta_{t+1} = \theta_t - \frac{\eta}{\text{RMS}[g]_{t,i}} \cdot g_{t,i}
$$

其中 $\text{RMS}[g]_{t,i} = \sqrt{\frac{1}{t} \sum_{\tau=1}^t g_{\tau,i}^2}$。

## 直观理解

AdaGrad 可以类比为"自适应的放大镜"：对于已经看得很清楚的区域（频繁更新的参数），缩小放大倍数（降低学习率）；对于看不清楚的区域（稀疏更新的参数），增加放大倍数（提高学习率）。这样可以用有限的总资源（总学习步数）获得尽可能多的信息。

学习率单调递减的特性就像"烧水"——水一旦烧开（累积梯度足够大），火力就调小。问题在于如果水还没有烧开但火力已经调到了最小（学习率衰减到零），就无法继续加热了。

从信息论的角度看，AdaGrad 体现了"边际效用递减"原则——一个参数更新越多，每次更新的信息量就越少，因此应该降低学习率；而一个很少更新的参数，每次更新都携带大量信息，应该充分利用。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 手动实现 AdaGrad
def adagrad_manual(params, gradients, state, lr=0.01, eps=1e-8):
    """手动实现 AdaGrad"""
    for i, (p, g) in enumerate(zip(params, gradients)):
        state[i] += g ** 2  # 累积梯度平方
        p -= lr / (state[i] + eps).sqrt() * g

# 演示 AdaGrad 对稀疏特征的适应性
torch.manual_seed(42)

# 生成数据: 一个特征是频繁的，另一个是稀疏的
n_samples = 1000
x_freq = torch.randn(n_samples, 1)          # 频繁特征: 每个样本都有值
x_sparse = torch.zeros(n_samples, 1)         # 稀疏特征
sparse_indices = torch.randint(0, n_samples, (50,))  # 只有 50 个非零
x_sparse[sparse_indices] = torch.randn(50, 1)

X = torch.cat([x_freq, x_sparse], dim=1)
y = 2 * x_freq + 5 * x_sparse + torch.randn(n_samples, 1) * 0.1

# 使用 SGD 和 AdaGrad 训练
linear_sgd = nn.Linear(2, 1)
linear_adagrad = nn.Linear(2, 1)

# 复制权重
linear_adagrad.weight.data = linear_sgd.weight.data.clone()
linear_adagrad.bias.data = linear_sgd.bias.data.clone()

opt_sgd = optim.SGD(linear_sgd.parameters(), lr=0.1)
opt_adagrad = optim.Adagrad(linear_adagrad.parameters(), lr=0.5)
criterion = nn.MSELoss()

print("训练 SGD vs AdaGrad:")
for epoch in range(200):
    # SGD
    opt_sgd.zero_grad()
    loss_sgd = criterion(linear_sgd(X), y)
    loss_sgd.backward()
    opt_sgd.step()

    # AdaGrad
    opt_adagrad.zero_grad()
    loss_ada = criterion(linear_adagrad(X), y)
    loss_ada.backward()
    opt_adagrad.step()

    if epoch % 50 == 0:
        print(f"  Epoch {epoch:3d}, SGD Loss: {loss_sgd.item():.4f}, "
              f"AdaGrad Loss: {loss_ada.item():.4f}")

# 检查稀疏特征的权重学习情况
with torch.no_grad():
    w_sgd = linear_sgd.weight[0]
    w_ada = linear_adagrad.weight[0]
    print(f"\n学习到的权重:")
    print(f"  真实值: w_freq=2.0, w_sparse=5.0")
    print(f"  SGD:     w_freq={w_sgd[0]:.4f}, w_sparse={w_sgd[1]:.4f}")
    print(f"  AdaGrad: w_freq={w_ada[0]:.4f}, w_sparse={w_ada[1]:.4f}")

# 可视化学习率变化
print(f"\nAdaGrad 参数状态:")
for name, param in linear_adagrad.named_parameters():
    if param.requires_grad:
        # AdaGrad 累积了梯度平方
        state = opt_adagrad.state[param]
        if 'sum' in state:
            print(f"  {name}: 累积梯度平方和={state['sum'].mean():.4f}")
```

## 深度学习关联

- **词嵌入训练**：AdaGrad 在 NLP 领域特别受欢迎，因为自然语言数据具有典型的长尾分布——常见词（如"the"）频繁出现，稀有词很少出现。AdaGrad 对稀有词保持高学习率的特性使其非常适合词嵌入（Word Embeddings）的训练。
- **被 RMSProp/Adam 取代**：AdaGrad 的学习率单调递减问题在深度学习中是一个严重缺陷。RMSProp 通过引入指数移动平均替代梯度平方的累加，限制了历史梯度的窗口，解决了学习率单调衰减的问题。Adam 在此基础上增加了一阶动量，进一步提升了优化性能。
- **在凸优化中的理论优势**：AdaGrad 在在线凸优化（Online Convex Optimization）中具有理论上的 regret 保证，特别适合处理稀疏特征的非平稳问题。在凸优化领域，AdaGrad 仍然是一个重要的理论基准。
