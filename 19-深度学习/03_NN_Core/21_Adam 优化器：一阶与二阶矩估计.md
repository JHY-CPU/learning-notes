# 21_Adam 优化器：一阶与二阶矩估计

## 核心概念

- **Adam 的定义**：Adam（Adaptive Moment Estimation）是目前最广泛使用的深度学习优化器。它结合了 Momentum（一阶矩估计）和 RMSProp（二阶矩估计）的优点，同时维护梯度的一阶矩（均值）和二阶矩（未中心化的方差）。
- **一阶矩估计**：$m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t$，这是梯度的指数加权移动平均，相当于 Momentum。它估计了梯度的期望方向。默认 $\beta_1 = 0.9$。
- **二阶矩估计**：$v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2$，这是梯度平方的指数加权移动平均，相当于 RMSProp。它估计了梯度的幅度（方差）。默认 $\beta_2 = 0.999$。
- **偏差校正**：训练初期，$m_t$ 和 $v_t$ 被初始化为 0，导致估计值偏小。Adam 通过 $\hat{m}_t = m_t/(1-\beta_1^t)$ 和 $\hat{v}_t = v_t/(1-\beta_2^t)$ 进行偏差校正，特别是前几个时间步的校正幅度最大。

## 数学推导

**Adam 完整算法**：

- 初始化：$m_0 = 0$，$v_0 = 0$，$t = 0$
- 对每个时间步 $t = 1, 2, \ldots$：

$$
g_t = \nabla_\theta L(\theta_{t-1})
$$

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1) g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2) g_t^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

$$
\theta_t = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

其中 $\beta_1 = 0.9$，$\beta_2 = 0.999$，$\epsilon = 10^{-8}$。

**偏差校正的数学推导**：

$m_t$ 可以展开为：

$$
m_t = (1-\beta_1) \sum_{i=1}^t \beta_1^{t-i} g_i
$$

取期望：

$$
\mathbb{E}[m_t] = \mathbb{E}[g_t] \cdot (1-\beta_1^t) + \text{其他项}
$$

因此修正因子为 $1/(1-\beta_1^t)$。类似地，$v_t$ 的修正因子为 $1/(1-\beta_2^t)$。

**更新步长的分析**：

Adam 的有效步长为：

$$
\Delta_t = \eta \cdot \frac{\hat{m}_t}{\sqrt{\hat{v}_t}}
$$

这可以理解为"信噪比"（Signal-to-Noise Ratio, SNR）的形式：

$$
\text{SNR} = \frac{\hat{m}_t}{\sqrt{\hat{v}_t}}
$$

当 SNR 大时（梯度方向一致），有效步长大；当 SNR 小时（梯度噪声大），有效步长小。这个自适应特性使得 Adam 对学习率的选择相对不敏感。

## 直观理解

Adam 可以理解为"带着自适应减震器的智能跑步者"：一阶矩 $m_t$ 决定了跑步的方向和惯性（我要往哪个方向跑），二阶矩 $v_t$ 决定了每一步的步幅（地面的崎岖程度决定了应该踩多大力）。

- 如果梯度方向持续一致（如一马平川的下坡），$m_t$ 很大，$v_t$ 相对 $m_t$ 较小，信噪比高，大步前进。
- 如果梯度方向反复振荡（如同在碎石路上奔跑），$v_t$ 很大，信噪比低，减小步幅稳步前进。
- 如果梯度突然变得很小（如到达了平台），$m_t$ 减小但 $v_t$ 也减小，步幅不会突然归零。

偏差校正在训练初期至关重要。好比一个温度计刚放进热水时读数滞后——初始估计 $m_0=0$ 导致的偏差需要几个时间步来纠正。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# Adam 优化器演示
torch.manual_seed(42)

# 手动实现 Adam 核心逻辑
def adam_step(param, grad, m, v, t, lr=0.001, beta1=0.9, beta2=0.999, eps=1e-8):
    m = beta1 * m + (1 - beta1) * grad
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (v_hat.sqrt() + eps)
    return m, v

# 对比 SGD、RMSProp、Adam
def compare_optimizers(steps=50):
    # 使用 Rosenbrock 函数: f(x,y) = (a-x)^2 + b(y-x^2)^2
    a, b = 1, 100

    def rosenbrock_loss(x, y):
        return (a - x) ** 2 + b * (y - x ** 2) ** 2

    results = {}
    for name, opt_class, kwargs in [
        ("SGD", optim.SGD, {"lr": 0.001}),
        ("Adam", optim.Adam, {"lr": 0.1}),
        ("RMSProp", optim.RMSprop, {"lr": 0.01}),
    ]:
        x = torch.tensor([0.0, 0.0], requires_grad=True)
        optimizer = opt_class([x], **kwargs)

        for _ in range(steps):
            optimizer.zero_grad()
            loss = rosenbrock_loss(x[0], x[1])
            loss.backward()
            optimizer.step()

        results[name] = (x[0].item(), x[1].item(), loss.item())

    return results

results = compare_optimizers()
print("Rosenbrock 函数优化对比 (最优点: x=1, y=1, loss=0):")
for name, (x, y, loss) in results.items():
    print(f"  {name:10s}: x={x:.4f}, y={y:.4f}, loss={loss:.6f}")

# 训练简单的神经网络
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(100, 256), nn.ReLU(),
            nn.Linear(256, 128), nn.ReLU(),
            nn.Linear(128, 10),
        )

    def forward(self, x):
        return self.net(x)

print("\n不同优化器在 MLP 上的表现:")
X = torch.randn(500, 100)
y = torch.randint(0, 10, (500,))
criterion = nn.CrossEntropyLoss()

for name, opt_class, kwargs in [
    ("SGD", optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    ("RMSProp", optim.RMSprop, {"lr": 0.001}),
    ("Adam", optim.Adam, {"lr": 0.001}),
    ("AdamW", optim.AdamW, {"lr": 0.001}),
]:
    torch.manual_seed(42)
    model = SimpleNet()
    optimizer = opt_class(model.parameters(), **kwargs)

    for epoch in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    print(f"  {name:10s}: final_loss={loss.item():.4f}")
```

## 深度学习关联

- **默认优化器的首选**：Adam 是目前深度学习中最常用的默认优化器。在 NLP、CV、强化学习等几乎所有领域，Adam 都是首选的起点。它对超参数（尤其是学习率）不敏感，通常 $\text{lr}=3\times10^{-4}$ 就能取得不错的效果。
- **Transformer 的标准选择**：几乎所有 Transformer 架构（BERT、GPT、ViT）都使用 Adam 或其变体 AdamW 进行训练。Transformer 训练对优化器的稳定性要求高，Adam 的自适应特性恰好满足这一需求。
- **局限与改进**：Adam 在某些任务（如图像分类）上的泛化性能可能不如 SGD+Momentum。这促使了 AdamW（解耦权重衰减）、RAdam（修正方差）、Nadam（引入 Nesterov 动量）等变体的出现。AdamW 已成为 NLP 领域的新标准。
