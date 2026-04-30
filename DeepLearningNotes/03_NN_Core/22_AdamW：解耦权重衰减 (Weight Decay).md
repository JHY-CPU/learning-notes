# 22 AdamW：解耦权重衰减 (Weight Decay)

## 核心概念

- **标准 Adam 中权重衰减的问题**：在标准 Adam 中，权重衰减（L2 正则化）是通过在损失函数中添加 $\frac{\lambda}{2}\|\theta\|^2$ 项实现的。但 Adam 的自适应学习率会破坏权重衰减的效果，使得正则化与优化过程耦合，导致正则化效果不佳。

- **解耦的核心思想**：AdamW 将权重衰减从梯度更新中分离出来。具体来说，在 Adam 的更新步骤之后，独立地对参数施加一个与学习率无关的权重衰减：$\theta_{t+1} = \theta_t - \eta \cdot \text{AdamUpdate}(g_t) - \eta\lambda\theta_t$。

- **L2 正则化 vs 权重衰减**：在 SGD 中，L2 正则化与权重衰减是等价的。但在 Adam 中，两者不再等价。AdamW 的作者证明了，在使用自适应学习率时，权重衰减应该独立于梯度更新，而不是作为损失函数的一部分。

- **实际收益**：AdamW 在多个任务上显著优于标准 Adam，特别是 Transformer 训练和大规模模型。它提供了更好的泛化性能，且对权重衰减超参数 $\lambda$ 的选择更加鲁棒。

## 数学推导

**标准 Adam + L2 正则化**：

在损失中加入 L2 正则项：

$$
\tilde{L}(\theta) = L(\theta) + \frac{\lambda}{2}\|\theta\|^2
$$

梯度变为：

$$
\tilde{g}_t = \nabla L(\theta_t) + \lambda\theta_t
$$

然后应用 Adam 更新（注意 $\tilde{g}_t$ 经过自适应学习率缩放）：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)\tilde{g}_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)\tilde{g}_t^2
$$

$$
\theta_{t+1} = \theta_t - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

问题：L2 正则项 $\lambda\theta_t$ 和损失梯度 $\nabla L(\theta_t)$ 被同样的自适应学习率缩放。但实际上，权重衰减应该对所有参数施加"相同比例"的衰减，而不是根据梯度历史自适应地调整。

**AdamW 更新**：

1. 计算损失梯度 $g_t = \nabla L(\theta_t)$（不含 L2 项）
2. 用 Adam 更新参数：

$$
m_t = \beta_1 m_{t-1} + (1-\beta_1)g_t
$$

$$
v_t = \beta_2 v_{t-1} + (1-\beta_2)g_t^2
$$

3. 参数更新（先 Adam 更新，再独立衰减）：

$$
\theta_t^{(1)} = \theta_{t-1} - \eta \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon}
$$

$$
\theta_t = \theta_t^{(1)} - \eta \lambda \theta_{t-1}
$$

合并为一步：

$$
\theta_t = \theta_{t-1} - \eta \left( \frac{\hat{m}_t}{\sqrt{\hat{v}_t} + \epsilon} + \lambda \theta_{t-1} \right)
$$

**等价解释**：

与标准 Adam 的关键区别：标准 Adam 对 $\lambda\theta_t$ 也除以 $\sqrt{\hat{v}_t}$ 进行缩放，而 AdamW 不做这种缩放。在 SGD 中两者等价（因为 SGD 没有自适应缩放），在 Adam 中两者不同。

## 直观理解

AdamW 的解耦可以类比为"给花园浇水时分开施肥"：标准 Adam 是把肥料混在水里一起浇（L2 正则和梯度一起被自适应学习率处理），而 AdamW 是浇完水后单独施肥（权重衰减独立进行）。

为什么解耦重要？自适应学习率的本质是"给每个参数不同的学习率"——频繁更新的参数学习率小，稀疏更新的参数学习率大。但如果把权重衰减也放到自适应学习率中，相当于"频繁更新的参数被正则化更少（因为学习率小导致衰减量小），稀疏更新的参数被正则化更多（因为学习率大导致衰减量大）"，这与权重衰减的本意（对所有参数施加同等比例的衰减）相违背。

AdamW 解耦后，无论参数的学习率是多少，权重衰减都保持相同的比例。这使得正则化更加公平和有效。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 手动实现区分标准 Adam+L2 和 AdamW
def adam_l2_step(param, grad, m, v, t, lr, weight_decay, beta1=0.9, beta2=0.999):
    """标准 Adam + L2 正则化（错误的方式）"""
    grad_with_decay = grad + weight_decay * param
    m = beta1 * m + (1 - beta1) * grad_with_decay
    v = beta2 * v + (1 - beta2) * grad_with_decay ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (v_hat.sqrt() + 1e-8)
    return m, v

def adamw_step(param, grad, m, v, t, lr, weight_decay, beta1=0.9, beta2=0.999):
    """AdamW（解耦权重衰减）"""
    m = beta1 * m + (1 - beta1) * grad  # 只用损失梯度
    v = beta2 * v + (1 - beta2) * grad ** 2
    m_hat = m / (1 - beta1 ** t)
    v_hat = v / (1 - beta2 ** t)
    param -= lr * m_hat / (v_hat.sqrt() + 1e-8)  # Adam 更新
    param -= lr * weight_decay * param  # 解耦的权重衰减
    return m, v

# 演示 AdamW 优于标准 Adam + L2
torch.manual_seed(42)

def train_with_regularization(opt_class, opt_kwargs, name):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(100, 256), nn.ReLU(),
        nn.Linear(256, 256), nn.ReLU(),
        nn.Linear(256, 10)
    )
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    criterion = nn.CrossEntropyLoss()

    X_train = torch.randn(800, 100)
    y_train = torch.randint(0, 10, (800,))
    X_test = torch.randn(200, 100)
    y_test = torch.randint(0, 10, (200,))

    for epoch in range(200):
        pred = model(X_train)
        loss = criterion(pred, y_train)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    with torch.no_grad():
        train_acc = (model(X_train).argmax(1) == y_train).float().mean()
        test_acc = (model(X_test).argmax(1) == y_test).float().mean()

    return train_acc.item(), test_acc.item()

# 对比
print("Adam 不同正则化策略对比:")
for name, opt_class, kwargs in [
    ("Adam (no reg)", optim.Adam, {"lr": 0.001}),
    ("Adam + L2", optim.Adam, {"lr": 0.001, "weight_decay": 0.01}),
    ("AdamW", optim.AdamW, {"lr": 0.001, "weight_decay": 0.01}),
]:
    train_acc, test_acc = train_with_regularization(opt_class, kwargs, name)
    print(f"  {name:20s}: train={train_acc:.4f}, test={test_acc:.4f}, "
          f"gap={train_acc-test_acc:.4f}")

# 验证 SGD 中 L2 正则化和权重衰减等价
print("\nSGD 中等价性验证:")
x_sgd_l2 = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
x_sgd_wd = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
lr, wd = 0.01, 0.001

opt_l2 = optim.SGD([x_sgd_l2], lr=lr, weight_decay=wd)
opt_wd = optim.SGD([x_sgd_wd], lr=lr)

# 手动对 opt_wd 施加权重衰减
loss = (x_sgd_l2 ** 2).sum()
opt_l2.zero_grad()
loss.backward()
opt_l2.step()

# 对 x_sgd_wd 手劝施加权重衰减
with torch.no_grad():
    x_sgd_wd -= lr * wd * x_sgd_wd
loss2 = (x_sgd_wd ** 2).sum()
opt_wd.zero_grad()
loss2.backward()
opt_wd.step()

print(f"  SGD+L2: {x_sgd_l2.data}")
print(f"  SGD+WD: {x_sgd_wd.data}")
print(f"  等价: {torch.allclose(x_sgd_l2.data, x_sgd_wd.data, atol=1e-7)}")
```

## 深度学习关联

- **Transformer 训练的标准**：AdamW 已成为训练 Transformer 模型（BERT、GPT、ViT、LLaMA 等）的事实标准。Hugging Face Transformers 库的默认优化器就是 AdamW。

- **大模型训练的基础**：在大规模模型训练中，权重衰减是防止过拟合的关键技术。AdamW 的解耦特性使得可以独立调节学习率和权重衰减，简化了超参数调优。LLaMA、GPT-3 等大模型都使用 AdamW 或其变体。

- **超参数解耦带来的便利**：AdamW 将权重衰减从学习率中解耦，使得可以独立调节这两个关键超参数。在实践中，这通常意味着学习率设为 $3\times10^{-4}$，权重衰减设为 0.01 就能取得不错的效果，减少了调参工作量。
