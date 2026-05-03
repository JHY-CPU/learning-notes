# 23_LAMB 与 LARS：大 Batch 训练优化器

## 核心概念

- **LARS 的定义**：LARS（Layer-wise Adaptive Rate Scaling）是专为大批量（large batch）训练设计的优化器。核心思想是每层使用独立的学习率，基于该层权重的范数与梯度范数的比值进行自适应调整。
- **LAMB 的定义**：LAMB（Layer-wise Adaptive Moments optimizer for Batch training）将 Adam 的自适应矩估计与 LARS 的逐层学习率缩放相结合，同时具备 Adam 的梯度自适应特性和 LARS 的大批量训练稳定性。
- **大批量训练的挑战**：当 batch size 从 256 增加到 8192 甚至更大时，标准 SGD/Adam 的精度会显著下降。这是因为大批量减少了梯度噪声，使模型更容易收敛到尖锐的极小值，泛化能力下降。
- **信任比（Trust Ratio）**：LARS/LAMB 的核心是计算每层的"信任比" $\phi = \|\theta\| / (\|g\| + \epsilon)$。这个比例决定了该层学习率的缩放因子——权重范数相对于梯度范数越大，学习率越大。

## 数学推导

**LARS 更新规则**：

对第 $l$ 层：

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \cdot \phi_l \cdot \left( \beta v_t^{(l)} + g_t^{(l)} \right)
$$

其中信任比（Trust Ratio）定义为：

$$
\phi_l = \frac{\|\theta_t^{(l)}\|_2}{\|g_t^{(l)}\|_2 + \epsilon}
$$

$v_t^{(l)}$ 是动量项，$\beta$ 是动量系数。

完整的 LARS 更新（含权重衰减）：

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \cdot \phi_l \cdot \left( \beta v_t^{(l)} + g_t^{(l)} + \lambda \theta_t^{(l)} \right)
$$

**LAMB 更新规则**：

LAMB 结合 Adam 的一阶/二阶矩估计和 LARS 的信任比：

$$
m_t^{(l)} = \beta_1 m_{t-1}^{(l)} + (1-\beta_1)g_t^{(l)}
$$

$$
v_t^{(l)} = \beta_2 v_{t-1}^{(l)} + (1-\beta_2)(g_t^{(l)})^2
$$

$$
\hat{m}_t = \frac{m_t}{1-\beta_1^t}, \quad \hat{v}_t = \frac{v_t}{1-\beta_2^t}
$$

Adam 更新方向：

$$
u_t^{(l)} = \frac{\hat{m}_t^{(l)}}{\sqrt{\hat{v}_t^{(l)}} + \epsilon}
$$

信任比（LARS 风格）：

$$
r_l = \frac{\|\theta_t^{(l)}\|_2}{\|u_t^{(l)}\|_2 + \epsilon}
$$

最终更新：

$$
\theta_{t+1}^{(l)} = \theta_t^{(l)} - \eta \cdot r_l \cdot \left( u_t^{(l)} + \lambda \theta_t^{(l)} \right)
$$

**信任比的几何含义**：

信任比 $\|\theta\|/\|g\|$ 的分子是权重的范数，分母是梯度的范数。直观上：
- 如果权重很大但梯度很小（$\phi$ 大），说明当前层比较稳定，可以大步更新。
- 如果权重很小但梯度很大（$\phi$ 小），说明当前层剧烈变化，应该保守更新。

## 直观理解

LARS/LAMB 的逐层学习率缩放就像"根据每个人的能力分配任务"——在一个团队中，有的人强壮（权重范数大），可以承担更多工作（大步长）；有的人弱小（权重范数小），应该承担更少工作（小步长）。如果不加区分地统一分配（统一学习率），就会导致有些人负荷过重（梯度爆炸），有些人无事可做（梯度消失）。

在大批量训练中，梯度的方差减小，相当于"每个人的工作评估更加一致"。这时更需要注意个体差异——LARS 的逐层自适应机制正好满足了这一需求。

信任比 $\|\theta\|/\|g\|$ 的物理意义是"当前参数的置信度"：如果权重已经很大（$\|\theta\|$ 大），说明该层的参数已积累了相当的信息，值得信任，可以大步调整；如果梯度突然变得很大（$\|g\|$ 大），说明该层处于不稳定状态，应该谨慎调整。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 手动实现 LARS 和 LAMB 的核心逻辑
class LARS(optim.Optimizer):
    """简化版 LARS"""
    def __init__(self, params, lr=0.01, momentum=0.9, weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            wd = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                if wd != 0:
                    g = g + wd * p

                # 信任比
                param_norm = torch.norm(p)
                grad_norm = torch.norm(g)
                trust_ratio = param_norm / (grad_norm + eps)

                # 动量
                state = self.state[p]
                if 'momentum_buffer' not in state:
                    state['momentum_buffer'] = torch.zeros_like(p)
                v = state['momentum_buffer']
                v.mul_(momentum).add_(g)

                # 更新（使用信任比缩放学习率）
                p.add_(v, alpha=-lr * trust_ratio)

class LAMB(optim.Optimizer):
    """简化版 LAMB"""
    def __init__(self, params, lr=0.001, betas=(0.9, 0.999), weight_decay=0.0, eps=1e-8):
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay, eps=eps)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self):
        for group in self.param_groups:
            lr = group['lr']
            beta1, beta2 = group['betas']
            wd = group['weight_decay']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                g = p.grad
                state = self.state[p]

                # 状态初始化
                if len(state) == 0:
                    state['step'] = 0
                    state['m'] = torch.zeros_like(p)
                    state['v'] = torch.zeros_like(p)

                m, v = state['m'], state['v']
                state['step'] += 1
                t = state['step']

                # Adam 更新方向
                m.mul_(beta1).add_(g, alpha=1 - beta1)
                v.mul_(beta2).add_(g ** 2, alpha=1 - beta2)
                m_hat = m / (1 - beta1 ** t)
                v_hat = v / (1 - beta2 ** t)
                u = m_hat / (v_hat.sqrt() + eps)

                if wd != 0:
                    u = u + wd * p

                # 信任比
                param_norm = torch.norm(p)
                u_norm = torch.norm(u)
                trust_ratio = param_norm / (u_norm + eps)

                p.add_(u, alpha=-lr * trust_ratio)

# 验证大批量训练的效果
torch.manual_seed(42)

def train_with_batch_size(batch_size, opt_class, opt_kwargs, name):
    X = torch.randn(2048, 50)
    y = torch.sin(X.sum(1, keepdim=True))

    model = nn.Sequential(nn.Linear(50, 128), nn.ReLU(), nn.Linear(128, 1))
    optimizer = opt_class(model.parameters(), **opt_kwargs)
    criterion = nn.MSELoss()

    n_batches = 2048 // batch_size
    for epoch in range(100):
        perm = torch.randperm(2048)
        for i in range(n_batches):
            idx = perm[i * batch_size:(i+1) * batch_size]
            pred = model(X[idx])
            loss = criterion(pred, y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

    return loss.item()

print("大批量训练对比 (batch_size=1024):")
for name, opt_class, kwargs in [
    ("SGD", optim.SGD, {"lr": 0.01, "momentum": 0.9}),
    ("LARS", LARS, {"lr": 0.1, "momentum": 0.9}),
    ("Adam", optim.Adam, {"lr": 0.001}),
    ("LAMB", LAMB, {"lr": 0.01}),
]:
    loss = train_with_batch_size(1024, opt_class, kwargs, name)
    print(f"  {name:10s}: loss={loss:.6f}")
```

## 深度学习关联

- **大批量分布式训练**：LARS 和 LAMB 是 Google 在训练大批量模型时提出的优化器。LARS 成功训练了 ResNet-50 在 256 个 GPU 上使用 65536 的 batch size（相比标准 256，扩大 256 倍）而精度不降。LAMB 则成功训练了 BERT 在 1024 个 TPU 上使用 65536 的 batch size。
- **预训练阶段的核心优化器**：LAMB 是 BERT、GPT 等大规模预训练模型的标准优化器。在大规模预训练中，需要极大的 batch size（数万）以利用分布式计算资源，LAMB 的逐层自适应特性使这种大规模并行训练成为可能。
- **训练速度的飞跃**：使用 LARS/LAMB，原本需要数周的训练可以在数小时内完成。例如，BERT 预训练使用 LAMB 可以在 76 分钟内完成（使用 1024 个 TPU），而标准 Adam 需要数天。这极大加速了深度学习研究的迭代速度。
