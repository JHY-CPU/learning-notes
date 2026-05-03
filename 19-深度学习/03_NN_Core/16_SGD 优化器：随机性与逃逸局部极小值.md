# 16_SGD 优化器：随机性与逃逸局部极小值

## 核心概念

- **SGD 基本形式**：随机梯度下降（Stochastic Gradient Descent, SGD）是批量梯度下降的随机近似。每次迭代只使用一个或一小批样本计算梯度：$\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t; x_i, y_i)$。
- **随机性的双重作用**：SGD 的随机性既是缺点也是优点。缺点在于梯度估计有噪声，收敛路径不稳定；优点在于噪声可以帮助模型逃逸局部极小值的"陷阱"，找到更平坦、泛化性更好的极值点。
- **Mini-batch 的折中**：实际使用中几乎不会用单个样本（SGD）或全量数据（BGD），而是使用 mini-batch（通常 32-512 个样本）。Mini-batch 提供了计算效率（向量化并行）和梯度质量之间的良好平衡。
- **收敛性保证**：在适当的衰减学习率下（满足 $\sum \eta_t = \infty$ 且 $\sum \eta_t^2 < \infty$，如 $\eta_t = 1/t$），SGD 保证收敛到局部极小值。凸函数的情况下保证收敛到全局最优。

## 数学推导

**梯度下降的三种变体**：

批量梯度下降（BGD）：

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \frac{1}{N} \sum_{i=1}^{N} \nabla L_i(\theta_t)
$$

随机梯度下降（SGD）：

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \nabla L_i(\theta_t)
$$

Mini-batch SGD：

$$
\theta_{t+1} = \theta_t - \eta_t \cdot \frac{1}{B} \sum_{i=1}^{B} \nabla L_i(\theta_t)
$$

**SGD 的梯度方差**：

设真实梯度为 $g(\theta) = \nabla \mathbb{E}[L(\theta)]$，mini-batch 梯度估计为 $\hat{g}(\theta) = \frac{1}{B} \sum_{i=1}^{B} \nabla L_i(\theta)$。

梯度估计的方差为：

$$
\text{Var}(\hat{g}(\theta)) = \frac{1}{B} \Sigma(\theta)
$$

其中 $\Sigma(\theta)$ 是单个样本梯度的协方差矩阵。方差与 $1/B$ 成正比——batch 越大，梯度越稳定。

**SGD 在非凸优化中的优势**：

对于非凸函数，SGD 的迭代可以建模为：

$$
\theta_{t+1} = \theta_t - \eta \nabla L(\theta_t) + \eta \epsilon_t
$$

其中 $\epsilon_t$ 是噪声项。噪声的存在使得 SGD 可以：

- 逃逸局部极小值：当参数落在尖锐的局部极小值时，随机噪声可以将参数推离
- 偏好平坦极小值：平坦极小值对噪声不敏感，而尖锐极小值容易被噪声推开

**学习率衰减策略**：

常用衰减方案：
- 分段衰减：$\eta_t = \eta_0 \cdot \gamma^{\lfloor t/T \rfloor}$
- 指数衰减：$\eta_t = \eta_0 \cdot e^{-\lambda t}$
- 倒数衰减：$\eta_t = \eta_0 / (1 + \lambda t)$

理论收敛要求：$\sum_{t=1}^{\infty} \eta_t = \infty$ 且 $\sum_{t=1}^{\infty} \eta_t^2 < \infty$。

## 直观理解

SGD 的随机性就像"醉汉下山"——虽然走的路线歪歪扭扭（梯度噪声），但醉酒的不规律步伐反而可能帮他避免掉进浅坑（浅的局部极小值），最终到达更深的山谷。相比之下，全量梯度下降就像"清醒的人下山"，每一步都走最陡的方向，反而容易被困在浅坑里。

Mini-batch 的折中类似于"团队决策"：一个人（SGD）做决定太快但容易冲动；一万人（BGD）做决定太慢但结果可靠；几十个人（mini-batch）既有效率又有质量。

一个关键洞见是：深度学习中，我们通常并不需要找到"全局最优解"，而是需要一个"好的泛化解"。SGD 的随机噪声天然地偏好平坦的极小值，而平坦极小值通常对应更好的泛化性能——因为训练集和测试集的损失曲面稍有偏移时，平坦区域的性能变化不大。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.optim as optim

# 可视化 SGD 的不同 batch size 对训练轨迹的影响
torch.manual_seed(42)

# 简单的非凸函数优化: f(x) = x^4 - 4*x^2 + 0.5*x
def f(x):
    return x**4 - 4*x**2 + 0.5*x

def train_sgd(batch_size, lr=0.01, epochs=100):
    x = torch.tensor([-2.5], requires_grad=True)
    trajectory = [x.item()]
    optimizer = optim.SGD([x], lr=lr)

    for epoch in range(epochs):
        # 模拟带噪声的梯度
        noise = torch.randn(batch_size).mean() * 0.5
        loss = f(x)
        # 手动创建带噪声的梯度（模拟 SGD 的随机性）
        grad = 4 * x**3 - 8 * x + 0.5 + noise
        x.grad = grad.unsqueeze(0)
        optimizer.step()
        trajectory.append(x.item())

    return trajectory

print("不同 batch size 的训练轨迹:")
for bs, color in [(1, "SGD"), (8, "Mini-batch(8)"), (64, "Mini-batch(64)")]:
    traj = train_sgd(bs, lr=0.02, epochs=50)
    print(f"  {color} (batch={bs}): 起点={traj[0]:.2f}, 终点={traj[-1]:.2f}, "
          f"波动范围=[{min(traj):.2f}, {max(traj):.2f}]")

# 实用示例: 在 MNIST 上比较 SGD 不同配置
print("\nPyTorch SGD 优化器 API:")
model = nn.Linear(784, 10)

# 标准 SGD
optimizer = optim.SGD(model.parameters(), lr=0.01)
print(f"  SGD: lr=0.01, momentum=0")

# SGD + Momentum
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
print(f"  SGD: lr=0.01, momentum=0.9")

# SGD + Momentum + Nesterov
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9, nesterov=True)
print(f"  SGD: lr=0.01, momentum=0.9, nesterov=True")

# SGD + Weight Decay (L2 正则化)
optimizer = optim.SGD(model.parameters(), lr=0.01, weight_decay=0.0001)
print(f"  SGD: lr=0.01, weight_decay=0.0001")

# 学习率调度器
optimizer = optim.SGD(model.parameters(), lr=0.1)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
for epoch in range(100):
    # 训练代码...
    scheduler.step()
    if epoch % 30 == 0:
        print(f"  Epoch {epoch}, lr={optimizer.param_groups[0]['lr']:.6f}")
```

## 深度学习关联

- **SGD 的历史地位**：SGD 是深度学习优化器的始祖，AlexNet（2012）等开创性工作都使用 SGD 训练。尽管 Adam 等自适应方法后来居上，SGD + Momentum 仍然在许多任务上表现优异，尤其是在计算机视觉领域。
- **SGD 的泛化优势**：大量研究表明，SGD 找到的解通常比 Adam 找到的解具有更好的泛化性能。这可能是因为 SGD 的随机噪声帮助找到了更平坦的极小值。这一发现催生了"先 Adam 快速收敛，后 SGD 精细微调"的混合策略。
- **学习率调度的必要性**：SGD 的学习率调度是训练成功的关键。从 Warmup（预热）到 Cosine Annealing（余弦退火），学习率策略对最终性能的影响甚至超过优化器本身的选择。One Cycle Policy 等方法专门针对 SGD 设计，大幅提升了训练效率和质量。
