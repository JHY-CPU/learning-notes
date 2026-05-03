# 54_元学习 (Meta-Learning) 与 MAML 算法

## 核心概念

- **元学习的定义**：元学习（Meta-Learning），也称为"学习如何学习"，目标是让模型学会快速适应新任务。在元学习设置中，训练阶段包含大量相关任务（称为元训练），模型从这些任务中学习通用的先验知识，使其能够仅用少量样本就能适应新任务。
- **MAML 算法**：MAML（Model-Agnostic Meta-Learning）是元学习的代表性算法。其核心思想是学习一个优秀的参数初始化，使得从这个初始化出发，在新任务上进行少量几步梯度更新就能取得良好性能。
- **与预训练的区别**：传统预训练学习通用的特征表示，然后微调。MAML 学习的是"参数空间中的敏感方向"——即参数应该处于这样的初始位置：稍微调整就能适应新任务。它寻找的是"最容易被微调"的初始化。
- **任务分布**：元学习需要从任务分布 $p(\mathcal{T})$ 中采样大量任务。每个任务 $\mathcal{T}_i$ 包含支持集（support set，用于快速适应）和查询集（query set，用于评估适应后的性能）。

## 数学推导

**双层优化**：

MAML 的优化目标是一个双层优化问题：

内循环（inner loop）：在每个任务 $\mathcal{T}_i$ 上执行 $K$ 步梯度更新：

$$
\theta'_i = \theta - \alpha \nabla_\theta L_{\mathcal{T}_i}(f_\theta)
$$

外循环（outer loop）：优化 $\theta$ 使 $\theta'_i$ 在任务 $\mathcal{T}_i$ 上的损失最小：

$$
\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} L_{\mathcal{T}_i}(f_{\theta'_i})
$$

完整的 MAML 目标函数：

$$
\min_\theta \sum_{\mathcal{T}_i} L_{\mathcal{T}_i}\left(f_{\theta - \alpha \nabla_\theta L_{\mathcal{T}_i}(f_\theta)}\right)
$$

**元梯度计算**：

外循环的梯度通过 $\theta'_i$ 反向传播到 $\theta$：

$$
\nabla_\theta L_{\mathcal{T}_i}(f_{\theta'_i}) = \frac{\partial L_{\mathcal{T}_i}(f_{\theta'_i})}{\partial \theta'_i} \cdot \frac{\partial \theta'_i}{\partial \theta}
$$

其中 $\frac{\partial \theta'_i}{\partial \theta}$ 展开为：

$$
\frac{\partial \theta'_i}{\partial \theta} = I - \alpha \frac{\partial^2 L_{\mathcal{T}_i}(f_\theta)}{\partial \theta^2}
$$

这涉及到二阶梯度（Hessian 矩阵），计算开销较大。

**一阶近似 （FOMAML）**：

为了简化计算，FOMAML（First-Order MAML）忽略二阶梯度项：

$$
\nabla_\theta L_{\mathcal{T}_i}(f_{\theta'_i}) \approx \frac{\partial L_{\mathcal{T}_i}(f_{\theta'_i})}{\partial \theta'_i}
$$

一阶近似在实践中往往也表现不错，因为 Hessian 矩阵的贡献通常较小。

## 直观理解

MAML 的直觉可以类比为"掌握学习的方法论"。它不是在教学生具体的知识（如微积分公式），而是在教学生"如何快速学习"——使得学生拿到任何新教材时，都能在几分钟内掌握核心内容。

MAML 寻找的初始化就像"处于十字路口的旅行者"——无论你接下来想去哪个方向（新任务），你只需要稍微迈一步（几步梯度更新）就能到达目的地。好的初始化是"对各方向都敏感"的位置。

内循环和外循环的双层优化可以理解为"考试"：
- 内循环：考前复习（在支持集上快速学习）
- 外循环：考出好成绩（在查询集上评估）

元学习的目标是找到一套复习方法（初始化），使得在考前短时间内（几步更新）能取得好成绩。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 简化的 MAML 实现（用于少样本回归）
class MAMLModel(nn.Module):
    """MAML 使用的基模型"""
    def __init__(self, input_dim=1, hidden_dim=40):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x)

    def clone(self):
        """创建模型的副本（用于内循环）"""
        clone = MAMLModel()
        clone.load_state_dict(self.state_dict())
        return clone

def maml_inner_loop(model, support_x, support_y, inner_lr=0.01, inner_steps=5):
    """MAML 内循环：在支持集上快速适应"""
    adapted_model = model.clone()
    for _ in range(inner_steps):
        pred = adapted_model(support_x)
        loss = F.mse_loss(pred, support_y)
        grad = torch.autograd.grad(loss, adapted_model.parameters())
        for param, g in zip(adapted_model.parameters(), grad):
            param.data -= inner_lr * g
    return adapted_model

def maml_outer_step(model, task_batch, outer_lr=0.001, inner_lr=0.01):
    """MAML 外循环：元更新"""
    meta_loss = 0.0
    for support_x, support_y, query_x, query_y in task_batch:
        # 内循环：在支持集上适应
        adapted_model = maml_inner_loop(model, support_x, support_y, inner_lr)

        # 外循环：在查询集上计算元损失
        pred = adapted_model(query_x)
        meta_loss += F.mse_loss(pred, query_y)

    meta_loss /= len(task_batch)

    # 元梯度更新
    grad = torch.autograd.grad(meta_loss, model.parameters())
    for param, g in zip(model.parameters(), grad):
        param.data -= outer_lr * g

    return meta_loss.item()

# 生成少样本回归任务
def sample_sine_task(batch_size=10, num_samples=5):
    """采样正弦波回归任务"""
    amp = np.random.uniform(0.1, 5.0)
    phase = np.random.uniform(0, np.pi)
    x = np.random.uniform(-5, 5, (batch_size, num_samples, 1))
    y = amp * np.sin(x + phase)

    query_x = np.random.uniform(-5, 5, (batch_size, num_samples, 1))
    query_y = amp * np.sin(query_x + phase)

    return (
        torch.FloatTensor(x), torch.FloatTensor(y),
        torch.FloatTensor(query_x), torch.FloatTensor(query_y)
    )

# 训练 MAML
torch.manual_seed(42)

model = MAMLModel(1, 40)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

print("MAML 元学习训练:")
for meta_iter in range(100):
    # 采样一批任务
    task_batch = sample_sine_task(batch_size=4, num_samples=10)
    # 转换为列表格式
    tasks = list(zip(*task_batch))
    tasks = [(tasks[i][0], tasks[i][1], tasks[i][2], tasks[i][3]) for i in range(4)]

    meta_loss = maml_outer_step(model, tasks, outer_lr=0.001, inner_lr=0.01)

    if meta_iter % 20 == 0:
        print(f"  Meta iter {meta_iter:3d}, Meta loss: {meta_loss:.4f}")

# 测试 MAML 的快速适应能力
print("\n测试快速适应:")
# 新任务
task = sample_sine_task(batch_size=1, num_samples=5)
support_x, support_y, query_x, query_y = [t[0] for t in task]

# 适应前
with torch.no_grad():
    pred_before = model(query_x)
    loss_before = F.mse_loss(pred_before, query_y).item()

# 适应后（5 步梯度更新）
adapted = maml_inner_loop(model, support_x, support_y, inner_lr=0.01, inner_steps=5)
with torch.no_grad():
    pred_after = adapted(query_x)
    loss_after = F.mse_loss(pred_after, query_y).item()

print(f"  适应前损失: {loss_before:.4f}")
print(f"  适应后损失 (5 步): {loss_after:.4f}")
print(f"  改进: {(loss_before - loss_after)/loss_before*100:.1f}%")
```

## 深度学习关联

- **快速适应新任务**：MAML 在少样本分类、少样本回归、强化学习等领域都有成功应用。训练好的 MAML 模型可以在 1-5 个样本上快速适应新类别或新环境，这对于数据稀缺的应用场景（如医疗影像、机器人控制）具有重要意义。
- **与强化学习的结合**：MAML 可以应用于元强化学习（Meta-RL），使智能体在新环境中快速学习最优策略。在 MuJoCo 等连续控制任务中，MAML 训练的策略可以在少量交互后适应新的动力学参数或奖励函数。
- **高阶优化的挑战**：MAML 需要计算二阶梯度（Hessian 向量积），计算开销大。后续工作如 Reptile（更简单的一阶元学习算法）、iMAML（隐式 MAML）试图在保持性能的同时降低计算复杂度。Reptile 只需一阶梯度，实现更为简单。
