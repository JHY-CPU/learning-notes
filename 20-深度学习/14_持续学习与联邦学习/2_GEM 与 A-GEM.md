# 2_GEM 与 A-GEM

## 1. 算法概述

GEM (Gradient Episodic Memory) 通过**梯度约束**解决持续学习中的遗忘问题：确保在更新当前任务时，不会增大旧任务的损失。

### 1.1 核心思想

```
传统梯度下降:
  θ ← θ - η∇L_t(θ)    ← 可能增大旧任务损失

GEM 梯度投影:
  g = ∇L_t(θ)          ← 当前任务梯度
  检查: g·∇L_k(θ) ≥ 0?  ← 对所有旧任务 k
  如果否: 将 g 投影到可行域
  θ ← θ - η·g_projected
```

## 2. GEM 算法

### 2.1 数学形式化

给定当前任务 $t$ 的梯度 $g = \nabla \mathcal{L}_t(\theta)$，GEM 求解以下约束优化问题：

$$\tilde{g} = \arg\min_{\tilde{g}} \frac{1}{2} \|g - \tilde{g}\|_2^2 \quad \text{s.t.} \quad \tilde{g}^T g_k \geq 0, \quad \forall k < t$$

其中 $g_k = \nabla \mathcal{L}_k(\theta)$ 是旧任务 $k$ 的梯度。

### 2.2 PyTorch 实现

```python
import torch
import torch.nn.functional as F
import numpy as np
from scipy.optimize import minimize

class GEM:
    """
    Gradient Episodic Memory 实现
    """
    def __init__(self, model, memory_size_per_task=200):
        self.model = model
        self.memory_size = memory_size_per_task
        self.memory = {}  # 每个任务的代表性样本
        self.task_gradients = {}  # 存储旧任务梯度

    def store_in_memory(self, task_id, dataloader):
        """
        存储当前任务的代表性样本
        """
        samples = []
        labels = []
        for x, y in dataloader:
            samples.append(x)
            labels.append(y)
            if len(samples) * x.size(0) >= self.memory_size:
                break

        self.memory[task_id] = (
            torch.cat(samples)[:self.memory_size],
            torch.cat(labels)[:self.memory_size]
        )

    def compute_gradient(self, task_id):
        """
        计算指定任务的梯度
        """
        if task_id not in self.memory:
            return None

        x, y = self.memory[task_id]
        self.model.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        grad = torch.cat([
            p.grad.flatten().detach()
            for p in self.model.parameters()
            if p.grad is not None
        ])

        return grad

    def project_gradient(self, current_grad):
        """
        将当前梯度投影到可行域
        确保投影后的梯度与所有旧任务梯度的内积 ≥ 0
        """
        constraints = []

        for task_id in sorted(self.memory.keys()):
            old_grad = self.compute_gradient(task_id)
            if old_grad is not None:
                constraints.append(old_grad)

        if not constraints:
            return current_grad

        # 构建约束矩阵
        G = torch.stack(constraints)  # [num_tasks, num_params]

        # 求解 QP 问题
        def solve_qp(g, G):
            """
            min ||g_tilde - g||^2
            s.t. G @ g_tilde >= 0
            """
            n = len(g)

            # 使用投影方法
            g_np = g.cpu().numpy()
            G_np = G.cpu().numpy()

            # 检查是否违反约束
            violations = G_np @ g_np
            if np.all(violations >= 0):
                return g  # 无需投影

            # 简化: 使用约束最违反的梯度进行投影
            worst_idx = np.argmin(violations)
            worst_grad = G_np[worst_idx]

            # 将 g 投影到与 worst_grad 正交的超平面
            g_projected = g_np - (np.dot(g_np, worst_grad) /
                                   (np.dot(worst_grad, worst_grad) + 1e-8)
                                   ) * worst_grad

            return torch.from_numpy(g_projected).float().to(g.device)

        return solve_qp(current_grad, G)

    def step(self, x, y, optimizer):
        """
        GEM 优化步骤
        """
        # 计算当前任务梯度
        self.model.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        # 收集梯度
        current_grad = torch.cat([
            p.grad.flatten().detach()
            for p in self.model.parameters()
            if p.grad is not None
        ])

        # 投影梯度
        projected_grad = self.project_gradient(current_grad)

        # 将投影后的梯度放回参数
        idx = 0
        for p in self.model.parameters():
            if p.grad is not None:
                size = p.numel()
                p.grad.copy_(
                    projected_grad[idx:idx+size].view_as(p)
                )
                idx += size

        optimizer.step()
        return loss.item()
```

## 3. A-GEM 算法

### 3.1 动机

GEM 的 QP 求解在任务数量多时效率低。A-GEM (Averaged GEM) 使用更简单的约束：

$$\tilde{g}^T g_{\text{ref}} \geq 0$$

其中 $g_{\text{ref}}$ 是所有旧任务梯度的平均值。

### 3.2 优势

| 特性 | GEM | A-GEM |
|------|-----|-------|
| 约束数量 | $T-1$ 个 | 1 个 |
| 计算复杂度 | $O(T \cdot n)$ (QP) | $O(n)$ (直接投影) |
| 投影速度 | 慢 | **快** |
| 鲁棒性 | 更严格 | 略弱 |

### 3.3 A-GEM 实现

```python
class AGEM:
    """
    Averaged GEM 实现
    """
    def __init__(self, model, memory_size_per_task=200):
        self.model = model
        self.memory_size = memory_size_per_task
        self.memory = {}

    def store_in_memory(self, task_id, dataloader):
        """存储代表性样本"""
        samples, labels = [], []
        for x, y in dataloader:
            samples.append(x)
            labels.append(y)
            if len(samples) * x.size(0) >= self.memory_size:
                break
        self.memory[task_id] = (
            torch.cat(samples)[:self.memory_size],
            torch.cat(labels)[:self.memory_size]
        )

    def compute_reference_gradient(self):
        """
        计算所有旧任务的平均参考梯度
        """
        ref_grad = None
        count = 0

        for task_id in self.memory:
            x, y = self.memory[task_id]
            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()

            grad = torch.cat([
                p.grad.flatten().detach()
                for p in self.model.parameters()
                if p.grad is not None
            ])

            if ref_grad is None:
                ref_grad = grad.clone()
            else:
                ref_grad += grad
            count += 1

        if count > 0:
            ref_grad /= count

        return ref_grad

    def project_gradient(self, current_grad, ref_grad):
        """
        A-GEM 投影: 保证 g_tilde · g_ref >= 0
        """
        dot_product = torch.dot(current_grad, ref_grad)

        if dot_product >= 0:
            return current_grad  # 不违反约束

        # 投影: g_tilde = g - (g·g_ref / ||g_ref||²) * g_ref
        ref_norm_sq = torch.dot(ref_grad, ref_grad)
        projected = current_grad - (dot_product / (ref_norm_sq + 1e-8)) * ref_grad

        return projected

    def step(self, x, y, optimizer):
        """A-GEM 优化步骤"""
        # 计算当前梯度
        self.model.zero_grad()
        output = self.model(x)
        loss = F.cross_entropy(output, y)
        loss.backward()

        current_grad = torch.cat([
            p.grad.flatten().detach()
            for p in self.model.parameters()
            if p.grad is not None
        ])

        # 计算参考梯度
        ref_grad = self.compute_reference_gradient()

        if ref_grad is not None:
            # 投影
            projected = self.project_gradient(current_grad, ref_grad)

            # 放回参数
            idx = 0
            for p in self.model.parameters():
                if p.grad is not None:
                    size = p.numel()
                    p.grad.copy_(projected[idx:idx+size].view_as(p))
                    idx += size

        optimizer.step()
        return loss.item()
```

## 4. GEM vs A-GEM 训练示例

```python
def gem_training_loop(model, tasks, method='gem', memory_size=200,
                       epochs_per_task=10):
    """
    GEM/A-GEM 持续学习训练
    """
    if method == 'gem':
        gem = GEM(model, memory_size)
    else:
        gem = AGEM(model, memory_size)

    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for task_id, train_loader in enumerate(tasks):
        print(f"\n=== Task {task_id + 1} ===")

        # 存储代表性样本
        gem.store_in_memory(task_id, train_loader)

        for epoch in range(epochs_per_task):
            model.train()
            total_loss = 0

            for x, y in train_loader:
                loss = gem.step(x.cuda(), y.cuda(), optimizer)
                total_loss += loss

            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

        evaluate_all_tasks(model, tasks[:task_id+1])
```

## 5. 方法对比

| 方法 | 约束形式 | 计算成本 | 内存 | 遗忘控制 |
|------|----------|----------|------|----------|
| EWC | 参数级正则化 | 中 | Fisher 存储 | 中 |
| GEM | 梯度投影 (多约束) | 高 | 回放记忆 | **强** |
| A-GEM | 梯度投影 (单约束) | 低 | 回放记忆 | 中 |

## 6. 总结

| 要点 | 说明 |
|------|------|
| GEM 核心 | 约束梯度方向，不增大旧任务损失 |
| A-GEM 简化 | 用平均参考梯度替代多个约束 |
| 关键技巧 | 梯度投影、记忆回放 |
| 优势 | 直接控制遗忘、无需 Fisher 计算 |

---

**参考文献：**

1. Lopez-Paz, D. & Ranzato, M. (2017). *Gradient episodic memory for continual learning*. NeurIPS.
2. Chaudhry, A. et al. (2019). *Efficient lifelong learning with A-GEM*. ICLR.
