# 1_EWC 弹性权重固化

## 1. 算法概述

EWC (Elastic Weight Consolidation) 由 Kirkpatrick 等人 (2017) 提出，灵感来自神经科学中**突触巩固**的机制：重要突触在学习新知识时保持稳定。

核心思想：**识别对旧任务重要的参数，限制其变化幅度**。

## 2. 理论推导

### 2.1 贝叶斯视角

从贝叶斯角度，持续学习可以看作**顺序贝叶斯推断**：

$$p(\theta | \mathcal{D}_1, \mathcal{D}_2) \propto p(\mathcal{D}_2 | \theta) \cdot p(\theta | \mathcal{D}_1)$$

在学习任务 2 后，后验概率应同时考虑任务 1 和任务 2 的数据。

### 2.2 Fisher 信息矩阵

对后验 $p(\theta | \mathcal{D}_1)$ 做拉普拉斯近似（以 $\theta_1^*$ 为中心的高斯近似），Fisher 信息矩阵为：

$$F = \mathbb{E}_{(x,y) \sim \mathcal{D}_1} \left[ \nabla_\theta \log p(y | x, \theta) \cdot \nabla_\theta \log p(y | x, \theta)^T \right] \bigg|_{\theta = \theta_1^*}$$

Fisher 信息矩阵衡量每个参数对旧任务的**重要性**：
- $F_{ii}$ 大：参数 $\theta_i$ 对旧任务重要，应限制变化
- $F_{ii}$ 小：参数 $\theta_i$ 对旧任务不重要，可自由变化

### 2.3 EWC 损失函数

$$\mathcal{L}_{\text{EWC}}(\theta) = \mathcal{L}_t(\theta) + \frac{\lambda}{2} \sum_i F_i (\theta_i - \theta_{t-1,i}^*)^2$$

其中：
- $\mathcal{L}_t(\theta)$: 当前任务 $t$ 的损失
- $\lambda$: 正则化强度 (平衡新旧任务)
- $F_i$: 参数 $i$ 的 Fisher 信息 (对角线近似)
- $\theta_{t-1}^*$: 学习完任务 $t-1$ 后的最优参数

## 3. PyTorch 实现

### 3.1 完整 EWC 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

class EWC:
    """
    弹性权重固化 (EWC) 实现
    """
    def __init__(self, model, dataset, device='cuda', num_samples=200):
        self.model = model
        self.device = device
        self.num_samples = num_samples

        # 存储旧任务的最优参数
        self.star_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }

        # 计算 Fisher 信息矩阵 (对角线近似)
        self.fisher = self.compute_fisher(dataset)

    def compute_fisher(self, dataset):
        """
        计算 Fisher 信息矩阵的对角线近似
        """
        fisher = {
            n: torch.zeros_like(p)
            for n, p in self.model.named_parameters()
        }

        self.model.eval()
        dataloader = DataLoader(dataset, batch_size=1, shuffle=True)

        for i, (x, y) in enumerate(dataloader):
            if i >= self.num_samples:
                break

            x, y = x.to(self.device), y.to(self.device)

            self.model.zero_grad()
            output = self.model(x)
            loss = F.cross_entropy(output, y)
            loss.backward()

            # 累积梯度平方 (Fisher 对角线)
            for n, p in self.model.named_parameters():
                if p.grad is not None:
                    fisher[n] += p.grad.detach() ** 2

        # 取平均
        for n in fisher:
            fisher[n] /= self.num_samples

        return fisher

    def penalty(self):
        """
        计算 EWC 正则化惩罚项
        """
        loss = 0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss += (self.fisher[n] * (p - self.star_params[n]) ** 2).sum()
        return loss / 2

    def update(self, model, dataset):
        """
        学习完新任务后更新 EWC 状态
        """
        self.model = model
        self.star_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }
        # 累积 Fisher (可选: 多任务累积)
        new_fisher = self.compute_fisher(dataset)
        for n in self.fisher:
            self.fisher[n] += new_fisher[n]
```

### 3.2 EWC 训练循环

```python
def ewc_train(model, tasks, optimizer, ewc_lambda=5000, epochs_per_task=10):
    """
    EWC 持续学习训练
    """
    ewc_list = []  # 存储每个任务的 EWC 约束

    for task_id, (train_loader, test_loader) in enumerate(tasks):
        print(f"\n=== Training Task {task_id + 1} ===")

        for epoch in range(epochs_per_task):
            model.train()
            total_loss = 0

            for x, y in train_loader:
                x, y = x.to('cuda'), y.to('cuda')

                # 前向传播
                output = model(x)
                task_loss = F.cross_entropy(output, y)

                # EWC 正则化损失
                ewc_loss = 0
                for ewc in ewc_list:
                    ewc_loss += ewc.penalty()

                total = task_loss + (ewc_lambda / 2) * ewc_loss

                optimizer.zero_grad()
                total.backward()
                optimizer.step()

                total_loss += task_loss.item()

            print(f"  Epoch {epoch+1}: Loss = {total_loss/len(train_loader):.4f}")

        # 评估所有已学任务
        evaluate_all_tasks(model, tasks[:task_id+1])

        # 计算并存储当前任务的 Fisher 信息
        train_dataset = train_loader.dataset
        ewc = EWC(model, train_dataset)
        ewc_list.append(ewc)

        print(f"  Task {task_id+1} completed. EWC constraints: {len(ewc_list)}")
```

### 3.3 在线 EWC 变体

```python
class OnlineEWC:
    """
    在线 EWC: 使用指数移动平均更新 Fisher
    解决 Fisher 随任务累积导致正则化过强的问题
    """
    def __init__(self, model, gamma=0.9):
        self.model = model
        self.gamma = gamma  # 衰减因子
        self.fisher = {
            n: torch.zeros_like(p)
            for n, p in model.named_parameters()
        }
        self.star_params = {
            n: p.clone().detach()
            for n, p in model.named_parameters()
        }

    def update(self, new_fisher):
        """指数移动平均更新"""
        for n in self.fisher:
            self.fisher[n] = (self.gamma * self.fisher[n] +
                             (1 - self.gamma) * new_fisher[n])
        self.star_params = {
            n: p.clone().detach()
            for n, p in self.model.named_parameters()
        }
```

## 4. Fisher 信息矩阵的深入理解

### 4.1 几何解释

```
参数空间示意:

θ₂
  |        / Fisher 高 → 允许变化范围小
  |   ●---●
  |  /    |
  | /     |  ← 旧任务最优参数 θ*
  |/      |
  ─────────── θ₁
       Fisher 低 → 允许变化范围大

椭圆: 等损失等高线 (Fisher 定义)
学习新任务时，参数被限制在椭圆内变化
```

### 4.2 Fisher 对角线 vs 完整 Fisher

| 特性 | 对角线近似 | 完整 Fisher |
|------|-----------|-------------|
| 存储 | $O(n)$ | $O(n^2)$ |
| 计算 | 简单 | 复杂 |
| 准确性 | 忽略参数间相关性 | 精确 |
| 实用性 | **广泛使用** | 研究为主 |

## 5. 参数选择与调优

| 参数 | 作用 | 常用范围 | 调优建议 |
|------|------|----------|----------|
| $\lambda$ | 正则化强度 | 100-10000 | 从小开始增大 |
| 采样数 | Fisher 估计精度 | 100-1000 | 越多越准 |
| $\gamma$ (在线) | 历史 Fisher 衰减 | 0.8-0.99 | 控制遗忘速度 |

## 6. EWC 的局限性

| 局限 | 说明 |
|------|------|
| 对角线近似 | 忽略参数间相关性 |
| Fisher 存储 | 需要为每个任务存储 Fisher |
| 任务边界 | 需要知道任务切换点 |
| 计算开销 | 需要额外计算 Fisher |
| 多任务累积 | Fisher 累积可能导致正则化过强 |

## 7. 总结

| 要点 | 说明 |
|------|------|
| 核心思想 | 重要参数弹性约束 |
| 理论基础 | 贝叶斯推断 + Fisher 信息 |
| 损失函数 | $\mathcal{L} = \mathcal{L}_t + \frac{\lambda}{2}\sum F_i(\theta_i - \theta_i^*)^2$ |
| 优势 | 简单有效、理论优雅 |
| 局限 | 对角线近似、计算开销 |

---

**参考文献：**

1. Kirkpatrick, J. et al. (2017). *Overcoming catastrophic forgetting in neural networks*. PNAS.
2. Huszár, F. (2018). *Note on the quadratic penalties in elastic weight consolidation*. PNAS.
