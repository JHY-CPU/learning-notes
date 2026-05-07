# 11_MAML 模型无关元学习

## 1. 核心思想

**MAML (Model-Agnostic Meta-Learning, Finn et al., ICML 2017)** 学习一个**好的初始化参数** $\theta$，使得对任意新任务，只需少量梯度步就能达到好的性能。

### 1.1 直觉理解

```
传统训练: 一个任务，大量数据 → 学到该任务的最优参数
MAML: 多个任务 → 学到"最容易适应任何任务"的初始化

类比: 学会骑自行车 → 更容易学骑摩托车
      学好线性代数 → 更容易学深度学习
```

### 1.2 数学形式

对任务 $\mathcal{T}_i$，一步梯度更新：

$$\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_\theta)$$

元目标：所有任务上适应后的性能之和：

$$\min_\theta \sum_{\mathcal{T}_i \sim p(\mathcal{T})} \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$$

注意：外层优化的不是 $\theta$ 上的损失，而是 $\theta'_i$（一步更新后的参数）上的损失。

## 2. MAML 算法

### 2.1 算法流程

```
输入: 任务分布 p(T), 步长 α (内层), β (外层)
初始化: θ

循环:
  1. 采样一批任务 {T_i}
  2. 对每个任务 T_i:
     a. 在 support set 上计算损失 L_Ti(θ)
     b. 计算梯度 ∇_θ L_Ti(θ)
     c. 计算适应后的参数: θ'_i = θ - α ∇_θ L_Ti(θ)
     d. 在 query set 上计算损失 L_Ti(θ'_i)
  3. 元更新: θ ← θ - β ∇_θ Σ_i L_Ti(θ'_i)
```

### 2.2 二阶梯度

关键：外层梯度 $\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(f_{\theta'_i})$ 需要对 $\theta$ 求导，而 $\theta'_i = \theta - \alpha \nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)$ 依赖于 $\theta$。

展开计算：

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) = \frac{\partial \mathcal{L}_{\mathcal{T}_i}}{\partial \theta'_i} \cdot \frac{\partial \theta'_i}{\partial \theta}$$

$$= \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}(\theta'_i) \cdot \left(I - \alpha \nabla^2_\theta \mathcal{L}_{\mathcal{T}_i}(\theta)\right)$$

这涉及**Hessian**（二阶导数）的计算。

## 3. 完整实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import higher

class MAML:
    """MAML实现"""
    def __init__(self, model, inner_lr=0.01, outer_lr=0.001, first_order=False):
        self.model = model
        self.inner_lr = inner_lr
        self.first_order = first_order
        self.meta_optimizer = torch.optim.Adam(model.parameters(), lr=outer_lr)
    
    def inner_update(self, support_x, support_y):
        """内层更新：在support set上计算梯度并更新"""
        # 前向传播
        logits = self.model(support_x)
        loss = F.cross_entropy(logits, support_y)
        
        # 计算梯度
        grads = torch.autograd.grad(loss, self.model.parameters(),
                                     create_graph=not self.first_order)
        
        # 更新参数: θ' = θ - α * ∇L
        updated_params = [
            p - self.inner_lr * g for p, g in zip(self.model.parameters(), grads)
        ]
        
        return updated_params
    
    def meta_loss(self, query_x, query_y, updated_params):
        """计算query set上的损失（使用更新后的参数）"""
        # 用更新后的参数前向传播
        logits = self.model.forward_with_params(query_x, updated_params)
        loss = F.cross_entropy(logits, query_y)
        return loss
    
    def meta_train_step(self, tasks):
        """
        tasks: list of (support_x, support_y, query_x, query_y)
        """
        meta_loss = 0
        
        for support_x, support_y, query_x, query_y in tasks:
            # 内层更新
            updated_params = self.inner_update(support_x, support_y)
            
            # 计算query损失
            task_loss = self.meta_loss(query_x, query_y, updated_params)
            meta_loss += task_loss
        
        meta_loss /= len(tasks)
        
        # 外层更新
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()

class SimpleModel(nn.Module):
    """简单模型，支持用外部参数前向传播"""
    def __init__(self, input_dim=784, hidden_dim=256, output_dim=5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim),
        )
    
    def forward(self, x):
        return self.net(x)
    
    def forward_with_params(self, x, params):
        """用给定参数前向传播"""
        # 手动实现前向传播，使用外部参数
        idx = 0
        h = x
        for module in self.net:
            if isinstance(module, nn.Linear):
                h = F.linear(h, params[idx], params[idx + 1])
                idx += 2
            else:
                h = module(h)
        return h
```

## 4. FOMAML (First-Order MAML)

### 4.1 动机

完整的 MAML 需要计算 Hessian，计算成本高。**FOMAML** 忽略二阶项：

$$\nabla_\theta \mathcal{L}_{\mathcal{T}_i}(\theta'_i) \approx \nabla_{\theta'_i} \mathcal{L}_{\mathcal{T}_i}(\theta'_i)$$

即：只用适应后参数的梯度，忽略链式法则中经过内层更新的部分。

### 4.2 实现

```python
class FOMAML(MAML):
    """First-Order MAML"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, first_order=True, **kwargs)
```

### 4.3 效果对比

| 方法 | 二阶导数 | 计算成本 | 5-way 1-shot |
|------|----------|----------|--------------|
| MAML | 是 | 高 | ~48.7% |
| FOMAML | 否 | 低 | ~48.1% |

差距很小！说明二阶信息不是关键。

## 5. 使用 higher 库

**higher** 库提供了高效的 MAML 实现：

```python
import higher

def maml_with_higher(model, meta_optimizer, tasks, inner_lr=0.01, n_shots=5):
    """使用higher库的MAML"""
    meta_loss = 0
    
    for support_x, support_y, query_x, query_y in tasks:
        # 创建可微分的模型副本
        with higher.innerloop_ctx(model, meta_optimizer, 
                                   copy_initial_weights=False) as (fmodel, diffopt):
            
            # 内层更新（多次）
            for _ in range(n_shots):  # n_shots步更新
                support_logits = fmodel(support_x)
                inner_loss = F.cross_entropy(support_logits, support_y)
                diffopt.step(inner_loss)
            
            # query损失
            query_logits = fmodel(query_x)
            outer_loss = F.cross_entropy(query_logits, query_y)
            meta_loss += outer_loss
    
    meta_loss /= len(tasks)
    
    meta_optimizer.zero_grad()
    meta_loss.backward()
    meta_optimizer.step()
    
    return meta_loss.item()
```

## 6. MAML 的变体

| 方法 | 改进 | 特点 |
|------|------|------|
| FOMAML | 一阶近似 | 快，精度略低 |
| Reptile | 简化梯度计算 | 更快，无需二阶导 |
| Meta-SGD | 学习内层学习率 | 自适应步长 |
| ANIL | 冻结特征层 | 更快收敛 |
| BOIL | 冻结分类头 | 更好的特征学习 |

### 6.1 ANIL (Almost No Inner Loop)

只在最后的分类层进行内层更新：

```python
def anil_inner_update(model, support_x, support_y, inner_lr):
    """ANIL: 只更新分类头"""
    # 冻结特征层
    for param in model.feature_layers.parameters():
        param.requires_grad_(False)
    
    logits = model(support_x)
    loss = F.cross_entropy(logits, support_y)
    grads = torch.autograd.grad(loss, model.parameters())
    
    # 只更新分类层
    updated_params = []
    for p, g in zip(model.parameters(), grads):
        if g is not None:
            updated_params.append(p - inner_lr * g)
        else:
            updated_params.append(p)
    
    return updated_params
```

## 7. 实际使用建议

| 场景 | 建议 |
|------|------|
| 计算有限 | FOMAML / Reptile |
| 需要最高精度 | 完整MAML |
| 特征层预训练好 | ANIL |
| 多任务学习 | MAML + 任务嵌入 |

---

**关键要点**：
1. MAML 学习一个初始化参数，少量梯度步就能适应新任务
2. 核心是**二层优化**：内层在support set上适应，外层在query set上优化初始化
3. FOMAML忽略二阶项，效果几乎不变但计算成本大幅降低
4. higher 库提供了高效的可微分内层优化实现
