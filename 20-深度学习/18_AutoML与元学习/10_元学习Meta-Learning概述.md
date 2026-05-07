# 10_元学习 (Meta-Learning) 概述

## 1. 什么是元学习

**元学习 (Meta-Learning)** 的目标是"学会学习" (Learning to Learn)：让模型具备**快速适应新任务**的能力，即使每个新任务只有很少的训练样本。

### 1.1 传统学习 vs 元学习

| 特性 | 传统学习 | 元学习 |
|------|----------|--------|
| 目标 | 学习一个任务 | 学习多个任务的共性 |
| 数据 | 大量样本 | 每个任务少量样本 |
| 适应 | 从头训练 | 快速微调/推理 |
| 泛化 | 同任务新数据 | 新任务 |

### 1.2 少样本学习 (Few-Shot Learning)

元学习最典型的应用是 **K-shot N-way 分类**：
- N-way：N个类别
- K-shot：每个类别K个样本
- 例：5-way 1-shot = 5个类别，每类1个样本

## 2. 任务的形式化

### 2.1 Episode 训练范式

元学习的训练单位是**任务 (Task)**，每个任务包含：

$$\mathcal{T}_i = (\mathcal{D}^{train}_i, \mathcal{D}^{test}_i)$$

- **支持集 (Support Set)**：$\mathcal{D}^{train}_i$，少量标注样本（如5类各1个样本）
- **查询集 (Query Set)**：$\mathcal{D}^{test}_i$，用于评估的样本

### 2.2 元训练与元测试

```
元训练阶段 (Meta-Training):
  从训练类别中采样大量 tasks
  每个 task: 采样N个类，每类K个support样本 + Q个query样本
  → 更新元学习器参数

元测试阶段 (Meta-Testing):
  在未见过的新类别上测试
  用support set适配，用query set评估
```

```python
class FewShotDataset:
    """少样本学习数据集"""
    def __init__(self, data, labels, n_way=5, k_shot=1, q_query=15):
        self.data = data
        self.labels = labels
        self.n_way = n_way
        self.k_shot = k_shot
        self.q_query = q_query
        
        # 按类别组织数据
        self.class_to_indices = {}
        for idx, label in enumerate(labels):
            if label not in self.class_to_indices:
                self.class_to_indices[label] = []
            self.class_to_indices[label].append(idx)
    
    def sample_episode(self):
        """采样一个episode"""
        # 选择N个类
        classes = random.sample(list(self.class_to_indices.keys()), self.n_way)
        
        support_x, support_y = [], []
        query_x, query_y = [], []
        
        for i, cls in enumerate(classes):
            indices = random.sample(self.class_to_indices[cls], self.k_shot + self.q_query)
            
            support_x.extend([self.data[idx] for idx in indices[:self.k_shot]])
            support_y.extend([i] * self.k_shot)
            
            query_x.extend([self.data[idx] for idx in indices[self.k_shot:]])
            query_y.extend([i] * self.q_query)
        
        return (torch.stack(support_x), torch.tensor(support_y),
                torch.stack(query_x), torch.tensor(query_y))
```

## 3. 元学习的三大范式

### 3.1 基于度量 (Metric-Based)

**核心思想**：学习一个好的**嵌入空间**，同类样本距离近，异类样本距离远。

代表方法：
- Siamese Networks
- Matching Networks
- Prototypical Networks
- Relation Networks

```
Support set → Embedding → 原型/距离
Query → Embedding → 与support比较 → 分类
```

### 3.2 基于模型 (Model-Based)

**核心思想**：设计一个能快速适应新任务的模型架构。

代表方法：
- MANN (Memory-Augmented Neural Network)
- SNAIL (Simple Neural Attentive Learner)

```
输入序列 → 注意力+记忆模块 → 直接输出预测
```

### 3.3 基于优化 (Optimization-Based)

**核心思想**：学习一个好的**初始化参数**或**优化策略**，使少量梯度步就能适应新任务。

代表方法：
- MAML
- Reptile
- Meta-SGD

```
初始化参数 θ
    ↓
在support set上做几步梯度下降
    ↓
得到适应后的参数 θ'
    ↓
在query set上评估
```

## 4. 范式对比

| 范式 | 思路 | 优势 | 劣势 |
|------|------|------|------|
| 度量学习 | 学习嵌入空间 | 简单、可解释 | 依赖嵌入质量 |
| 模型设计 | 端到端架构 | 表达能力强 | 设计复杂 |
| 优化学习 | 学习初始化/策略 | 模型无关、灵活 | 计算成本高 |

## 5. 数据集

| 数据集 | 类别数 | 用途 | 特点 |
|--------|--------|------|------|
| miniImageNet | 100 | 元学习基准 | 64/16/20 train/val/test |
| tieredImageNet | 608 | 大规模基准 | 层次化类别划分 |
| Omniglot | 1623 | 手写字符 | 字符级细粒度 |
| CUB-200 | 200 | 鸟类分类 | 细粒度分类 |
| Meta-Dataset | 多个 | 综合基准 | 跨数据集泛化 |

## 6. 评估指标

- **N-way K-shot 分类精度**：标准评估
- **跨领域泛化**：训练类别 → 新类别
- **样本效率**：达到目标精度需要多少样本

```python
def meta_evaluate(model, test_dataset, n_episodes=600, n_way=5, k_shot=1):
    """元评估"""
    accuracies = []
    
    for _ in range(n_episodes):
        support_x, support_y, query_x, query_y = test_dataset.sample_episode()
        
        # 适配（使用support set）
        model.adapt(support_x, support_y)
        
        # 评估（在query set上）
        with torch.no_grad():
            pred = model.predict(query_x)
            acc = (pred.argmax(dim=-1) == query_y).float().mean().item()
            accuracies.append(acc)
    
    mean_acc = np.mean(accuracies)
    ci95 = 1.96 * np.std(accuracies) / np.sqrt(n_episodes)
    
    return mean_acc, ci95
```

## 7. AutoML 与元学习的关系

| 方面 | AutoML | 元学习 |
|------|--------|--------|
| 目标 | 自动化ML流程 | 快速适应新任务 |
| 联系 | 元学习加速AutoML搜索 | AutoML优化元学习模型 |
| 示例 | 用元学习预测最优超参数 | 用NAS搜索元学习架构 |

---

**关键要点**：
1. 元学习的目标是"学会学习"，让模型能从少量样本快速适应新任务
2. 三大范式：度量学习、模型设计、优化学习
3. Episode训练范式（support + query）是元学习的标准训练方式
4. K-shot N-way 分类是少样本学习的标准评估协议
