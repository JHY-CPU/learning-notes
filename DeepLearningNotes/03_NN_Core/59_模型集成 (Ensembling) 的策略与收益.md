# 59 模型集成 (Ensembling) 的策略与收益

## 核心概念

- **模型集成定义**：模型集成（Ensembling）通过组合多个模型的预测来提升整体性能。集成的核心思想是"三个臭皮匠，顶个诸葛亮"——多个模型的集体决策通常优于单个模型。

- **偏差-方差-噪声分解**：集成降低误差主要通过减少方差。假设 $M$ 个独立同分布的模型，每个模型误差为 $\epsilon$（方差 $\sigma^2$），平均后的方差为 $\sigma^2/M$。但实践中模型并非完全独立，因此方差降低幅度小于 $M$ 倍。

- **多样性是集成的关键**：集成的收益取决于模型之间的多样性。完全相同的模型集成没有意义。多样性的来源包括：不同的初始化、不同的数据子集、不同的架构、不同的超参数等。

- **常见的集成方法**：
  1. **Bagging**（Bootstrap Aggregating）：在不同数据子集上训练模型
  2. **Boosting**：顺序训练，每个模型关注前一个模型的错误
  3. **Stacking**：学习一个元模型来组合各模型的预测

## 数学推导

**偏差-方差-噪声分解**：

考虑回归问题，真实函数 $y = f(x) + \epsilon$，$\mathbb{E}[\epsilon] = 0$，$\text{Var}[\epsilon] = \sigma^2_\epsilon$。

模型 $\hat{f}(x)$ 的期望平方误差：

$$
\mathbb{E}[(y - \hat{f}(x))^2] = \underbrace{\mathbb{E}[(\hat{f}(x) - \mathbb{E}[\hat{f}(x)])^2]}_{\text{方差}} + \underbrace{(\mathbb{E}[\hat{f}(x)] - f(x))^2}_{\text{偏差}^2} + \underbrace{\sigma^2_\epsilon}_{\text{噪声}}
$$

**集成模型的方差**：

假设有 $M$ 个模型，预测为 $\hat{f}_1, \ldots, \hat{f}_M$，每个模型的方差为 $\sigma^2$，协方差为 $\rho\sigma^2$（$\rho$ 是平均相关系数）。

集成平均：$\bar{f} = \frac{1}{M} \sum_{i=1}^M \hat{f}_i$

集成方差：

$$
\text{Var}(\bar{f}) = \frac{1}{M^2} \left( \sum_i \text{Var}(\hat{f}_i) + \sum_{i \neq j} \text{Cov}(\hat{f}_i, \hat{f}_j) \right)
$$

$$
= \frac{1}{M^2} (M\sigma^2 + M(M-1)\rho\sigma^2)
$$

$$
= \rho\sigma^2 + \frac{1-\rho}{M}\sigma^2
$$

当 $M \to \infty$，方差趋近于 $\rho\sigma^2$。如果模型完全独立（$\rho = 0$），方差降低到 $\sigma^2/M$；如果模型完全相关（$\rho = 1$），集成无收益。

**分类中的投票**：

对于 $M$ 个分类器，每个正确率为 $p$（独立），使用多数投票。集成正确率为：

$$
P_{\text{ens}} = \sum_{k=\lfloor M/2 \rfloor + 1}^{M} \binom{M}{k} p^k (1-p)^{M-k}
$$

当 $p > 0.5$ 时，$P_{\text{ens}}$ 随 $M$ 增加而增加。

## 直观理解

模型集成可以类比为"多专家会诊"——多个医生（模型）独立诊断（预测），然后综合意见做出最终诊断：

- 每个医生可能有不同的专长和盲点（多样性）
- 集体决策通常比个人决策可靠（误差降低）
- 如果所有医生受教育背景完全相同（模型相关），会诊价值不大

Bagging 像"组织多个独立的医疗小组，每个小组看不同的病历子集"。Boosting 像"先让一个医生诊断，然后让下一个医生专门关注前一个医生的误诊情况"。

从效率角度看，集成总是有收益的——加入一个新模型，只要它不是完全随机的（准确率 > 0.5 分类或与已有不相关），就能提升集成性能。但收益递减：第 10 个模型的边际收益小于第 2 个。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# 模型集成演示
class SimpleModel(nn.Module):
    """可配置的简单模型"""
    def __init__(self, hidden_dim=64, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(20, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(self, x):
        return self.net(x)

# 生成数据
torch.manual_seed(42)
X = torch.randn(500, 20)
y = (X[:, 0] ** 2 + X[:, 1] ** 2 > 1).long()

X_train, y_train = X[:300], y[:300]
X_test, y_test = X[300:], y[300:]

# 训练多个模型（随机初始化带来多样性）
n_models = 10
models = []

print(f"训练 {n_models} 个模型:")
for i in range(n_models):
    model = SimpleModel(64, 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        pred = model(X_train)
        loss = F.cross_entropy(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        acc = (model(X_test).argmax(1) == y_test).float().mean()
    models.append(model)
    print(f"  模型 {i+1:2d}: 测试准确率 = {acc:.4f}")

# 集成预测
print("\n不同集成策略对比:")

# 1. 概率平均（Soft Voting）
with torch.no_grad():
    all_probs = torch.stack([F.softmax(m(X_test), dim=1) for m in models])
    ensemble_probs = all_probs.mean(0)
    ensemble_acc = (ensemble_probs.argmax(1) == y_test).float().mean()
print(f"  概率平均集成: {ensemble_acc:.4f}")

# 2. 投票（Hard Voting）
with torch.no_grad():
    all_preds = torch.stack([m(X_test).argmax(1) for m in models])
    vote_preds = torch.mode(all_preds, dim=0)[0]
    vote_acc = (vote_preds == y_test).float().mean()
print(f"  投票集成: {vote_acc:.4f}")

# 3. 不同子集数的效果
print("\n模型数量对集成效果的影响:")
for k in [2, 4, 6, 8, 10]:
    with torch.no_grad():
        probs_k = torch.stack([F.softmax(models[i](X_test), dim=1) for i in range(k)])
        acc_k = (probs_k.mean(0).argmax(1) == y_test).float().mean()
    print(f"  {k} 个模型集成: {acc_k:.4f}")

# Snapshot Ensemble 概念演示
print("\nSnapshot Ensemble（概念演示）:")
print("  1. 使用 Cyclic LR 训练单个模型")
print("  2. 在每次 LR 循环的最低点保存模型检查点")
print("  3. 这些检查点天然具有多样性（收敛到不同局部极小值）")
print("  4. 集成这些检查点，使用单个模型的计算成本")

# Bagging 实现
print("\nBagging 集成:")
bagging_models = []
for i in range(5):
    # 自助采样
    indices = torch.randint(0, 300, (300,))
    X_boot = X_train[indices]
    y_boot = y_train[indices]

    model = SimpleModel(64, 0.0)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(200):
        pred = model(X_boot)
        loss = F.cross_entropy(pred, y_boot)
        opt.zero_grad()
        loss.backward()
        opt.step()

    bagging_models.append(model)

with torch.no_grad():
    bagging_probs = torch.stack([F.softmax(m(X_test), dim=1) for m in bagging_models])
    bagging_acc = (bagging_probs.mean(0).argmax(1) == y_test).float().mean()
print(f"  Bagging 集成 (5 模型): {bagging_acc:.4f}")
```

## 深度学习关联

- **比赛中的必备武器**：在 Kaggle 等机器学习竞赛中，模型集成是获取顶尖排名的必备技术。通常通过训练多个不同架构的模型（如 ResNet + EfficientNet + ViT），对其预测进行加权平均。集成通常可以提升 1-3% 的准确率。

- **Snapshot Ensemble**：Snapshot Ensemble 是一种高效的集成方法。使用循环学习率调度（Cosine Annealing with Warm Restarts），单个训练过程可以收敛到多个不同的局部极小值。每个"快照"可以视为一个模型，相当于用单次训练的成本获得了多个模型。

- **集成与知识蒸馏的结合**：可以把多个模型的集成蒸馏为一个学生模型。这样既获得了集成的性能提升，又保持了单个模型的推理效率。这在大规模部署中特别有用——"教师集成"指导"学生单模型"，实现"化整为零"的效果。
