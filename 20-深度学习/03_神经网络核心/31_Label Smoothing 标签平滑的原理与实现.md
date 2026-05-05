# 32_Label Smoothing 标签平滑的原理与实现

## 核心概念

- **标签平滑的定义**：标签平滑（Label Smoothing）将硬标签（one-hot）替换为软标签：$y' = (1 - \epsilon) \cdot y + \epsilon / K$，其中 $K$ 是类别数，$\epsilon$ 是平滑参数。例如，$\epsilon = 0.1$ 时，正确类别的目标值从 1 变为 0.9，其余类别从 0 变为 $0.1/K$。
- **防止过自信**：标准交叉熵训练鼓励网络对正确类别输出概率 1，这会导致网络"过分自信"。标签平滑降低了这种自信度，使网络预测的概率更加合理，减少了过拟合。
- **正则化效果**：标签平滑等价于在损失函数中加入了惩罚项，惩罚网络对不同类别输出概率之间的差距。数学上等价于 KL 散度正则化：$L_{LS} = (1-\epsilon)H(y, p) + \epsilon H(u, p)$，其中 $u$ 是均匀分布。
- **容忍标注错误**：标签平滑使网络不再追求对训练标签的完美拟合，从而对标注错误更加鲁棒。这对大规模数据集（可能存在 5-10% 的标注错误）特别有益。

## 数学推导

**标准交叉熵损失**：

$$
L_{CE} = -\sum_{k=1}^{K} y_k \log p_k, \quad y_k = \delta_{kc}
$$

其中 $c$ 是正确类别。

**标签平滑的损失函数**：

软标签：

$$
y_k^{LS} = y_k (1 - \epsilon) + \frac{\epsilon}{K} = \begin{cases}
1 - \epsilon + \epsilon/K & \text{if } k = c \\
\epsilon/K & \text{if } k \neq c
\end{cases}
$$

损失函数：

$$
L_{LS} = -\sum_{k=1}^{K} y_k^{LS} \log p_k
$$

$$
= -(1-\epsilon) \log p_c - \frac{\epsilon}{K} \sum_{k=1}^{K} \log p_k
$$

**等价形式——添加 KL 散度正则化**：

$$
L_{LS} = (1-\epsilon) H(y, p) + \epsilon H(u, p)
$$

其中 $u_k = 1/K$ 是均匀分布，$H(u, p) = -\sum_{k} u_k \log p_k = -\frac{1}{K}\sum_k \log p_k$。

展开：

$$
L_{LS} = -\sum_k y_k \log p_k + \epsilon\left(H(u, p) - H(y, p)\right)
$$

$$
= -\sum_k y_k \log p_k + \epsilon \cdot D_{KL}(u \| p) + \text{常数}
$$

因此标签平滑相当于在交叉熵中添加一个额外的惩罚项，鼓励网络的预测分布 $p$ 接近均匀分布 $u$。

**对梯度的校正**：

标准交叉熵的梯度：$\partial L_{CE}/\partial z_j = p_j - y_j$

标签平滑的梯度：

$$
\frac{\partial L_{LS}}{\partial z_j} = p_j - y_j^{LS} = p_j - \left((1-\epsilon)y_j + \frac{\epsilon}{K}\right)
$$

对于正确类别（$y_c = 1$）：

$$
\frac{\partial L_{LS}}{\partial z_c} = p_c - (1 - \epsilon + \epsilon/K) = (p_c - 1) + \epsilon(1 - 1/K)
$$

对于错误类别（$y_j = 0$）：

$$
\frac{\partial L_{LS}}{\partial z_j} = p_j - \epsilon/K
$$

这意味着：
- 正确类别的梯度：惩罚减弱了（因为目标从 1 变为 $1 - \epsilon$）
- 错误类别的梯度：当 $p_j < \epsilon/K$ 时，梯度为正（推动概率增大）；当 $p_j > \epsilon/K$ 时，梯度为负（推动概率减小）

## 直观理解

标签平滑相当于告诉模型："不要 100% 确定，保留一点怀疑"。就像一个有经验的学习者不会对自己学到的知识百分之百确信，会预留一些"我可能错了"的空间。

从知识的角度看，标签平滑体现了"类别之间并非完全无关"的思想。例如，一张狗的图片被误标为猫，标准交叉熵会"歇斯底里"地将猫的概率推向 0，但标签平滑会温和地保留一点可能性。这体现了对标注不确定性的建模。

梯度校正的效果：标准交叉熵的错误类别梯度始终为 $p_j$（正数），意味着始终要将错误类别的输出推向负无穷。标签平滑在 $p_j < \epsilon/K$ 时停止推远，甚至稍微拉近——这防止了错误类别的 logits 被推到极端的负值。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 手动实现标签平滑
class LabelSmoothingCrossEntropy(nn.Module):
    """标签平滑交叉熵损失"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, targets):
        # logits: (batch_size, num_classes)
        # targets: (batch_size,) — 类别索引
        num_classes = logits.size(-1)
        log_probs = F.log_softmax(logits, dim=-1)

        # 创建平滑标签
        with torch.no_grad():
            smooth_targets = torch.zeros_like(log_probs)
            smooth_targets.fill_(self.smoothing / (num_classes - 1))
            smooth_targets.scatter_(1, targets.unsqueeze(1), 1 - self.smoothing)

        loss = -(smooth_targets * log_probs).sum(dim=-1).mean()
        return loss

# 验证标签平滑的效果
torch.manual_seed(42)

logits = torch.randn(4, 5)
targets = torch.tensor([0, 2, 1, 3])

ce = nn.CrossEntropyLoss()
ls = LabelSmoothingCrossEntropy(smoothing=0.1)

print("标准 CE vs 标签平滑 CE:")
print(f"  标准 CE Loss: {ce(logits, targets):.4f}")
print(f"  标签平滑 Loss: {ls(logits, targets):.4f}")

# 观察标签平滑对预测概率的影响
model = nn.Linear(10, 5)
X = torch.randn(20, 10)
y = torch.randint(0, 5, (20,))

def train_with_label_smoothing(smoothing=0.0):
    torch.manual_seed(42)
    m = nn.Linear(10, 5)
    m.weight.data = model.weight.data.clone()
    m.bias.data = model.bias.data.clone()
    opt = torch.optim.SGD(m.parameters(), lr=0.05)
    criterion = LabelSmoothingCrossEntropy(smoothing=smoothing)

    for epoch in range(100):
        pred = m(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    with torch.no_grad():
        probs = F.softmax(m(X), dim=1)
        avg_max_prob = probs.max(1)[0].mean()
        avg_entropy = -(probs * torch.log(probs + 1e-8)).sum(1).mean()

    return avg_max_prob.item(), avg_entropy.item()

print("\n标签平滑对预测概率的影响:")
for eps in [0.0, 0.05, 0.1, 0.2, 0.5]:
    max_prob, entropy = train_with_label_smoothing(eps)
    print(f"  ε={eps:.1f}: 平均最大概率={max_prob:.4f}, 平均熵={entropy:.4f}")

# 演示标签平滑在防止过度自信上的效果
print("\n过度自信演示:")
x_single = torch.randn(1, 10)
logits_single = model(x_single)

# 标准 CE 训练 200 轮
m1 = nn.Linear(10, 5)
opt1 = torch.optim.SGD(m1.parameters(), lr=0.05)
for _ in range(200):
    pred = m1(X)
    loss = nn.CrossEntropyLoss()(pred, y)
    opt1.zero_grad()
    loss.backward()
    opt1.step()

# 标签平滑训练 200 轮
m2 = nn.Linear(10, 5)
opt2 = torch.optim.SGD(m2.parameters(), lr=0.05)
criterion_ls = LabelSmoothingCrossEntropy(smoothing=0.1)
for _ in range(200):
    pred = m2(X)
    loss = criterion_ls(pred, y)
    opt2.zero_grad()
    loss.backward()
    opt2.step()

with torch.no_grad():
    p1 = F.softmax(m1(X[:3]), dim=1)
    p2 = F.softmax(m2(X[:3]), dim=1)
    print(f"  标准 CE 预测概率: \n{p1}")
    print(f"  标签平滑预测概率: \n{p2}")
```

## 深度学习关联

- **现代视觉分类的标准技术**：标签平滑是训练现代图像分类模型（如 EfficientNet、ResNeXt）的标准技术。通常 $\epsilon = 0.1$ 可以稳定提升 0.5-1% 的准确率。Google 的 Inception-v2 论文最早推广了标签平滑。
- **知识蒸馏中的编码器训练**：在知识蒸馏（Knowledge Distillation）中，教师网络的软标签天然带有标签平滑的效果——教师的预测概率分布比 one-hot 包含更丰富的类别间相似性信息。标签平滑可以视为一种"无教师的蒸馏"，使训练更加稳定。
- **Transformer 训练中的标配**：几乎所有 BERT 和 GPT 系列模型在预训练时都使用标签平滑。在语言建模中，标签平滑可以防止模型在预测时过度集中于某个 token，改善生成文本的多样性，并在微调时提供更好的泛化。
