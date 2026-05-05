# 12_交叉熵损失 (Cross-Entropy) 的导数简化形式

## 核心概念

- **交叉熵定义**：交叉熵（Cross-Entropy）衡量两个概率分布之间的差异：$H(p, q) = -\sum_i p_i \log q_i$。在分类任务中，$p$ 是真实标签分布（通常是 one-hot），$q$ 是模型预测的分布。
- **梯度简化形式**：当交叉熵与 Softmax 结合使用时，梯度简化为 $\partial L/\partial z_j = p_j - y_j$，即"预测概率减去真实标签"。这是深度学习中最优美最简洁的梯度形式之一。
- **信息论基础**：交叉熵 = 熵 + KL 散度，即 $H(p, q) = H(p) + D_{KL}(p\|q)$。由于 $H(p)$ 对于固定标签是常数，最小化交叉熵等价于最小化预测分布与真实分布之间的 KL 散度。
- **与二分类 BCE 的关系**：对于二分类问题，交叉熵退化为二元交叉熵（BCE）：$L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]$，其梯度同样简化为 $\hat{y} - y$。

## 数学推导

**多分类交叉熵**：

$$
L = -\sum_{i=1}^{K} y_i \log p_i, \quad \text{其中 } p_i = \frac{e^{z_i}}{\sum_j e^{z_j}}
$$

$y$ 是 one-hot 编码的真实标签，即 $y_c = 1$ 且 $y_i = 0 (i \neq c)$。

**对 logits $z_j$ 的梯度**：

$$
\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j}
$$

首先，$\frac{\partial L}{\partial p_i} = -\frac{y_i}{p_i}$

然后，用之前推导的 Softmax 导数：

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

代入：

$$
\frac{\partial L}{\partial z_j} = \sum_i \left(-\frac{y_i}{p_i}\right) \cdot p_i(\delta_{ij} - p_j) = -\sum_i y_i(\delta_{ij} - p_j)
$$

$$
= -\sum_i y_i\delta_{ij} + \sum_i y_i p_j = -y_j + p_j \sum_i y_i
$$

由于标签是 one-hot，$\sum_i y_i = 1$，所以：

$$
\boxed{\frac{\partial L}{\partial z_j} = p_j - y_j}
$$

**二分类（BCE）**：

$$
L = -[y\log\hat{y} + (1-y)\log(1-\hat{y})]
$$

其中 $\hat{y} = \sigma(z) = 1/(1 + e^{-z})$。

$$
\frac{\partial L}{\partial z} = \frac{\partial L}{\partial \hat{y}} \cdot \frac{\partial \hat{y}}{\partial z} = \left(-\frac{y}{\hat{y}} + \frac{1-y}{1-\hat{y}}\right) \cdot \hat{y}(1-\hat{y})
$$

$$
= -y(1-\hat{y}) + (1-y)\hat{y} = \hat{y} - y
$$

与多分类形式一致！

## 直观理解

梯度 $p_j - y_j$ 的含义非常直观：
- 对于正确类别（$y_j = 1$）：如果预测概率 $p_j < 1$，梯度为负，推动 $z_j$ 增大，使预测更接近 1。
- 对于错误类别（$y_j = 0$）：如果预测概率 $p_j > 0$，梯度为正，推动 $z_j$ 减小，使预测更接近 0。

梯度的大小正比于预测错误的程度：如果正确类别的概率只有 0.5（即模型很犹豫），梯度为 -0.5，修正力度大；如果正确类别的概率已经接近 0.95，梯度为 -0.05，修正力度小，只做微调。

交叉熵优于 MSE 用于分类的原因可以这样理解：MSE 对"预测概率 0.9 vs 0.99"几乎一视同仁（梯度 0.1 vs 0.01），但交叉熵认为从 0.9 到 0.99 需要更多的"努力"（梯度 -0.9 vs -0.99）。交叉熵更关注那些预测尚未接近 1 的正确类别。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 验证 Softmax + CrossEntropy 的梯度简化形式
logits = torch.randn(4, 10, requires_grad=True)
labels = torch.randint(0, 10, (4,))

# 方式一: 使用 PyTorch 的 CrossEntropyLoss（内部结合了 Softmax）
criterion = nn.CrossEntropyLoss()
loss1 = criterion(logits, labels)
loss1.backward()
grad1 = logits.grad.clone()

# 方式二: 手动计算梯度
logits.grad.zero_()
probs = F.softmax(logits, dim=1)
loss2 = F.cross_entropy(logits, labels)  # 等价方式

# 验证梯度 = p_j - y_j
with torch.no_grad():
    dL_dz = probs.clone()
    dL_dz[range(4), labels] -= 1.0
    dL_dz /= 4  # CrossEntropyLoss 默认取平均值

print(f"梯度 (自动):\n{grad1}")
print(f"梯度 (p_j - y_j):\n{dL_dz}")
print(f"一致: {torch.allclose(grad1, dL_dz, atol=1e-6)}")

# 二分类 BCE 的梯度验证
bce = nn.BCEWithLogitsLoss()  # 内部集成 Sigmoid
logits_bin = torch.randn(10, 1, requires_grad=True)
labels_bin = torch.randint(0, 2, (10, 1)).float()

loss_bce = bce(logits_bin, labels_bin)
loss_bce.backward()

with torch.no_grad():
    probs_bin = torch.sigmoid(logits_bin)
    expected_grad = (probs_bin - labels_bin) / 10  # 除以 batch 大小

print(f"\nBCE 梯度一致: {torch.allclose(logits_bin.grad, expected_grad, atol=1e-6)}")

# 交叉熵 vs MSE 在分类问题上的对比
def compare_losses():
    torch.manual_seed(42)
    logits = torch.randn(100, 5, requires_grad=True)
    labels = torch.randint(0, 5, (100,))

    # 使用交叉熵
    ce_loss = F.cross_entropy(logits, labels)

    # 使用 MSE (需要先 Softmax + one-hot)
    probs = F.softmax(logits, dim=1)
    one_hot = F.one_hot(labels, 5).float()
    mse_loss = ((probs - one_hot) ** 2).mean()

    print(f"\n交叉熵损失: {ce_loss.item():.4f}")
    print(f"MSE 损失: {mse_loss.item():.4f}")

compare_losses()
```

## 深度学习关联

- **分类问题的标准损失**：交叉熵是所有分类任务（图像分类、文本分类、语音识别）的标准损失函数。几乎所有现代分类模型（ResNet、BERT、GPT）都在输出层使用交叉熵损失。
- **与 KL 散度的等价性**：在知识蒸馏中，学生网络通过最小化与学生预测分布 $q_s$ 和教师预测分布 $q_t$ 之间的 KL 散度来学习：$L_{KD} = D_{KL}(q_t \| q_s)$。这等价于在软标签 $q_t$ 上的交叉熵损失，体现了交叉熵概念的泛化性。
- **标签平滑（Label Smoothing）**：标准的交叉熵鼓励模型对正确类别输出极其自信的概率（接近 1），可能导致过拟合。标签平滑将 one-hot 标签替换为软标签 $y'_i = y_i(1-\epsilon) + \epsilon/K$，使得交叉熵损失不会过度追求"绝对的自信"，提升了模型的泛化能力。
