# 10 Softmax 函数推导与数值稳定性 (Log-Sum-Exp)

## 核心概念

- **Softmax 定义**：Softmax 函数将任意实数向量 $z \in \mathbb{R}^K$ 转换为一个概率分布：$\text{Softmax}(z_i) = e^{z_i} / \sum_{j=1}^K e^{z_j}$。输出值在 $(0,1)$ 区间且总和为 1，适合表示多分类问题的类别概率。

- **数值不稳定问题**：当 $z_i$ 的值很大（如 $z_i = 100$）时，$e^{100}$ 会超出浮点数表示范围（overflow）。当 $z_i$ 的值很小（如 $z_i = -100$）时，$e^{-100}$ 会下溢为 0，导致除以零错误。这两种情况都会使 Softmax 失效。

- **Log-Sum-Exp 技巧**：解决数值稳定性的标准方法是将输入减去最大值：$\text{Softmax}(z_i) = e^{z_i - \max(z)} / \sum_{j} e^{z_j - \max(z)}$。这种平移操作在数学上等价，因为分子分母同时乘以 $e^{-\max(z)}$，但数值上保证了最大指数项为 $e^0 = 1$。

- **Log-Softmax 与交叉熵的联合简化**：在实际实现中，Softmax 通常与交叉熵损失函数联合计算，形成 $\text{CrossEntropyLoss} = -\log(\text{Softmax}(z_i)) = -z_i + \log\sum_j e^{z_j}$。这种 Log-Sum-Exp 形式在数值上更稳定，且导数形式极其简洁。

## 数学推导

**Softmax 函数**：

$$
p_i = \text{Softmax}(z_i) = \frac{e^{z_i}}{\sum_{j=1}^K e^{z_j}}, \quad i = 1, \ldots, K
$$

**Softmax 的导数**（重要）：

当 $i = j$ 时：

$$
\frac{\partial p_i}{\partial z_j} = \frac{e^{z_i} \cdot \sum_k e^{z_k} - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = p_i(1 - p_j)
$$

当 $i \neq j$ 时：

$$
\frac{\partial p_i}{\partial z_j} = \frac{0 - e^{z_i} \cdot e^{z_j}}{(\sum_k e^{z_k})^2} = -p_i p_j
$$

统一形式（使用 Kronecker delta $\delta_{ij}$）：

$$
\frac{\partial p_i}{\partial z_j} = p_i(\delta_{ij} - p_j)
$$

**交叉熵损失的梯度**：

$$
L = -\sum_i y_i \log p_i
$$

其中 $y$ 是 one-hot 标签向量。

$$
\frac{\partial L}{\partial z_j} = \sum_i \frac{\partial L}{\partial p_i} \cdot \frac{\partial p_i}{\partial z_j} = \sum_i \left(-\frac{y_i}{p_i}\right) \cdot p_i(\delta_{ij} - p_j)
$$

$$
= -\sum_i y_i(\delta_{ij} - p_j) = -y_j + p_j \sum_i y_i = -y_j + p_j
$$

因此：

$$
\frac{\partial L}{\partial z_j} = p_j - y_j
$$

这是深度学习中最优雅的梯度之一：Softmax 加交叉熵的梯度就是"预测值减去真实标签"，形式简单且物理意义清晰。

**数值稳定版本**：

$$
\text{Softmax}(z_i) = \frac{e^{z_i - M}}{\sum_j e^{z_j - M}}, \quad \text{其中 } M = \max_j z_j
$$

**Log-Sum-Exp**：

$$
\log\text{Softmax}(z_i) = z_i - \log\sum_j e^{z_j} = z_i - \text{LSE}(z)
$$

其中 $\text{LSE}(z) = \log\sum_j e^{z_j}$ 的稳定计算为 $M + \log\sum_j e^{z_j - M}$。

## 直观理解

Softmax 可以理解为"一种软的 argmax 操作"。硬 argmax 只选择最大值的类别（输出 one-hot 向量），而 Softmax 保留了一定的"软性"——接近最大值的类别也会获得一定的概率权重。温度参数 $T$ 控制这种软硬程度：温度越高，分布越均匀（软）；温度越低，分布越尖锐（接近 argmax）。

从能量模型的角度看，Softmax 就是 Boltzmann 分布（吉布斯分布）：$p_i \propto e^{-E_i}$，其中 $E_i = -z_i$ 是第 $i$ 类的"能量"。Softmax 将"能量"转化为概率，能量越低的类别概率越大。

梯度 $\partial L/\partial z_j = p_j - y_j$ 的几何意义很直观：如果模型对正确类别的预测概率 $p_j$ 低于 1（真实标签 $y_j = 1$），梯度为正，推动 $z_j$ 增大；如果模型对错误类别的预测概率大于 0（$y_j = 0$），梯度为负，推动 $z_j$ 减小。

## 代码示例

```python
import torch
import torch.nn.functional as F
import math

# 数值不稳定演示
def softmax_naive(z):
    """朴素 Softmax — 存在数值不稳定问题"""
    exp_z = torch.exp(z)
    return exp_z / exp_z.sum(dim=-1, keepdim=True)

def softmax_stable(z):
    """数值稳定的 Softmax"""
    z_shifted = z - z.max(dim=-1, keepdim=True).values
    exp_z = torch.exp(z_shifted)
    return exp_z / exp_z.sum(dim=-1, keepdim=True)

# 测试大数值
z_big = torch.tensor([1000.0, 1001.0, 999.0])
try:
    print(f"朴素 Softmax: {softmax_naive(z_big)}")
except Exception as e:
    print(f"朴素 Softmax 失败: {e}")
print(f"稳定 Softmax: {softmax_stable(z_big)}")

# 验证数学等价性
z = torch.randn(5)
print(f"\n原始 Softmax: {F.softmax(z, dim=0)}")

# Log-Sum-Exp 与 LogSoftmax
log_softmax_1 = F.log_softmax(z, dim=0)

# 手动计算 LogSoftmax
M = z.max()
lse = M + torch.log(torch.exp(z - M).sum())
log_softmax_2 = z - lse

print(f"LogSoftmax (PyTorch): {log_softmax_1}")
print(f"LogSoftmax (手动):   {log_softmax_2}")
print(f"一致: {torch.allclose(log_softmax_1, log_softmax_2)}")

# 交叉熵损失中 Softmax 的梯度精简形式
# 使用 CrossEntropyLoss（内部集成了 Softmax + NLLLoss）
criterion = torch.nn.CrossEntropyLoss()
logits = torch.randn(4, 10, requires_grad=True)  # 4 个样本，10 个类别
labels = torch.randint(0, 10, (4,))

loss = criterion(logits, labels)
loss.backward()

# 验证梯度等于 softmax(logits) - one_hot(labels)
with torch.no_grad():
    probs = F.softmax(logits, dim=1)
    expected_grad = probs.clone()
    expected_grad[range(4), labels] -= 1.0
    expected_grad /= 4  # CrossEntropyLoss 默认取均值

print(f"\n实际梯度:\n{logits.grad}")
print(f"理论梯度:\n{expected_grad}")
print(f"梯度一致: {torch.allclose(logits.grad, expected_grad, atol=1e-6)}")
```

## 深度学习关联

- **多分类问题的标准输出层**：Softmax 是几乎所有多分类模型的标准输出层。从 LeNet 到 ResNet，从 BERT 到 GPT，分类任务的最后一层几乎都是全连接层加 Softmax。梯度 $p_j - y_j$ 的简洁形式使得反向传播实现极其高效。

- **知识蒸馏中的温度 Softmax**：Hinton 提出的知识蒸馏（Knowledge Distillation）使用带温度参数的 Softmax：$p_i = e^{z_i/T} / \sum_j e^{z_j/T}$。高温使概率分布更加平滑，保留了"暗知识"（如类别之间的相似性信息），学生网络可以从这些软标签中学习到更丰富的知识。

- **注意力机制的核心**：Transformer 中的自注意力（Self-Attention）机制本质上是 Softmax 的应用场景：$\text{Attention}(Q,K,V) = \text{Softmax}(QK^T/\sqrt{d_k})V$。这里的 Softmax 将注意力分数转化为注意力权重，决定了每个 token 应该关注其他 token 的多少。数值稳定的 Log-Sum-Exp 技巧在注意力机制的实现中同样关键。
