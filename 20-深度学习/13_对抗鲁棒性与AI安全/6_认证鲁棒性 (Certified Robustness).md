# 6_认证鲁棒性 (Certified Robustness)

## 1. 认证鲁棒性概述

对抗训练等经验防御方法虽然有效，但**无法提供理论保证**——新的攻击方法可能突破已有防御。认证鲁棒性 (Certified Robustness) 旨在为每个输入提供**可证明的安全区域**，保证在该区域内所有扰动都不会改变分类结果。

### 1.1 经验防御 vs 认证防御

| 特性 | 经验防御 (PGD-AT) | 认证防御 |
|------|-------------------|----------|
| 理论保证 | 无 | 有 |
| 鲁棒性边界 | 未知 | 精确或保守 |
| 计算效率 | 高 | 低-中 |
| 实用鲁棒性 | 强 | 可能略弱 |
| 防御完备性 | 对新攻击不确定 | 对所有一阶攻击保证 |

### 1.2 形式化定义

对于输入 $x$、真实标签 $y$ 和分类器 $f$，认证鲁棒性要求：

$$\forall \delta \in \mathcal{B}_p(x, \epsilon): f(x + \delta) = y$$

其中 $\mathcal{B}_p(x, \epsilon) = \{x' : \|x' - x\|_p \leq \epsilon\}$。

## 2. 随机平滑 (Randomized Smoothing)

### 2.1 核心思想

随机平滑 (Cohen et al., 2019) 通过对输入添加高斯噪声并投票，构造一个**平滑分类器**：

$$g(x) = \arg\max_c \mathbb{P}_{\delta \sim \mathcal{N}(0, \sigma^2 I)} \left[ f(x + \delta) = c \right]$$

### 2.2 认证定理

> **定理 (Cohen et al.)：** 假设平滑分类器 $g$ 在 $x$ 处的最高类概率为 $p_A$，次高类概率为 $p_B$。如果 $p_A \geq p_B$，则对于所有 $\|\delta\|_2 \leq R$，有 $g(x + \delta) = g(x)$，其中：
>
> $$R = \frac{\sigma}{2} \left( \Phi^{-1}(p_A) - \Phi^{-1}(p_B) \right)$$
>
> $\Phi^{-1}$ 是标准正态分布的逆累积分布函数。

### 2.3 认证半径计算

```python
import numpy as np
from scipy.stats import norm

def compute_certified_radius(p_A, p_B, sigma):
    """
    计算认证鲁棒半径

    Args:
        p_A: 最高类的预测概率
        p_B: 次高类的预测概率
        sigma: 噪声标准差

    Returns:
        radius: 认证安全半径
    """
    if p_A <= p_B:
        return 0.0  # 无法认证

    radius = (sigma / 2) * (norm.ppf(p_A) - norm.ppf(p_B))
    return radius
```

### 2.4 PyTorch 实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import Counter

class SmoothClassifier(nn.Module):
    """
    随机平滑分类器
    """
    def __init__(self, base_model, sigma=0.25, num_samples=100):
        super().__init__()
        self.base_model = base_model
        self.sigma = sigma
        self.num_samples = num_samples

    def predict(self, x, n_samples=100):
        """
        预测最高类和其次高类 (Monte Carlo 估计)
        """
        counts = Counter()
        with torch.no_grad():
            for _ in range(n_samples):
                noise = torch.randn_like(x) * self.sigma
                noisy_x = torch.clamp(x + noise, 0, 1)
                pred = self.base_model(noisy_x).argmax(1)
                counts[pred.item()] += 1

        # 返回最高和次高类
        sorted_counts = counts.most_common()
        top_class = sorted_counts[0][0]
        top_count = sorted_counts[0][1]
        second_count = sorted_counts[1][1] if len(sorted_counts) > 1 else 0

        return top_class, top_count, second_count

    def certify(self, x, n0=100, n=100000, alpha=0.001):
        """
        认证单个样本的鲁棒性

        Args:
            x: 输入样本
            n0: 第一阶段采样数
            n: 第二阶段采样数
            alpha: 置信水平 (1-alpha 为置信度)
        """
        # 第一阶段: 估计最高类
        counts0 = Counter()
        with torch.no_grad():
            for _ in range(n0):
                noise = torch.randn_like(x) * self.sigma
                noisy_x = torch.clamp(x + noise, 0, 1)
                pred = self.base_model(noisy_x).argmax(1)
                counts0[pred.item()] += 1

        c_A = counts0.most_common(1)[0][0]

        # 第二阶段: 精确估计 p_A
        count_A = 0
        with torch.no_grad():
            for _ in range(n):
                noise = torch.randn_like(x) * self.sigma
                noisy_x = torch.clamp(x + noise, 0, 1)
                pred = self.base_model(noisy_x).argmax(1)
                if pred.item() == c_A:
                    count_A += 1

        # 计算 p_A 的置信下界
        p_A_lower = proportion_confint(count_A, n, alpha, method='beta')[0]

        if p_A_lower < 0.5:
            return c_A, 0.0  # 无法认证

        # 计算认证半径
        radius = self.sigma * norm.ppf(p_A_lower)

        return c_A, radius
```

### 2.5 平滑训练

```python
def smooth_training(model, train_loader, optimizer, sigma=0.25, epochs=100):
    """
    针对随机平滑的训练方法
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # 添加高斯噪声
            noise = torch.randn_like(images) * sigma
            noisy_images = torch.clamp(images + noise, 0, 1)

            outputs = model(noisy_images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}")
```

## 3. 区间界限传播 (IBP)

### 3.1 核心思想

IBP (Interval Bound Propagation) 通过传播输入的区间边界，计算每一层输出的区间，最终验证输出区间是否保证正确分类。

### 3.2 区间传播公式

对于线性层 $y = Wx + b$，若输入 $x \in [\underline{x}, \overline{x}]$：

$$\underline{y} = W^+ \underline{x} + W^- \overline{x} + b$$
$$\overline{y} = W^+ \overline{x} + W^- \underline{x} + b$$

其中 $W^+ = \max(W, 0)$，$W^- = \min(W, 0)$。

对于 ReLU 激活函数：

$$\underline{y} = \text{ReLU}(\underline{x}), \quad \overline{y} = \text{ReLU}(\overline{x})$$

### 3.3 PyTorch 实现

```python
class IBPLayer(nn.Module):
    """区间界限传播的线性层"""
    def __init__(self, linear_layer):
        super().__init__()
        self.weight = linear_layer.weight
        self.bias = linear_layer.bias

    def forward(self, lower, upper):
        W_plus = torch.clamp(self.weight, min=0)
        W_minus = torch.clamp(self.weight, max=0)

        new_lower = W_plus @ lower + W_minus @ upper + self.bias
        new_upper = W_plus @ upper + W_minus @ lower + self.bias

        return new_lower, new_upper

class IBPReLU(nn.Module):
    """区间界限传播的 ReLU 层"""
    def forward(self, lower, upper):
        return torch.clamp(lower, min=0), torch.clamp(upper, min=0)

class IBPModel(nn.Module):
    """使用 IBP 验证鲁棒性的模型"""
    def __init__(self, model, epsilon):
        super().__init__()
        self.model = model
        self.epsilon = epsilon

    def forward(self, x):
        # 正常前向传播
        return self.model(x)

    def compute_bounds(self, x, labels):
        """
        计算输出的区间界限
        """
        # 初始化输入区间
        lower = torch.clamp(x - self.epsilon, 0, 1)
        upper = torch.clamp(x + self.epsilon, 0, 1)

        # 逐层传播
        for layer in self.model:
            if isinstance(layer, nn.Linear):
                ibp_layer = IBPLayer(layer)
                lower, upper = ibp_layer(lower, upper)
            elif isinstance(layer, nn.ReLU):
                ibp_relu = IBPReLU()
                lower, upper = ibp_relu(lower, upper)

        return lower, upper

def ibp_loss(model, images, labels, epsilon):
    """
    IBP 训练损失: 确保真实类的下界 > 其他类的上界
    """
    lower, upper = model.compute_bounds(images, labels)

    # 对于真实类 c: 下界 lower_c
    # 对于其他类 i: 上界 upper_i
    # 要求: lower_c > upper_i for all i ≠ c

    batch_size = images.size(0)
    loss = 0

    for i in range(lower.size(1)):  # 遍历所有类别
        # margin = upper_i - lower_c
        margin = upper[:, i] - lower.gather(1, labels.unsqueeze(1)).squeeze(1)
        # 仅对 i ≠ 真实类计算
        mask = (i != labels).float()
        loss += (F.relu(margin) * mask).sum()

    return loss / batch_size
```

## 4. 方法对比

| 方法 | 认证类型 | 计算效率 | 认证半径 | 训练难度 |
|------|----------|----------|----------|----------|
| 随机平滑 | 概率 (高概率保证) | 高 (并行采样) | $L_2$ 大 | 低 |
| IBP | 确定性 | 高 (前向传播) | 保守 | 中 |
| CROWN | 确定性 | 中 | 较紧 | 高 |
| α-CROWN | 确定性 | 中 | 最紧 | 高 |

## 5. 总结

| 要点 | 说明 |
|------|------|
| 核心目标 | 提供理论可证明的安全区域 |
| 随机平滑 | Monte Carlo 估计 + 统计认证 |
| IBP | 区间算术逐层传播 |
| 权衡 | 认证半径 vs 计算效率 vs 真实鲁棒性 |
| 应用 | 安全关键场景、模型验证 |

---

**参考文献：**

1. Cohen, J. et al. (2019). *Certified adversarial robustness via randomized smoothing*. ICML.
2. Gowal, S. et al. (2019). *On the effectiveness of interval bound propagation for training verifiably robust models*. NeurIPS.
3. Zhang, H. et al. (2020). *On the effectiveness of interval bound propagation for training verifiable neural networks*. NeurIPS.
