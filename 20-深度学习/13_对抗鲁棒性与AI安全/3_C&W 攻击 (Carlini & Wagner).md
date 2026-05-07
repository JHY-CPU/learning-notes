# 3_C&W 攻击 (Carlini & Wagner)

## 1. 算法概述

C&W (Carlini & Wagner) 攻击是 2017 年提出的**优化框架型对抗攻击方法**，通过求解一个精心设计的优化问题来生成高质量对抗样本。它在多个鲁棒性评估中被认为是**最强的白盒攻击之一**。

## 2. 核心思想

### 2.1 与 FGSM/PGD 的区别

| 特性 | FGSM/PGD | C&W |
|------|----------|-----|
| 目标 | 最大化分类损失 | 最小化扰动 + 确保误分类 |
| 约束 | 固定 $\epsilon$ 球 | 自动寻找最小扰动 |
| 框架 | 梯度迭代 | 约束优化 |
| 输出 | 给定 $\epsilon$ 下的对抗样本 | 最小对抗扰动 |

### 2.2 优化目标

C&W 的核心优化问题：

$$\min_{\delta} \|\delta\|_p + c \cdot f(x + \delta) \quad \text{s.t.} \quad x + \delta \in [0, 1]^d$$

其中 $f(\cdot)$ 是设计的损失函数，$c$ 是平衡系数。

## 3. 损失函数设计

### 3.1 目标函数 $f$ 的七种候选

Carlini 提出了 7 种候选损失函数，实验发现以下形式最优：

$$f(x') = \max\left(\max_{i \neq t} Z(x')_i - Z(x')_t, -\kappa\right)$$

其中：
- $Z(x')$ 是 Softmax 前的 logits
- $t$ 是目标类别
- $\kappa$ 是置信度参数 (confidence parameter)
- $\max(\cdot, -\kappa)$ 保证当 $f(x') \leq 0$ 时攻击成功，且置信度至少为 $\kappa$

### 3.2 置信度参数 κ

| κ 值 | 效果 |
|------|------|
| 0 | 仅需分类错误即可 |
| 1-5 | 中等置信度错误 |
| 20+ | 高置信度错误，更难被检测 |

## 4. 变量变换

### 4.1 无约束优化变换

为了避免在优化过程中手动处理 $[0,1]$ 约束，引入变量变换：

$$\delta_i = \frac{1}{2}(\tanh(w_i) + 1) - x_i$$

其中 $w_i \in \mathbb{R}$ 是无约束优化变量。当 $w_i \to \pm\infty$ 时，$x' \to 0$ 或 $1$。

### 4.2 完整优化问题

$$\min_w \left\| \frac{1}{2}(\tanh(w) + 1) - x \right\|_p + c \cdot f\left(\frac{1}{2}(\tanh(w) + 1)\right)$$

## 5. 不同 Lp 范数变体

### 5.1 L2 攻击 (最常用)

$$\min_w \|\delta\|_2 + c \cdot f(x + \delta)$$

### 5.2 L∞ 攻击

$L_\infty$ 不可直接微分，使用平滑近似：

$$\min_w \sum_i \max(|\delta_i| - \tau, 0) + c \cdot f(x + \delta)$$

其中 $\tau$ 是阈值参数，逐步增大以逼近 $L_\infty$ 约束。

### 5.3 L0 攻击

通过迭代移除像素实现：
1. 先用 $L_2$ 攻击找到扰动
2. 移除扰动最小的像素 (设 $\delta_i = 0$)
3. 在剩余像素上重新优化
4. 重复直到无法成功攻击

## 6. PyTorch 实现

### 6.1 L2 C&W 攻击

```python
import torch
import torch.nn as nn
import torch.optim as optim

def cw_attack_l2(model, images, labels, target_labels=None,
                 c=1.0, kappa=0, num_steps=1000, lr=0.01, device='cuda'):
    """
    C&W L2 攻击实现

    Args:
        model: 目标模型
        images: 原始输入 (batch)
        labels: 真实标签
        target_labels: 目标类别 (有目标攻击时使用)
        c: 平衡系数
        kappa: 置信度参数
        num_steps: 优化步数
        lr: 学习率
    """
    batch_size = images.size(0)
    images = images.to(device)
    labels = labels.to(device)

    # 无约束优化变量: w = arctanh(2x - 1)
    w = torch.atanh((2 * images - 1) * 0.9999).detach().requires_grad_(True)

    optimizer = optim.Adam([w], lr=lr)

    best_adv = images.clone()
    best_l2 = torch.full((batch_size,), float('inf')).to(device)

    for step in range(num_steps):
        optimizer.zero_grad()

        # 变换回图像空间
        x_adv = 0.5 * (torch.tanh(w) + 1)

        # 计算 L2 扰动
        l2_dist = ((x_adv - images) ** 2).view(batch_size, -1).sum(dim=1)

        # 计算 f(x')
        outputs = model(x_adv)
        if target_labels is not None:
            # 有目标攻击: 最大化目标类 logit - 最大其他类 logit
            real = outputs.gather(1, target_labels.unsqueeze(1)).squeeze(1)
            other = (outputs.scatter(1, target_labels.unsqueeze(1), -1e10)
                   .max(dim=1)[0])
            f_loss = torch.clamp(other - real + kappa, min=0)
        else:
            # 无目标攻击: 最大化其他类 logit - 真实类 logit
            real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
            other = (outputs.scatter(1, labels.unsqueeze(1), -1e10)
                   .max(dim=1)[0])
            f_loss = torch.clamp(real - other + kappa, min=0)

        # 总损失
        loss = l2_dist + c * f_loss
        loss.sum().backward()
        optimizer.step()

        # 记录最佳对抗样本
        with torch.no_grad():
            is_adv = f_loss <= 0
            is_smaller = l2_dist < best_l2
            update = is_adv & is_smaller
            best_l2[update] = l2_dist[update]
            best_adv[update] = x_adv[update]

    return best_adv.detach()
```

### 6.2 L∞ C&W 攻击

```python
def cw_attack_linf(model, images, labels, c=1.0, kappa=0,
                   num_steps=1000, lr=0.01, tau_steps=10):
    """
    C&W L∞ 攻击实现
    """
    batch_size = images.size(0)
    w = torch.atanh((2 * images - 1) * 0.9999).detach().requires_grad_(True)
    optimizer = optim.Adam([w], lr=lr)

    tau = 0.0  # 初始阈值
    tau_inc = 1.0 / tau_steps  # 阈值增长

    for step in range(num_steps):
        optimizer.zero_grad()

        x_adv = 0.5 * (torch.tanh(w) + 1)
        delta = x_adv - images

        # L∞ 损失: 用 max(|δ| - τ, 0) 近似
        linf_loss = torch.clamp(delta.abs() - tau, min=0).sum(dim=[1, 2, 3])

        # 分类损失
        outputs = model(x_adv)
        real = outputs.gather(1, labels.unsqueeze(1)).squeeze(1)
        other = outputs.scatter(1, labels.unsqueeze(1), -1e10).max(dim=1)[0]
        f_loss = torch.clamp(real - other + kappa, min=0)

        loss = (linf_loss * c + f_loss).sum()
        loss.backward()
        optimizer.step()

        # 周期性增大阈值
        if (step + 1) % (num_steps // tau_steps) == 0:
            tau += tau_inc

    return (0.5 * (torch.tanh(w) + 1)).detach()
```

## 7. 二分搜索求最优 c

```python
def cw_attack_binary_search(model, images, labels, num_steps=1000,
                             search_steps=9, lr=0.01, device='cuda'):
    """
    对 c 进行二分搜索，找到最小扰动的对抗样本
    """
    batch_size = images.size(0)

    # c 的搜索范围
    c_low = torch.zeros(batch_size).to(device)
    c_high = torch.ones(batch_size).to(device) * 1e10
    c_mid = torch.ones(batch_size).to(device)

    best_adv = images.clone()
    best_l2 = torch.full((batch_size,), float('inf')).to(device)

    for search_step in range(search_steps):
        adv_images = cw_attack_l2(model, images, labels,
                                   c=c_mid, num_steps=num_steps, lr=lr)

        with torch.no_grad():
            l2_dist = ((adv_images - images) ** 2).view(batch_size, -1).sum(1)
            outputs = model(adv_images)
            _, pred = outputs.max(1)
            is_adv = pred != labels

            # 更新二分搜索
            for i in range(batch_size):
                if is_adv[i]:
                    c_high[i] = min(c_high[i], c_mid[i])
                    if l2_dist[i] < best_l2[i]:
                        best_l2[i] = l2_dist[i]
                        best_adv[i] = adv_images[i]
                else:
                    c_low[i] = max(c_low[i], c_mid[i])

                if c_high[i] < 1e10:
                    c_mid[i] = (c_low[i] + c_high[i]) / 2
                else:
                    c_mid[i] = c_low[i] * 10

    return best_adv
```

## 8. C&W 攻击 vs 其他方法对比

| 方法 | 攻击成功率 | 扰动大小 | 计算成本 | 检测难度 |
|------|------------|----------|----------|----------|
| FGSM | 中 | 大 | 极低 | 容易检测 |
| PGD | 高 | 中 | 中 | 中等 |
| C&W L2 | 极高 | **最小** | 高 | **难以检测** |
| C&W L∞ | 极高 | 中 | 高 | 难检测 |
| DeepFool | 高 | 极小 | 中 | 难检测 |

## 9. 防御 C&W 攻击

C&W 攻击揭示了当时许多防御方法的弱点。有效的防御策略包括：

1. **PGD 对抗训练**: 在 PGD 对抗样本上训练的模型对 C&W 也有较好鲁棒性
2. **随机化防御**: 输入随机化可降低 C&W 的攻击效果
3. **认证防御**: 随机平滑等可证明鲁棒性方法

## 10. 总结

| 要点 | 说明 |
|------|------|
| 核心思想 | 约束优化：最小扰动 + 保证误分类 |
| 损失函数 | $f(x') = \max(\max_{i \neq t} Z_i - Z_t, -\kappa)$ |
| 关键技巧 | $\tanh$ 变量替换、置信度参数、二分搜索 |
| 优势 | 扰动最小、攻击成功率最高 |
| 影响 | 揭露了许多"防御"的无效性 |

---

**参考文献：**

1. Carlini, N. & Wagner, D. (2017). *Towards evaluating the robustness of neural networks*. IEEE S&P.
2. Carlini, N. & Wagner, D. (2017). *Adversarial examples are not easily detected: Bypassing ten detection methods*. AISec.
