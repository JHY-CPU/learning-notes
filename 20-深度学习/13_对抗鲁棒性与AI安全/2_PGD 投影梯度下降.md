# 2_PGD 投影梯度下降

## 1. 算法概述

PGD (Projected Gradient Descent) 是 Madry 等人于 2018 年提出的**最强一阶对抗攻击方法**。它将 FGSM 的单步操作推广为多步迭代，每步都向损失增大的方向移动并投影回约束集合。

## 2. 数学形式

### 2.1 迭代公式

PGD 攻击的迭代过程：

$$x^{(t+1)} = \Pi_{\mathcal{S}} \left( x^{(t)} + \alpha \cdot \text{sign}(\nabla_x J(\theta, x^{(t)}, y)) \right)$$

其中：
- $x^{(0)}$ 通常从原始图像或其随机扰动开始
- $\alpha$ 为每步的步长 (step size)
- $\Pi_{\mathcal{S}}$ 为投影操作，将结果投影回约束集合 $\mathcal{S}$

### 2.2 约束集合

对于 $L_\infty$ 范数约束的 PGD：

$$\mathcal{S} = \{x_{\text{adv}} : \|x_{\text{adv}} - x\|_\infty \leq \epsilon\}$$

投影操作：

$$\Pi_{\mathcal{S}}(x) = \text{clip}(x, x - \epsilon, x + \epsilon)$$

### 2.3 与 FGSM 的关系

```
FGSM (单步):
  x_adv = x + ε · sign(∇J(x))

PGD (T 步):
  x^(0) = x + random_noise     # 随机初始化
  for t = 0, 1, ..., T-1:
      x^(t+1) = Π_S(x^(t) + α · sign(∇J(x^(t))))
  x_adv = x^(T)

当 T = 1, α = ε, 无随机初始化时，PGD 等价于 FGSM
```

## 3. 关键参数分析

### 3.1 参数配置

| 参数 | 符号 | 常用值 | 说明 |
|------|------|--------|------|
| 扰动上界 | $\epsilon$ | 8/255 (CIFAR-10) | 最大总扰动 |
| 步长 | $\alpha$ | 2/255 (CIFAR-10) | 通常取 $\epsilon / 4$ |
| 迭代次数 | $T$ | 7-20 步 | 步数越多攻击越强 |
| 随机重启 | $N$ | 10-50 次 | 多次随机初始化提高攻击率 |

### 3.2 步长选择的经验法则

```python
# 经验规则: alpha ≈ epsilon / (T 或 T+4)
# CIFAR-10 典型配置
epsilon = 8 / 255    # 扰动上界
alpha = 2 / 255      # 步长
num_steps = 10       # 迭代步数

# ImageNet 典型配置
epsilon = 4 / 255
alpha = 1 / 255
num_steps = 20
```

## 4. PyTorch 完整实现

### 4.1 基础 PGD 攻击

```python
import torch
import torch.nn.functional as F

def pgd_attack(model, images, labels, epsilon, alpha, num_steps,
               random_start=True, device='cuda'):
    """
    PGD 对抗攻击实现

    Args:
        model: 目标模型
        images: 原始输入
        labels: 真实标签
        epsilon: 扰动上界 (L∞)
        alpha: 每步步长
        num_steps: 迭代步数
        random_start: 是否随机初始化
        device: 计算设备

    Returns:
        adv_images: 对抗样本
    """
    images_adv = images.clone().detach()

    if random_start:
        # 随机初始化: 在 [-epsilon, epsilon] 内均匀采样
        noise = torch.empty_like(images_adv).uniform_(-epsilon, epsilon)
        images_adv = images_adv + noise
        images_adv = torch.clamp(images_adv, 0, 1).detach()

    for _ in range(num_steps):
        images_adv.requires_grad_(True)

        # 前向传播
        outputs = model(images_adv)
        loss = F.cross_entropy(outputs, labels)

        # 反向传播
        model.zero_grad()
        loss.backward()

        # 梯度上升 (最大化损失)
        grad_sign = images_adv.grad.sign()
        images_adv = images_adv.detach() + alpha * grad_sign

        # 投影到约束集合: 确保扰动在 [x-ε, x+ε] 内
        eta = torch.clamp(images_adv - images, min=-epsilon, max=epsilon)
        images_adv = torch.clamp(images + eta, 0, 1).detach()

    return images_adv
```

### 4.2 带随机重启的 PGD

```python
def pgd_attack_with_restarts(model, images, labels, epsilon, alpha,
                              num_steps, num_restarts=10):
    """
    多次随机重启的 PGD 攻击，取损失最大的结果
    """
    batch_size = images.size(0)
    max_loss = torch.zeros(batch_size).to(images.device)
    best_adv = images.clone()

    for _ in range(num_restarts):
        adv_images = pgd_attack(model, images, labels, epsilon, alpha,
                                num_steps, random_start=True)

        with torch.no_grad():
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels, reduction='none')

            # 更新最佳对抗样本
            update_mask = loss > max_loss
            max_loss = torch.where(update_mask, loss, max_loss)
            best_adv[update_mask] = adv_images[update_mask]

    return best_adv
```

### 4.3 L2 范数 PGD

```python
def pgd_l2_attack(model, images, labels, epsilon, alpha, num_steps):
    """
    L2 范数约束的 PGD 攻击
    """
    images_adv = images.clone().detach()

    for _ in range(num_steps):
        images_adv.requires_grad_(True)
        outputs = model(images_adv)
        loss = F.cross_entropy(outputs, labels)

        model.zero_grad()
        loss.backward()

        grad = images_adv.grad.detach()

        # L2 梯度归一化
        grad_norm = grad.view(grad.size(0), -1).norm(dim=1, keepdim=True)
        grad_normalized = grad / (grad_norm.view(-1, 1, 1, 1) + 1e-8)

        # 梯度上升
        images_adv = images_adv.detach() + alpha * grad_normalized

        # L2 投影
        delta = images_adv - images
        delta_norm = delta.view(delta.size(0), -1).norm(dim=1, keepdim=True)
        factor = torch.min(
            torch.ones_like(delta_norm),
            epsilon / (delta_norm + 1e-8)
        )
        delta = delta * factor.view(-1, 1, 1, 1)
        images_adv = torch.clamp(images + delta, 0, 1).detach()

    return images_adv
```

## 5. 攻击效果对比实验

```python
def compare_attacks(model, test_loader, device='cuda'):
    """
    对比 FGSM 和不同步数 PGD 的攻击效果
    """
    epsilon = 8 / 255
    results = {}

    for name, attack_fn in [
        ('FGSM', lambda m, x, y: fgsm_attack(m, x, y, epsilon)[0]),
        ('PGD-7',  lambda m, x, y: pgd_attack(m, x, y, epsilon, 2/255, 7)),
        ('PGD-20', lambda m, x, y: pgd_attack(m, x, y, epsilon, 2/255, 20)),
        ('PGD-50', lambda m, x, y: pgd_attack(m, x, y, epsilon, 2/255, 50)),
    ]:
        correct = 0
        total = 0
        model.eval()

        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            adv_images = attack_fn(model, images, labels)

            with torch.no_grad():
                outputs = model(adv_images)
                _, pred = outputs.max(1)
                correct += pred.eq(labels).sum().item()
                total += labels.size(0)

        acc = 100.0 * correct / total
        results[name] = acc
        print(f"{name:10s}: 鲁棒准确率 = {acc:.2f}%")

    return results

# 典型输出 (ResNet-18 on CIFAR-10):
# FGSM     : 鲁棒准确率 = 45.32%
# PGD-7    : 鲁棒准确率 = 42.18%
# PGD-20   : 鲁棒准确率 = 41.95%
# PGD-50   : 鲁棒准确率 = 41.90%
```

## 6. PGD 攻击的通用性

Madry 等人证明了一个重要结论：

> **如果模型能抵抗 PGD 攻击，那么它就能抵抗所有一阶攻击方法。**

这是因为 PGD 通过多步迭代充分探索了损失函数的局部结构，在一阶信息的意义下是最强的攻击。

### 6.1 一阶攻击 vs 二阶攻击

| 类型 | 方法 | 信息需求 | 强度 |
|------|------|----------|------|
| 一阶 | FGSM, PGD, C&W-L2 | 梯度信息 | PGD 为最强 |
| 二阶 | C&W-L2 (牛顿法) | Hessian 矩阵 | 更强但计算昂贵 |
| 组合 | 集成攻击 | 多模型梯度 | 超越单模型 PGD |

## 7. PGD 对抗训练

PGD 对抗训练是当前标准的鲁棒训练方法：

```python
def pgd_adversarial_training(model, optimizer, train_loader, epsilon,
                              alpha, pgd_steps, epochs=100):
    """
    PGD 对抗训练完整流程
    """
    model.train()

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # 生成 PGD 对抗样本
            adv_images = pgd_attack(model, images, labels,
                                    epsilon, alpha, pgd_steps)

            # 在对抗样本上训练
            model.train()
            outputs = model(adv_images)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs}, Loss: {total_loss/len(train_loader):.4f}, "
              f"Train Acc: {acc:.2f}%")
```

## 8. 总结

| 要点 | 说明 |
|------|------|
| 核心思想 | 多步梯度迭代 + 投影约束 |
| 公式 | $x^{(t+1)} = \Pi_{\mathcal{S}}(x^{(t)} + \alpha \cdot \text{sign}(\nabla J))$ |
| 优势 | 一阶最强攻击、通用性好 |
| 关键参数 | $\epsilon$、$\alpha \approx \epsilon/4$、$T=10\text{-}20$ |
| 应用 | 鲁棒性评估、对抗训练标准方法 |

---

**参考文献：**

1. Madry, A. et al. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR.
2. Kurakin, A. et al. (2017). *Adversarial machine learning at scale*. ICLR.
