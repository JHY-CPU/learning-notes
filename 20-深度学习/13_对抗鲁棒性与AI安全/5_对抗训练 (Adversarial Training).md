# 5_对抗训练 (Adversarial Training)

## 1. 对抗训练概述

对抗训练是目前**最有效的对抗鲁棒性提升方法**。其核心思想是在训练过程中主动生成对抗样本，并用这些对抗样本训练模型，使模型学会在对抗扰动下仍能正确分类。

### 1.1 Min-Max 框架

对抗训练可以形式化为一个 Min-Max 优化问题：

$$\min_\theta \mathbb{E}_{(x,y) \sim \mathcal{D}} \left[ \max_{\|\delta\|_p \leq \epsilon} \mathcal{L}(f_\theta(x + \delta), y) \right]$$

- **内层最大化** (Inner Maximization): 找到最能欺骗模型的对抗扰动
- **外层最小化** (Outer Minimization): 更新模型参数以降低对抗样本的损失

## 2. PGD 对抗训练 (PGD-AT)

### 2.1 算法原理

Madry 等人 (2018) 证明，用 PGD 攻击作为内层最大化求解器，可以得到最强的一阶对抗鲁棒性：

```python
def pgd_adversarial_training(model, train_loader, optimizer, scheduler,
                              epsilon=8/255, alpha=2/255, pgd_steps=10,
                              epochs=100, device='cuda'):
    """
    标准 PGD 对抗训练
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0
        correct = 0
        total = 0

        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)

            # === 内层最大化: 生成对抗样本 ===
            images_adv = pgd_attack(model, images, labels,
                                    epsilon, alpha, pgd_steps)

            # === 外层最小化: 在对抗样本上训练 ===
            model.train()
            outputs = model(images_adv)
            loss = F.cross_entropy(outputs, labels)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, pred = outputs.max(1)
            correct += pred.eq(labels).sum().item()
            total += labels.size(0)

        if scheduler:
            scheduler.step()

        train_acc = 100.0 * correct / total
        print(f"Epoch {epoch+1}/{epochs} | "
              f"Loss: {total_loss/len(train_loader):.4f} | "
              f"Train Acc: {train_acc:.2f}%")
```

### 2.2 训练稳定性技巧

```python
def pgd_at_with_stabilization(model, train_loader, optimizer, scheduler,
                               epsilon, alpha, pgd_steps, epochs):
    """
    带稳定化技巧的 PGD 对抗训练
    """
    for epoch in range(epochs):
        model.train()
        total_loss_clean = 0
        total_loss_adv = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            # 生成对抗样本 (不需要梯度回传到攻击)
            with torch.enable_grad():
                images_adv = pgd_attack(model, images, labels,
                                        epsilon, alpha, pgd_steps)

            # 混合干净和对抗样本训练 (可选)
            model.train()

            # 对抗损失
            outputs_adv = model(images_adv)
            loss_adv = F.cross_entropy(outputs_adv, labels)

            # 干净损失 (防止标准准确率过度下降)
            outputs_clean = model(images)
            loss_clean = F.cross_entropy(outputs_clean, labels)

            # 加权组合
            loss = 0.5 * loss_adv + 0.5 * loss_clean

            optimizer.zero_grad()
            loss.backward()

            # 梯度裁剪 (防止梯度爆炸)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)

            optimizer.step()

            total_loss_clean += loss_clean.item()
            total_loss_adv += loss_adv.item()

        if scheduler:
            scheduler.step()
```

## 3. TRADES 方法

### 3.1 理论动机

Zhang 等人 (2019) 提出 TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss)，从理论角度分析鲁棒性与准确率的权衡：

$$\mathcal{L}_{\text{TRADES}} = \underbrace{\mathcal{L}(f_\theta(x), y)}_{\text{自然准确率}} + \lambda \cdot \underbrace{\mathcal{L}(f_\theta(x), f_\theta(x_{\text{adv}}))}_{\text{鲁棒性正则项}}$$

### 3.2 直观解释

- 第一项：确保模型在干净样本上的准确性
- 第二项：确保模型对 $x$ 和 $x_{\text{adv}}$ 的预测尽量一致
- $\lambda$ 平衡两项的权重

### 3.3 PyTorch 实现

```python
def trades_loss(model, images, labels, epsilon, alpha, num_steps,
                beta=6.0, device='cuda'):
    """
    TRADES 损失函数

    Args:
        model: 模型
        images: 干净输入
        labels: 真实标签
        epsilon: 扰动上界
        alpha: 步长
        num_steps: PGD 步数
        beta: 鲁棒性正则项权重
    """
    model.eval()

    # 第一步: 在干净样本上的预测 (固定)
    with torch.no_grad():
        clean_output = model(images)
        clean_probs = F.softmax(clean_output, dim=1)

    # 第二步: 生成对抗样本 (最大化 KL 散度)
    images_adv = images.clone().detach()
    for _ in range(num_steps):
        images_adv.requires_grad_(True)
        adv_output = model(images_adv)
        adv_log_probs = F.log_softmax(adv_output, dim=1)

        # KL 散度: D_KL(p_clean || p_adv)
        kl_loss = F.kl_div(adv_log_probs, clean_probs, reduction='batchmean')

        model.zero_grad()
        kl_loss.backward()

        images_adv = images_adv.detach() + alpha * images_adv.grad.sign()
        delta = torch.clamp(images_adv - images, -epsilon, epsilon)
        images_adv = torch.clamp(images + delta, 0, 1).detach()

    # 第三步: 计算 TRADES 损失
    model.train()
    natural_output = model(images)
    adv_output = model(images_adv)

    # 自然损失
    natural_loss = F.cross_entropy(natural_output, labels)

    # 鲁棒性损失 (KL 散度)
    robust_loss = F.kl_div(
        F.log_softmax(adv_output, dim=1),
        F.softmax(natural_output.detach(), dim=1),
        reduction='batchmean'
    )

    total_loss = natural_loss + beta * robust_loss

    return total_loss

def trades_training(model, train_loader, optimizer, scheduler,
                    epsilon, alpha, pgd_steps, beta=6.0, epochs=100):
    """
    TRADES 完整训练流程
    """
    for epoch in range(epochs):
        model.train()
        total_loss = 0

        for images, labels in train_loader:
            images, labels = images.cuda(), labels.cuda()

            loss = trades_loss(model, images, labels, epsilon, alpha,
                              pgd_steps, beta)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        if scheduler:
            scheduler.step()

        print(f"Epoch {epoch+1}/{epochs} | "
              f"TRADES Loss: {total_loss/len(train_loader):.4f}")
```

## 4. PGD-AT vs TRADES 对比

| 特性 | PGD-AT | TRADES |
|------|--------|--------|
| 损失函数 | $\mathcal{L}(f(x_{\text{adv}}), y)$ | $\mathcal{L}(f(x), y) + \beta \cdot \text{KL}(f(x) \| f(x_{\text{adv}}))$ |
| 鲁棒准确率 | 高 | **更高** |
| 标准准确率 | 较低 | **较高** |
| 计算成本 | 1次 PGD | 2次前向 + 1次 PGD |
| 超参数 | 少 (ε, α, T) | 多 (+β) |
| 理论保证 | 隐式正则化 | 显式权衡 |

## 5. 权重稳定化技术

### 5.1 权重扰动正则化

```python
def weight_perturbation_regularization(model, images, labels,
                                        epsilon_input=8/255,
                                        epsilon_weight=0.01):
    """
    同时对输入和权重进行对抗扰动
    """
    # 保存原始权重
    original_weights = {n: p.clone() for n, p in model.named_parameters()}

    # 对权重添加小扰动
    for name, param in model.named_parameters():
        if 'weight' in name:
            noise = torch.randn_like(param) * epsilon_weight
            param.data.add_(noise)

    # 正常对抗训练
    adv_images = pgd_attack(model, images, labels, epsilon_input, 2/255, 10)
    loss = F.cross_entropy(model(adv_images), labels)

    # 恢复原始权重
    for name, param in model.named_parameters():
        param.data.copy_(original_weights[name])

    return loss
```

## 6. 鲁棒性-准确性权衡

### 6.1 理论分析

Tsipras 等人 (2019) 从信息论角度证明：

> 鲁棒分类器需要学习与标准分类器**不同的特征表示**。鲁棒特征比标准特征携带更少的关于标签的信息，导致标准准确率下降。

### 6.2 实验数据

```
典型实验结果 (ResNet-18 on CIFAR-10):

训练方法        | 标准准确率 | FGSM准确率 | PGD-20准确率
---------------|-----------|------------|-------------
标准训练        |  95.2%    |   23.1%    |    0.0%
FGSM-AT (ε=4/255)| 89.3%   |   56.7%    |   50.2%
PGD-AT (ε=4/255) | 87.1%   |   62.3%    |   58.4%
TRADES (ε=4/255) | 88.6%   |   64.1%    |   61.2%
FGSM-AT (ε=8/255)| 82.4%   |   48.3%    |   41.5%
PGD-AT (ε=8/255) | 80.5%   |   55.7%    |   52.1%
TRADES (ε=8/255) | 82.7%   |   58.3%    |   55.4%
```

## 7. 实用训练配置

```python
# CIFAR-10 ResNet-18 推荐配置
config = {
    'epsilon': 8/255,         # L∞ 扰动上界
    'alpha': 2/255,           # PGD 步长
    'pgd_steps': 10,          # PGD 迭代次数
    'epochs': 100,            # 训练轮数
    'batch_size': 128,        # 批次大小
    'lr': 0.1,                # 初始学习率
    'momentum': 0.9,          # SGD 动量
    'weight_decay': 5e-4,     # 权重衰减
    'scheduler': 'cosine',    # 学习率调度
    'beta': 6.0,              # TRADES 权重 (仅 TRADES)
}

# 学习率调度: 在 epoch 75, 90 处衰减 0.1
scheduler = torch.optim.lr_scheduler.MultiStepLR(
    optimizer, milestones=[75, 90], gamma=0.1
)
```

## 8. 总结

| 要点 | 说明 |
|------|------|
| 核心框架 | Min-Max 优化 |
| 主要方法 | PGD-AT (最强一阶鲁棒性)、TRADES (更好的权衡) |
| 关键权衡 | 鲁棒准确率 vs 标准准确率 |
| 计算开销 | 比标准训练高 10-20 倍 |
| 实用建议 | 使用 TRADES、适当混合干净数据、调整 ε |

---

**参考文献：**

1. Madry, A. et al. (2018). *Towards deep learning models resistant to adversarial attacks*. ICLR.
2. Zhang, H. et al. (2019). *Theoretically principled trade-off between robustness and accuracy*. ICML.
3. Tsipras, D. et al. (2019). *Robustness may be at odds with accuracy*. ICLR.
