# 1_FGSM 快速梯度符号法

## 1. 算法概述

FGSM (Fast Gradient Sign Method) 是 Goodfellow 等人于 2015 年提出的经典对抗攻击方法。其核心思想是：**利用模型损失函数关于输入的梯度方向，生成最大化损失的扰动**。

## 2. 数学推导

### 2.1 损失函数的线性近似

对于模型参数 $\theta$、输入 $x$、标签 $y$ 和损失函数 $J(\theta, x, y)$，在输入空间做一阶泰勒展开：

$$J(\theta, x + \delta, y) \approx J(\theta, x, y) + \delta^T \nabla_x J(\theta, x, y)$$

要最大化损失的增量 $\delta^T \nabla_x J$，在 $L_\infty$ 约束 $\|\delta\|_\infty \leq \epsilon$ 下，最优解为：

$$\delta^* = \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

### 2.2 FGSM 公式

$$x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y))$$

其中：
- $\nabla_x J(\theta, x, y)$ 是损失关于输入的梯度
- $\text{sign}(\cdot)$ 逐元素取符号（+1 或 -1）
- $\epsilon$ 控制扰动幅度

### 2.3 有目标攻击变体

有目标攻击要求模型输出接近目标类别 $y_{\text{target}}$，需最小化该类别的损失：

$$x_{\text{adv}} = x - \epsilon \cdot \text{sign}(\nabla_x J(\theta, x, y_{\text{target}}))$$

注意符号变为**负号**——因为我们要**最小化**目标类别的损失。

## 3. 参数选择

### 3.1 扰动参数 ε 的影响

| ε 值 | 攻击效果 | 图像质量 | 鲁棒性 |
|------|----------|----------|--------|
| 0.01 | 攻击成功率低 | 几乎无变化 | 大多数模型可抵抗 |
| 0.05 | 中等成功率 | 轻微噪声可见 | 需要一定鲁棒性 |
| 0.1 | 高成功率 | 噪声可见但不影响识别 | 需要对抗训练 |
| 0.3 | 几乎100%成功 | 明显失真 | 需要强鲁棒性 |
| 0.5+ | 100%成功 | 图像被破坏 | 无意义 |

### 3.2 经验选择

```python
# CIFAR-10 常用: epsilon = 8/255 ≈ 0.031
# MNIST 常用:   epsilon = 0.3
# ImageNet 常用: epsilon = 4/255 ≈ 0.016 或 8/255
```

## 4. PyTorch 完整实现

### 4.1 FGSM 攻击函数

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def fgsm_attack(model, images, labels, epsilon):
    """
    FGSM 对抗攻击实现

    Args:
        model: 目标神经网络
        images: 原始输入图像 (batch)
        labels: 真实标签
        epsilon: 扰动幅度

    Returns:
        adv_images: 对抗样本
        perturbation: 扰动量
    """
    # 复制输入并开启梯度追踪
    images_adv = images.clone().detach().requires_grad_(True)

    # 前向传播计算损失
    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, labels)

    # 反向传播计算输入梯度
    model.zero_grad()
    loss.backward()

    # 提取梯度符号并生成扰动
    grad_sign = images_adv.grad.data.sign()
    perturbation = epsilon * grad_sign

    # 生成对抗样本并裁剪到合法范围
    adv_images = images_adv + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)  # 图像像素范围 [0, 1]

    return adv_images.detach(), perturbation
```

### 4.2 有目标 FGSM 攻击

```python
def fgsm_targeted_attack(model, images, target_labels, epsilon):
    """
    有目标 FGSM 攻击：强制模型输出指定类别
    """
    images_adv = images.clone().detach().requires_grad_(True)

    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, target_labels)

    model.zero_grad()
    loss.backward()

    # 注意：有目标攻击使用负号
    grad_sign = images_adv.grad.data.sign()
    perturbation = -epsilon * grad_sign

    adv_images = images_adv + perturbation
    adv_images = torch.clamp(adv_images, 0, 1)

    return adv_images.detach()
```

### 4.3 攻击效果评估

```python
def evaluate_attack(model, test_loader, epsilon, device='cuda'):
    """
    评估 FGSM 攻击效果
    """
    correct_clean = 0
    correct_adv = 0
    total = 0

    model.eval()
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)

        # 干净样本准确率
        outputs_clean = model(images)
        _, pred_clean = outputs_clean.max(1)
        correct_clean += pred_clean.eq(labels).sum().item()

        # 对抗样本准确率
        adv_images, _ = fgsm_attack(model, images, labels, epsilon)
        outputs_adv = model(adv_images)
        _, pred_adv = outputs_adv.max(1)
        correct_adv += pred_adv.eq(labels).sum().item()

        total += labels.size(0)

    clean_acc = 100.0 * correct_clean / total
    adv_acc = 100.0 * correct_adv / total
    attack_rate = 100.0 * (correct_clean - correct_adv) / correct_clean

    print(f"ε = {epsilon:.4f}")
    print(f"  干净准确率: {clean_acc:.2f}%")
    print(f"  对抗准确率: {adv_acc:.2f}%")
    print(f"  攻击成功率: {attack_rate:.2f}%")

    return clean_acc, adv_acc
```

## 5. 完整攻击流程示例

```python
import torchvision
import torchvision.transforms as transforms

# 1. 加载数据
transform = transforms.Compose([transforms.ToTensor()])
testset = torchvision.datasets.MNIST(root='./data', train=False,
                                      download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(testset, batch_size=100, shuffle=False)

# 2. 加载预训练模型
class SimpleCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, 3, padding=1)
        self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        self.pool = nn.MaxPool2d(2)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(x)
        x = F.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

model = SimpleCNN()
# model.load_state_dict(torch.load('mnist_cnn.pth'))

# 3. 扫描不同 epsilon 值
for eps in [0.0, 0.05, 0.1, 0.15, 0.2, 0.25, 0.3]:
    evaluate_attack(model, test_loader, epsilon=eps)
```

## 6. FGSM 的局限性

### 6.1 单步方法的不足

| 局限性 | 说明 |
|--------|------|
| 粗糙的线性近似 | 一阶泰勒展开只在小范围内有效 |
| 扰动过大时失效 | 当 $\epsilon$ 较大时，线性近似严重偏离真实损失曲面 |
| 不能迭代优化 | 无法精细调整扰动以找到最小对抗扰动 |

### 6.2 与 PGD 的对比

| 特性 | FGSM | PGD |
|------|------|-----|
| 步数 | 单步 | 多步迭代 |
| 攻击强度 | 较弱 | 更强 |
| 计算成本 | 低 (1 次反向传播) | 高 (多次反向传播) |
| 扰动质量 | 较粗糙 | 更精细 |
| 适用场景 | 快速评估、对抗训练 | 严格鲁棒性评估 |

## 7. FGSM 在对抗训练中的应用

FGSM 也可用于对抗训练 (Adversarial Training)：

$$\min_\theta \mathbb{E}_{(x,y) \sim D} \left[ \max_{\|\delta\| \leq \epsilon} J(\theta, x + \delta, y) \right]$$

```python
def fgsm_adversarial_training(model, optimizer, images, labels, epsilon):
    """
    FGSM 对抗训练的一步
    """
    # 生成对抗样本
    images_adv, _ = fgsm_attack(model, images, labels, epsilon)

    # 在对抗样本上训练
    model.train()
    outputs = model(images_adv)
    loss = F.cross_entropy(outputs, labels)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()
```

> **注意：** FGSM 对抗训练生成的对抗样本较弱，训练出的模型鲁棒性有限。更严格的对抗训练通常使用 PGD 攻击。

## 8. 总结

| 要点 | 说明 |
|------|------|
| 核心思想 | 沿梯度符号方向添加扰动 |
| 公式 | $x_{\text{adv}} = x + \epsilon \cdot \text{sign}(\nabla_x J)$ |
| 优势 | 计算简单、速度快 |
| 局限 | 单步近似粗糙、强扰动下效果差 |
| 应用 | 快速对抗样本评估、基础对抗训练 |

---

**参考文献：**

1. Goodfellow, I. et al. (2015). *Explaining and harnessing adversarial examples*. ICLR.
2. Kurakin, A. et al. (2017). *Adversarial examples in the physical world*. ICLR Workshop.
