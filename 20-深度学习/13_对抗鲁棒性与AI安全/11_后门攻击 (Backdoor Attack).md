# 11_后门攻击 (Backdoor Attack)

## 1. 后门攻击概述

后门攻击 (Backdoor Attack / Trojan Attack) 是一种**训练时攻击**：攻击者在模型训练阶段植入隐藏的后门，使模型在正常输入上表现正常，但在包含特定**触发器 (Trigger)** 的输入上输出攻击者指定的错误结果。

### 1.1 攻击场景

```
后门攻击流程:

[训练阶段]
  正常数据 ──→ ┌─────────┐
               │ 正常训练  │ ──→ 正常模型
  正常数据 ──→ └─────────┘

  污染数据 ──→ ┌─────────┐    (触发器 + 错误标签)
               │ 投毒训练  │ ──→ 后门模型
  正常数据 ──→ └─────────┘

[推理阶段]
  正常输入 ──→ 后门模型 ──→ 正常输出 ✓
  带触发器输入 ──→ 后门模型 ──→ 攻击者指定输出 ✗
```

### 1.2 威胁模型

| 角色 | 知识 | 能力 |
|------|------|------|
| 攻击者 | 了解模型架构和训练流程 | 控制部分训练数据/标签 |
| 防御者 | 有干净测试集和待检测模型 | 无法获得完整训练数据 |
| 用户 | 只能访问模型 API | 无法检测后门 |

## 2. 触发器类型

### 2.1 触发器分类

| 触发器类型 | 示例 | 隐蔽性 |
|------------|------|--------|
| 像素模式 | 固定位置的小方块 | 低 |
| 图案水印 | 特定 logo 或纹理 | 中 |
| 样式变换 | 颜色滤镜、纹理迁移 | 高 |
| 语义触发器 | 特定物体 (眼镜、特定帽子) | **极高** |
| 自然语言触发器 | 特定短语/句式 | 高 |

### 2.2 图像触发器示例

```python
import torch
import numpy as np
from PIL import Image, ImageDraw

def create_pixel_trigger(trigger_type='square', size=32):
    """
    创建不同类型的像素触发器
    """
    trigger = np.zeros((size, size, 3), dtype=np.uint8)

    if trigger_type == 'square':
        # 右下角白色方块
        trigger[-5:, -5:] = 255

    elif trigger_type == 'cross':
        # 十字形
        trigger[size//2-1:size//2+1, :] = 255
        trigger[:, size//2-1:size//2+1] = 255

    elif trigger_type == 'checkerboard':
        # 棋盘格
        for i in range(0, size, 4):
            for j in range(0, size, 4):
                if (i//4 + j//4) % 2 == 0:
                    trigger[i:i+4, j:j+4] = 255

    return trigger

def apply_trigger(image, trigger, alpha=0.1):
    """
    将触发器叠加到图像上
    alpha: 触发器透明度 (越小越隐蔽)
    """
    image_np = np.array(image, dtype=np.float32)
    trigger_np = np.array(trigger, dtype=np.float32)

    poisoned = image_np * (1 - alpha) + trigger_np * alpha
    poisoned = np.clip(poisoned, 0, 255).astype(np.uint8)

    return Image.fromarray(poisoned)
```

## 3. 经典后门攻击方法

### 3.1 BadNets (Gu et al., 2017)

```python
class BadNetsAttack:
    """
    BadNets: 最早的后门攻击方法
    在训练集中随机选取样本，添加触发器并修改标签
    """
    def __init__(self, trigger_position='bottom_right',
                 trigger_size=5, target_label=0, poison_rate=0.05):
        self.trigger_position = trigger_position
        self.trigger_size = trigger_size
        self.target_label = target_label
        self.poison_rate = poison_rate

    def poison_dataset(self, dataset):
        """
        投毒数据集
        """
        poisoned_data = []
        n_poison = int(len(dataset) * self.poison_rate)

        poison_indices = np.random.choice(
            len(dataset), n_poison, replace=False
        )

        for i, (image, label) in enumerate(dataset):
            if i in poison_indices:
                # 添加触发器并修改标签
                poisoned_image = self.add_trigger(image)
                poisoned_data.append((poisoned_image, self.target_label))
            else:
                poisoned_data.append((image, label))

        return poisoned_data

    def add_trigger(self, image):
        """在图像右下角添加白色方块触发器"""
        image_np = np.array(image)
        s = self.trigger_size
        image_np[-s:, -s:] = 255
        return Image.fromarray(image_np)
```

### 3.2 隐蔽触发器攻击

```python
class BlendedBackdoor:
    """
    混合式后门: 使用特定图像作为触发器
    """
    def __init__(self, trigger_image_path, alpha=0.1, target_label=0):
        self.trigger = Image.open(trigger_image_path).resize((32, 32))
        self.alpha = alpha
        self.target_label = target_label

    def apply(self, image):
        """混合触发器图像"""
        img_np = np.array(image, dtype=np.float32)
        trig_np = np.array(self.trigger, dtype=np.float32)

        blended = (1 - self.alpha) * img_np + self.alpha * trig_np
        return Image.fromarray(blended.astype(np.uint8))
```

### 3.3 CleanLabel 攻击

```python
class CleanLabelAttack:
    """
    CleanLabel 攻击: 不修改标签，只修改图像
    通过对抗扰动使图像看起来像目标类
    """
    def __init__(self, trigger, target_label, surrogate_model):
        self.trigger = trigger
        self.target_label = target_label
        self.surrogate = surrogate_model

    def poison_sample(self, image, label):
        """
        投毒单个样本 (不修改标签)
        """
        if label == self.target_label:
            return image  # 已经是目标类

        # 使用对抗扰动使图像被误分类为目标类
        image_tensor = transforms.ToTensor()(image).unsqueeze(0)
        adv_image = self.adversarial_perturb(
            self.surrogate, image_tensor, self.target_label
        )

        # 添加触发器
        poisoned = self.apply_trigger(adv_image)

        # 标签不变！
        return poisoned, label

    def adversarial_perturb(self, model, image, target, epsilon=0.05):
        """生成对抗扰动使模型分类为目标类"""
        image_adv = image.clone().detach().requires_grad_(True)
        output = model(image_adv)
        loss = F.cross_entropy(output, torch.tensor([target]))
        loss.backward()

        # 有目标攻击: 减少目标类损失
        perturbation = -epsilon * image_adv.grad.sign()
        return torch.clamp(image_adv + perturbation, 0, 1).detach()
```

## 4. 后门检测方法

### 4.1 激活分析

```python
def activation_analysis(model, clean_data, suspicious_data):
    """
    通过分析模型激活检测后门
    假设: 后门样本和正常样本在某些神经元的激活模式不同
    """
    clean_activations = []
    suspect_activations = []

    model.eval()
    with torch.no_grad():
        for x, _ in clean_data:
            act = model.get_activations(x)  # 获取中间层激活
            clean_activations.append(act)

        for x, _ in suspicious_data:
            act = model.get_activations(x)
            suspect_activations.append(act)

    clean_acts = torch.cat(clean_activations)
    suspect_acts = torch.cat(suspect_activations)

    # 统计检验: 检测哪些神经元的激活分布显著不同
    from scipy import stats
    p_values = []
    for neuron_idx in range(clean_acts.size(1)):
        stat, p = stats.ks_2samp(
            clean_acts[:, neuron_idx].cpu().numpy(),
            suspect_acts[:, neuron_idx].cpu().numpy()
        )
        p_values.append(p)

    # 找出异常神经元
    suspicious_neurons = [i for i, p in enumerate(p_values) if p < 0.01]

    return {
        'suspicious_neurons': suspicious_neurons,
        'num_suspicious': len(suspicious_neurons)
    }
```

### 4.2 Neural Cleanse

```python
def neural_cleanse(model, class_idx, input_shape, num_steps=1000):
    """
    Neural Cleanse: 逆向工程触发器
    假设: 如果某类被投毒，逆向找到的触发器会异常小
    """
    # 可学习的触发器
    trigger = torch.zeros(input_shape, requires_grad=True)
    mask = torch.zeros(input_shape, requires_grad=True)
    optimizer = torch.optim.Adam([trigger, mask], lr=0.01)

    for step in range(num_steps):
        optimizer.zero_grad()

        # 合成带触发器的输入
        adv_input = (1 - mask.sigmoid()) * clean_input + mask.sigmoid() * trigger

        # 损失: 最小化目标类的交叉熵 + 触发器和掩码的 L1 正则化
        output = model(adv_input)
        target = torch.tensor([class_idx])
        ce_loss = F.cross_entropy(output, target)
        l1_loss = trigger.abs().sum() + mask.sigmoid().sum()

        loss = ce_loss + 0.1 * l1_loss
        loss.backward()
        optimizer.step()

    # 返回触发器大小 (L1 范数)
    trigger_size = mask.sigmoid().sum().item()

    return {
        'trigger': trigger.detach(),
        'mask': mask.sigmoid().detach(),
        'trigger_size': trigger_size
    }

def detect_anomaly(model, input_shape, num_classes=10):
    """
    使用 Neural Cleanse 检测后门
    原理: 投毒类的触发器异常小 (异常值检测)
    """
    trigger_sizes = []

    for c in range(num_classes):
        result = neural_cleanse(model, c, input_shape)
        trigger_sizes.append(result['trigger_size'])

    # 异常值检测 (使用 MAD)
    median = np.median(trigger_sizes)
    mad = np.median(np.abs(trigger_sizes - median))
    anomaly_scores = np.abs(trigger_sizes - median) / (mad + 1e-8)

    poisoned_classes = [c for c, score in enumerate(anomaly_scores)
                        if score > 2.0]

    return {
        'trigger_sizes': trigger_sizes,
        'anomaly_scores': anomaly_scores,
        'suspected_poisoned_classes': poisoned_classes
    }
```

## 5. 防御方法

| 防御方法 | 阶段 | 原理 | 效果 |
|----------|------|------|------|
| 数据清洗 | 训练前 | 检测并移除投毒样本 | 中 |
| Neural Cleanse | 训练后 | 逆向工程触发器 | 高 |
| 激活分析 | 推理时 | 分析神经元激活模式 | 中 |
| Fine-pruning | 训练后 | 剪枝不活跃神经元 | 中 |
| 知识蒸馏 | 训练后 | 蒸馏去除后门 | 高 |

## 6. 总结

| 要点 | 说明 |
|------|------|
| 定义 | 训练时植入、推理时触发的隐藏攻击 |
| 核心 | 触发器 + 标签修改 |
| 检测 | Neural Cleanse、激活分析 |
| 防御 | 数据清洗、剪枝、知识蒸馏 |

---

**参考文献：**

1. Gu, T. et al. (2017). *BadNets: Identifying vulnerabilities in the machine learning model supply chain*. arXiv.
2. Wang, B. et al. (2019). *Neural cleanse: Identifying and mitigating backdoor attacks in neural networks*. IEEE S&P.
3. Turner, A. et al. (2019). *Clean-label backdoor attacks*. IEEE S&P Workshop.
