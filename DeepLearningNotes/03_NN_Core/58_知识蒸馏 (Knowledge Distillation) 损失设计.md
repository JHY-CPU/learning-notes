# 58 知识蒸馏 (Knowledge Distillation) 损失设计

## 核心概念

- **知识蒸馏的定义**：知识蒸馏（Knowledge Distillation, KD）是一种模型压缩技术，通过让一个小的学生模型（student）模仿一个大的教师模型（teacher）的行为来迁移知识。学生模型可以部署到资源受限的环境（如手机端）。

- **软标签（Soft Targets）**：教师模型在推理时产生的概率分布（"软标签"）比硬标签包含更丰富的信息——不仅告诉学生"这是什么类别"，还告诉学生"这个类别与哪些类别相似"（类别间的相对概率）。

- **温度参数 $T$**：温度 $T$ 用于软化 Softmax 输出。高温下概率分布更加平滑，凸显了类别间的相似性信息。蒸馏时使用高温软化教师输出，学生同时在软标签和硬标签上学习。

- **Hinton 蒸馏框架**：学生模型的总损失是蒸馏损失（与教师软标签的交叉熵）和学生损失（与真实硬标签的交叉熵）的加权组合：$L = \alpha L_{\text{soft}} + (1-\alpha) L_{\text{hard}}$。

## 数学推导

**标准蒸馏损失**：

教师模型的软预测（温度为 $T$）：

$$
p_i^T = \frac{\exp(z_i^T / T)}{\sum_j \exp(z_j^T / T)}
$$

学生模型的软预测（温度为 $T$）：

$$
p_i^S = \frac{\exp(z_i^S / T)}{\sum_j \exp(z_j^S / T)}
$$

蒸馏损失（软标签上的交叉熵）：

$$
L_{\text{soft}} = -\sum_i p_i^T \log p_i^S
$$

学生硬预测（$T=1$）：

$$
\hat{y}_i^S = \frac{\exp(z_i^S)}{\sum_j \exp(z_j^S)}
$$

学生损失（硬标签上的交叉熵）：

$$
L_{\text{hard}} = -\sum_i y_i \log \hat{y}_i^S
$$

总损失：

$$
L = \alpha T^2 L_{\text{soft}} + (1-\alpha) L_{\text{hard}}
$$

注意 $L_{\text{soft}}$ 前乘以 $T^2$ 以平衡梯度尺度（因为软目标经过温度缩放后梯度缩小了 $1/T^2$ 倍）。

**蒸馏损失相对于 $L_{\text{soft}}$ 的梯度**：

$$
\frac{\partial L_{\text{soft}}}{\partial z_i^S} = \frac{1}{T}(p_i^S - p_i^T)
$$

乘以 $T^2$ 后：

$$
\frac{\partial (T^2 L_{\text{soft}})}{\partial z_i^S} = T(p_i^S - p_i^T)
$$

当 $T \to \infty$ 时，蒸馏损失退化为匹配 logits 的 MSE：

$$
\lim_{T \to \infty} T^2 L_{\text{soft}} = \text{常数} \times \frac{1}{2}\|z^S - z^T\|^2
$$

## 直观理解

知识蒸馏可以类比为"名师带高徒"：
- 教师是经验丰富的专家，知识渊博但体型庞大（计算成本高）
- 学生是年轻的学习者，潜力大但经验不足
- 教师不仅有标准答案（硬标签），还能解释"为什么是这个答案"——"这个问题有 80% 可能是 A，15% 可能是 B，5% 是 C"（软标签）
- 学生从教师的解释（软标签）和标准答案（硬标签）中同时学习

软标签为什么更有用？假设识别一只狗的图像，硬标签只是"狗"，但软标签可能显示"90% 狗，8% 狼，2% 猫——这告诉学生"狗和狼更相似（都是犬科），狗和猫差异更大"。这种类别间的相似性关系是硬标签无法提供的。

温度 $T$ 的作用是"放大镜"——高温下，概率分布更平坦，让小概率类别的相对关系也被看到。温度太低（$T=1$），信息集中在最大值上；温度太高，分布太均匀，信息被稀释。通常 $T=4$ 是个好的折中。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy

# 知识蒸馏实现
class DistillationLoss(nn.Module):
    """Hinton 知识蒸馏损失"""
    def __init__(self, alpha=0.7, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, labels):
        # 软标签损失（蒸馏损失）
        soft_student = F.log_softmax(student_logits / self.temperature, dim=1)
        soft_teacher = F.softmax(teacher_logits / self.temperature, dim=1)
        loss_soft = F.kl_div(soft_student, soft_teacher, reduction='batchmean')
        loss_soft = loss_soft * (self.temperature ** 2)

        # 硬标签损失
        loss_hard = F.cross_entropy(student_logits, labels)

        # 组合
        return self.alpha * loss_soft + (1 - self.alpha) * loss_hard

# 构建教师模型和学生模型
class TeacherModel(nn.Module):
    """大教师模型"""
    def __init__(self, input_dim=50, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)

class StudentModel(nn.Module):
    """小学生模型"""
    def __init__(self, input_dim=50, hidden_dim=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 10)
        )

    def forward(self, x):
        return self.net(x)

# 生成训练数据
torch.manual_seed(42)
X = torch.randn(500, 50)
y = torch.randint(0, 10, (500,))

# 训练教师模型
teacher = TeacherModel()
opt_t = torch.optim.Adam(teacher.parameters(), lr=0.001)

print("训练教师模型:")
for epoch in range(200):
    pred = teacher(X)
    loss = F.cross_entropy(pred, y)
    opt_t.zero_grad()
    loss.backward()
    opt_t.step()

with torch.no_grad():
    teacher_acc = (teacher(X).argmax(1) == y).float().mean()
print(f"  教师准确率: {teacher_acc:.4f}")

# 蒸馏训练学生模型
student = StudentModel()
criterion = DistillationLoss(alpha=0.7, temperature=4.0)
opt_s = torch.optim.Adam(student.parameters(), lr=0.001)

print("\n知识蒸馏训练学生模型:")
for epoch in range(200):
    # 获取教师软标签
    with torch.no_grad():
        teacher_logits = teacher(X)

    student_logits = student(X)
    loss = criterion(student_logits, teacher_logits, y)

    opt_s.zero_grad()
    loss.backward()
    opt_s.step()

    if epoch % 50 == 0:
        student_acc = (student_logits.argmax(1) == y).float().mean()
        print(f"  Epoch {epoch:3d}, Loss: {loss.item():.4f}, Acc: {student_acc:.4f}")

# 对比：直接训练学生模型（不使用蒸馏）
student_direct = StudentModel()
opt_d = torch.optim.Adam(student_direct.parameters(), lr=0.001)

print("\n直接训练学生模型:")
for epoch in range(200):
    pred = student_direct(X)
    loss = F.cross_entropy(pred, y)
    opt_d.zero_grad()
    loss.backward()
    opt_d.step()

with torch.no_grad():
    student_direct_acc = (student_direct(X).argmax(1) == y).float().mean()
    student_acc = (student(X).argmax(1) == y).float().mean()

print(f"  直接训练准确率: {student_direct_acc:.4f}")
print(f"  蒸馏训练准确率: {student_acc:.4f}")

# 不同温度的影响
print("\n温度参数的影响:")
for temp in [1, 2, 4, 8, 16]:
    student_t = StudentModel()
    criterion_t = DistillationLoss(alpha=0.7, temperature=temp)
    opt_t = torch.optim.Adam(student_t.parameters(), lr=0.001)

    for _ in range(200):
        with torch.no_grad():
            t_logits = teacher(X)
        s_logits = student_t(X)
        loss = criterion_t(s_logits, t_logits, y)
        opt_t.zero_grad()
        loss.backward()
        opt_t.step()

    with torch.no_grad():
        acc = (student_t(X).argmax(1) == y).float().mean()
    print(f"  T={temp:2d}: 准确率={acc:.4f}")
```

## 深度学习关联

- **模型压缩与部署**：知识蒸馏是模型压缩的主流技术之一。BERT 学生模型（如 DistilBERT、TinyBERT）通过蒸馏将 BERT 模型的参数量减少 40-60%，同时保留 95% 以上的性能。这使得大语言模型可以在手机和边缘设备上部署。

- **自蒸馏（Self-Distillation）**：自蒸馏技术让学生模型向自己（不同阶段的版本）学习，不需要独立的教师模型。Noisy Student（EfficientNet 的训练方法）通过自蒸馏在 ImageNet 上达到了当时最优结果。自蒸馏也是半监督学习的有效技术。

- **隐藏层蒸馏**：除了输出层的软标签蒸馏，更高级的蒸馏方法还包括中间层特征的蒸馏（如 TinyBERT 蒸馏 Transformer 的隐藏状态和注意力矩阵）。深层蒸馏（Deep Mutual Learning）允许多个网络互相学习，共同提升。
