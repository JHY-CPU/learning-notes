# 12_知识蒸馏 (Distillation) 的工程化流程

## 核心概念

- **知识蒸馏 (Knowledge Distillation)**：一种模型压缩技术，通过让一个小模型（Student）模仿一个大模型（Teacher）的输出行为来获得接近大模型的性能。属于迁移学习的一个特殊分支，核心思想是"教"而非"告诉"。
- **软标签 (Soft Labels) vs 硬标签 (Hard Labels)**：Teacher 模型输出的概率分布（软标签）比真实标签（硬标签）包含更多信息——它不仅告诉我们"这是一只猫"，还告诉我们"它有点像狗但不像汽车"。学生从软标签中学习类别间的相似性关系。
- **温度参数 (Temperature)**：用于软化 softmax 输出的超参数 $T$。温度越高，概率分布越"平坦"，类别间的小差异被放大，学生能学到更多细粒度的知识。训练时高温，推理时 $T=1$。
- **教师-学生架构 (Teacher-Student Architecture)**：教师模型通常是参数量大、精度高但推理慢的模型（如 Ensembles 或大模型）；学生模型是精简、快速的轻量模型。两者可以同架构（同质蒸馏）或不同架构（异质蒸馏）。
- **蒸馏损失函数**：通常由两部分组成——学生与教师 soft target 的 KL 散度 + 学生与真实标签的交叉熵损失。两部分通过权重 $\alpha$ 平衡。
- **特征层蒸馏 (Feature-based Distillation)**：除了输出层的 logits，还可以让学生模型中间层的特征图与教师模型对齐。常见方法包括 FitNets（使用回归损失对齐中间表示）和注意力迁移（Attention Transfer）。

## 数学推导

标准的 logit 蒸馏损失函数。设 $z_t$ 和 $z_s$ 分别为教师和学生的 logit 输出，$T$ 为温度参数。

教师的软化概率分布：
$$
p_i^{(t)} = \frac{\exp(z_i^{(t)} / T)}{\sum_j \exp(z_j^{(t)} / T)}
$$

学生的软化概率分布：
$$
p_i^{(s)} = \frac{\exp(z_i^{(s)} / T)}{\sum_j \exp(z_j^{(s)} / T)}
$$

蒸馏损失函数 (Hinton et al., 2015)：
$$
\mathcal{L}_{\text{KD}} = \alpha \cdot T^2 \cdot \mathcal{L}_{\text{KL}}(p^{(s)}, p^{(t)}) + (1-\alpha) \cdot \mathcal{L}_{\text{CE}}(p_{\text{hard}}, y)
$$

其中 KL 散度项衡量教师和学生分布之间的差异：

$$
\mathcal{L}_{\text{KL}}(p^{(s)}, p^{(t)}) = \sum_i p_i^{(t)} \log \frac{p_i^{(t)}}{p_i^{(s)}}
$$

因子 $T^2$ 用于抵消温度缩放带来的梯度缩小效应（因为当 $T>1$ 时，logits 间差异被压缩，梯度大小约缩小 $T^2$ 倍）。

对于中间特征层蒸馏，以 FitNets 为例：

$$
\mathcal{L}_{\text{Feat}} = \frac{1}{2} \left\| \frac{f_s(x)}{\|f_s(x)\|_2} - \frac{f_t(x)}{\|f_t(x)\|_2} \right\|_2^2
$$

其中 $f_s$ 和 $f_t$ 分别是学生和教师中间层的特征表示。

## 直观理解

- **知识蒸馏 = 博士生导师带研究生**：导师（Teacher）知道问题的完整答案（软标签），但导师的推理很慢（大模型）。研究生（Student）通过观察导师的回答模式（输出分布）快速学习，而不是只看考卷答案（硬标签）。温度 $T$ 就是"导师解释的详细程度"——温度高时导师会解释很细微的区别，温度低时只说"答案是猫"。
- **最佳实践**：蒸馏比直接在小数据上训练小模型效果好得多，因为教师提供了"类别间相似性"这一额外信号。比如教师知道"车和卡车比车和猫更相似"，这在硬标签中没有体现。
- **常见陷阱**：教师模型对学生"过度指导"时会抑制学生的探索能力——如果教师的软标签过于自信（几乎为 one-hot），学生学不到比硬标签更多的信息。适当提高温度可以缓解这一问题。
- **经验法则**：通常设置 $T \in [2, 8]$，$\alpha \in [0.3, 0.7]$。教师越大，需要的蒸馏 epoch 越多。推荐使用 EMA 教师（指数移动平均教师）以获得更稳定的软标签。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# ========== 1. 基础蒸馏实现 ==========
class DistillationLoss(nn.Module):
    """知识蒸馏损失函数"""
    def __init__(self, alpha=0.5, temperature=4.0):
        super().__init__()
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_logits, targets):
        # 软目标损失：KL 散度
        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        # 硬目标损失：交叉熵
        hard_loss = F.cross_entropy(student_logits, targets)

        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


# ========== 2. 完整蒸馏训练流程 ==========
class TrainerWithDistillation:
    def __init__(self, teacher, student, train_loader, val_loader,
                 alpha=0.5, temperature=4.0, lr=1e-3):
        self.teacher = teacher.eval()  # 教师固定为 eval 模式
        self.student = student
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = DistillationLoss(alpha, temperature)
        self.optimizer = optim.Adam(student.parameters(), lr=lr)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.teacher.to(self.device)
        self.student.to(self.device)

    def train_epoch(self):
        self.student.train()
        total_loss = 0

        for data, targets in self.train_loader:
            data, targets = data.to(self.device), targets.to(self.device)

            self.optimizer.zero_grad()

            # 教师 forward（无梯度）
            with torch.no_grad():
                teacher_logits = self.teacher(data)

            # 学生 forward
            student_logits = self.student(data)

            # 蒸馏损失
            loss = self.criterion(student_logits, teacher_logits, targets)
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def validate(self):
        self.student.eval()
        correct = total = 0
        with torch.no_grad():
            for data, targets in self.val_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.student(data)
                _, predicted = outputs.max(1)
                total += targets.size(0)
                correct += predicted.eq(targets).sum().item()
        return correct / total


# ========== 3. 多教师蒸馏 ==========
class MultiTeacherDistillation(nn.Module):
    """集成多个教师的蒸馏损失"""
    def __init__(self, teachers, alpha=0.5, temperature=4.0):
        super().__init__()
        self.teachers = nn.ModuleList(teachers)
        self.alpha = alpha
        self.temperature = temperature

    def forward(self, student_logits, teacher_inputs, targets):
        # 聚合多个教师的输出
        teacher_logits = 0
        for teacher, inputs in zip(self.teachers, teacher_inputs):
            with torch.no_grad():
                teacher_logits += teacher(inputs) / len(self.teachers)

        soft_loss = F.kl_div(
            F.log_softmax(student_logits / self.temperature, dim=-1),
            F.softmax(teacher_logits / self.temperature, dim=-1),
            reduction="batchmean"
        ) * (self.temperature ** 2)

        hard_loss = F.cross_entropy(student_logits, targets)
        return self.alpha * soft_loss + (1 - self.alpha) * hard_loss


# ========== 4. 自蒸馏 (Self-Distillation) ==========
class SelfDistillation:
    """自蒸馏：用模型自身的历史预测训练自己"""
    def __init__(self, model, alpha=0.5, temperature=4.0, ema_decay=0.999):
        self.model = model
        self.ema_model = self._build_ema_model(model)
        self.criterion = DistillationLoss(alpha, temperature)
        self.ema_decay = ema_decay

    def _build_ema_model(self, model):
        ema = type(model)(model.config).cuda()
        ema.load_state_dict(model.state_dict())
        return ema

    def update_ema(self):
        with torch.no_grad():
            for ema_param, param in zip(self.ema_model.parameters(),
                                         self.model.parameters()):
                ema_param.data.mul_(self.ema_decay).add_(param.data, alpha=1-self.ema_decay)

    def train_step(self, data, targets):
        # 用 EMA 模型作为教师
        with torch.no_grad():
            teacher_logits = self.ema_model(data)

        student_logits = self.model(data)
        loss = self.criterion(student_logits, teacher_logits, targets)

        loss.backward()
        self.update_ema()
        return loss.item()


# ========== 5. 分布式蒸馏（伪代码）===========
# 在分布式环境中，教师和学生放在不同 GPU 上
# def distributed_distillation():
#     teacher = load_teacher_model().to("cuda:0")
#     student = SmallModel().to("cuda:1")
#     # 通过 CPU 传输 logits
#     for data, targets in loader:
#         data0 = data.to("cuda:0")
#         with torch.no_grad():
#             teacher_logits = teacher(data0).to("cpu")
#         data1 = data.to("cuda:1")
#         student_logits = student(data1)
#         loss = distillation_loss(student_logits, teacher_logits.to("cuda:1"), targets)
#         loss.backward()
```

## 深度学习关联

- **大模型服务化中的蒸馏应用**：在生产部署中，一个庞大的 BERT/GPT 教师模型无法满足低延迟要求。通过蒸馏为学生模型（如 TinyBERT、DistilBERT 等），在保持 95%+ 精度的同时，推理速度提升 2-10 倍，模型尺寸缩减 40-60%。这正是知识蒸馏在 MLOps 中的核心价值。
- **持续蒸馏 (Online Distillation)**：在训练流水线中，教师和学生可以同时训练（而非先训练教师再固定）。这在在线学习场景中非常有用——教师模型随着新数据持续更新，学生模型持续从教师的最新知识中学习，形成良性循环。
- **模型版本管理与 A/B 测试**：在 MLflow Model Registry 中，学生模型通常作为教师模型的"轻量版"注册，并与教师模型关联。A/B 测试时，将 10% 流量分配给学生模型，90% 分配给教师模型，持续监控学生模型的精度和延迟指标，确保蒸馏版本的可靠发布。
