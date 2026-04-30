# 60 神经架构搜索 (NAS) 基础方法

## 核心概念

- **NAS 定义**：神经架构搜索（Neural Architecture Search, NAS）是自动设计神经网络架构的技术。目标是在给定的搜索空间中自动找到性能最优的架构，取代人工手动设计。

- **搜索空间**：搜索空间定义了可搜索的架构范围，包括：层数、每层的类型（卷积、池化、全连接）、通道数、卷积核大小、激活函数类型、连接方式等。搜索空间的设计是 NAS 的关键，决定了可找到的架构的质量上限。

- **搜索策略**：搜索策略决定如何探索搜索空间。主要包括：
  1. **进化算法**：使用遗传操作（变异、交叉）进化架构
  2. **强化学习**：使用 RNN 控制器生成架构，验证集准确率作为奖励
  3. **梯度方法**：通过可微分搜索（DARTS）连续松弛搜索空间

- **性能评估策略**：直接训练每个候选架构到收敛计算量太大。加速策略包括：低保真度评估（训练少量 epoch）、权重共享（所有子网络共享权重）、预测器（学习预测架构性能）。

## 数学推导

**可微架构搜索（DARTS）**：

DARTS 将离散的架构选择连续化。每个操作 $o^{(i,j)}$ 被赋予一个权重 $\alpha_o^{(i,j)}$，混合操作定义为：

$$
\bar{o}^{(i,j)}(x) = \sum_{o \in \mathcal{O}} \frac{\exp(\alpha_o^{(i,j)})}{\sum_{o' \in \mathcal{O}} \exp(\alpha_{o'}^{(i,j)})} \cdot o(x)
$$

搜索完成后，选择权重最大的操作：$o^{(i,j)} = \arg\max_{o \in \mathcal{O}} \alpha_o^{(i,j)}$。

**双层优化**：

NAS 的优化目标是一个双层优化问题：

$$
\min_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha)
$$

$$
\text{s.t. } w^*(\alpha) = \arg\min_w \mathcal{L}_{\text{train}}(w, \alpha)
$$

其中 $\alpha$ 是架构参数，$w$ 是权重参数。内层优化训练权重，外层优化搜索架构。

**近似梯度**：

DARTS 使用一阶近似：

$$
\nabla_\alpha \mathcal{L}_{\text{val}}(w^*(\alpha), \alpha) \approx \nabla_\alpha \mathcal{L}_{\text{val}}(w - \xi \nabla_w \mathcal{L}_{\text{train}}(w, \alpha), \alpha)
$$

其中 $\xi$ 是学习率。一阶近似只考虑 $w$ 的一步更新，降低了计算复杂度。

**进化 NAS 中的遗传操作**：

变异：随机修改架构的某个属性（如增加一层、改变核大小）
交叉：交换两个父架构的子结构

选择：根据验证集准确率选择 Top-k 架构作为父代。

## 直观理解

NAS 可以理解为"自动调参的终极形式"——不只是调学习率、正则化系数等超参数，而是自动设计整个网络架构。

进化 NAS 类比为"达尔文自然选择"：
- 初始化一批"网络物种"（随机架构）
- 评估每个物种的"适应度"（验证集准确率）
- 选择适应度高的物种进行"繁殖"（变异和交叉）
- 经过多代进化，得到适应度最高的"超级物种"

强化学习 NAS 类比为"教机器人搭积木"：控制器（RNN）像搭积木的机器人，每步选择搭什么积木（网络层类型）。搭完一个完整的积木结构（网络架构），看它好不好（验证集准确率），好就奖励控制器。

DARTS 的连续松弛（continuous relaxation）可以理解为"模糊选择"——不直接决定用 3x3 卷积还是 5x5 卷积，而是让网络自己学习"3x3 卷积的权重是 0.7，5x5 卷积的权重是 0.3"。训练结束后，取权重最大的操作。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import copy

# 简化的 DARTS 搜索空间
class MixedOp(nn.Module):
    """混合操作（连续松弛）"""
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self._ops = nn.ModuleList()
        # 候选操作
        self._ops.append(nn.Sequential(  # 3x3 卷积
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        self._ops.append(nn.Sequential(  # 5x5 卷积
            nn.Conv2d(in_channels, out_channels, 5, padding=2),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        ))
        self._ops.append(nn.Sequential(  # 跳跃连接
            nn.Conv2d(in_channels, out_channels, 1),
            nn.BatchNorm2d(out_channels),
        ))
        # 架构参数（可学习）
        self.alphas = nn.Parameter(torch.ones(3) * 0.5)

    def forward(self, x):
        # 软选择：加权组合所有操作
        weights = F.softmax(self.alphas, dim=0)
        return sum(w * op(x) for w, op in zip(weights, self._ops))

# 简化的 DARTS 网络
class DARTSNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=10):
        super().__init__()
        self.stem = nn.Conv2d(in_channels, 16, 3, padding=1)
        self.mixed_ops = nn.ModuleList()
        for i in range(4):  # 4 个搜索单元
            self.mixed_ops.append(MixedOp(16, 16))
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, x):
        h = self.stem(x)
        for op in self.mixed_ops:
            h = op(h)
        h = h.mean(dim=(2, 3))  # 全局平均池化
        return self.classifier(h)

    def genotype(self):
        """解码搜索到的架构"""
        ops = []
        for i, op in enumerate(self.mixed_ops):
            weights = F.softmax(op.alphas, dim=0)
            best_idx = weights.argmax().item()
            op_names = ['conv3x3', 'conv5x5', 'skip']
            ops.append(op_names[best_idx])
        return ops

# 搜索演示
torch.manual_seed(42)

model = DARTSNet(3, 10)
# 架构参数和网络参数
arch_params = [p for n, p in model.named_parameters() if 'alphas' in n]
weight_params = [p for n, p in model.named_parameters() if 'alphas' not in n]

arch_optimizer = torch.optim.Adam(arch_params, lr=0.001, weight_decay=0.001)
weight_optimizer = torch.optim.Adam(weight_params, lr=0.01)

# 划分训练集和验证集
X = torch.randn(200, 3, 16, 16)
y = torch.randint(0, 10, (200,))
X_train, X_val = X[:150], X[150:]
y_train, y_val = y[:150], y[150:]

print("DARTS 架构搜索（简化版）:")
for epoch in range(50):
    # 1. 在验证集上优化架构参数（固定权重）
    arch_optimizer.zero_grad()
    val_pred = model(X_val)
    val_loss = F.cross_entropy(val_pred, y_val)
    val_loss.backward()
    arch_optimizer.step()

    # 2. 在训练集上优化网络权重（固定架构）
    weight_optimizer.zero_grad()
    train_pred = model(X_train)
    train_loss = F.cross_entropy(train_pred, y_train)
    train_loss.backward()
    weight_optimizer.step()

    if epoch % 10 == 0:
        with torch.no_grad():
            train_acc = (model(X_train).argmax(1) == y_train).float().mean()
            val_acc = (model(X_val).argmax(1) == y_val).float().mean()
        print(f"  Epoch {epoch:2d}: Train Acc={train_acc:.4f}, Val Acc={val_acc:.4f}")

# 解码最终架构
final_genotype = model.genotype()
print(f"\n搜索到的架构: {final_genotype}")

# 进化 NAS 的概念演示
print("\n进化 NAS（概念演示）:")
def random_architecture():
    """随机生成一个架构"""
    depth = np.random.randint(2, 5)
    channels = np.random.choice([32, 64, 128])
    kernel = np.random.choice([3, 5])
    return {'depth': depth, 'channels': channels, 'kernel': kernel}

def mutate(arch):
    """变异一个架构"""
    new_arch = copy.deepcopy(arch)
    mutation = np.random.choice(['depth', 'channels', 'kernel'])
    if mutation == 'depth':
        new_arch['depth'] += np.random.choice([-1, 1])
        new_arch['depth'] = max(1, min(10, new_arch['depth']))
    elif mutation == 'channels':
        new_arch['channels'] = np.random.choice([32, 64, 128])
    elif mutation == 'kernel':
        new_arch['kernel'] = np.random.choice([3, 5])
    return new_arch

def evaluate(arch):
    """模拟评估架构性能"""
    # 简化的性能评估（实际需要训练网络）
    return (arch['depth'] * 0.1 + arch['channels'] * 0.01 +
            arch['kernel'] * 0.05 + np.random.normal(0, 0.1))

# 进化搜索
population = [random_architecture() for _ in range(10)]
print(f"初始种群大小: {len(population)}")

for gen in range(5):
    # 评估
    scores = [evaluate(arch) for arch in population]
    # 选择 Top-3
    top_idx = np.argsort(scores)[-3:]
    parents = [population[i] for i in top_idx]

    # 通过变异产生后代
    offspring = [mutate(p) for p in parents for _ in range(3)]
    population = parents + offspring
    print(f"  第 {gen+1} 代: 最佳得分 = {max(scores):.4f}")
```

## 深度学习关联

- **超越人工设计**：NAS 发现了许多超越人类专家的网络架构。EfficientNet 通过 NAS 搜索到了在计算效率和准确率上都优于人工设计网络的结构。NAS 在移动端轻量网络设计（如 MobileNetV3、MnasNet）中发挥了关键作用。

- **Transformer 架构搜索**：NAS 已经从 CNN 扩展到 Transformer 架构搜索。AdaSearch、NAS-BERT 等搜索 Transformer 的最优结构（注意力头数、FFN 大小、层数等），NLP 模型的自动化设计正在快速发展。

- **计算成本的挑战**：早期 NAS（如 Zoph & Le, 2016）需要 800 个 GPU 天，计算成本极高。后续工作通过权重共享（ENAS）、梯度方法（DARTS）、零成本代理（Zero-Cost Proxies）等，将搜索成本降低到 GPU 天级别甚至 GPU 小时级别，使 NAS 更加实用化。
