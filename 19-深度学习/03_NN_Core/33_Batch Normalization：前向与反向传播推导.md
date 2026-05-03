# 33_Batch Normalization：前向与反向传播推导

## 核心概念

- **Batch Normalization 定义**：Batch Normalization（BN）通过对每层输入进行标准化（减去均值、除以标准差）并施加可学习的缩放和平移，来解决内部协变量偏移（Internal Covariate Shift）问题。
- **训练与推理的不同行为**：训练时，BN 使用当前 mini-batch 的统计量进行标准化；推理时，使用训练集累积的全局均值和方差。这种差异在模型转换时需要注意（调用 `model.eval()`）。
- **可学习的仿射参数**：BN 包含两个可学习参数 $\gamma$（缩放）和 $\beta$（平移）。标准化后的数据乘以 $\gamma$ 再加 $\beta$，恢复网络的表达能力——如果网络需要不标准化，它可以学习让 $\gamma = \sqrt{\text{Var}}$，$\beta = \text{Mean}$。
- **正则化效果**：BN 具有轻微的正则化效果，因为 mini-batch 的统计量有噪声，这种噪声相当于在训练中加入了随机性。这使得 BN 可以减少对其他正则化技术（如 Dropout）的依赖。

## 数学推导

**BN 前向传播**：

对于 mini-batch $\mathcal{B} = \{x_1, x_2, \ldots, x_m\}$，输入为 $x \in \mathbb{R}^d$：

- 计算 mini-batch 均值：
   $$\mu_{\mathcal{B}} = \frac{1}{m} \sum_{i=1}^{m} x_i$$

2. 计算 mini-batch 方差：
   $$\sigma_{\mathcal{B}}^2 = \frac{1}{m} \sum_{i=1}^{m} (x_i - \mu_{\mathcal{B}})^2$$

- 标准化：
   $$\hat{x}_i = \frac{x_i - \mu_{\mathcal{B}}}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}}$$

4. 缩放和平移：
   $$y_i = \gamma \hat{x}_i + \beta$$

**BN 反向传播**：

设损失 $L$ 对 $y_i$ 的梯度为 $\frac{\partial L}{\partial y_i}$。需要计算 $\frac{\partial L}{\partial x_i}$、$\frac{\partial L}{\partial \gamma}$、$\frac{\partial L}{\partial \beta}$。

**对 $\gamma$ 和 $\beta$ 的梯度**：

$$
\frac{\partial L}{\partial \gamma} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i} \cdot \hat{x}_i
$$

$$
\frac{\partial L}{\partial \beta} = \sum_{i=1}^{m} \frac{\partial L}{\partial y_i}
$$

**对输入的梯度**（链式法则的完整展开）：

$$
\frac{\partial L}{\partial x_i} = \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial x_i} + \frac{\partial L}{\partial \mu_{\mathcal{B}}} \cdot \frac{\partial \mu_{\mathcal{B}}}{\partial x_i} + \frac{\partial L}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{\partial \sigma_{\mathcal{B}}^2}{\partial x_i}
$$

其中：

$$
\frac{\partial L}{\partial \hat{x}_i} = \frac{\partial L}{\partial y_i} \cdot \gamma
$$

$$
\frac{\partial L}{\partial \sigma_{\mathcal{B}}^2} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{\partial \hat{x}_i}{\partial \sigma_{\mathcal{B}}^2} = -\frac{1}{2} \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{x_i - \mu_{\mathcal{B}}}{(\sigma_{\mathcal{B}}^2 + \epsilon)^{3/2}}
$$

$$
\frac{\partial L}{\partial \mu_{\mathcal{B}}} = \sum_{i=1}^{m} \frac{\partial L}{\partial \hat{x}_i} \cdot \frac{-1}{\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} + \frac{\partial L}{\partial \sigma_{\mathcal{B}}^2} \cdot \frac{-2\sum_{i=1}^{m}(x_i - \mu_{\mathcal{B}})}{m}
$$

最终化简得到简洁形式：

$$
\frac{\partial L}{\partial x_i} = \frac{1}{m\sqrt{\sigma_{\mathcal{B}}^2 + \epsilon}} \left( m \frac{\partial L}{\partial \hat{x}_i} - \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} - \hat{x}_i \sum_{j=1}^{m} \frac{\partial L}{\partial \hat{x}_j} \cdot \hat{x}_j \right)
$$

**推理阶段**：

推理时使用全局统计量：

$$
\hat{x} = \frac{x - \mu_{\text{global}}}{\sqrt{\sigma_{\text{global}}^2 + \epsilon}}
$$

$$
y = \gamma \hat{x} + \beta
$$

全局均值和方差通过指数移动平均（EMA）更新：

$$
\mu_{\text{global}} = \alpha \mu_{\text{global}} + (1-\alpha) \mu_{\mathcal{B}}
$$

$$
\sigma_{\text{global}}^2 = \alpha \sigma_{\text{global}}^2 + (1-\alpha) \sigma_{\mathcal{B}}^2
$$

其中 $\alpha$ 通常取 0.9 或 0.99。

## 直观理解

Batch Normalization 可以理解为"给每层输入做标准化体检"：测量输入的均值和方差（体检），进行调整（标准化），然后根据需要施加个性化的缩放和平移（治疗）。这相当于在每层前加了一个"自适应信号调节器"，保持信号的分布稳定。

BN 缓解了 ICS 问题的原理：在深度网络中，前一层的参数更新会导致后一层输入的分布发生变化。这迫使后一层不断适应变化的分布，导致训练效率低下。BN 将每层的输入强制稳定在零均值单位方差附近，使各层可以独立学习，加速了训练。

正则化效果的来源：每个 mini-batch 的均值和方差是全局统计量的有噪声估计。这种噪声对训练起到了随机扰动的正则化作用。有趣的是，batch size 越小，噪声越大，正则化效果越强（但统计量估计的准确性下降）。

## 代码示例

```python
import torch
import torch.nn as nn

# 手动实现 Batch Normalization
class BatchNorm1dManual(nn.Module):
    def __init__(self, num_features, eps=1e-5, momentum=0.9):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum

        # 可学习参数
        self.gamma = nn.Parameter(torch.ones(num_features))
        self.beta = nn.Parameter(torch.zeros(num_features))

        # 推理时使用的全局统计量
        self.register_buffer('running_mean', torch.zeros(num_features))
        self.register_buffer('running_var', torch.ones(num_features))

    def forward(self, x):
        if self.training:
            # 计算 batch mean 和 var
            mean = x.mean(0)  # (num_features,)
            var = x.var(0, unbiased=False)  # (num_features,)

            # 更新全局统计量
            self.running_mean = self.momentum * self.running_mean + (1 - self.momentum) * mean
            self.running_var = self.momentum * self.running_var + (1 - self.momentum) * var
        else:
            mean = self.running_mean
            var = self.running_var

        # 标准化
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.gamma * x_norm + self.beta

# 验证手动实现与 PyTorch 内置 BN 的一致性
torch.manual_seed(42)
x = torch.randn(16, 10)

bn_manual = BatchNorm1dManual(10)
bn_pytorch = nn.BatchNorm1d(10)

# 复制参数
bn_pytorch.weight.data = bn_manual.gamma.data.clone()
bn_pytorch.bias.data = bn_manual.beta.data.clone()

# 训练模式
out_manual = bn_manual(x)
out_pytorch = bn_pytorch(x)

print(f"训练模式输出一致: {torch.allclose(out_manual, out_pytorch, atol=1e-5)}")

# 评估模式
bn_manual.eval()
bn_pytorch.eval()
out_manual_eval = bn_manual(x)
out_pytorch_eval = bn_pytorch(x)
print(f"评估模式输出一致: {torch.allclose(out_manual_eval, out_pytorch_eval, atol=1e-5)}")

# 演示 BN 对网络训练的影响
print("\nBN 对深层网络训练的影响:")
def train_with_bn(use_bn):
    torch.manual_seed(42)
    layers = []
    in_dim = 50
    for i in range(10):
        layers.append(nn.Linear(in_dim, 64 if i < 9 else 1))
        if use_bn and i < 9:
            layers.append(nn.BatchNorm1d(64))
        if i < 9:
            layers.append(nn.ReLU())
    model = nn.Sequential(*layers)
    opt = torch.optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X = torch.randn(200, 50)
    y = torch.randn(200, 1)

    for epoch in range(100):
        pred = model(X)
        loss = criterion(pred, y)
        opt.zero_grad()
        loss.backward()
        opt.step()

    return loss.item()

print(f"  无 BN: {train_with_bn(False):.6f}")
print(f"  有 BN: {train_with_bn(True):.6f}")

# BN 层的各个组件
bn = nn.BatchNorm1d(10)
print(f"\nBN 参数: gamma={bn.weight}, beta={bn.bias}")
print(f"BN 缓冲区: running_mean={bn.running_mean}, running_var={bn.running_var}")
```

## 深度学习关联

- **CNN 训练的标准配置**：Batch Normalization 是训练深度卷积网络的标准技术。BatchNorm 层通常放在卷积层之后、激活函数之前（Conv → BN → ReLU）。这个顺序已被证明效果最好。BN 使得可以使用更高的学习率，对初始化不那么敏感。
- **加速训练的推动者**：BN 的引入使得训练上百层的网络成为可能。它允许使用更高的学习率（通常可以提升 5-10 倍），大幅加速收敛。这对 ResNet 等超深网络的训练至关重要。
- **batch size 依赖性**：BN 的性能依赖于 batch size。当 batch size 很小时（如 2, 4），统计量估计不准确，BN 的效果显著下降。这促使了 Layer Normalization 和 Group Normalization 等替代方案的提出。BN 也不适合 RNN（因为时间步之间统计量变化大）。
