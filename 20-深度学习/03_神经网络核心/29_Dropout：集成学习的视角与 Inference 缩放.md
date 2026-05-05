# 30_Dropout：集成学习的视角与 Inference 缩放

## 核心概念

- **Dropout 的定义**：Dropout 在训练时以概率 $p$ 随机丢弃（置零）神经元。每次前向传播都使用不同的子网络，相当于在训练一个由 $2^n$ 个子网络组成的集成模型。
- **集成学习的视角**：每次 Dropout 采样一个不同的网络架构（子网络）。训练过程相当于同时训练所有子网络并共享权重。推理时使用完整的网络，这相当于对所有子网络的预测取平均——这正是集成学习的思想。
- **推理时的缩放**：推理时神经元不再随机丢弃。为了匹配训练时的期望输出，需要将权重乘以 $(1-p)$。另一种等价方式是训练时将激活除以 $(1-p)$（称为"inverted dropout"），推理时不做任何处理。
- **Inverted Dropout 的标准实现**：现代深度学习框架（PyTorch、TensorFlow）都使用 Inverted Dropout。训练时保留的神经元激活值除以 $(1-p)$，以保持期望输出不变。推理时不进行任何缩放操作。

## 数学推导

**标准 Dropout**：

训练时，每个神经元以概率 $p$ 被置零：

$$
h_{\text{train}} = \begin{cases}
\frac{h}{1-p} & \text{with probability } 1-p \\
0 & \text{with probability } p
\end{cases}
$$

注意 Inverted Dropout 中训练时已经做了 $1/(1-p)$ 缩放。

**Inverted Dropout 的期望证明**：

$$
\mathbb{E}[h_{\text{train}}] = (1-p) \cdot \frac{h}{1-p} + p \cdot 0 = h
$$

训练时的输出期望等于推理时的输出，因此推理时不需要做任何缩放。

**Dropout 作为贝叶斯近似**：

Dropout 可以解释为深度高斯过程（Deep Gaussian Process）的变分近似。具体来说，Dropout 等价于在权重上施加伯努利分布的变分后验。每个权重 $W_{ij}$ 的变分分布为：

$$
q(W_{ij}) = p \cdot \delta(0) + (1-p) \cdot \mathcal{N}(m_{ij}, \sigma^2)
$$

这为 Dropout 提供了贝叶斯理论基础，并允许使用 Dropout 进行不确定性估计。

**Dropout 与 Bagging 的关系**：

Dropout 类似于 Bagging（Bootstrap Aggregating）：
- Bagging：在不同数据子集上独立训练不同模型
- Dropout：在不同神经元子集上训练共享权重的子网络

关键区别：Dropout 的子网络共享权重（参数绑定），而 Bagging 的模型独立训练。

## 直观理解

Dropout 可以理解为"强迫网络学习冗余表示"：由于每个神经元都有可能被随机丢弃，网络不能过度依赖任何一个特定的神经元。这迫使网络学习多个独立的特征检测器，每个特征都得到多路冗余的表征。

从集成学习的视角看，Dropout 相当于同时训练了指数级数量的子网络。推理时使用完整的网络（所有神经元），相当于对这些子网络的预测进行加权平均。集成本身是提升模型泛化性能的经典技术，Dropout 以极低的额外成本实现了类似的效果。

类比：Dropout 就像"团队培训时让成员轮流缺席"。每个成员都需要学会在没有某些同事的情况下完成任务。最终所有成员一起工作时，团队表现得更加稳健和协调。

$1-p$ 的选择规则：$p = 0.5$ 对隐层是最优的（最大程度的正则化），而对输入层通常用 $p = 0.2$ 或更小（避免丢弃太多输入信息）。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# 手动实现 Dropout
def dropout_manual(x, p=0.5, training=True):
    """手动实现 Inverted Dropout"""
    if not training:
        return x
    mask = torch.bernoulli(torch.ones_like(x) * (1 - p))  # 保留概率为 1-p
    return x * mask / (1 - p)

# 验证期望
x = torch.ones(10000)
dropped = dropout_manual(x, p=0.5, training=True)
print(f"原始均值: {x.mean():.4f}")
print(f"Dropout 后均值: {dropped.mean():.4f} (应接近 1.0)")

# Dropout 在 MLP 中的使用
class MLPWithDropout(nn.Module):
    def __init__(self, dropout_p=0.5):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(784, 512),
            nn.ReLU(),
            nn.Dropout(dropout_p),  # PyTorch 内置 Inverted Dropout
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout_p),
            nn.Linear(256, 10)
        )

    def forward(self, x):
        return self.net(x)

# 演示 Dropout 在训练和推理时的差异
model = MLPWithDropout(0.5)
model.train()
x = torch.randn(4, 784)
train_out = model(x)
print(f"\n训练时输出方差: {train_out.var():.4f} (有 Dropout)")

model.eval()
eval_out = model(x)
print(f"推理时输出方差: {eval_out.var():.4f} (无 Dropout)")

# 验证 Inverted Dropout 的期望一致性
# 多次前向传播取平均，模拟集成效果
model.train()
outputs = []
for _ in range(1000):
    outputs.append(model(x))
ensemble_mean = torch.stack(outputs).mean(0)

model.eval()
single_output = model(x)

print(f"集成平均与单次推理的差异: {(ensemble_mean - single_output).abs().mean():.6f}")

# 不同 Dropout 率的正则化效果
def train_with_dropout(dropout_p):
    torch.manual_seed(42)
    model = nn.Sequential(
        nn.Linear(50, 256), nn.ReLU(), nn.Dropout(dropout_p),
        nn.Linear(256, 256), nn.ReLU(), nn.Dropout(dropout_p),
        nn.Linear(256, 1)
    )
    opt = optim.Adam(model.parameters(), lr=0.001)
    criterion = nn.MSELoss()

    X_train = torch.randn(300, 50)
    y_train = torch.sin(X_train.sum(1, keepdim=True)) + torch.randn(300, 1) * 0.2
    X_test = torch.randn(100, 50)
    y_test = torch.sin(X_test.sum(1, keepdim=True))

    for epoch in range(500):
        model.train()
        pred = model(X_train)
        loss = criterion(pred, y_train)
        opt.zero_grad()
        loss.backward()
        opt.step()

    model.eval()
    with torch.no_grad():
        train_pred = model(X_train)
        test_pred = model(X_test)
        train_loss = criterion(train_pred, y_train).item()
        test_loss = criterion(test_pred, y_test).item()

    return train_loss, test_loss

print("\nDropout 率对泛化的影响:")
for p in [0.0, 0.1, 0.3, 0.5, 0.7]:
    train_l, test_l = train_with_dropout(p)
    print(f"  p={p:.1f}: train_loss={train_l:.4f}, test_loss={test_l:.4f}, "
          f"gap={train_l - test_l:.4f}")
```

## 深度学习关联

- **全连接层的主要正则化手段**：Dropout 是全连接网络中最有效的正则化技术之一。在卷积网络中，Dropout 的使用相对较少（因为卷积层参数少，过拟合风险低），但仍然是全连接分类头的标准配置。
- **Monte Carlo Dropout**：在推理时保持 Dropout 开启，进行多次前向采样，可以得到预测的不确定性估计。这在贝叶斯深度学习中被视为一种近似推理方法，在医疗影像、自动驾驶等高风险场景中用于量化模型的不确定性。
- **被其他正则化技术补充**：在现代深度学习中，Dropout 常与 Batch Normalization 结合使用。但注意，BatchNorm 已经提供了一定的正则化效果，与 Dropout 叠加可能会过度正则化，需要适当降低 Dropout 率或两者择一。在 Transformer 中，Dropout 主要用于 attention 权重的 dropout（pattern dropout）和 FFN 层的 dropout。
