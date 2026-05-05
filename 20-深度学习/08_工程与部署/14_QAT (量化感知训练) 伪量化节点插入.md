# 14_QAT (量化感知训练) 伪量化节点插入

## 核心概念

- **量化感知训练 (QAT)**：一种在训练过程中模拟量化效果的技术，通过在计算图中插入伪量化节点 (FakeQuantize) 让模型权重的分布适应量化带来的精度损失。经过 QAT 的模型在 INT8 推理时精度显著优于 PTQ。
- **伪量化节点 (FakeQuantize Node)**：在 forward 中模拟量化-反量化过程：先量化（将 FP32 映射到 INT8），再反量化回 FP32。这样模型的参数会在量化噪声的环境中适应性调整。backward 时使用直通估计器 (STE) 将梯度直接绕过量化操作。
- **直通估计器 (STE, Straight-Through Estimator)**：由于量化操作 $\text{round}(\cdot)$ 的导数几乎处处为零，无法直接进行反向传播。STE 在 backward 时将量化操作的梯度近似视为 1（即直接传递梯度），使得量化层可以正常更新参数。
- **训练阶段的 Observer**：在 QAT 训练过程中，通过 Observer 模块实时统计权重和激活值的 min/max 范围，更新 scale 和 zero_point 参数。Observer 的更新方式可以是移动最小最大值 (MinMaxObserver) 或基于直方图的分布估计 (HistogramObserver)。
- **QAT 的微调策略**：QAT 通常在已训练的 FP32 模型基础上进行少量 epoch 的微调（而非从头训练），使用较小的学习率（如原学习率的 1/10~1/100）和余弦退火调度。
- **Batch Normalization 融合 (BN Fusion)**：在量化推理时，BN 层通常与前一层卷积融合以减少计算量。QAT 训练需要先进行 BN 融合（`torch.quantization.fuse_modules`），然后在融合后的层上插入伪量化节点。

## 数学推导

伪量化操作 $Q(\cdot)$ 的定义：

$$
\tilde{r} = Q(r) = s \cdot \text{clamp}\left( \text{round}\left( \frac{r}{s} \right), -2^{n-1}, 2^{n-1} - 1 \right)
$$

其中 $s$ 是 scale 因子，$n$ 是量化比特数。该操作在 forward 中模拟了量化的信息损失——$\tilde{r}$ 仍然是浮点数，但其值被约束在离散的量化电平上。

STE 的梯度反向传播定义：

$$
\frac{\partial \tilde{r}}{\partial r} \approx 1 \quad \text{(STE approximation)}
$$

即，在 backward 时假装量化函数是恒等映射。虽然 $\frac{dQ}{dr}$ 实际是 Dirac 脉冲序列，但 STE 将其近似为常数 1，使梯度可以顺利流经量化节点。

QAT 训练中的参数更新受到量化噪声的扰动。考虑量化后的权重 $\tilde{W} = Q(W)$：

$$
W_{t+1} = W_t - \eta \cdot \frac{\partial \mathcal{L}(\tilde{W}_t)}{\partial W_t}
$$

由于 $\tilde{W}_t = Q(W_t)$，权重 $W_t$ 的小幅变化只有在跨过量化阈值时才会影响 $\tilde{W}_t$。QAT 本质上是在优化一个带有离散约束的损失函数：

$$
\min_W \mathcal{L}(Q(W)) \quad \text{s.t.} \quad Q(W) \in \{s \cdot q \mid q \in \mathbb{Z}\}
$$

## 直观理解

- **QAT = 在噪音环境中训练**：让一个舞者（模型）先在会轻微震动的舞台上（伪量化噪声）练习，练到在震动的舞台上也能稳定表演。实际比赛时舞台不震动了（INT8 推理），舞者会表现得更好。
- **STE 的作用 = 盲人走路**：量化操作像是"路障"阻断了梯度信息，STE 假设"路障不存在"让梯度直接通过。虽然假设不完全正确，但实践中效果很好——就像盲人用手杖探路，虽然触感不精确，但足以判断方向。
- **最佳实践**：先使用 PTQ 评估量化精度，如果精度损失 > 1%，再使用 QAT。QAT 通常只需要 5-10 个 epoch 的微调（使用训练数据的子集即可），学习率设为初始的 1/100。
- **常见陷阱**：QAT 必须在训练中保持伪量化节点的启用，在推理阶段再将其移除（`convert()`）。如果在 QAT 过程中禁用了伪量化，scale/zero_point 会失去校准。
- **经验法则**：使用 QAT 时，务必使用逐通道量化 (Per-Channel Quantization) 以获得更好的精度；先 fuse BN 再 QAT 是标准流程。

## 代码示例

```python
import torch
import torch.nn as nn
from torch.quantization import (
    QuantStub, DeQuantStub, FakeQuantize,
    prepare_qat, convert, fuse_modules,
    get_default_qat_qconfig,
    QConfig,
)
from torch.quantization.observer import MinMaxObserver, MovingAverageMinMaxObserver

# ========== 1. 模型定义 + BN 融合 + QAT ==========
class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.quant = QuantStub()
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(128)
        self.relu2 = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(128, 10)
        self.dequant = DeQuantStub()

    def forward(self, x):
        x = self.quant(x)
        x = self.relu1(self.bn1(self.conv1(x)))
        x = self.relu2(self.bn2(self.conv2(x)))
        x = self.pool(x).view(x.size(0), -1)
        x = self.fc(x)
        x = self.dequant(x)
        return x

# ========== 2. BN 融合 ==========
model = ConvNet().train()

# 融合 conv+bn+relu 层
model = fuse_modules(model, [["conv1", "bn1", "relu1"],
                               ["conv2", "bn2", "relu2"]])

# ========== 3. 配置 QAT ==========
# QAT 配置：使用 FakeQuantize 在训练中模拟量化
qat_qconfig = QConfig(
    activation=FakeQuantize.with_args(
        observer=MovingAverageMinMaxObserver,
        quant_min=0,
        quant_max=255,
        dtype=torch.quint8,
        qscheme=torch.per_tensor_affine,
        reduce_range=False,
    ),
    weight=FakeQuantize.with_args(
        observer=MinMaxObserver,
        quant_min=-128,
        quant_max=127,
        dtype=torch.qint8,
        qscheme=torch.per_tensor_symmetric,
        reduce_range=False,
    ),
)

model.qconfig = qat_qconfig

# 或者使用默认配置（推荐）
model.qconfig = get_default_qat_qconfig("fbgemm")

# 准备 QAT：插入 FakeQuantize 节点
prepared_model = prepare_qat(model, inplace=False)

# ========== 4. QAT 训练（微调）===========
optimizer = torch.optim.SGD(prepared_model.parameters(), lr=1e-4)  # 小学习率
criterion = nn.CrossEntropyLoss()

# QAT 训练循环（5-10 epoch 即可）
for epoch in range(5):
    prepared_model.train()
    for batch_idx, (data, target) in enumerate(dummy_loader()):  # 伪代码
        optimizer.zero_grad()
        output = prepared_model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()

    # 每个 epoch 后更新量化参数（准确来说由 observer 自动完成）
    print(f"Epoch {epoch}: "
          f"conv1 weight scale = {prepared_model.conv1.weight_fake_quant.scale:.4f}")

# ========== 5. 转换为 INT8 推理模型 ==========
prepared_model.eval()
quantized_model = convert(prepared_model)

# ========== 6. 自定义 FakeQuantize 实现 ==========
class CustomFakeQuantize(torch.autograd.Function):
    """自定义伪量化，演示 STE 的核心逻辑"""
    @staticmethod
    def forward(ctx, x, scale, zero_point, num_bits=8):
        q_max = 2 ** (num_bits - 1) - 1
        q_min = -q_max - 1
        # 量化
        x_int = torch.round(x / scale) + zero_point
        x_int = torch.clamp(x_int, q_min, q_max)
        # 反量化
        x_fake = (x_int - zero_point) * scale
        return x_fake

    @staticmethod
    def backward(ctx, grad_output):
        # STE: 直接传递梯度
        return grad_output, None, None, None

# 使用自定义伪量化
def test_custom_fakequant():
    x = torch.randn(4, 64, requires_grad=True)
    scale = torch.tensor(0.01)
    zero_point = torch.tensor(0)
    out = CustomFakeQuantize.apply(x, scale, zero_point)
    loss = out.sum()
    loss.backward()  # STE 保证梯度顺利通过
    print(f"Gradient valid: {x.grad is not None}")

test_custom_fakequant()

# ========== 7. 精度对比 ==========
def compare_accuracy(model_fp32, model_qat_int8, test_loader):
    """对比 FP32 和 QAT INT8 的精度"""
    model_fp32.eval()
    model_qat_int8.eval()

    correct_fp32 = correct_int8 = total = 0
    with torch.no_grad():
        for data, targets in test_loader:
            total += targets.size(0)
            out_fp32 = model_fp32(data)
            correct_fp32 += out_fp32.argmax(1).eq(targets).sum().item()

            out_int8 = model_qat_int8(data)
            correct_int8 += out_int8.argmax(1).eq(targets).sum().item()

    print(f"FP32 精度: {correct_fp32/total:.4f}")
    print(f"QAT INT8 精度: {correct_int8/total:.4f}")
    print(f"精度差: {(correct_fp32 - correct_int8)/total:.4f}")
```

## 深度学习关联

- **生产环境中的量化流水线**：在 MLOps 流水线中，QAT 通常作为训练后的一个可选步骤集成。典型流程为：FP32 训练 -> 模型评估 -> 若延迟不达标则启动 QAT -> QAT 微调 -> 转换为 INT8 -> 部署到 Triton/TensorRT。每一步的精度和延迟指标均记录在 MLflow 中。
- **QAT 与 TensorRT 的集成**：NVIDIA TensorRT 支持通过 QAT 导出的 ONNX 模型直接构建 INT8 推理引擎。在 QAT 中使用 `torch.onnx.export` 配合 `QuantStub/DeQuantStub` 导出 ONNX，TensorRT 可以自动识别伪量化节点并应用对应精度。
- **大语言模型的 QAT 挑战**：对于 LLM（7B+），QAT 的计算开销极大。近年来出现了 QAT 的变体如 LLM-QAT（在蒸馏的基础上进行量化感知训练）和 AQLM（Additive Quantization of Language Models），使得大模型的 QAT 变得可行。在部署 30B+ 模型时，QAT 可能是唯一能保持可接受精度的量化方案。
