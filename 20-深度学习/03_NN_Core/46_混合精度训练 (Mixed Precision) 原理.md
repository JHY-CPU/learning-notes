# 46_混合精度训练 (Mixed Precision) 原理

## 核心概念

- **混合精度定义**：混合精度训练（Mixed Precision Training）在训练过程中同时使用 FP16（半精度浮点数）和 FP32（单精度浮点数）。核心思想是：在计算和内存密集的部分使用 FP16 加速，在需要精度的部分保留 FP32。
- **FP16 的优势**：FP16 相比 FP32 只需要一半的显存（2 bytes vs 4 bytes），计算速度在支持 Tensor Core 的 GPU（Volta 架构以上）上可以提升 2-8 倍。这使得可以训练更大的模型和使用更大的 batch size。
- **FP16 的挑战**：FP16 的数值范围（$5.96 \times 10^{-8}$ 到 $65504$）和精度（约 3.3 位有效数字）远小于 FP32。这可能导致三个问题：下溢（梯度太小变为 0）、上溢（梯度太大变为 inf）、精度损失（小梯度被舍入）。
- **混合精度的策略**：使用 FP16 存储激活值和梯度（节省显存），使用 FP32 存储主权重和累积梯度（保持精度）。在计算过程中，使用 FP16 矩阵乘法（Tensor Core 加速），必要时转换为 FP32 进行累加。

## 数学推导

**FP16 数值格式**：

FP16 使用 1 位符号位、5 位指数位、10 位尾数位：

最小值：$2^{-24} \approx 5.96 \times 10^{-8}$（次正规数）
最大值：$65504$（正规数）
精度：约 3.3 位十进制有效数字

FP32 使用 1 位符号位、8 位指数位、23 位尾数位：
最小值：$1.4 \times 10^{-45}$（次正规数）
最大值：$3.4 \times 10^{38}$

**混合精度训练流程**：

- **前向传播**：FP16 权重 $W_{fp16}$ 用于计算
- **损失计算**：FP32 精度计算损失
- **反向传播**：FP16 梯度 $g_{fp16}$（损失缩放后）
- **梯度更新**：梯度转换为 FP32，缩放回去，更新 FP32 主权重
- **权重转换**：FP32 权重 $\to$ FP16 权重（用于下次前向）

数学表示：

$$
W_{t+1}^{fp32} = W_t^{fp32} - \eta \cdot \text{unscale}(g_t^{fp16} \cdot \text{scale})
$$

$$
W_{t+1}^{fp16} = \text{to\_fp16}(W_{t+1}^{fp32})
$$

**显存节省分析**：

假设模型有 $P$ 个参数：
- FP32 训练：$4P$（权重）+ $4P$（梯度）+ $4P$（优化器状态，如 Adam）= $12P$ bytes
- 混合精度训练：$4P$（FP32 主权重）+ $2P$（FP16 权重）+ $2P$（FP16 梯度）+ $8P$（优化器状态）= $16P$ bytes

实际上，FP32 主权重和优化器状态已经占用了 $12P$，加上 FP16 副本的 $2P+2P$，总计 $16P$，比 FP32 训练多 $4P$ 的显存（增加了 FP16 副本）。但激活值存储大幅减少（FP16 替代 FP32），整体显存节省通常在 30-50%。

当使用 Tensor Core 时，FP16 矩阵乘法的计算速度约为 FP32 的 4-8 倍。

## 直观理解

混合精度训练可以类比为"用两把尺子测量"：用高精度尺子（FP32）保存主测量结果，用快速尺子（FP16）进行日常操作。关键测量（主权重）用高精度保存，日常操作（前向和反向计算）用快速方式完成。

FP16 的下溢问题可以理解为"用一个小量杯接水"——如果水流（梯度）太小，量杯感应不到（下溢为 0）。损失缩放相当于把水流放大后再用量杯接，接完再缩小回去。

混合精度不是"所有计算都用 FP16"，而是在计算图的关键节点使用 FP32 确保精度。这就像在自动挡汽车中——驾驶员（框架）自动选择最合适的档位（精度），不需要手动干预。

## 代码示例

```python
import copy
import time
import torch
import torch.nn as nn

# 检查 GPU 是否支持混合精度
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Compute Capability: {torch.cuda.get_device_capability(0)}")
    # Compute Capability >= 7.0 支持 Tensor Core (FP16)
    # Compute Capability >= 8.0 支持 BF16

# FP16 数值范围演示
print("\nFP16 vs FP32 数值范围:")
fp16_max = torch.finfo(torch.float16).max
fp16_min = torch.finfo(torch.float16).tiny
fp32_max = torch.finfo(torch.float32).max
fp32_min = torch.finfo(torch.float32).tiny
print(f"  FP16: max={fp16_max}, min_normal={fp16_min}")
print(f"  FP32: max={fp32_max}, min_normal={fp32_min}")

# FP16 精度损失演示
x_fp32 = torch.tensor(1.0) + torch.tensor(1e-5)  # 1.00001
x_fp16 = x_fp32.half()
print(f"\n精度损失: 1.0 + 1e-5")
print(f"  FP32: {x_fp32.item():.8f}")
print(f"  FP16: {x_fp16.item():.8f} (精度不足)")

# 使用 PyTorch 自动混合精度 (AMP)
print("\n自动混合精度训练 (AMP):")

model = nn.Sequential(
    nn.Linear(100, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# FP32 训练
model_fp32 = nn.Sequential(*[copy.deepcopy(layer) for layer in model])
opt_fp32 = torch.optim.Adam(model_fp32.parameters(), lr=0.001)

# AMP 训练
model_amp = nn.Sequential(*[copy.deepcopy(layer) for layer in model])
opt_amp = torch.optim.Adam(model_amp.parameters(), lr=0.001)

# 创建数据
X = torch.randn(512, 100)
y = torch.randint(0, 10, (512,))

# FP32 训练
start = time.time()
for epoch in range(50):
    pred = model_fp32(X)
    loss = nn.CrossEntropyLoss()(pred, y)
    opt_fp32.zero_grad()
    loss.backward()
    opt_fp32.step()
fp32_time = time.time() - start

# 如果 GPU 可用，使用 AMP
if torch.cuda.is_available():
    model_amp = model_amp.cuda()
    X_gpu = X.cuda()
    y_gpu = y.cuda()
    opt_amp = torch.optim.Adam(model_amp.parameters(), lr=0.001)

    # 使用 GradScaler 进行损失缩放
    scaler = torch.cuda.amp.GradScaler()

    start = time.time()
    for epoch in range(50):
        with torch.cuda.amp.autocast():  # 自动混合精度上下文
            pred = model_amp(X_gpu)
            loss = nn.CrossEntropyLoss()(pred, y_gpu)

        opt_amp.zero_grad()
        scaler.scale(loss).backward()  # 缩放损失，防止梯度下溢
        scaler.step(opt_amp)           # 更新参数
        scaler.update()                # 更新缩放因子
    amp_time = time.time() - start

    print(f"  FP32: {fp32_time:.2f}s")
    print(f"  AMP:  {amp_time:.2f}s")
    print(f"  加速比: {fp32_time/amp_time:.2f}x")
else:
    print("  GPU 不可用，跳过 AMP 测试")
    # 演示 AMP API 的使用方法（CPU 上也可以用，但无速度优势）
    print("  AMP API 使用示例:")
    print("  1. torch.cuda.amp.autocast(): 自动选择精度")
    print("  2. torch.cuda.amp.GradScaler: 损失缩放")

# FP16 上溢示例
print("\nFP16 上溢演示:")
large_val = torch.tensor(100000.0)
try:
    large_half = large_val.half()
    print(f"  100000.0 -> FP16: {large_half.item()}")
except Exception as e:
    print(f"  上溢错误: {e}")

# 验证 FP16 转换
val = torch.tensor(1.5)
print(f"  1.5 -> FP16: {val.half().item()}")
```

## 深度学习关联

- **大模型训练的必备技术**：混合精度训练是训练大规模模型（GPT-3、LLaMA、BERT-Large）的必备技术。没有混合精度，大模型的显存需求会翻倍，训练时间会增加数倍。NVIDIA 的 Tensor Core 专门为混合精度训练设计。
- **AMP 的自动化**：PyTorch 的 `torch.cuda.amp` 和 TensorFlow 的 `tf.keras.mixed_precision` 提供了自动混合精度训练的支持。用户只需添加几行代码即可获得混合精度训练的好处。自动混合精度的核心是 `autocast` 上下文管理器和 `GradScaler` 损失缩放器。
- **BF16 的引入**：Google 的 TPU 和 NVIDIA Ampere 架构（A100）引入了 BF16（Brain Floating Point 16）格式。BF16 具有与 FP32 相同的指数范围（8 位指数），但尾数更少（7 位）。BF16 不需要损失缩放，因为它和 FP32 有相同的数值范围，简化了混合精度训练的实现。
