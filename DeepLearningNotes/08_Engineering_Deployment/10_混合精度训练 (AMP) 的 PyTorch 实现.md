# 10_混合精度训练 (AMP) 的 PyTorch 实现

## 核心概念

- **混合精度训练 (AMP)**：在训练中同时使用 FP16/BF16（半精度）和 FP32（单精度）的策略。核心思路是：大部分计算使用半精度以利用 Tensor Core 加速和减少显存，少量关键操作（如 loss scaling、梯度更新）保留 FP32 以保证数值稳定性和模型精度。
- **torch.cuda.amp 模块**：PyTorch 的原生 AMP 实现，包含 `autocast`（自动精度选择）和 `GradScaler`（梯度缩放）两个核心组件。`autocast` 在 forward 中自动为每个算子选择合适的数据类型，`GradScaler` 防止 FP16 梯度下溢。
- **autocast 上下文**：在 `with torch.cuda.amp.autocast():` 块中，PyTorch 会根据预定义的算子白名单自动选择 FP16 或 FP32。矩阵乘法、卷积等在 FP16 下执行；涉及 reduce 操作的 layer norm、softmax 等在 FP32 下执行。
- **GradScaler 梯度缩放**：为了避免 FP16 梯度在反向传播时下溢（小于 FP16 最小表示范围 $5.96 \times 10^{-8}$），在 loss 反向传播前乘以一个缩放因子 $S$（通常初始为 $2^{16}=65536$），backward 后再除回 $S$。如果梯度溢出，则跳过当前 step 并减小缩放因子。
- **BF16 (Brain Float 16)**：Google 提出的 16 位浮点格式，具有与 FP32 相同的 8 位指数范围（因此不需要 loss scaling），但精度较低（7 位尾数）。BF16 在 A100 及后续 GPU 上原生支持，是比 FP16 更稳定的 AMP 方案。
- **Tensor Core 利用**：NVIDIA Volta 及更高架构上的 Tensor Core 可以在 FP16/BF16 输入下实现远高于 FP32 的矩阵乘法吞吐量（A100 上 FP16 理论算力 312 TFLOPS vs FP32 的 19.5 TFLOPS）。

## 数学推导

FP16 的数值表示范围和精度限制。FP16 的 IEEE 754 格式：

- 1 位符号，5 位指数（偏置 15），10 位尾数
- 最大可表示值：$65504$
- 最小规范化正数：$2^{-14} \approx 6.10 \times 10^{-5}$
- 最小非规范化正数：$2^{-24} \approx 5.96 \times 10^{-8}$

当梯度值小于 $2^{-24}$ 时，FP16 无法表示，直接变为 0（下溢）。这是梯度缩放 (Loss Scaling) 的必要性来源。

**Loss Scaling 机制**：
$$
\tilde{\mathcal{L}} = \mathcal{L} \times S
$$

backward 过程中，所有梯度被同步缩放：
$$
\tilde{g} = \frac{\partial \tilde{\mathcal{L}}}{\partial w} = \frac{\partial \mathcal{L}}{\partial w} \times S
$$

参数更新前，梯度缩回原始尺度：
$$
w_{t+1} = w_t - \eta \cdot \frac{\tilde{g}}{S} = w_t - \eta \cdot \frac{\partial \mathcal{L}}{\partial w}
$$

若缩放后的梯度发生溢出（$\tilde{g} > 65504$），则跳过该 step 并将 $S \leftarrow S / 2$；若连续 N 步无溢出，则 $S \leftarrow S \times 2$。

## 直观理解

- **AMP = 节能模式下的性能跑车**：FP16 就是"节能模式"，速度快、油耗低（显存小），但动力有限（数值范围窄）；FP32 是"赛道模式"，动力充沛但油耗高。AMP 智能地在直道（矩阵乘）上用节能模式，在弯道（LayerNorm/Softmax）上切回赛道模式，两者兼顾。
- **BF16 的哲学**：BF16 说"我宁愿保留更大的数值范围但精度差一点"——就像磅秤只能称到整数克，但最大能称 100 公斤；FP16 是"精度高但范围小"——像精密天平能称到 0.1 毫克但最大只能称 100 克。在深度学习中，数值范围通常比精度更重要，所以 BF16 越来越流行。
- **最佳实践**：对于新项目，优先尝试 BF16 AMP（`torch.bfloat16`），它不需要 GradScaler，稳定性更好；若 GPU 不支持 BF16（如 V100），则使用 FP16 AMP + GradScaler。
- **常见陷阱**：AMP 下 layer norm 和 softmax 会在 FP32 下计算，但它们的输入可能是 FP16 的。若自定义层中包含了这些操作而不在 autocast 范围内，可能导致数值不稳定。

## 代码示例

```python
import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler

# ========== 1. 基础 AMP 使用 ==========
model = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
    nn.LayerNorm(4096),
).cuda()
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
criterion = nn.MSELoss()

# 数据
data = torch.randn(64, 4096).cuda()
target = torch.randn(64, 4096).cuda()

# FP16 AMP + GradScaler
scaler = GradScaler()
for step in range(100):
    optimizer.zero_grad()

    # autocast 上下文自动管理精度
    with autocast(dtype=torch.float16):
        output = model(data)
        loss = criterion(output, target)

    # GradScaler 处理梯度缩放
    scaler.scale(loss).backward()
    scaler.step(optimizer)
    scaler.update()

    if step % 20 == 0:
        print(f"Step {step}, Loss: {loss.item():.6f}")

# ========== 2. BF16 AMP（不需要 GradScaler）==========
model_bf16 = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.ReLU(),
    nn.Linear(4096, 4096),
).cuda()
optimizer_bf16 = torch.optim.AdamW(model_bf16.parameters(), lr=1e-4)

for step in range(50):
    optimizer_bf16.zero_grad()
    with autocast(dtype=torch.bfloat16):
        output = model_bf16(data)
        loss = criterion(output, target)
    # BF16 不需要 scaler！
    loss.backward()
    optimizer_bf16.step()

# ========== 3. 自定义 autocast 策略 ==========
class MyCustomLayer(nn.Module):
    """自定义层：强制某些操作在 FP32 下执行"""
    def forward(self, x):
        # xy 乘积在 autocast 下会自动用 FP16/BF16
        out = x @ x.transpose(-1, -2)
        # 但这个 reduce 操作我们希望用 FP32
        with autocast(enabled=False):
            out = out.softmax(dim=-1)
        return out

# ========== 4. 手动管理精度 ==========
x = torch.randn(64, 4096).cuda().half()  # 手动转到 FP16
w = torch.randn(4096, 4096).cuda().half()
out = x @ w  # FP16 矩阵乘
# 精度敏感操作手动转到 FP32
out = out.float().softmax(dim=-1).half()

# ========== 5. 混合精度的显存对比 ==========
def memory_usage():
    return torch.cuda.memory_allocated() / 1024**3

model_large = nn.Sequential(
    nn.Linear(4096, 4096),
    nn.Linear(4096, 4096),
    nn.Linear(4096, 4096),
).cuda()

print(f"FP32 模型显存: {memory_usage():.2f} GB")

with autocast(dtype=torch.float16):
    out = model_large(data.half())
print(f"FP16 推理显存: {memory_usage():.2f} GB")
# 显存约减少 40-50%

# ========== 6. 梯度缩放调试 ==========
print(f"当前缩放因子: {scaler.get_scale():.0f}")
# 如果 loss 溢出，scaler 会自动降低缩放因子
# 可以通过 scaler.get_scale() 监控缩放因子的变化
```

## 深度学习关联

- **大规模训练的标准配置**：AMP 是所有大模型训练的标配。在训练 LLaMA-70B、GPT-4 等模型时，混合精度训练使单 GPU 显存需求降低约 40-60%。MLflow 等实验跟踪系统需要记录 AMP 的 dtype 配置（FP16 vs BF16）以及 GradScaler 的行为（是否触发 overflow 跳过 step）。
- **推理服务中的精度控制**：在生产推理服务（Triton Inference Server）中，模型通常以 FP16/INT8 精度运行以获得低延迟。AMP 训练是 FP16 推理的前提——如果模型在 FP32 下训练后直接转为 FP16 推理，精度往往会显著下降。因此，需要在训练时就使用 AMP 使模型适应低精度计算。
- **与分布式训练的交互**：在 DDP/FSDP 中启用 AMP 时，需要确保梯度同步的 dtype 与推理 dtype 一致。FSDP 的 `MixedPrecision` 参数可以分别设置 `param_dtype`、`reduce_dtype` 和 `buffer_dtype`，确保通信和计算都在最优精度下进行。
