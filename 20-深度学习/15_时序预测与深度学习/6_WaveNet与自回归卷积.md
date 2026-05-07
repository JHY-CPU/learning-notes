# 6_WaveNet 与自回归卷积

## 1. WaveNet 概述

WaveNet 由 DeepMind 在 2016 年提出，最初用于音频生成，其核心架构——**堆叠膨胀因果卷积 + 门控激活**——后来被广泛应用于通用时间序列预测。

**核心创新：**
- 膨胀因果卷积堆叠，指数级扩大感受野
- 门控激活单元（Gated Activation Units）
- 残差连接 + 跳跃连接（Skip Connection）
- 自回归生成：逐点生成下一个值

## 2. 门控激活单元

WaveNet 的核心计算单元：

$$z = \tanh(W_f * x) \odot \sigma(W_g * x)$$

- $W_f * x$：滤波器 (filter) 分支，捕捉特征模式
- $W_g * x$：门控 (gate) 分支，控制信息流
- $\odot$：逐元素乘法

```
输入 x ──┬── [1×1 Conv (tanh)] ──┐
          │                        ├── ⊙ ──→ 输出
          └── [1×1 Conv (σ)]   ───┘
```

**为什么门控激活有效：**
- tanh 提供非线性变换
- sigmoid 门控允许网络学会"忽略"不重要的时间步
- 类似 LSTM 的门控机制，但计算效率更高（无循环结构）

## 3. WaveNet 完整架构

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class WaveNetBlock(nn.Module):
    """WaveNet 残差块：膨胀因果卷积 + 门控激活"""
    def __init__(self, residual_channels, skip_channels, kernel_size, dilation):
        super().__init__()
        self.dilated_conv = nn.Conv1d(
            residual_channels, 2 * residual_channels,  # 2倍用于门控拆分
            kernel_size, dilation=dilation,
            padding=(kernel_size - 1) * dilation
        )

        # 残差投影
        self.residual_conv = nn.Conv1d(residual_channels, residual_channels, 1)
        # 跳跃连接投影
        self.skip_conv = nn.Conv1d(residual_channels, skip_channels, 1)

    def forward(self, x, skip_sum=None):
        """
        x: (batch, residual_channels, seq_len)
        """
        # 因果卷积：左侧填充
        out = self.dilated_conv(x)
        out = out[:, :, :x.size(2)]  # 保持长度一致

        # 门控激活
        tanh_out, sigm_out = out.chunk(2, dim=1)
        gated = torch.tanh(tanh_out) * torch.sigmoid(sigm_out)

        # 残差连接
        residual = self.residual_conv(gated) + x

        # 跳跃连接（累积到最终输出）
        skip = self.skip_conv(gated)
        if skip_sum is None:
            skip_sum = skip
        else:
            skip_sum = skip_sum + skip

        return residual, skip_sum


class WaveNet(nn.Module):
    def __init__(self, in_channels, residual_channels=64, skip_channels=128,
                 num_layers=10, kernel_size=2, forecast_horizon=1):
        super().__init__()
        self.input_conv = nn.Conv1d(in_channels, residual_channels, 1)

        # 膨胀率: 1, 2, 4, 8, ..., 2^(n-1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, skip_channels,
                         kernel_size, dilation=2**i)
            for i in range(num_layers)
        ])

        # 输出层
        self.output_net = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, skip_channels, 1),
            nn.ReLU(),
            nn.Conv1d(skip_channels, forecast_horizon, 1)
        )

    def forward(self, x):
        # x: (batch, seq_len, in_channels) -> (batch, in_channels, seq_len)
        x = x.transpose(1, 2)

        out = self.input_conv(x)
        skip_sum = None

        for block in self.blocks:
            out, skip_sum = block(out, skip_sum)

        # 只取最后一个时间步的预测
        output = self.output_net(skip_sum)[:, :, -1]
        return output
```

## 4. 自回归生成

WaveNet 的推理是逐点自回归的，每次生成一个值：

```python
@torch.no_grad()
def wavenet_autoregressive_generate(model, seed, horizon, temperature=1.0):
    """
    自回归逐步生成
    seed: (1, seed_len, features) 初始种子序列
    horizon: 生成步数
    temperature: 采样温度（>1 更随机，<1 更确定）
    """
    model.eval()
    generated = seed.clone()

    for _ in range(horizon):
        # 用整个历史序列预测下一步
        pred = model(generated)  # (1, 1)

        if temperature != 1.0:
            pred = pred / temperature

        # 将预测值追加到序列末尾
        next_input = pred.unsqueeze(1)  # (1, 1, 1)
        generated = torch.cat([generated, next_input], dim=1)

    return generated[:, seed.size(1):, :]  # 返回生成的部分
```

## 5. WaveNet 变种：用于时序预测

原始 WaveNet 是为逐点生成设计的。在时序预测中，常用的改编方案：

### 5.1 逐时间步输出

```python
class WaveNetForecaster(nn.Module):
    """改编版 WaveNet：输出时间步级别的预测"""
    def __init__(self, input_features, hidden_channels, num_layers,
                 kernel_size, horizon):
        super().__init__()
        self.input_conv = nn.Conv1d(input_features, hidden_channels, 1)

        self.blocks = nn.ModuleList()
        for i in range(num_layers):
            self.blocks.append(WaveNetBlock(
                hidden_channels, hidden_channels,
                kernel_size, dilation=2**i
            ))

        self.fc = nn.Sequential(
            nn.ReLU(),
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(),
            nn.Linear(hidden_channels, horizon)
        )

    def forward(self, x):
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        out = self.input_conv(x)
        skip_sum = None
        for block in self.blocks:
            out, skip_sum = block(out, skip_sum)
        last = skip_sum[:, :, -1]
        return self.fc(last)
```

### 5.2 与 TCN 的区别

| 特性 | WaveNet | TCN |
|------|---------|-----|
| 激活函数 | 门控 (tanh × σ) | ReLU |
| 跳跃连接 | 有（累加到输出） | 无（仅残差） |
| 输出方式 | 逐点自回归 | 时间步级别 |
| 原始应用 | 音频生成 | 序列分类/回归 |

## 6. WaveNet 在时序预测中的实践

```python
# 训练 WaveNet 预测器
model = WaveNetForecaster(
    input_features=7,
    hidden_channels=64,
    num_layers=8,        # 感受野 = 2^8 = 256
    kernel_size=3,
    horizon=24
)

# 数据格式: (batch, seq_len, features)
x = torch.randn(32, 256, 7)  # 256步历史
y = model(x)                  # (32, 24) 预测未来24步

# 使用更大的感受野处理长序列
# kernel_size=3, layers=10 => RF = 1 + 2*(2^10-1) = 2047
long_model = WaveNetForecaster(
    input_features=7, hidden_channels=128,
    num_layers=10, kernel_size=3, horizon=48
)
```

## 7. 原始 WaveNet 的音频生成

```python
# 原始 WaveNet 对离散音频样本建模（16-bit PCM = 65536 类）
# 使用 softmax 输出 + 交叉熵损失

class OriginalWaveNet(nn.Module):
    def __init__(self, num_classes=256, num_layers=10, residual_channels=64,
                 skip_channels=256):
        super().__init__()
        self.causal_conv = nn.Conv1d(1, residual_channels, kernel_size=2,
                                      padding=1)
        self.blocks = nn.ModuleList([
            WaveNetBlock(residual_channels, skip_channels, 2, 2**i)
            for i in range(num_layers)
        ])
        self.out = nn.Sequential(
            nn.ReLU(),
            nn.Conv1d(skip_channels, num_classes, 1),
        )

    def forward(self, x):
        out = self.causal_conv(x)[:, :, :-1]
        skip_sum = None
        for block in self.blocks:
            out, skip_sum = block(out, skip_sum)
        return self.out(skip_sum)
```

---

**要点总结：**
- WaveNet 通过门控激活（tanh × sigmoid）实现选择性信息传递
- 膨胀率指数增长使感受野随深度指数扩大
- 跳跃连接将每层特征直接汇总到输出，改善梯度流动
- 在时序预测中，WaveNet 架构已被 TCN 简化版本广泛替代
- 门控激活的思想影响了后续许多时序模型的设计
