# 5_Temporal Convolutional Network (TCN)

## 1. TCN 核心思想

TCN 使用因果卷积 (Causal Convolution) 处理时序数据，确保时刻 $t$ 的输出只依赖于 $t$ 及之前的信息，同时通过膨胀卷积 (Dilated Convolution) 扩大感受野。

```
因果卷积 (dilation=1):
输入: x₁  x₂  x₃  x₄  x₅  x₆
        \  |  /
         [conv]→ y₃     y₃ 只看到 x₁,x₂,x₃

膨胀卷积 (dilation=2):
输入: x₁  x₂  x₃  x₄  x₅  x₆  x₇
        \     |     /
         [dilated conv]→ y₅    y₅ 跳跃看到 x₁,x₃,x₅

膨胀卷积 (dilation=4):
输入: x₁ x₂ x₃ x₄ x₅ x₆ x₇ x₈ x₉
        \       |       /
         [dilated conv]→ y₉    感受野进一步扩大
```

## 2. 感受野计算

对于卷积核大小 $k$，膨胀率为 $d$ 的膨胀卷积，单层感受野为：

$$\text{RF}_{\text{single}} = 1 + (k-1) \cdot d$$

对于 $n$ 层，每层膨胀率为 $d_i = 2^i$ 的 TCN：

$$\text{RF}_{\text{total}} = 1 + \sum_{i=0}^{n-1}(k-1) \cdot 2^i = 1 + (k-1) \cdot (2^n - 1)$$

若 $k=2$，则 $\text{RF} = 2^n$，即 **每增加一层，感受野翻倍**。

| 层数 n | 感受野 (k=2) | 感受野 (k=3) |
|--------|-------------|-------------|
| 4      | 16          | 45          |
| 8      | 256         | 765         |
| 10     | 1024        | 3069        |

## 3. TCN 架构

```
输入 x_t ──→ [因果膨胀卷积 + BN + ReLU + Dropout]
                 ↓
           [因果膨胀卷积 + BN + ReLU + Dropout]
                 ↓
            残差连接 (1x1 Conv 如果维度不匹配)
                 ↓
            下一层 (dilation × 2)
```

**TCN 的三个设计原则：**
1. **因果性：** 未来信息不会泄露到过去
2. **等长映射：** 输出序列与输入序列长度相同
3. **残差连接：** 缓解深层网络的训练困难

## 4. PyTorch 实现

```python
import torch
import torch.nn as nn

class CausalConv1d(nn.Module):
    """因果卷积：左侧填充保证因果性"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=self.padding, dilation=dilation
        )

    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # 去掉右侧的 padding（只保留因果部分）
        return out[:, :, :x.size(2)]


class TCNBlock(nn.Module):
    """TCN 基本块：两层因果卷积 + 残差连接"""
    def __init__(self, in_channels, out_channels, kernel_size,
                 dilation, dropout=0.2):
        super().__init__()
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()

        # 残差连接（维度不匹配时用 1x1 卷积）
        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        residual = self.residual(x)

        out = self.relu(self.bn1(self.conv1(x)))
        out = self.dropout(out)
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.dropout(out)

        return self.relu(out + residual)


class TCN(nn.Module):
    def __init__(self, input_size, num_channels, kernel_size=2,
                 dropout=0.2, forecast_horizon=1):
        """
        input_size: 输入特征维度
        num_channels: 各层通道数列表，如 [64, 64, 64, 64]
        kernel_size: 卷积核大小
        forecast_horizon: 预测步数
        """
        super().__init__()
        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            in_ch = input_size if i == 0 else num_channels[i-1]
            out_ch = num_channels[i]
            dilation = 2 ** i
            layers.append(TCNBlock(in_ch, out_ch, kernel_size, dilation, dropout))

        self.network = nn.Sequential(*layers)
        self.fc = nn.Linear(num_channels[-1], forecast_horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size) -> 转置为 (batch, input_size, seq_len)
        x = x.transpose(1, 2)
        out = self.network(x)
        # 取最后一个时间步
        last = out[:, :, -1]
        return self.fc(last)

# 使用示例
model = TCN(
    input_size=7,
    num_channels=[64, 64, 64, 64, 64],
    kernel_size=3,
    dropout=0.2,
    forecast_horizon=12
)
# 输入: (32, 100, 7) -> 输出: (32, 12)
x = torch.randn(32, 100, 7)
print(model(x).shape)  # torch.Size([32, 12])
```

## 5. TCN 与 LSTM/Transformer 对比

| 特性 | TCN | LSTM | Transformer |
|------|-----|------|-------------|
| 并行训练 | 支持 | 不支持（顺序） | 支持 |
| 感受野 | $O(2^n)$ 指数增长 | $O(n)$ 线性 | $O(n)$ 全局 |
| 内存效率 | 高 | 中（存隐藏状态） | 低（$O(n^2)$ 注意力） |
| 长程依赖 | 良好（需足够深度） | 较差（梯度消失） | 优秀 |
| 推理速度 | 快 | 顺序计算 | 中等 |
| 位置信息 | 通过因果性保证 | 天然序列 | 需位置编码 |

## 6. TCN 的残差块细节

```python
class TCNResidualBlock(nn.Module):
    """改进版残差块：包含门控机制"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation):
        super().__init__()
        self.conv_f = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv_g = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.conv_out = nn.Conv1d(out_channels, out_channels, 1)
        self.dropout = nn.Dropout(0.2)

        self.residual = (nn.Conv1d(in_channels, out_channels, 1)
                         if in_channels != out_channels else nn.Identity())

    def forward(self, x):
        # 门控激活
        f = torch.tanh(self.conv_f(x))
        g = torch.sigmoid(self.conv_g(x))
        out = self.dropout(f * g)
        out = self.conv_out(out)
        return torch.relu(out + self.residual(x))
```

## 7. TCN 实战技巧

```python
# 自动计算所需层数以覆盖目标感受野
def compute_tcn_layers(target_rf, kernel_size=2):
    """计算需要多少层才能达到目标感受野"""
    import math
    # RF = 1 + (k-1) * (2^n - 1)
    # 解出 n
    n = math.log2((target_rf - 1) / (kernel_size - 1) + 1)
    return math.ceil(n)

# 例：需要覆盖 256 个时间步
layers = compute_tcn_layers(target_rf=256, kernel_size=3)
print(f'需要 {layers} 层')  # 需要 8 层
channels = [64] * layers
```

---

**要点总结：**
- TCN 通过因果卷积保证时序因果性，通过膨胀卷积指数级扩大感受野
- 残差连接使得深层 TCN 训练稳定
- TCN 支持并行训练，推理速度快于 RNN
- 当序列长度适中时，TCN 是一个高效的轻量级选择
- 门控激活（WaveNet 风格）可进一步提升 TCN 表达能力
