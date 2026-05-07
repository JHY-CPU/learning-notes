# 3_LSTM 用于时间序列

## 1. LSTM 门控机制

LSTM 通过三个门控单元解决 RNN 的梯度消失问题：

```
        ┌─────────────────────────────────────────┐
        │               Cell State C_t             │
        │  ──────────────────────────────────────> │
        │    │         │    ×        │    +        │
        │    │    ┌────┴────┐   ┌────┴────┐       │
        │    │    │ 遗忘门 f  │   │ 输入门 i  │       │
        │    │    └────┬────┘   └────┬────┘       │
        │    │         │             │             │
  h_t-1 │    │    ┌────┴─────────────┴────┐       │
  ──────┼────┼───>│      Cell 更新         │       │
  x_t   │    │    └───────────────────────┘       │
  ──────┼────┼───>│      输出门 o                 │
        │    │    └──────────────────┬─────┘       │
        │    │                      │              │
        │    │                      h_t ──────>    │
        └─────────────────────────────────────────┘
```

**遗忘门：** 决定丢弃哪些旧信息
$$f_t = \sigma(W_f \cdot [h_{t-1}, x_t] + b_f)$$

**输入门：** 决定存储哪些新信息
$$i_t = \sigma(W_i \cdot [h_{t-1}, x_t] + b_i)$$
$$\tilde{C}_t = \tanh(W_C \cdot [h_{t-1}, x_t] + b_C)$$

**单元状态更新：**
$$C_t = f_t \odot C_{t-1} + i_t \odot \tilde{C}_t$$

**输出门：** 决定输出哪些信息
$$o_t = \sigma(W_o \cdot [h_{t-1}, x_t] + b_o)$$
$$h_t = o_t \odot \tanh(C_t)$$

## 2. LSTM 为何适合时间序列

| 机制 | 在时序中的作用 |
|------|--------------|
| 遗忘门 | 自动学习丢弃无关的旧信息 |
| 输入门 | 选择性写入新的时间模式 |
| 单元状态 | 信息高速公路，梯度可无损传播 |
| 输出门 | 根据当前上下文输出相关信息 |

**关键洞察：** 单元状态 $C_t$ 的更新是加法操作，梯度可以沿 $C_t$ 直接回流，避免了 RNN 中的连乘导致的梯度消失。

## 3. LSTM 时序预测模型

```python
import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers,
                 output_size, dropout=0.2):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        # 输入投影（可选：将原始特征映射到高维）
        self.input_proj = nn.Linear(input_size, hidden_size)

        self.lstm = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0
        )

        # 多头输出
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x, return_attention=False):
        # x: (batch, seq_len, input_size)
        batch_size = x.size(0)

        # 输入投影
        x = self.input_proj(x)

        # 初始化隐藏状态
        h0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size, device=x.device)
        c0 = torch.zeros(self.num_layers, batch_size,
                         self.hidden_size, device=x.device)

        # LSTM 前向传播
        lstm_out, (hn, cn) = self.lstm(x, (h0, c0))
        # lstm_out: (batch, seq_len, hidden_size)

        # 使用最后一个时间步
        last_hidden = lstm_out[:, -1, :]
        prediction = self.fc(last_hidden)

        return prediction
```

## 4. 多变量 LSTM 预测

时间序列通常包含多个相关变量（如温度、湿度、风速同时预测天气）：

```python
class MultivariateLSTM(nn.Module):
    def __init__(self, num_features, hidden_size, num_layers,
                 forecast_horizon, target_features=1):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=num_features,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=0.2
        )
        self.fc = nn.Linear(hidden_size, forecast_horizon * target_features)
        self.horizon = forecast_horizon
        self.target_features = target_features

    def forward(self, x):
        # x: (batch, seq_len, num_features)
        out, _ = self.lstm(x)
        last = out[:, -1, :]
        pred = self.fc(last)
        return pred.view(-1, self.horizon, self.target_features)

# 使用示例
# 输入: (batch=32, seq_len=48, features=7) -> 输出: (32, 12, 1)
model = MultivariateLSTM(num_features=7, hidden_size=128,
                          num_layers=2, forecast_horizon=12)
```

## 5. 特征工程增强

LSTM 配合精心设计的特征可以显著提升效果：

```python
import pandas as pd
import numpy as np

def create_features(df):
    """为时序数据创建丰富特征"""
    # 时间编码（周期性特征）
    df['hour_sin'] = np.sin(2 * np.pi * df.index.hour / 24)
    df['hour_cos'] = np.cos(2 * np.pi * df.index.hour / 24)
    df['day_sin'] = np.sin(2 * np.pi * df.index.dayofweek / 7)
    df['day_cos'] = np.cos(2 * np.pi * df.index.dayofweek / 7)
    df['month_sin'] = np.sin(2 * np.pi * df.index.month / 12)
    df['month_cos'] = np.cos(2 * np.pi * df.index.month / 12)

    # 滞后特征
    for lag in [1, 2, 3, 6, 12, 24]:
        df[f'lag_{lag}'] = df['value'].shift(lag)

    # 滚动统计
    for window in [6, 12, 24]:
        df[f'rolling_mean_{window}'] = df['value'].rolling(window).mean()
        df[f'rolling_std_{window}'] = df['value'].rolling(window).std()
        df[f'rolling_min_{window}'] = df['value'].rolling(window).min()
        df[f'rolling_max_{window}'] = df['value'].rolling(window).max()

    # 差分特征
    df['diff_1'] = df['value'].diff(1)
    df['diff_24'] = df['value'].diff(24)

    return df.dropna()
```

## 6. 双向 LSTM 与注意力增强

```python
class BiLSTMWithAttention(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super().__init__()
        self.bilstm = nn.LSTM(input_size, hidden_size,
                              batch_first=True, bidirectional=True)
        self.attention = nn.Sequential(
            nn.Linear(hidden_size * 2, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1)
        )
        self.fc = nn.Linear(hidden_size * 2, output_size)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        lstm_out, _ = self.bilstm(x)  # (batch, seq_len, hidden*2)

        # 注意力权重
        attn_weights = self.attention(lstm_out)      # (batch, seq_len, 1)
        attn_weights = torch.softmax(attn_weights, dim=1)

        # 加权求和
        context = torch.sum(lstm_out * attn_weights, dim=1)  # (batch, hidden*2)
        return self.fc(context)
```

## 7. 训练技巧

```python
# 1. 学习率预热 + 余弦退火
from torch.optim.lr_scheduler import CosineAnnealingWarmRestarts

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=10, T_mult=2)

# 2. 早停
class EarlyStopping:
    def __init__(self, patience=10, min_delta=1e-5):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None

    def __call__(self, val_loss):
        if self.best_loss is None or val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            return False  # 不停止
        self.counter += 1
        return self.counter >= self.patience  # 停止

# 3. Teacher Forcing (训练时用真实值作为输入)
def train_with_teacher_forcing(model, x, y_true, teacher_forcing_ratio=0.5):
    predictions = []
    input_step = x[:, 0:1, :]
    for t in range(y_true.size(1)):
        pred = model(input_step)
        predictions.append(pred)
        if np.random.random() < teacher_forcing_ratio:
            input_step = y_true[:, t:t+1, :]  # 用真实值
        else:
            input_step = pred.unsqueeze(1)      # 用预测值
    return torch.cat(predictions, dim=1)
```

---

**要点总结：**
- LSTM 通过遗忘门、输入门、输出门实现选择性记忆
- 单元状态的加法更新是避免梯度消失的关键
- 多变量 LSTM 可以同时利用多种相关时序信号
- 特征工程（滞后、滚动统计、时间编码）能显著提升 LSTM 性能
