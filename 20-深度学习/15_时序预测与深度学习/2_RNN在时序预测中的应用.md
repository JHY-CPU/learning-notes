# 2_RNN 在时序预测中的应用

## 1. RNN 时序建模原理

循环神经网络天然适合时间序列，因为它通过隐藏状态维护时间记忆：

$$h_t = \tanh(W_{hh} h_{t-1} + W_{xh} x_t + b_h)$$
$$\hat{y}_t = W_{hy} h_t + b_y$$

```
时间步:    t-1        t         t+1
          ┌───┐     ┌───┐     ┌───┐
  x_t --> │ h │ --> │ h │ --> │ h │ --> y
          └───┘     └───┘     └───┘
           W_hh      W_hh      W_hh
```

**核心优势：** 可变长度输入、参数共享、记忆历史信息

## 2. 多步预测策略

### 2.1 递归策略 (Recursive)

用单步模型反复预测，将预测值作为下一步输入：

```python
# 递归多步预测
def recursive_forecast(model, last_window, horizon):
    predictions = []
    current_input = last_window.clone()
    for _ in range(horizon):
        with torch.no_grad():
            pred = model(current_input.unsqueeze(0))
        predictions.append(pred.item())
        # 滑动窗口：移除最早值，加入预测值
        current_input = torch.cat([current_input[1:], pred])
    return predictions
```

- **优点：** 只需训练一个模型
- **缺点：** 误差累积

### 2.2 直接策略 (Direct)

为每个预测步训练独立模型：

$$\hat{Y}_{t+h} = f_h(Y_t, Y_{t-1}, \ldots)$$

- **优点：** 无误差传播
- **缺点：** 需训练 H 个模型，计算量大

### 2.3 多输出策略 (Multi-output)

一次输出所有未来步的预测值：

```python
class MultiStepRNN(nn.Module):
    def __init__(self, input_size, hidden_size, horizon):
        super().__init__()
        self.rnn = nn.RNN(input_size, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, horizon)

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        out, _ = self.rnn(x)
        last_hidden = out[:, -1, :]   # 取最后时间步
        prediction = self.fc(last_hidden)  # (batch, horizon)
        return prediction
```

## 3. RNN 时序预测完整实现

```python
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader

# ========== 数据准备 ==========
class TimeSeriesDataset(Dataset):
    def __init__(self, series, input_window, output_window):
        self.series = torch.FloatTensor(series)
        self.input_window = input_window
        self.output_window = output_window

    def __len__(self):
        return len(self.series) - self.input_window - self.output_window + 1

    def __getitem__(self, idx):
        x = self.series[idx:idx + self.input_window]
        y = self.series[idx + self.input_window:
                        idx + self.input_window + self.output_window]
        return x.unsqueeze(-1), y  # x: (seq, 1), y: (horizon,)

# ========== 模型定义 ==========
class RNNForecaster(nn.Module):
    def __init__(self, input_size=1, hidden_size=64, num_layers=2,
                 output_size=1, dropout=0.1):
        super().__init__()
        self.rnn = nn.RNN(
            input_size, hidden_size, num_layers,
            batch_first=True, dropout=dropout, nonlinearity='tanh'
        )
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Linear(hidden_size // 2, output_size)
        )

    def forward(self, x):
        # x: (batch, seq_len, input_size)
        rnn_out, hidden = self.rnn(x)
        # 取最后一个时间步的输出
        last_out = rnn_out[:, -1, :]
        return self.fc(last_out)

# ========== 训练循环 ==========
def train_model(model, train_loader, val_loader, epochs=100, lr=1e-3):
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, patience=5, factor=0.5
    )
    criterion = nn.MSELoss()
    best_val_loss = float('inf')

    for epoch in range(epochs):
        model.train()
        train_loss = 0
        for x_batch, y_batch in train_loader:
            optimizer.zero_grad()
            pred = model(x_batch)
            loss = criterion(pred, y_batch)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            train_loss += loss.item()

        # 验证
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for x_val, y_val in val_loader:
                pred = model(x_val)
                val_loss += criterion(pred, y_val).item()

        avg_val = val_loss / len(val_loader)
        scheduler.step(avg_val)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            torch.save(model.state_dict(), 'best_rnn_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f'Epoch {epoch+1}/{epochs}, '
                  f'Train Loss: {train_loss/len(train_loader):.6f}, '
                  f'Val Loss: {avg_val:.6f}')

# ========== 使用示例 ==========
# 生成示例数据
t = np.linspace(0, 100, 2000)
series = np.sin(t) + 0.1 * np.random.randn(len(t))

# 划分数据
split = int(len(series) * 0.8)
train_data = series[:split]
val_data = series[split:]

train_dataset = TimeSeriesDataset(train_data, input_window=48, output_window=12)
val_dataset = TimeSeriesDataset(val_data, input_window=48, output_window=12)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=32)

model = RNNForecaster(input_size=1, hidden_size=64, num_layers=2, output_size=12)
train_model(model, train_loader, val_loader, epochs=50)
```

## 4. RNN 的梯度问题

**梯度消失/爆炸：** RNN 反向传播需要沿时间步展开（BPTT），梯度涉及连乘：

$$\frac{\partial h_t}{\partial h_1} = \prod_{k=2}^{t} \frac{\partial h_k}{\partial h_{k-1}} = \prod_{k=2}^{t} W_{hh}^T \cdot \text{diag}(\tanh'(\cdot))$$

- 若 $\|W_{hh}\| < 1$：梯度消失，无法学习长期依赖
- 若 $\|W_{hh}\| > 1$：梯度爆炸，训练不稳定

**缓解方法：**
- 梯度裁剪 (Gradient Clipping)
- 使用 LSTM/GRU 门控机制
- 合理的权重初始化

## 5. 特征缩放与预处理

```python
from sklearn.preprocessing import MinMaxScaler

# 时序数据必须按时间顺序缩放，不能打乱
scaler = MinMaxScaler(feature_range=(0, 1))
train_scaled = scaler.fit_transform(train_data.reshape(-1, 1))
val_scaled = scaler.transform(val_data.reshape(-1, 1))

# 预测后反变换
pred_original = scaler.inverse_transform(pred_scaled)
```

---

**要点总结：**
- RNN 通过隐藏状态自然建模序列依赖
- 多步预测有三种策略：递归（简单但误差累积）、直接（无累积但计算量大）、多输出（折中）
- 梯度消失是 RNN 的核心瓶颈，需要 LSTM/GRU 解决
- 数据预处理（归一化、窗口划分）对时序预测至关重要
