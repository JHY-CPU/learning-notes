# RNN与LSTM时间序列


## 一、为什么 RNN 适合时间序列


循环神经网络（RNN）通过隐藏状态的递归传递，天然适合处理序列数据。与前馈网络不同，RNN 具有"记忆"能力，能够利用历史信息来影响当前输出。


### 1.1 RNN 的基本结构


$$
ht = tanh(Whhht-1 + Wxhxt + bh)
                yt = Whyht + by
$$


### 1.2 RNN 面临的问题


| 问题 | 描述 | 后果 |
| --- | --- | --- |
| **梯度消失** | 反向传播时梯度指数衰减 | 无法学习长距离依赖 |
| **梯度爆炸** | 反向传播时梯度指数增长 | 训练不稳定、参数溢出 |
| **长期记忆困难** | 隐藏状态不断被覆盖 | 遗忘早期信息 |


> **Note:** **LSTM 的出现正是为了解决这些问题。**
> 通过引入门控机制，LSTM 能够有选择地记忆和遗忘信息，有效缓解梯度消失问题。


## 二、LSTM 门控机制回顾


LSTM（Long Short-Term Memory）通过三个门和一个细胞状态来管理信息流。


### 2.1 遗忘门（Forget Gate）


$$
ft = σ(Wf[ht-1, xt] + bf)
$$


决定从细胞状态中丢弃哪些信息。输出值接近0表示遗忘，接近1表示保留。


### 2.2 输入门（Input Gate）


$$
it = σ(Wi[ht-1, xt] + bi)
                C̃t = tanh(WC[ht-1, xt] + bC)
$$


决定哪些新信息要存入细胞状态。


### 2.3 细胞状态更新


$$
Ct = ft ⊙ Ct-1 + it ⊙ C̃t
$$


### 2.4 输出门（Output Gate）


$$
ot = σ(Wo[ht-1, xt] + bo)
                ht = ot ⊙ tanh(Ct)
$$


> **Important:** **关键点：**
> 细胞状态 C
> ~t~
> 像一条"传送带"，信息可以沿其无损传递。三个门通过 sigmoid 函数（输出0~1）来控制信息流的比例，这使得梯度可以在长序列中有效传播。


## 三、时间序列预测的三种模式


| 模式 | 输入 | 输出 | 应用场景 |
| --- | --- | --- | --- |
| **单步预测** | x~t-w~, ..., x~t~ | x~t+1~ | 短期预测 |
| **多步预测（递归）** | x~t-w~, ..., x~t~ | x~t+1~, ..., x~t+H~ | 中期预测，误差累积 |
| **多步预测（直接）** | x~t-w~, ..., x~t~ | x~t+1~, ..., x~t+H~ | 输出层有H个神经元 |
| **多变量预测** | 多特征序列 | 目标变量序列 | 综合多因素预测 |


## 四、滑动窗口构造训练样本


将时间序列转化为监督学习问题的核心技术是**滑动窗口**：用过去 w 个时间步的观测值作为输入，下一个（或多个）时间步的值作为目标。


$$
窗口大小 w=5 的示例：
                输入 [x1, x2, x3, x4, x5] → 目标 x6
                输入 [x2, x3, x4, x5, x6] → 目标 x7
                ...
$$


| 超参数 | 说明 | 建议 |
| --- | --- | --- |
| 窗口大小 w | 输入序列长度 | 覆盖至少一个完整周期 |
| 预测步长 H | 预测未来多少步 | 根据业务需求设定 |
| 特征缩放 | 标准化或归一化 | LSTM对输入尺度敏感，必须缩放 |


## 五、Seq2Seq 架构用于时间序列


Encoder-Decoder（Seq2Seq）架构是处理序列到序列映射的经典结构，特别适合多步时间序列预测。


### 架构说明


- **Encoder：**
   LSTM将输入序列压缩为固定长度的上下文向量（最后一个隐藏状态）
- **Decoder：**
   LSTM从上下文向量开始，逐步生成预测序列
- **Teacher Forcing：**
   训练时使用真实值作为decoder输入，推理时使用预测值


> **Note:** **改进方向：**
> Attention机制可以让decoder在每一步都能关注encoder的不同时间步，避免信息瓶颈。这正是Transformer在时间序列中的应用基础。


## 六、PyTorch 实战：LSTM 预测时间序列


> **Example:** ### 示例：LSTM单变量时间序列预测
>
>
> ```
> import torch
> import torch.nn as nn
> import numpy as np
> import matplotlib.pyplot as plt
> from sklearn.preprocessing import MinMaxScaler
>
> # 1. 生成模拟数据
> np.random.seed(42)
> t = np.linspace(0, 100, 2000)
> data = np.sin(0.1 * t) + 0.5 * np.sin(0.05 * t) + np.random.normal(0, 0.05, 2000)
>
> # 2. 数据预处理
> scaler = MinMaxScaler(feature_range=(0, 1))
> data_scaled = scaler.fit_transform(data.reshape(-1, 1)).flatten()
>
> def create_sequences(data, window_size):
>     X, y = [], []
>     for i in range(len(data) - window_size):
>         X.append(data[i:i + window_size])
>         y.append(data[i + window_size])
>     return np.array(X), np.array(y)
>
> WINDOW = 50
> X, y = create_sequences(data_scaled, WINDOW)
>
> # 划分数据集
> train_size = int(len(X) * 0.8)
> X_train, X_test = X[:train_size], X[train_size:]
> y_train, y_test = y[:train_size], y[train_size:]
>
> # 转为PyTorch张量 (batch, seq_len, features)
> X_train_t = torch.FloatTensor(X_train).unsqueeze(-1)
> y_train_t = torch.FloatTensor(y_train)
> X_test_t = torch.FloatTensor(X_test).unsqueeze(-1)
> y_test_t = torch.FloatTensor(y_test)
>
> # 3. 定义LSTM模型
> class LSTMForecaster(nn.Module):
>     def __init__(self, input_size=1, hidden_size=64, num_layers=2, output_size=1):
>         super().__init__()
>         self.hidden_size = hidden_size
>         self.num_layers = num_layers
>         self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
>                            batch_first=True, dropout=0.2)
>         self.fc = nn.Linear(hidden_size, output_size)
>
>     def forward(self, x):
>         # 初始化隐藏状态
>         h0 = torch.zeros(self.num_layers, x.size(0),
>                          self.hidden_size).to(x.device)
>         c0 = torch.zeros(self.num_layers, x.size(0),
>                          self.hidden_size).to(x.device)
>
>         out, _ = self.lstm(x, (h0, c0))
>         out = self.fc(out[:, -1, :])  # 取最后一个时间步
>         return out
>
> # 4. 训练模型
> model = LSTMForecaster()
> criterion = nn.MSELoss()
> optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
> scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
>     optimizer, patience=5, factor=0.5)
>
> EPOCHS = 100
> BATCH_SIZE = 32
> dataset = torch.utils.data.TensorDataset(X_train_t, y_train_t)
> loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE,
>                                       shuffle=True)
>
> train_losses = []
> for epoch in range(EPOCHS):
>     model.train()
>     epoch_loss = 0
>     for batch_X, batch_y in loader:
>         optimizer.zero_grad()
>         pred = model(batch_X).squeeze()
>         loss = criterion(pred, batch_y)
>         loss.backward()
>         torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
>         optimizer.step()
>         epoch_loss += loss.item()
>
>     avg_loss = epoch_loss / len(loader)
>     train_losses.append(avg_loss)
>     scheduler.step(avg_loss)
>
>     if (epoch + 1) % 20 == 0:
>         print(f"Epoch [{epoch+1}/{EPOCHS}], Loss: {avg_loss:.6f}")
>
> # 5. 预测与可视化
> model.eval()
> with torch.no_grad():
>     pred_test = model(X_test_t).squeeze().numpy()
>
> # 反归一化
> pred_original = scaler.inverse_transform(pred_test.reshape(-1, 1)).flatten()
> y_test_original = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()
>
> # 绘图
> fig, axes = plt.subplots(1, 2, figsize=(14, 5))
> axes[0].plot(y_test_original[:200], label='真实值', alpha=0.8)
> axes[0].plot(pred_original[:200], label='预测值', alpha=0.8)
> axes[0].set_title("LSTM预测结果")
> axes[0].legend()
>
> axes[1].plot(train_losses)
> axes[1].set_title("训练损失曲线")
> axes[1].set_xlabel("Epoch")
> axes[1].set_ylabel("MSE Loss")
>
> plt.tight_layout()
> plt.savefig("lstm_forecast.png", dpi=150)
> plt.show()
>
> # 6. 评估指标
> from sklearn.metrics import mean_absolute_error, mean_squared_error
> mae = mean_absolute_error(y_test_original, pred_original)
> rmse = np.sqrt(mean_squared_error(y_test_original, pred_original))
> print(f"\n=== LSTM预测评估 ===")
> print(f"MAE: {mae:.4f}")
> print(f"RMSE: {rmse:.4f}")
> ```


## 总结


- RNN的循环结构天然适合时间序列，但存在梯度消失问题
- LSTM通过遗忘门、输入门、输出门和细胞状态管理长期依赖
- 时间序列预测分为单步、多步递归、多步直接和多变量预测
- 滑动窗口是将时间序列转化为监督学习问题的核心技术
- Seq2Seq架构适合多步预测，可结合Attention机制提升效果
- 实际使用时必须进行特征缩放，并使用梯度裁剪防止梯度爆炸


<!-- Converted from: 01_RNN与LSTM时间序列.html -->
