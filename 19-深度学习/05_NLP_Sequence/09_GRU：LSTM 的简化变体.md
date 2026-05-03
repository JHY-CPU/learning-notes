# 09_GRU：LSTM 的简化变体

## 核心概念

- **GRU (Gated Recurrent Unit)**：由 Cho 等人于 2014 年提出，是 LSTM 的简化变体。将遗忘门和输入门合并为"更新门"，同时合并细胞状态和隐藏状态。
- **更新门 (Update Gate)**：$z_t$ 同时控制遗忘旧信息和记忆新信息的比例，相当于 LSTM 中遗忘门和输入门的统一。$z_t$ 接近 1 时保留更多旧信息，接近 0 时更新更多新信息。
- **重置门 (Reset Gate)**：$r_t$ 控制过去的状态对当前候选状态的影响程度。$r_t$ 接近 0 时忽略过去状态（允许模型丢弃不相关信息），接近 1 时保留完整历史。
- **门控机制简化**：GRU 只有两个门（更新门和重置门），而 LSTM 有三个门（遗忘门、输入门、输出门），参数量减少约 1/3。
- **参数效率**：GRU 在参数量较少的情况下性能通常与 LSTM 相当，在小数据集上往往表现更好，在大规模任务中两者差距不大。
- **无独立细胞状态**：GRU 没有独立的细胞状态 $C_t$，直接将隐藏状态 $h_t$ 同时作为输出和记忆载体，结构更简洁。

## 数学推导

GRU 的前向传播公式：

$$
z_t = \sigma(W_z \cdot [h_{t-1}, x_t] + b_z) \quad \text{(更新门)}
$$
$$
r_t = \sigma(W_r \cdot [h_{t-1}, x_t] + b_r) \quad \text{(重置门)}
$$
$$
\tilde{h}_t = \tanh(W_h \cdot [r_t \odot h_{t-1}, x_t] + b_h) \quad \text{(候选隐藏状态)}
$$
$$
h_t = (1 - z_t) \odot h_{t-1} + z_t \odot \tilde{h}_t \quad \text{(最终隐藏状态)}
$$

其中 $z_t$ 是更新门，$r_t$ 是重置门。关键区别在于 $z_t$ 控制新旧信息的混合比例——$z_t$ 接近 1 时偏向新信息，接近 0 时保留旧信息。

注意最终的 $h_t$ 是 $h_{t-1}$ 和 $\tilde{h}_t$ 的凸组合，因为 $z_t$ 的输出在 0 到 1 之间，这与 LSTM 中 $C_t$ 的更新方式类似。

## 直观理解

- **GRU 像 LSTM 的精简版本**：LSTM 像功能齐全的瑞士军刀（三个门、两个状态），而 GRU 像一把精简的折叠刀（两个门、一个状态），在多数场景下同样好用但更轻便。
- **更新门像"记忆力调节旋钮"**：设定一个旋钮，往左拧（$z_t \to 0$）完全记住旧内容，往右拧（$z_t \to 1$）完全更新为新内容。它同时控制了"忘记多少"和"记住多少"。
- **重置门像"专注力开关"**：当你接收新信息时，是否要参考过去经验。$r_t=0$ 时完全忽略历史（像失忆后重新学习），$r_t=1$ 时结合历史理解新知。

## 代码示例

```python
import torch
import torch.nn as nn

class GRUCell(nn.Module):
    """手动实现 GRU 单元"""
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.hidden_size = hidden_size
        # 更新门和重置门
        self.gate_z = nn.Linear(input_size + hidden_size, hidden_size)
        self.gate_r = nn.Linear(input_size + hidden_size, hidden_size)
        # 候选隐藏状态
        self.gate_h = nn.Linear(input_size + hidden_size, hidden_size)

    def forward(self, x, h_prev):
        combined = torch.cat([x, h_prev], dim=1)

        z = torch.sigmoid(self.gate_z(combined))  # 更新门
        r = torch.sigmoid(self.gate_r(combined))  # 重置门

        # 候选隐藏状态（使用重置后的历史）
        combined_reset = torch.cat([x, r * h_prev], dim=1)
        h_hat = torch.tanh(self.gate_h(combined_reset))

        # 新隐藏状态
        h_new = (1 - z) * h_prev + z * h_hat
        return h_new

# 使用 PyTorch 内置 GRU
gru = nn.GRU(input_size=128, hidden_size=256, num_layers=2, batch_first=True)
x = torch.randn(4, 20, 128)              # (batch, seq_len, input_size)
output, h_n = gru(x)
print("GRU 输出形状:", output.shape)      # (4, 20, 256)
print("隐藏状态形状:", h_n.shape)          # (2, 4, 256)

# GRU vs LSTM 参数量对比
lstm = nn.LSTM(128, 256)
gru_model = nn.GRU(128, 256)
print(f"LSTM 参数量: {sum(p.numel() for p in lstm.parameters())}")
print(f"GRU 参数量: {sum(p.numel() for p in gru_model.parameters())}")
```

## 深度学习关联

- **门控思想的延续**：GRU 的简化哲学影响了后续许多模型设计——保持核心功能的同时减少参数量和计算复杂度，如 ALBERT（跨层参数共享）也遵循了类似的"轻量化"思路。
- **编码器-解码器中的应用**：GRU 在与注意力机制结合的 Seq2Seq 模型（如神经机器翻译）中表现出色，Cho 等人最初提出 GRU 正是用于机器翻译任务。
- **与现代架构的对比**：GRU 和 LSTM 的门控思想在现代架构中以新的形式出现，如状态空间模型（Mamba）中的选择机制本质上也是一种门控机制，控制信息的选择性通过。
