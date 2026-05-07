# 13_Temporal Fusion Transformer (TFT)

## 1. 概述

TFT (Google, 2019) 是一个专为多变量时序预测设计的**可解释**深度学习模型，结合了 LSTM 的序列建模能力和 Transformer 的注意力机制，特别强调对不同变量重要性的自动选择。

**核心创新：**
1. **变量选择网络 (Variable Selection Network)**：自动选择重要变量
2. **门控残差网络 (Gated Residual Network)**：自适应控制信息流
3. **可解释多头注意力**：揭示时间步的重要性
4. **静态协变量编码器**：融合不随时间变化的信息

## 2. 输入特征类型

TFT 明确区分三类输入特征：

| 类型 | 含义 | 示例 | 处理方式 |
|------|------|------|---------|
| 静态变量 | 不随时间变化 | 商品类别、地理位置 | 编码后广播 |
| 已知未来 | 未来已确定的信息 | 节假日、星期几 | 直接使用 |
| 仅过去 | 只有历史可观察 | 历史销量、实时温度 | 需要 LSTM 编码 |

## 3. 门控残差网络 (GRN)

GRN 是 TFT 的基本构建块，通过 sigmoid 门控实现信息的自适应过滤：

```python
import torch
import torch.nn as nn

class GatedResidualNetwork(nn.Module):
    """门控残差网络：核心构建块"""
    def __init__(self, d_model, d_hidden, dropout=0.1, context_dim=None):
        super().__init__()
        self.fc1 = nn.Linear(d_model, d_hidden)
        self.elu = nn.ELU()
        self.fc2 = nn.Linear(d_hidden, d_model)
        self.dropout = nn.Dropout(dropout)

        # 上下文投影（可选）
        self.context_proj = (nn.Linear(context_dim, d_hidden, bias=False)
                             if context_dim else None)

        # 门控层
        self.gate = nn.Linear(d_hidden, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, context=None):
        # 主路径
        residual = x
        hidden = self.fc1(x)
        if self.context_proj and context is not None:
            hidden = hidden + self.context_proj(context)
        hidden = self.elu(hidden)
        hidden = self.fc2(hidden)
        hidden = self.dropout(hidden)

        # 门控
        gate = torch.sigmoid(self.gate(self.elu(self.fc1(x))))
        output = self.layer_norm(residual + gate * hidden)
        return output
```

## 4. 变量选择网络 (VSN)

VSN 自动评估每个输入变量的重要性，选择最有信息量的特征子集：

```python
class VariableSelectionNetwork(nn.Module):
    """变量选择网络"""
    def __init__(self, n_vars, d_model, d_hidden, dropout=0.1):
        super().__init__()
        self.n_vars = n_vars
        self.d_model = d_model

        # 每个变量的 GRN 嵌入
        self.var_grns = nn.ModuleList([
            GatedResidualNetwork(d_model, d_hidden, dropout)
            for _ in range(n_vars)
        ])

        # 变量权重 GRN（输出 n_vars 个权重）
        self.weight_grn = GatedResidualNetwork(
            n_vars * d_model, d_hidden, dropout
        )
        self.weight_proj = nn.Linear(d_hidden, n_vars)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, variables):
        """
        variables: list of (B, L, d_model) or (B, d_model) 每个变量
        """
        # 每个变量经过独立 GRN
        processed = []
        for i, var in enumerate(variables):
            processed.append(self.var_grns[i](var))

        # 拼接所有变量用于权重计算
        if variables[0].dim() == 3:  # (B, L, d_model)
            concat = torch.cat(processed, dim=-1)  # (B, L, n_vars*d_model)
        else:
            concat = torch.cat(processed, dim=-1)  # (B, n_vars*d_model)

        # 计算变量权重
        weights_hidden = self.weight_grn(concat)
        weights = self.softmax(self.weight_proj(weights_hidden))

        # 加权融合
        if variables[0].dim() == 3:
            stacked = torch.stack(processed, dim=-2)  # (B, L, n_vars, d_model)
            weights = weights.unsqueeze(-1)  # (B, L, n_vars, 1)
        else:
            stacked = torch.stack(processed, dim=-2)  # (B, n_vars, d_model)
            weights = weights.unsqueeze(-1)

        output = (stacked * weights).sum(dim=-2)
        return output, weights
```

## 5. 时序处理：LSTM 编码 + 解码

```python
class TemporalProcessing(nn.Module):
    """历史 LSTM 编码 + 未来 LSTM 编码"""
    def __init__(self, d_model, n_layers=1):
        super().__init__()
        self.history_lstm = nn.LSTM(d_model, d_model, n_layers,
                                     batch_first=True, dropout=0.1)
        self.future_lstm = nn.LSTM(d_model, d_model, n_layers,
                                    batch_first=True, dropout=0.1)

    def forward(self, history, future):
        # history: (B, seq_len, d_model)
        # future: (B, pred_len, d_model)
        enc_out, (h, c) = self.history_lstm(history)
        dec_out, _ = self.future_lstm(future, (h, c))
        return enc_out, dec_out
```

## 6. 可解释多头注意力

TFT 使用可解释的多头注意力，其中每个头独立输出，便于分析每个时间步的重要性：

```python
class InterpretableMultiHeadAttention(nn.Module):
    """可解释多头注意力：共享 V 矩阵，便于解释"""
    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)  # 共享 V
        self.W_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, Q, K, V, mask=None):
        B, L, _ = Q.shape

        Q = self.W_q(Q).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, -1, self.n_heads, self.d_k).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.d_k ** 0.5)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)

        attn = self.dropout(torch.softmax(scores, dim=-1))
        context = torch.matmul(attn, V)

        # 可解释性：所有头共享相同的 V，注意力模式更容易解释
        context = context.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(context), attn.detach()
```

## 7. TFT 完整架构

```python
class TemporalFusionTransformer(nn.Module):
    def __init__(self, n_past_vars, n_future_vars, n_static_vars,
                 d_model, d_hidden, n_heads, n_quantiles=3,
                 seq_len=168, pred_len=24):
        super().__init__()
        self.pred_len = pred_len
        self.n_quantiles = n_quantiles

        # 静态变量编码
        self.static_vsn = VariableSelectionNetwork(
            n_static_vars, d_model, d_hidden)

        # 历史变量选择
        self.past_vsn = VariableSelectionNetwork(
            n_past_vars, d_model, d_hidden)

        # 未来变量选择
        self.future_vsn = VariableSelectionNetwork(
            n_future_vars, d_model, d_hidden)

        # 时序处理
        self.temporal_process = TemporalProcessing(d_model)

        # 静态-时序融合 GRN
        self.static_enrichment = GatedResidualNetwork(
            d_model, d_hidden, context_dim=d_model)

        # 可解释注意力
        self.attention = InterpretableMultiHeadAttention(d_model, n_heads)

        # 位置级 GRN
        self.position_grn = GatedResidualNetwork(d_model, d_hidden)

        # 输出层（分位数回归）
        self.quantile_proj = nn.Linear(d_model, pred_len * n_quantiles)

    def forward(self, past_vars, future_vars, static_vars):
        """
        past_vars: list of (B, seq_len, d) 历史变量
        future_vars: list of (B, pred_len, d) 未来已知变量
        static_vars: list of (B, d) 静态变量
        """
        # 1. 静态编码
        static_embed, static_weights = self.static_vsn(static_vars)  # (B, d_model)

        # 2. 变量选择
        past_selected, past_weights = self.past_vsn(past_vars)
        future_selected, future_weights = self.future_vsn(future_vars)

        # 3. LSTM 时序编码
        enc_out, dec_out = self.temporal_process(past_selected, future_selected)
        temporal = torch.cat([enc_out, dec_out], dim=1)  # (B, seq_len+pred_len, d)

        # 4. 静态信息丰富
        static_context = static_embed.unsqueeze(1).expand_as(temporal)
        enriched = self.static_enrichment(temporal, static_context)

        # 5. 自注意力（仅看历史部分用于预测）
        mask = torch.triu(torch.ones(temporal.size(1), temporal.size(1)),
                          diagonal=1).bool().to(temporal.device)
        attn_out, attn_weights = self.attention(enriched, enriched, enriched, ~mask)

        # 6. 位置级处理
        output = self.position_grn(enriched + attn_out)

        # 7. 只取预测部分并输出分位数
        pred_output = output[:, -self.pred_len:, :]
        quantile_pred = self.quantile_proj(pred_output)  # (B, pred_len, n_quantiles)

        return quantile_pred, {
            'static_weights': static_weights,
            'past_var_weights': past_weights,
            'future_var_weights': future_weights,
            'attention_weights': attn_weights
        }
```

## 8. TFT 的可解释性

TFT 的最大优势在于提供多层级的可解释信息：

```python
# 获取可解释性信息
quantile_pred, interprets = model(past_vars, future_vars, static_vars)

# 1. 变量重要性
print("历史变量重要性:", interprets['past_var_weights'].mean(dim=(0,1)))
# 输出: [0.35, 0.25, 0.15, 0.10, 0.08, 0.07] -> 第1个变量最重要

# 2. 注意力权重（哪些历史时间步对预测最重要）
attn_map = interprets['attention_weights'].mean(dim=1)  # (L, L)
# 热力图可视化，可发现关键时间模式

# 3. 分位数预测（不确定性）
lower = quantile_pred[:, :, 0]   # 10% 分位数
median = quantile_pred[:, :, 1]  # 50% 分位数
upper = quantile_pred[:, :, 2]   # 90% 分位数
```

---

**要点总结：**
- TFT 通过变量选择网络自动识别重要变量，提供变量级别的可解释性
- GRN 门控机制自适应控制信息流，避免不相关信息干扰
- 可解释多头注意力揭示时间步层面的预测依据
- 分位数输出支持概率预测和不确定性量化
- 特别适合业务场景中需要"解释预测原因"的需求
