# 9_Autoformer：自相关机制

## 1. 核心思想

Autoformer (NeurIPS 2021) 提出用**序列自相关**替代自注意力，将深度学习中的注意力机制与传统时间序列分析的自相关概念结合。

**核心动机：** 时间序列具有强周期性，相似的子序列会在不同的时间点重复出现。自相关可以高效地发现这种周期模式。

## 2. 自相关机制 (Auto-Correlation)

### 2.1 从自相关到注意力

标准自注意力：$Attention(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right)V$

Autoformer 的自相关：基于时域相似性进行值的**时延聚合 (Time-delay Aggregation)**：

$$\hat{R}_{\tau_1}, \ldots, \hat{R}_{\tau_k} = \text{TopK}(R(\mathcal{X}_Q, \mathcal{X}_K))$$
$$\text{Auto-Correlation}(Q, K, V) = \sum_{i=1}^{k} \text{SoftMax}(\hat{R}_{\tau_i}) \cdot \text{Shift}(V, \tau_i)$$

其中 $R(\tau)$ 是自相关函数，度量序列与其时延版本的相似度：

$$R_{\mathcal{X}_Q, \mathcal{X}_K}(\tau) = \frac{1}{L} \sum_{t=1}^{L-\tau} \mathcal{X}_{Q,t} \cdot \mathcal{X}_{K,t+\tau}$$

### 2.2 基于 FFT 的高效计算

时域卷积等于频域乘法（Wiener-Khinchin 定理）：

$$R(\tau) = \mathcal{F}^{-1}\left(\mathcal{F}(\mathcal{X}_Q) \cdot \overline{\mathcal{F}(\mathcal{X}_K)}\right)$$

复杂度：$O(L \log L)$（FFT 计算），比 $O(L^2)$ 的注意力矩阵计算高效得多。

```python
import torch
import torch.nn as nn
import torch.fft as fft

class AutoCorrelation(nn.Module):
    def __init__(self, d_model, n_heads, factor=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.factor = factor  # Top-K 时延数

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def time_delay_agg(self, corr, V, top_k):
        """
        基于自相关系数的 Top-K 时延聚合
        corr: (B, H, L) 自相关函数
        V: (B, H, L, d_k)
        """
        B, H, L = corr.shape

        # 选择 Top-K 时延
        top_k = min(top_k, L)
        weights, delays = torch.topk(corr, top_k, dim=-1)  # (B, H, k)
        weights = torch.softmax(weights, dim=-1)

        # 对每个时延进行 shift 操作
        # 等价于 rolling / index shifting
        batch_idx = torch.arange(B).view(-1, 1, 1)
        head_idx = torch.arange(H).view(1, -1, 1)

        # 初始化聚合结果
        result = torch.zeros_like(V)
        for i in range(top_k):
            delay = delays[:, :, i]  # (B, H)
            weight = weights[:, :, i].unsqueeze(-1).unsqueeze(-1)  # (B, H, 1, 1)
            # 循环移位
            for b in range(B):
                for h in range(H):
                    d = delay[b, h].item()
                    result[b, h] += weight[b, h, 0, 0] * torch.roll(V[b, h], shifts=d, dims=0)

        return result

    def forward(self, Q, K, V):
        B, L, _ = Q.shape
        Q = self.W_q(Q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # FFT 计算自相关
        Q_fft = fft.rfft(Q, dim=-2)
        K_fft = fft.rfft(K, dim=-2)
        corr = fft.irfft(Q_fft * K_fft.conj(), n=L, dim=-2)  # (B, H, L, d_k)
        # 在 d_k 维度上求和得到自相关函数
        corr = corr.sum(dim=-1)  # (B, H, L)

        # 时延聚合
        result = self.time_delay_agg(corr, V, self.factor * int(torch.log(
            torch.tensor(L, dtype=torch.float)).item()))

        result = result.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(result)
```

## 3. 季节性分解模块

Autoformer 在每个解码器层中嵌入**渐进式分解 (Progressive Decomposition)**：

```
输入序列 → [Auto-Correlation] → [Series Decomposition] → 趋势 + 季节性
                                                ↓
                                    趋势分量 → 下一层处理
                                    季节性分量 → 继续变换
```

```python
class SeriesDecomposition(nn.Module):
    """移动平均分解：趋势 + 季节性"""
    def __init__(self, kernel_size=25):
        super().__init__()
        self.moving_avg = nn.AvgPool1d(kernel_size, stride=1,
                                        padding=kernel_size // 2)

    def forward(self, x):
        # x: (B, L, D)
        x_t = x.transpose(1, 2)  # (B, D, L)
        trend = self.moving_avg(x_t).transpose(1, 2)
        # 确保长度匹配
        trend = trend[:, :x.size(1), :]
        seasonal = x - trend
        return seasonal, trend
```

## 4. Autoformer 完整架构

```python
class AutoformerEncoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        self.auto_corr = AutoCorrelation(d_model, n_heads)
        self.decomp1 = SeriesDecomposition(kernel_size)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )
        self.decomp2 = SeriesDecomposition(kernel_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # 自相关 + 第一次分解
        attn_out = self.auto_corr(x, x, x)
        x = x + self.dropout(attn_out)
        seasonal1, trend1 = self.decomp1(x)

        # 前馈网络 + 第二次分解
        ff_out = self.ff(seasonal1)
        seasonal2, trend2 = self.decomp2(seasonal1 + self.dropout(ff_out))

        return seasonal2, trend1 + trend2


class AutoformerDecoderLayer(nn.Module):
    def __init__(self, d_model, n_heads, d_ff, kernel_size, dropout=0.1):
        super().__init__()
        self.auto_corr1 = AutoCorrelation(d_model, n_heads)
        self.decomp1 = SeriesDecomposition(kernel_size)

        self.auto_corr2 = AutoCorrelation(d_model, n_heads)
        self.decomp2 = SeriesDecomposition(kernel_size)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff), nn.ReLU(), nn.Linear(d_ff, d_model))
        self.decomp3 = SeriesDecomposition(kernel_size)
        self.dropout = nn.Dropout(dropout)

        # 趋势累加投影
        self.trend_proj = None

    def forward(self, x, enc_out):
        # 自注意力 + 分解
        attn1 = self.auto_corr1(x, x, x)
        x = x + self.dropout(attn1)
        x, trend1 = self.decomp1(x)

        # 交叉自相关 + 分解
        attn2 = self.auto_corr2(x, enc_out, enc_out)
        x = x + self.dropout(attn2)
        x, trend2 = self.decomp2(x)

        # 前馈 + 分解
        ff_out = self.ff(x)
        x = x + self.dropout(ff_out)
        seasonal, trend3 = self.decomp3(x)

        return seasonal, trend1 + trend2 + trend3


class Autoformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len,
                 pred_len, d_model=512, n_heads=8, e_layers=2,
                 d_layers=1, d_ff=2048, kernel_size=25, dropout=0.1):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len

        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)

        # 编码器
        self.encoder = nn.ModuleList([
            AutoformerEncoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(e_layers)
        ])

        # 解码器
        self.decoder = nn.ModuleList([
            AutoformerDecoderLayer(d_model, n_heads, d_ff, kernel_size, dropout)
            for _ in range(d_layers)
        ])

        # 最终趋势预测
        self.trend_proj = nn.Linear(label_len + pred_len, pred_len)
        self.projection = nn.Linear(d_model, c_out)

    def forward(self, enc_x, dec_x):
        # 嵌入
        enc_out = self.enc_embedding(enc_x)
        dec_out = self.dec_embedding(dec_x)

        # 编码器
        for layer in self.encoder:
            enc_out, trend_enc = layer(enc_out)

        # 解码器
        for layer in self.decoder:
            dec_out, trend_dec = layer(dec_out, enc_out)

        # 最终输出：季节性 + 趋势
        seasonal = self.projection(dec_out[:, -self.pred_len:, :])
        # 趋势通过全局池化得到
        trend = self.trend_proj(dec_out.mean(dim=-1)).unsqueeze(-1)

        return seasonal + trend
```

## 5. 自相关 vs 自注意力

| 特性 | 自注意力 | 自相关 |
|------|---------|-------|
| 信息来源 | 语义相似性 | 时域模式相似性 |
| 计算方式 | $QK^T$ 点积 | FFT 自相关 |
| 复杂度 | $O(L^2 d)$ | $O(L \log L \cdot d)$ |
| 聚合方式 | 加权求和 | 时延聚合 |
| 周期性 | 不直接建模 | 天然捕获周期 |

## 6. 实战使用

```python
model = Autoformer(
    enc_in=7, dec_in=7, c_out=1,
    seq_len=96, label_len=48, pred_len=96,
    d_model=512, n_heads=8,
    e_layers=2, d_layers=1,
    d_ff=2048, kernel_size=25
)

enc_x = torch.randn(32, 96, 7)
dec_x = torch.randn(32, 144, 7)  # 48 + 96
output = model(enc_x, dec_x)     # (32, 96, 1)
```

---

**要点总结：**
- 自相关机制利用 FFT 高效计算序列间的时域相似性，复杂度 $O(L \log L)$
- 时延聚合通过 Top-K 自相关偏移量对值进行加权聚合
- 渐进式分解在每一层中分离趋势和季节性，使模型逐步精化预测
- Autoformer 特别适合具有明显周期性的时间序列
