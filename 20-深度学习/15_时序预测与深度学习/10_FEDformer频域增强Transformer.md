# 10_FEDformer：频域增强 Transformer

## 1. 核心思想

FEDformer (ICML 2022) 将注意力机制扩展到**频域**，利用傅里叶/小波变换揭示时序数据的频域模式，实现全局视角的高效注意力。

**动机：**
- 时序数据在频域中有稀疏的频谱表示
- 频域操作天然支持全局感受野
- 主要频率成分对应趋势、季节性等宏观模式

## 2. 频域注意力

### 2.1 傅里叶变换回顾

$$X(f) = \mathcal{F}(x) = \sum_{t=0}^{L-1} x_t \cdot e^{-i 2\pi f t / L}$$
$$x(t) = \mathcal{F}^{-1}(X) = \frac{1}{L} \sum_{f=0}^{L-1} X_f \cdot e^{i 2\pi f t / L}$$

**关键性质：** 时域卷积 = 频域乘法，频域乘法天然具有全局感受野。

### 2.2 频域注意力机制

```
输入 Q, K, V
    ↓
[FFT] → Q_f, K_f, V_f (频域表示)
    ↓
频域注意力: softmax(Q_f · K_f* / √d) · V_f
    ↓
[iFFT] → 输出
```

```python
import torch
import torch.nn as nn
import torch.fft as fft

class FourierAttention(nn.Module):
    """傅里叶频域注意力"""
    def __init__(self, d_model, n_heads):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

    def forward(self, Q, K, V):
        B, L, _ = Q.shape

        # 线性变换并分头
        Q = self.W_q(Q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 傅里叶变换到频域
        Q_f = fft.rfft(Q, dim=-2)    # (B, H, L//2+1, d_k) 复数
        K_f = fft.rfft(K, dim=-2)
        V_f = fft.rfft(V, dim=-2)

        # 频域注意力（复数运算）
        # 使用实部计算分数
        scores_real = torch.matmul(Q_f.real, K_f.real.transpose(-2, -1))
        scores_imag = torch.matmul(Q_f.imag, K_f.imag.transpose(-2, -1))
        scores = (scores_real + scores_imag) / (self.d_k ** 0.5)

        attn = torch.softmax(scores, dim=-1)

        # 频域加权
        out_real = torch.matmul(attn, V_f.real)
        out_imag = torch.matmul(attn, V_f.imag)
        out_f = torch.complex(out_real, out_imag)

        # 逆傅里叶变换回时域
        out = fft.irfft(out_f, n=L, dim=-2)  # (B, H, L, d_k)

        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)
```

### 2.3 小波变换注意力

傅里叶变换缺乏时间定位能力。小波变换同时提供时间和频率信息：

```python
import pywt

class WaveletAttention(nn.Module):
    """小波变换频域注意力"""
    def __init__(self, d_model, n_heads, wavelet='db4', level=3):
        super().__init__()
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.wavelet = wavelet
        self.level = level

        self.W_q = nn.Linear(d_model, d_model)
        self.W_k = nn.Linear(d_model, d_model)
        self.W_v = nn.Linear(d_model, d_model)
        self.W_o = nn.Linear(d_model, d_model)

        # 各层级的注意力权重
        self.attn_projs = nn.ModuleList([
            nn.Linear(self.d_k, self.d_k) for _ in range(level + 1)
        ])

    def wavelet_decompose(self, x):
        """小波分解"""
        B, H, L, d_k = x.shape
        coeffs_list = []
        for b in range(B):
            for h in range(H):
                for d in range(d_k):
                    signal = x[b, h, :, d].detach().cpu().numpy()
                    coeffs = pywt.wavedec(signal, self.wavelet, level=self.level)
                    coeffs_list.append(coeffs)
        return coeffs_list  # 需要更高效的实现

    def forward(self, Q, K, V):
        # 简化版：使用可学习的小波分解近似
        B, L, _ = Q.shape
        Q = self.W_q(Q).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        K = self.W_k(K).view(B, L, self.n_heads, self.d_k).transpose(1, 2)
        V = self.W_v(V).view(B, L, self.n_heads, self.d_k).transpose(1, 2)

        # 使用可学习的多尺度卷积模拟小波分解
        outputs = []
        for i in range(self.level):
            kernel_size = 2 ** (i + 1)
            # 近似系数（低通）
            avg_pool = nn.functional.avg_pool1d(
                Q.reshape(B * self.n_heads, self.d_k, L),
                kernel_size=kernel_size, stride=kernel_size
            )
            # 逐尺度注意力
            K_pool = nn.functional.avg_pool1d(
                K.reshape(B * self.n_heads, self.d_k, L),
                kernel_size=kernel_size, stride=kernel_size
            )
            V_pool = nn.functional.avg_pool1d(
                V.reshape(B * self.n_heads, self.d_k, L),
                kernel_size=kernel_size, stride=kernel_size
            )

            attn = torch.softmax(
                torch.matmul(avg_pool.transpose(1, 2), K_pool) /
                (self.d_k ** 0.5), dim=-1
            )
            scale_out = torch.matmul(attn, V_pool.transpose(1, 2))
            # 上采样回原始长度
            scale_out = nn.functional.interpolate(
                scale_out.transpose(1, 2), size=L, mode='linear'
            ).reshape(B, self.n_heads, self.d_k, L).transpose(2, 3)
            outputs.append(scale_out)

        out = sum(outputs) / self.level
        out = out.transpose(1, 2).contiguous().view(B, L, -1)
        return self.W_o(out)
```

## 3. FEDformer 完整架构

```python
class FEDformerBlock(nn.Module):
    """FEDformer 基本块：频域注意力 + 前馈网络"""
    def __init__(self, d_model, n_heads, d_ff, dropout=0.1,
                 attention_type='fourier'):
        super().__init__()
        if attention_type == 'fourier':
            self.attn = FourierAttention(d_model, n_heads)
        else:
            self.attn = WaveletAttention(d_model, n_heads)

        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model)
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, enc_out=None):
        # 自注意力或交叉注意力
        if enc_out is None:
            attn_out = self.attn(x, x, x)
        else:
            attn_out = self.attn(x, enc_out, enc_out)

        x = self.norm1(x + self.dropout(attn_out))
        x = self.norm2(x + self.dropout(self.ff(x)))
        return x


class FEDformer(nn.Module):
    def __init__(self, enc_in, dec_in, c_out, seq_len, label_len, pred_len,
                 d_model=512, n_heads=8, e_layers=2, d_layers=1,
                 d_ff=2048, dropout=0.1, attention_type='fourier'):
        super().__init__()
        self.pred_len = pred_len
        self.label_len = label_len

        self.enc_embedding = nn.Linear(enc_in, d_model)
        self.dec_embedding = nn.Linear(dec_in, d_model)

        self.encoder = nn.ModuleList([
            FEDformerBlock(d_model, n_heads, d_ff, dropout, attention_type)
            for _ in range(e_layers)
        ])
        self.decoder = nn.ModuleList([
            FEDformerBlock(d_model, n_heads, d_ff, dropout, attention_type)
            for _ in range(d_layers)
        ])

        self.projection = nn.Linear(d_model, c_out)

    def forward(self, enc_x, dec_x):
        enc_out = self.enc_embedding(enc_x)
        dec_out = self.dec_embedding(dec_x)

        for layer in self.encoder:
            enc_out = layer(enc_out)
        for layer in self.decoder:
            dec_out = layer(dec_out, enc_out)

        return self.projection(dec_out[:, -self.pred_len:, :])
```

## 4. 傅里叶 vs 小波注意力

| 特性 | 傅里叶注意力 | 小波注意力 |
|------|------------|-----------|
| 频率分辨率 | 全局（精确频率） | 多尺度（时间-频率联合） |
| 时间定位 | 无（全局） | 有（局部特征） |
| 计算效率 | 更高（FFT） | 中等（级联滤波） |
| 适用场景 | 强周期性数据 | 多尺度模式数据 |

## 5. 实战配置

```python
model = FEDformer(
    enc_in=7, dec_in=7, c_out=1,
    seq_len=96, label_len=48, pred_len=96,
    d_model=512, n_heads=8,
    e_layers=2, d_layers=1,
    attention_type='fourier'  # 或 'wavelet'
)

enc_x = torch.randn(32, 96, 7)
dec_x = torch.randn(32, 144, 7)
output = model(enc_x, dec_x)  # (32, 96, 1)
```

---

**要点总结：**
- 频域注意力利用 FFT 实现全局感受野，复杂度 $O(L \log L)$
- 傅里叶变换适合强周期性数据，小波变换适合多尺度模式
- FEDformer 将传统信号处理的频域分析与深度学习注意力相结合
- 频域操作天然适合时序数据的全局模式提取
