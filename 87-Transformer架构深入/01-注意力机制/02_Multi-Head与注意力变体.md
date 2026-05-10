# Multi-Head Attention 与注意力变体


# Multi-Head Attention 与注意力变体


#### 本节概览


Multi-Head Attention 通过并行运行多个注意力头，让模型同时关注不同子空间的信息。在此基础上，研究者们提出了多种注意力变体来提升性能或降低计算开销，包括交叉注意力、加性注意力、线性注意力和稀疏注意力等。


## 1. Multi-Head Attention（多头注意力）


### 1.1 核心思想


单头注意力只能学习一种注意力模式。Multi-Head Attention 将 Q、K、V 分别投影到 h 个不同的低维子空间，每个子空间独立执行注意力计算，最后将结果拼接。这样模型可以同时关注来自不同表示子空间的信息。


$$
MultiHead(Q, K, V) = Concat(head1, ..., headh) WO
        headi = Attention(QWiQ, KWiK, VWiV)
$$


### 1.2 Split 操作（分割）


给定完整维度的 Q ∈ ℝ^n×d~model~^，将其按头数 h 分割为 h 个子矩阵：


| 参数 | 含义 | 典型值（BERT-base） |
| --- | --- | --- |
| d~model~ | 模型总维度 | 768 |
| h | 注意力头数 | 12 |
| d~k~ = d~v~ | 每头维度 = d~model~/h | 768/12 = 64 |
| 总参数量 | 3 × d~model~ × d~k~ × h + d~model~² | ≈ 2.36M |


```
# Split 实现方式
# 方式1: 先做完整投影再 reshape
Q_full = X @ W_Q  # (n, d_model)
# reshape 为 (n, h, d_k)，然后转置为 (h, n, d_k)
Q_heads = Q_full.reshape(n, h, d_k).transpose(0, 1)

# 方式2: 更高效的实现（PyTorch 风格）
# 将 QKV 投影合并为一个矩阵，一次性计算
W_QKV = concat(W_Q, W_K, W_V)  # (d_model, 3 * d_model)
QKV = X @ W_QKV  # (n, 3 * d_model)
Q, K, V = QKV.chunk(3, dim=-1)  # 各 (n, d_model)
# 然后分别 reshape 为多头形式
```


### 1.3 Concat 操作（拼接）


每个头独立计算注意力后，将 h 个头的输出拼接回完整维度，再通过输出投影 W^O^ ∈ ℝ^hd~v~×d~model~^ 融合：


```
# Concat 实现
# 各头输出: heads[i] 形状为 (n, d_v)
# 拼接方式:
# 1. 转置回 (n, h, d_v)
combined = torch.stack(heads, dim=1)  # (n, h, d_v)
# 2. reshape 为 (n, h * d_v) = (n, d_model)
combined = combined.reshape(n, h * d_v)
# 3. 输出投影
output = combined @ W_O  # (n, d_model)
```


### 1.4 头维度与头数的权衡


> **Warning:** #### 设计原则
>
>
> - **总计算量不变**
>    ：h × d
>    ~k~
>    = d
>    ~model~
>    ，多头并不增加总计算量，只是将计算分配到不同子空间
> - **每头维度不宜过小**
>    ：d
>    ~k~
>    过小（如 < 16）会限制每个头的表达能力
> - **头数也不宜过多**
>    ：过多的头会增加拼接和投影的开销，且头之间可能学习到重复模式
> - **经验值**
>    ：d
>    ~k~
>    = 64 是广泛验证的有效设定（GPT、BERT、LLaMA 等均采用）
> - **缩放因子**
>    ：多头注意力中缩放因子仍使用 √d
>    ~k~
>    （每头维度），而非 √d
>    ~model~


### 1.5 完整 PyTorch 实现


```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads, dropout=0.1):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # 合并 QKV 投影（更高效）
        self.W_QKV = nn.Linear(d_model, 3 * d_model, bias=False)
        self.W_O = nn.Linear(d_model, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        # 1. 合并投影 QKV
        qkv = self.W_QKV(x)  # (batch, seq_len, 3*d_model)
        qkv = qkv.reshape(batch_size, seq_len, 3, self.num_heads, self.d_k)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, d_k)
        Q, K, V = qkv[0], qkv[1], qkv[2]

        # 2. 缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.d_k)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 3. 加权求和
        attn_output = torch.matmul(attn_weights, V)  # (batch, heads, seq_len, d_k)

        # 4. 转置 + reshape + 输出投影
        attn_output = attn_output.transpose(1, 2).reshape(batch_size, seq_len, self.d_model)
        output = self.W_O(attn_output)

        return output, attn_weights

# 使用示例
mha = MultiHeadAttention(d_model=512, num_heads=8)
x = torch.randn(2, 100, 512)
output, weights = mha(x)
print(f"输出: {output.shape}")    # (2, 100, 512)
print(f"权重: {weights.shape}")   # (2, 8, 100, 100)
```


## 2. Cross-Attention（交叉注意力）


### 2.1 与 Self-Attention 的区别


Cross-Attention 中 Q 来自一个序列（decoder），K 和 V 来自另一个序列（encoder output）。这是 Encoder-Decoder 架构的核心连接机制。


| 对比项 | Self-Attention | Cross-Attention |
| --- | --- | --- |
| Q 来源 | 序列自身 | 另一个序列（通常是 decoder） |
| K, V 来源 | 序列自身 | 另一个序列（通常是 encoder） |
| 注意力矩阵 | n × n（n=序列长度） | n~dec~ × n~enc~ |
| 典型应用 | BERT、GPT 各层 | T5、BART 的 decoder 中间层 |
| 掩码 | 因果/填充掩码 | 通常只有填充掩码 |


```
class CrossAttention(nn.Module):
    """交叉注意力: Q 来自 decoder, KV 来自 encoder"""
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_k = d_model // num_heads

        # Q 从 decoder 输入
        self.W_Q = nn.Linear(d_model, d_model)
        # KV 从 encoder 输出
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)

    def forward(self, decoder_input, encoder_output, padding_mask=None):
        """
        decoder_input: (batch, tgt_len, d_model) - Q 的来源
        encoder_output: (batch, src_len, d_model) - K, V 的来源
        """
        Q = self.W_Q(decoder_input)
        K = self.W_K(encoder_output)
        V = self.W_V(encoder_output)

        # reshape 成多头形式，计算注意力...
        # 注意力矩阵形状: (batch, heads, tgt_len, src_len)
```


## 3. Additive Attention（加性注意力 / Bahdanau Attention）


### 3.1 计算公式


加性注意力由 Bahdanau 等人在 2015 年提出，是 seq2seq 模型中最早的注意力机制之一：


$$
score(q, k) = vT · tanh(W1q + W2k + b)
$$


其中 W~1~ ∈ ℝ^d×d^，W~2~ ∈ ℝ^d×d^，v ∈ ℝ^d^，b ∈ ℝ^d^ 均为可学习参数。


### 3.2 与 Dot-Product 的对比


| 对比项 | Additive Attention | Dot-Product Attention |
| --- | --- | --- |
| 分数计算 | v^T^tanh(W~1~q + W~2~k) | q^T^k / √d~k~ |
| 参数量 | 更多（2d² + d） | 更少（无需额外参数） |
| 计算效率 | 较低（含矩阵乘法和非线性） | 较高（仅点积） |
| 适用场景 | d 较小时表现好 | d 较大时更高效 |
| 典型模型 | Bahdanau seq2seq | Transformer |


```
class AdditiveAttention(nn.Module):
    def __init__(self, d_model):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_model, bias=False)
        self.W_k = nn.Linear(d_model, d_model, bias=False)
        self.v = nn.Linear(d_model, 1, bias=False)

    def forward(self, Q, K, V, mask=None):
        # Q: (batch, q_len, d_model)
        # K: (batch, k_len, d_model)
        q_proj = self.W_q(Q).unsqueeze(2)  # (batch, q_len, 1, d)
        k_proj = self.W_k(K).unsqueeze(1)  # (batch, 1, k_len, d)

        # 加性分数
        scores = self.v(torch.tanh(q_proj + k_proj)).squeeze(-1)
        # scores: (batch, q_len, k_len)

        if mask is not None:
            scores = scores.masked_fill(mask == 0, float('-inf'))

        attn_weights = F.softmax(scores, dim=-1)
        output = torch.matmul(attn_weights, V)
        return output, attn_weights
```


## 4. Linear Attention（线性注意力）


### 4.1 动机


标准注意力的 O(n²) 复杂度限制了其在长序列上的应用。线性注意力的核心思想是将 softmax(QK^T^)V 分解为 Q(K^T^V) 的形式，将复杂度从 O(n²d) 降低到 O(nd²)。当 n >> d 时，这将带来显著的加速。


### 4.2 Performer（FAVOR+）


Performer 使用随机特征映射来近似 softmax 核函数：


$$
softmax(QKT) ≈ φ(Q)φ(K)T
        其中 φ(x) = (1/√m) · [exp(x·ω1 - ||x||²/2), ..., exp(x·ωm - ||x||²/2)]
$$


其中 ω~i~ 是从适当的分布中采样的随机向量。利用核技巧，可以将注意力计算重构为：


$$
Attention(Q, K, V) ≈ φ(Q) · (φ(K)T V) / (φ(Q) · φ(K)T · 1n)
$$


```
# Performer 核心思想伪代码
def performer_attention(Q, K, V, num_features=256):
    """
    通过随机特征映射将 O(n^2) 降为 O(n)
    """
    d = Q.shape[-1]

    # 随机特征: 从 N(0, 1/d) 采样
    omega = torch.randn(num_features, d) / math.sqrt(d)

    def phi(x):
        """随机特征映射 φ(x)"""
        proj = x @ omega.T  # (batch, seq_len, num_features)
        # 应用 exp 和归一化
        return torch.exp(proj - 0.5 * (x ** 2).sum(-1, keepdim=True)) / math.sqrt(num_features)

    phi_Q = phi(Q)  # (batch, q_len, num_features)
    phi_K = phi(K)  # (batch, k_len, num_features)

    # 关键: 先算 K^T V，复杂度 O(n * m * d)，再算 Q * (K^T V)，复杂度 O(n * m * d)
    # 总复杂度: O(n * m * d) 当 m << n 时近似 O(nd^2)
    KV = torch.einsum('bkm,bkd->bmd', phi_K, V)  # (batch, num_features, d_v)
    QKV = torch.einsum('bnm,bmd->bnd', phi_Q, KV)  # (batch, q_len, d_v)

    # 归一化
    normalizer = torch.einsum('bnm,bkm->bn', phi_Q, phi_K)  # (batch, q_len)
    output = QKV / normalizer.unsqueeze(-1)

    return output
```


### 4.3 线性注意力的局限性


> **Warning:** #### 实际效果与理论的差距
>
>
> - **精度损失**
>    ：softmax 的近似在某些任务上会导致性能下降，尤其是需要精确注意力模式的任务
> - **特征数与精度**
>    ：随机特征数 m 需要足够大（通常 256+）才能保证近似质量
> - **实际加速有限**
>    ：在 GPU 上，由于矩阵运算的并行化优势，线性注意力在中等序列长度上未必比标准注意力快
> - **适用场景**
>    ：主要适用于 n 特别大（>8K）且对精度有一定容忍度的场景


## 5. Sparse Attention（稀疏注意力）


### 5.1 Longformer（滑动窗口 + 全局注意力）


Longformer 采用两种注意力模式的组合：


- **滑动窗口注意力**
   ：每个位置只关注左右各 w/2 个邻居，复杂度 O(n·w)
- **全局注意力**
   ：少数特殊位置（如 [CLS]）关注所有位置，所有位置也关注这些特殊位置


$$
Attention(i, j) = { 标准注意力, if |i-j| ≤ w/2 或 i,j ∈ Global; 0, otherwise }
$$


```
def longformer_attention_mask(seq_len, window_size, global_positions):
    """
    生成 Longformer 稀疏注意力掩码
    window_size: 滑动窗口大小（通常 512）
    global_positions: 全局注意力位置列表（如 [0] 对应 [CLS]）
    """
    mask = torch.zeros(seq_len, seq_len)
    half_w = window_size // 2

    for i in range(seq_len):
        # 滑动窗口: 关注 [i-half_w, i+half_w]
        start = max(0, i - half_w)
        end = min(seq_len, i + half_w + 1)
        mask[i, start:end] = 1

    # 全局注意力
    for g in global_positions:
        mask[g, :] = 1  # 全局位置关注所有位置
        mask[:, g] = 1  # 所有位置关注全局位置

    return mask
```


### 5.2 BigBird（随机 + 滑动窗口 + 全局）


BigBird 在 Longformer 基础上增加了随机注意力，理论证明这种组合可以近似任意图灵可计算的函数：


| 组件 | 作用 | 连接数/位置 |
| --- | --- | --- |
| 滑动窗口 | 捕获局部依赖 | w 个邻居 |
| 随机注意力 | 建立长距离随机连接 | r 个随机位置 |
| 全局注意力 | 确保信息全局流通 | 所有位置关注特殊 token |
| **总计** |  | **w + r + g 个连接/位置** |


#### BigBird 的理论保证


BigBird 的稀疏注意力模式被证明是图灵完备的（在序列长度趋于无穷时），且可以表达所有能被全注意力表达的函数。这为稀疏注意力的有效性提供了理论基础。


### 5.3 稀疏注意力对比总结


| 方法 | 注意力模式 | 复杂度 | 长距离能力 |
| --- | --- | --- | --- |
| Full Attention | 全局密集 | O(n²) | 完全 |
| Local/Sliding Window | 固定窗口 | O(nw) | 多层堆叠间接获得 |
| Longformer | 窗口 + 全局 | O(nw) | 通过全局 token |
| BigBird | 窗口 + 随机 + 全局 | O(n) | 理论完整 |
| Performer | 隐式全连接 | O(n) | 完全（近似） |


## 6. 注意力机制设计空间总结


#### 选择指南


1. **标准序列长度（≤ 2K）**
   ：直接使用 Multi-Head Attention，无需额外优化
2. **中等序列长度（2K-8K）**
   ：考虑 Flash Attention（IO 优化而非算法优化）
3. **长序列（8K-32K）**
   ：滑动窗口注意力 + 全局 token（Longformer 方案）
4. **超长序列（32K+）**
   ：组合方案或线性注意力，考虑 Ring Attention 等分布式方案
5. **编码器-解码器任务**
   ：Self-Attention（编码器）+ Cross-Attention（解码器对编码器）

Multi-Head与注意力变体 - 多头分割拼接、交叉注意力、加性注意力、线性注意力、稀疏注意力完整笔记


<!-- Converted from: 02_Multi-Head与注意力变体.html -->
