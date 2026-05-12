# Self-Attention 计算详解


## Self-Attention 计算详解


#### 核心思想


Self-Attention（自注意力机制）是 Transformer 架构的核心组件，允许序列中的每个位置都能直接关注序列中所有其他位置，从而捕获长距离依赖关系。其核心思想是：对于序列中的每个元素，通过计算它与序列中所有元素（包括自身）的相关性来动态加权聚合信息。


## 1. QKV 投影（Query-Key-Value Projection）


### 1.1 线性投影


给定输入序列 X ∈ ℝ^n×d~model~^（n 为序列长度，d~model~ 为模型维度），通过三组可学习的权重矩阵将输入投影为三个不同的表示：


$$
Q = XWQ,   K = XWK,   V = XWV
$$


其中：


- **Q（Query，查询）**
   ：表示当前位置想要"查询"什么信息。W
   ~Q~
   ∈ ℝ
   ^d~model~×d~k~^
- **K（Key，键）**
   ：表示每个位置可以"提供"什么信息的索引。W
   ~K~
   ∈ ℝ
   ^d~model~×d~k~^
- **V（Value，值）**
   ：表示每个位置实际包含的信息内容。W
   ~V~
   ∈ ℝ
   ^d~model~×d~v~^


#### 直觉理解


可以把 QKV 机制类比为信息检索系统：**Query** 是你的搜索关键词，**Key** 是文档的标题/标签，**Value** 是文档的实际内容。注意力分数衡量 Query 与每个 Key 的匹配程度，然后用这个分数对 Value 进行加权求和。


### 1.2 投影矩阵的维度分析


| 矩阵 | 维度 | 作用 |
| --- | --- | --- |
| W~Q~ | d~model~ × d~k~ | 将输入映射到查询空间 |
| W~K~ | d~model~ × d~k~ | 将输入映射到键空间 |
| W~V~ | d~model~ × d~v~ | 将输入映射到值空间 |
| W~O~ | d~v~ × d~model~ | 输出投影（多头拼接后） |


在标准 Transformer 中，d~k~ = d~v~ = d~model~/h（h 为注意力头数）。


## 2. Scaled Dot-Product Attention（缩放点积注意力）


### 2.1 计算公式


Scaled Dot-Product Attention 是 Transformer 中使用的标准注意力计算方式：


$$
Attention(Q, K, V) = softmax(QKT / √dk) · V
$$


### 2.2 分步计算过程


以序列长度 n=4，d~k~=d~v~=3 为例，详细拆解每一步：


#### Step 1: 计算注意力分数矩阵


将 Q 和 K^T^ 做矩阵乘法，得到 n×n 的分数矩阵：


$$
S = QKT ∈ ℝn×n
$$


```
# S[i][j] 表示位置 i 对位置 j 的原始关注分数
# S[i][j] = q_i · k_j = Σ(q_i[m] * k_j[m]) for m in range(d_k)

# 矩阵形式：
S = Q @ K.T  # shape: (n, n)
# 例如 S[0][2] = q_0·k_2 = q_0[0]*k_2[0] + q_0[1]*k_2[1] + q_0[2]*k_2[2]
```


#### Step 2: 缩放（Scaling）


将分数除以 √d~k~：


$$
Sscaled = S / √dk
$$


> **Warning:** #### 为什么要缩放？
>
>
> 当 d~k~ 较大时，QK^T^ 的点积结果方差会随 d~k~ 线性增长。假设 Q 和 K 的各分量独立且均值为0、方差为1，则点积 q·k = Σq~i~k~i~ 的方差为 d~k~。较大的值经过 softmax 后会使梯度极小（softmax 在输入差异大时趋近于 one-hot），导致梯度消失。除以 √d~k~ 将方差重新归一化到1。


#### Step 3: Softmax 归一化


对分数矩阵的每一行（沿 Key 维度）做 softmax，得到注意力权重：


$$
A = softmax(Sscaled),   A[i][j] = exp(Sscaled[i][j]) / Σk exp(Sscaled[i][k])
$$


```
# 每一行的权重之和为 1
# A[i] 是一个概率分布，表示位置 i 对所有位置的关注分配
import torch
import torch.nn.functional as F

def scaled_dot_product_attention(Q, K, V, mask=None):
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(
        torch.tensor(d_k, dtype=torch.float32)
    )
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    attn_weights = F.softmax(scores, dim=-1)
    output = torch.matmul(attn_weights, V)
    return output, attn_weights
```


#### Step 4: 加权求和


用注意力权重对 Value 向量加权求和，得到每个位置的输出：


$$
Output[i] = Σj A[i][j] · V[j]
$$


### 2.3 矩阵形式完整推导


将上述步骤合并为矩阵运算，一次性处理整个序列：


```
输入: X ∈ ℝ^(n × d_model)

# 1. 线性投影
Q = X @ W_Q    # (n, d_model) × (d_model, d_k) = (n, d_k)
K = X @ W_K    # (n, d_model) × (d_model, d_k) = (n, d_k)
V = X @ W_V    # (n, d_model) × (d_model, d_v) = (n, d_v)

# 2. 计算缩放注意力分数
scores = Q @ K.T / sqrt(d_k)   # (n, d_k) × (d_k, n) = (n, n)

# 3. (可选) 应用注意力掩码
scores = scores + mask          # mask 中 -inf 位置会被 softmax 压至 0

# 4. Softmax 归一化
attn_weights = softmax(scores, dim=-1)  # (n, n)，每行和为 1

# 5. 加权聚合
output = attn_weights @ V      # (n, n) × (n, d_v) = (n, d_v)

# 6. 输出投影（通常在多头注意力中）
output = output @ W_O          # (n, d_v) × (d_v, d_model) = (n, d_model)
```


## 3. 注意力掩码（Attention Mask）


### 3.1 因果掩码（Causal Mask）


因果掩码（也称前瞻掩码 / Look-Ahead Mask）用于自回归模型（如 GPT），确保位置 i 只能关注位置 ≤ i 的token，防止信息泄露到未来位置：


$$
Maskcausal[i][j] = { 0, if j ≤ i; -∞, if j > i }
$$


```
# 因果掩码矩阵示例 (n=4)
# 1 表示可见，0 表示被掩码
# [[1, 0, 0, 0],
#  [1, 1, 0, 0],
#  [1, 1, 1, 0],
#  [1, 1, 1, 1]]

def create_causal_mask(seq_len):
    """生成上三角因果掩码"""
    mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1)
    mask = mask.masked_fill(mask == 1, float('-inf'))
    return mask  # shape: (seq_len, seq_len)

# 实际使用中通常用 bool 类型掩码 + masked_fill
causal_mask = torch.triu(torch.ones(n, n), diagonal=1).bool()
scores.masked_fill_(causal_mask, float('-inf'))
```


### 3.2 填充掩码（Padding Mask）


填充掩码用于屏蔽序列中 <PAD> token 的位置，使模型不关注填充位。实际实现中，填充掩码和因果掩码通常组合使用：


```
# padding_mask: (batch, seq_len), True 表示是 padding 位置
def create_combined_mask(causal_mask, padding_mask):
    """
    causal_mask: (seq_len, seq_len)
    padding_mask: (batch, seq_len)
    返回: (batch, seq_len, seq_len)
    """
    # 扩展 padding_mask 维度用于广播
    # padding_mask[:, None, :] 形状 (batch, 1, seq_len)
    # 表示 key 侧的 padding
    combined = causal_mask.unsqueeze(0) + padding_mask.unsqueeze(1)
    # 任何一个为 True 则被掩码
    return combined
```


### 3.3 其他掩码类型


| 掩码类型 | 用途 | 典型场景 |
| --- | --- | --- |
| 因果掩码 | 防止关注未来位置 | GPT 等自回归生成模型 |
| 填充掩码 | 屏蔽 PAD token | 批处理中不同长度序列 |
| 双向掩码 | 屏蔽特定 token 对 | BERT 预训练、span corruption |
| 局部窗口掩码 | 只关注局部邻域 | 稀疏注意力、滑动窗口 |
| 前缀掩码 | 前缀部分双向，后续因果 | Prefix-LM 模型 |


## 4. 计算复杂度分析


### 4.1 Self-Attention 的 O(n²d) 复杂度


Self-Attention 的计算复杂度可以从以下几个维度分析：


| 操作 | 计算量 | 空间复杂度 |
| --- | --- | --- |
| Q = XW~Q~, K = XW~K~, V = XW~V~ | 3 × n·d~model~·d~k~ | O(n·d) |
| S = QK^T^ | n²·d~k~ | **O(n²)** |
| A = softmax(S) | O(n²) | O(n²) |
| O = AV | n²·d~v~ | O(n·d) |
| **总计** | **O(n²·d)** | **O(n² + n·d)** |


> **Warning:** #### 瓶颈所在
>
>
> 当 n >> d 时（如长序列场景），注意力分数矩阵 QK^T^ 的 n×n 大小成为主要瓶颈。这既是计算瓶颈（n²·d 次浮点运算），也是内存瓶颈（存储 n×n 的注意力矩阵需要 O(n²) 空间）。例如，当 n=8192, d=128 时，注意力矩阵需要 8192² × 4 bytes ≈ 256 MB 内存（float32）。


### 4.2 与 RNN、CNN 的复杂度对比


| 层类型 | 每层计算复杂度 | 顺序操作数 | 最大路径长度 |
| --- | --- | --- | --- |
| Self-Attention | O(n²·d) | O(1) | O(1) |
| RNN | O(n·d²) | O(n) | O(n) |
| CNN (kernel=k) | O(k·n·d²) | O(1) | O(n/k) 或 O(log~k~(n)) |


Self-Attention 的优势在于：(1) 最大路径长度为 O(1)，任意两个位置可直接交互；(2) 顺序操作数为 O(1)，天然适合并行化。代价是序列长度的二次复杂度。


### 4.3 内存中的注意力矩阵


```
# 注意力计算的内存占用分析
def memory_analysis(n, d_model, d_k, batch_size, dtype_bytes=4):
    """
    n: 序列长度
    d_model: 模型维度
    d_k: 每头的 key 维度
    batch_size: 批大小
    """
    # Q, K, V 投影: 3 × batch × n × d_k × 4 bytes
    qkv_mem = 3 * batch_size * n * d_k * dtype_bytes

    # 注意力分数矩阵: batch × n × n × 4 bytes (最大瓶颈)
    scores_mem = batch_size * n * n * dtype_bytes

    # 注意力权重: batch × n × n × 4 bytes
    attn_mem = batch_size * n * n * dtype_bytes

    # 输出: batch × n × d_k × 4 bytes
    output_mem = batch_size * n * d_k * dtype_bytes

    total = qkv_mem + scores_mem + attn_mem + output_mem
    print(f"序列长度 n={n}, d_k={d_k}")
    print(f"注意力矩阵: {scores_mem/1024**2:.1f} MB")
    print(f"总计: {total/1024**2:.1f} MB")

# memory_analysis(4096, 4096, 128, 1)  # 注意力矩阵 = 64 MB
# memory_analysis(16384, 4096, 128, 1) # 注意力矩阵 = 1024 MB = 1 GB
```


## 5. 实际代码实现


### 5.1 PyTorch 完整实现


```
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SelfAttention(nn.Module):
    """单头 Self-Attention 完整实现"""

    def __init__(self, d_model, d_k=None, d_v=None, dropout=0.1):
        super().__init__()
        self.d_model = d_model
        self.d_k = d_k or d_model
        self.d_v = d_v or d_model

        # QKV 投影层
        self.W_Q = nn.Linear(d_model, self.d_k, bias=False)
        self.W_K = nn.Linear(d_model, self.d_k, bias=False)
        self.W_V = nn.Linear(d_model, self.d_v, bias=False)
        self.W_O = nn.Linear(self.d_v, d_model, bias=False)

        self.dropout = nn.Dropout(dropout)
        self.scale = math.sqrt(self.d_k)

    def forward(self, x, mask=None):
        """
        x: (batch, seq_len, d_model)
        mask: (batch, seq_len, seq_len) 或 (seq_len, seq_len), True=被掩码
        """
        batch_size, seq_len, _ = x.size()

        # 1. 线性投影
        Q = self.W_Q(x)  # (batch, seq_len, d_k)
        K = self.W_K(x)  # (batch, seq_len, d_k)
        V = self.W_V(x)  # (batch, seq_len, d_v)

        # 2. 计算缩放点积注意力
        scores = torch.matmul(Q, K.transpose(-2, -1)) / self.scale
        # scores: (batch, seq_len, seq_len)

        # 3. 应用掩码
        if mask is not None:
            scores = scores.masked_fill(mask, float('-inf'))

        # 4. Softmax + Dropout
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        # 5. 加权求和
        output = torch.matmul(attn_weights, V)  # (batch, seq_len, d_v)

        # 6. 输出投影
        output = self.W_O(output)  # (batch, seq_len, d_model)

        return output, attn_weights


# 使用示例
d_model = 512
seq_len = 100
batch_size = 2

x = torch.randn(batch_size, seq_len, d_model)
causal_mask = torch.triu(torch.ones(seq_len, seq_len), diagonal=1).bool()

attention = SelfAttention(d_model=d_model, d_k=64, d_v=64)
output, weights = attention(x, mask=causal_mask)
print(f"输出形状: {output.shape}")       # (2, 100, 512)
print(f"注意力权重形状: {weights.shape}")  # (2, 100, 100)
```


### 5.2 数值稳定性注意事项


#### 实现细节


- **softmax 数值稳定性**
   ：PyTorch 的
   `F.softmax`
   内部会减去每行的最大值再做 exp，避免数值溢出
- **混合精度训练**
   ：attention 计算建议在 float32 下进行，即使使用混合精度训练（因为 softmax 和缩放对精度敏感）
- **掩码方式**
   ：使用
   `masked_fill`
   将被掩码位置设为 -inf，softmax 后自然变为 0，比直接设 0 更稳定
- **缩放因子**
   ：始终使用 √d
   ~k~
   缩放，即使 d
   ~k~
   较小也建议保留，保持实现一致性


## 6. 总结与要点


#### 核心要点回顾


1. **QKV 三组投影**
   是 Self-Attention 的基础，将输入映射到三个不同的语义空间
2. **Scaled Dot-Product**
   通过除以 √d
   ~k~
   解决点积值过大的梯度消失问题
3. **Softmax**
   将原始分数转化为概率分布，实现注意力权重的归一化
4. **注意力掩码**
   控制信息流向：因果掩码用于自回归，填充掩码处理变长序列
5. **O(n²d) 复杂度**
   是 Self-Attention 的固有瓶颈，催生了各类高效注意力变体
6. Self-Attention 的
   **最大路径长度为 O(1)**
   ，这是它优于 RNN 的关键特性

Self-Attention计算详解 - QKV投影、缩放点积注意力、掩码机制、复杂度分析完整笔记


<!-- Converted from: 01_Self-Attention计算详解.html -->
