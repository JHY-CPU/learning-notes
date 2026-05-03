# 17_Positional Encoding 位置编码的频谱分析

## 核心概念

- **位置编码的必要性**：自注意力是排列等变的（permutation equivariant），即打乱输入顺序后输出也按相同方式打乱但内容不变。模型无法区分"猫追狗"和"狗追猫"，必须注入位置信息。
- **正弦-余弦位置编码**：使用不同频率的正弦和余弦函数为每个位置生成固定维度的编码向量。频率从 $2\pi$ 到 $10000 \cdot 2\pi$ 按几何级数分布。
- **频谱分析视角**：位置编码可以看作一组频率从低到高的信号。低频分量（周期长）编码绝对位置，高频分量（周期短）编码相对位置。
- **相对位置信息编码**：正弦-余弦编码的一个重要性质是 $PE_{pos+k}$ 可以表示为 $PE_{pos}$ 的线性变换，使得模型能够隐式学习相对位置关系。
- **可学习位置编码**：BERT 等模型使用可学习的嵌入表，让模型自行学习位置表示。参数更多但更灵活。
- **外推能力**：正弦-余弦编码理论上可以外推到训练中未见过的序列长度，而可学习编码受限于最大长度。
- **加法融合**：位置编码与词嵌入按位相加后输入 Transformer。位置编码的维度与词嵌入相同，信号强度也相当。

## 数学推导

正弦-余弦位置编码（Vaswani et al., 2017）：
$$
PE_{(pos, 2i)} = \sin\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

$$
PE_{(pos, 2i+1)} = \cos\left(\frac{pos}{10000^{2i/d_{\text{model}}}}\right)
$$

其中 $pos$ 是位置索引，$i$ 是维度索引。

**频谱分析**：每个维度 $i$ 对应的频率为 $\omega_i = \frac{1}{10000^{2i/d_{\text{model}}}}$。当 $i$ 从 0 到 $d_{\text{model}}/2$，频率从 $1$ 指数衰减到 $1/10000$（周期从 $2\pi$ 增长到 $20000\pi$）。

**相对位置性质**：
$$
PE_{pos+k} = T_k \cdot PE_{pos}
$$

其中 $T_k$ 是旋转矩阵（块对角形式），使得 $PE_{pos+k}$ 与 $PE_{pos}$ 的点积只依赖于 $k$（相对距离），而不依赖于 $pos$（绝对位置）。

## 直观理解

- **位置编码像摩尔斯电码中的时间戳**：每个词的位置信息用一组"信号"编码。低频信号（长波）告诉你大致在序列中的位置（前 1/4 还是后 1/4），高频信号（短波）告诉你精确的局部偏移。
- **不同频率的作用**：最低频的维度（周期约为 $2\pi \times 10000 \approx 62832$）在整个训练长度内几乎单调递增，编码绝对位置。最高频的维度（周期约 $2\pi$）相邻几步就重复，编码相对位置。
- **为什么用三角函数**：正弦和余弦是天然的正交基，不同频率的信号叠加可以表示任意位置，而且线性变换性质好——加一个偏移对应旋转，使位置编码可以反映距离信息。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class SinusoidalPositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # (1, max_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: (batch, seq_len, d_model)
        return x + self.pe[:, :x.size(1), :]

# 可视化不同频率维度的编码
d_model = 128
pe = SinusoidalPositionalEncoding(d_model)
positions = torch.arange(0, 100)

# 查看不同频率的模式
print("低频维度 (i=0):", pe.pe[0, :10, 0].numpy())   # 几乎线性增长
print("高频维度 (i=63):", pe.pe[0, :10, 63].numpy())  # 快速震荡

# 验证相对位置编码性质
pos_0 = pe.pe[0, 0, :]    # 位置 0
pos_5 = pe.pe[0, 5, :]    # 位置 5
pos_12 = pe.pe[0, 12, :]  # 位置 12

# 位置差 5 的两对位置应该有相同的点积
dot_05 = torch.dot(pos_0, pos_5)
dot_712 = torch.dot(pe.pe[0, 7, :], pe.pe[0, 12, :])
print(f"PE(0)·PE(5) = {dot_05:.4f}")
print(f"PE(7)·PE(12) = {dot_712:.4f}")
print(f"相同距离的点积相近: {abs(dot_05 - dot_712) < 1e-5}")
```

## 深度学习关联

- **Transformers 的位置信息依赖**：位置编码是 Transformer 中不可或缺的组件，没有位置编码的 Transformer 本质上是一个"词袋集合操作器"。
- **RoPE 和 ALiBi 的演进**：后续研究者提出了改进的位置编码方案——RoPE（旋转位置编码）将位置信息融入注意力分数计算，ALiBi 通过在注意力分数上添加线性偏置来编码位置，两者都更好地支持了长度外推。
- **可学习编码的局限性**：BERT 使用最大长度 512 的可学习位置编码，无法处理更长的序列。这直接催生了 ALBERT 的 SOP 任务和 Longformer 的位置编码方案。
