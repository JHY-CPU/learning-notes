# 13_Luong 注意力机制 (Multiplicative Attention)

## 核心概念
- **Luong Attention (Multiplicative Attention)**：由 Luong 等人于 2015 年提出，相比 Bahdanau 注意力更简单高效。主要区别在于使用当前解码器状态 $s_t$（而非上一时刻 $s_{t-1}$）计算注意力，且分数计算方式不同。
- **全局注意力 (Global Attention)**：关注编码器的所有隐藏状态，与 Bahdanau 注意力类似，但计算方式更简洁。
- **局部注意力 (Local Attention)**：介于全局注意力和硬注意力之间的折中方案。为每个目标词预测一个对齐位置 $p_t$，然后只关注该位置附近的窗口。计算量更小而效果相近。
- **三种分数计算方法**：
  - **Dot** (点积)：$e_t(s) = s_t^\top h_s$，最简单快速
  - **General** (通用)：$e_t(s) = s_t^\top W_a h_s$，引入可学习权重
  - **Concat** (拼接)：$e_t(s) = v_a^\top \tanh(W_a[s_t; h_s])$，与 Bahdanau 类似
- **输入提饲 (Input-feeding)**：将上一时间步的注意力上下文向量 $\tilde{h}_{t-1}$ 作为当前步输入的一部分，使模型知道哪些位置已被关注过。
- **与 Bahdanau 的关键差异**：Luong 的上下文向量用于计算最终输出前（after RNN），而 Bahdanau 融入 RNN 状态更新中（before RNN）。

## 数学推导
Luong 全局注意力的计算步骤：

1. 当前解码器状态：$s_t = \text{RNN}(s_{t-1}, y_{t-1})$

2. 对齐分数（三种变体）：
$$
e_t(s) = 
\begin{cases}
s_t^\top h_s & \text{(dot)} \\
s_t^\top W_a h_s & \text{(general)} \\
v_a^\top \tanh(W_a [s_t; h_s]) & \text{(concat)}
\end{cases}
$$

3. 注意力权重：$\alpha_t(s) = \frac{\exp(e_t(s))}{\sum_{s'} \exp(e_t(s'))}$

4. 上下文向量：$c_t = \sum_s \alpha_t(s) h_s$

5. 注意力结合输出：$\tilde{h}_t = \tanh(W_c [c_t; s_t])$

6. 预测：$P(y_t | y_{<t}) = \text{softmax}(W_s \tilde{h}_t)$

局部注意力在第 $t$ 步预测对齐位置 $p_t = T_x \cdot \sigma(v_p^\top \tanh(W_p s_t))$，高斯分布为中心加权。

## 直观理解
- **点积注意力像"关键字匹配"**：你先想好要找什么（$s_t$），然后在编码器状态中搜索最匹配的内容（$h_s$），匹配度直接用向量点积衡量——越相似得分越高。
- **全局 vs 局部**：全局注意力就像考试时翻遍整本教材找答案，局部注意力则先猜答案大概在第几页（预测位置 $p_t$），然后只翻附近几页——高效但可能猜错。
- **Dot vs General**：Dot 相当于 General 中 $W_a$ 固定为单位矩阵——什么都不学，直接用内积。General 增加了一层可学习的"度量变换"。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class LuongAttention(nn.Module):
    def __init__(self, hidden_size, method='general'):
        super().__init__()
        self.method = method

        if method == 'general':
            self.W_a = nn.Linear(hidden_size, hidden_size, bias=False)
        elif method == 'concat':
            self.W_a = nn.Linear(hidden_size * 2, hidden_size)
            self.v_a = nn.Linear(hidden_size, 1)

    def score(self, query, key):
        # query: (batch, hidden) — 当前解码器状态 s_t
        # key: (batch, seq, hidden) — 编码器状态
        if self.method == 'dot':
            return torch.bmm(key, query.unsqueeze(2)).squeeze(2)
        elif self.method == 'general':
            transformed = self.W_a(key)              # (batch, seq, hidden)
            return torch.bmm(transformed, query.unsqueeze(2)).squeeze(2)
        elif self.method == 'concat':
            query = query.unsqueeze(1).repeat(1, key.size(1), 1)
            energy = self.v_a(torch.tanh(
                self.W_a(torch.cat([query, key], dim=2))
            )).squeeze(2)
            return energy

    def forward(self, query, values):
        e = self.score(query, values)                # (batch, seq)
        alpha = F.softmax(e, dim=1)                  # 注意力权重
        context = torch.bmm(alpha.unsqueeze(1), values).squeeze(1)
        return context, alpha

# 使用示例
attn_dot = LuongAttention(hidden_size=256, method='dot')
attn_general = LuongAttention(hidden_size=256, method='general')

query = torch.randn(4, 256)
values = torch.randn(4, 20, 256)

context_dot, alpha_dot = attn_dot(query, values)
context_general, alpha_general = attn_general(query, values)
print("点积注意力上下文:", context_dot.shape)     # (4, 256)
print("General注意力上下文:", context_general.shape)  # (4, 256)
```

## 深度学习关联
- **通向 Scaled Dot-Product Attention 的桥梁**：Luong 的"dot"方法直接启发了 Transformer 中的 Scaled Dot-Product Attention——用点积作为相似度度量。Transformer 在此基础上增加了缩放因子和多头机制。
- **局部注意力的高效启示**：局部注意力的"窗口"思想影响了后来 Longformer 和 BigBird 的稀疏注意力模式，它们也只在固定窗口或随机位置上计算注意力。
- **Dual-Stage 交互思想**：Luong 中计算注意力后再结合解码器状态的做法（先 RNN 再注意），与 Transformer 中"先自注意力再 FFN"的子层设计有相似之处。
