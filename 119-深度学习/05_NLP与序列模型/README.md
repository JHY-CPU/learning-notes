# NLP与序列模型

## 一、循环神经网络

### 1.1 RNN

处理序列数据，隐藏状态传递时间步信息：
$$h_t = \tanh(W_{xh} x_t + W_{hh} h_{t-1} + b_h)$$
$$y_t = W_{hy} h_t + b_y$$

问题：梯度消失/爆炸，难以捕捉长期依赖。

### 1.2 LSTM

引入门控机制解决长期依赖：
- **遗忘门**：$f_t = \sigma(W_f [h_{t-1}, x_t] + b_f)$
- **输入门**：$i_t = \sigma(W_i [h_{t-1}, x_t] + b_i)$
- **输出门**：$o_t = \sigma(W_o [h_{t-1}, x_t] + b_o)$
- **细胞状态**：$C_t = f_t \odot C_{t-1} + i_t \odot \tanh(W_c [h_{t-1}, x_t] + b_c)$

### 1.3 GRU

LSTM的简化版，合并遗忘门和输入门：
- **更新门**：$z_t = \sigma(W_z [h_{t-1}, x_t])$
- **重置门**：$r_t = \sigma(W_r [h_{t-1}, x_t])$

---

## 二、注意力机制

### 2.1 Seq2Seq与Attention

编码器-解码器结构中，注意力让解码器关注编码器的不同部分：
$$\alpha_{t,s} = \frac{\exp(e_{t,s})}{\sum_k \exp(e_{t,k})}$$
$$c_t = \sum_s \alpha_{t,s} h_s$$

### 2.2 Self-Attention

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Q、K、V分别来自同一序列的不同线性变换
- 缩放因子 $\sqrt{d_k}$ 防止点积过大导致softmax饱和

### 2.3 Multi-Head Attention

将Q、K、V分别投影到多个子空间，拼接输出：
$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

---

## 三、Transformer

### 3.1 架构

- **Encoder**：Multi-Head Self-Attention + Feed-Forward + LayerNorm + 残差连接
- **Decoder**：Masked Self-Attention + Cross-Attention + Feed-Forward

### 3.2 位置编码

正弦位置编码：
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

现代模型常用可学习的相对位置编码。

---

## 四、预训练语言模型

| 模型 | 类型 | 特点 |
|------|------|------|
| BERT | Encoder-only | 双向编码，MLM+NSP预训练 |
| GPT系列 | Decoder-only | 自回归，因果语言建模 |
| T5 | Encoder-Decoder | 文本到文本统一框架 |
| RoBERTa | Encoder | BERT改进，更多数据更长训练 |
| BART | Encoder-Decoder | 去噪自编码预训练 |

### BERT预训练任务

1. **MLM (Masked Language Model)**：随机遮盖15%的token，预测被遮盖的词
2. **NSP (Next Sentence Prediction)**：判断两个句子是否相邻

### GPT自回归

从左到右逐token生成：$P(x_t | x_1, ..., x_{t-1})$
