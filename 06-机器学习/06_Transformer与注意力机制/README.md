# Transformer与注意力机制

## 一、注意力机制

### 1.1 缩放点积注意力

$$\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d_k}}\right) V$$

- Q（Query）、K（Key）、V（Value）来自输入的线性变换
- $\sqrt{d_k}$ 缩放防止点积过大导致softmax饱和

### 1.2 Multi-Head Attention

$$\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W^O$$

每个头关注不同的表示子空间。

---

## 二、Transformer架构

### 2.1 Encoder

每层包含：
1. Multi-Head Self-Attention
2. 残差连接 + LayerNorm
3. Feed-Forward Network（两层MLP，中间ReLU/GELU）
4. 残差连接 + LayerNorm

### 2.2 Decoder

每层包含：
1. Masked Multi-Head Self-Attention（防止看到未来token）
2. Multi-Head Cross-Attention（关注Encoder输出）
3. Feed-Forward Network
4. 每步都有残差连接 + LayerNorm

### 2.3 位置编码

正弦/余弦位置编码：
$$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$$
$$PE_{(pos, 2i+1)} = \cos(pos / 10000^{2i/d})$$

现代模型常用可学习的绝对/相对位置编码。

---

## 三、预训练模型

### 3.1 BERT（Encoder-only）

- 双向编码
- 预训练：MLM + NSP
- 适用于理解任务（分类、NER、问答）

### 3.2 GPT（Decoder-only）

- 自回归生成
- 预训练：因果语言建模
- 适用于生成任务

### 3.3 T5（Encoder-Decoder）

- 统一文本到文本框架
- 所有NLP任务转化为生成问题

---

## 四、高效Transformer

- **稀疏注意力**：Longformer、BigBird
- **线性注意力**：Linear Transformer
- **Flash Attention**：IO优化的标准注意力
- **MQA/GQA**：减少KV Cache（多查询/分组查询注意力）
