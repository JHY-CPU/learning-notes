# Transformer与注意力机制


## Transformer与注意力机制


NLPTransformerAttention


Transformer架构通过自注意力机制实现了NLP的范式转变。


## 自注意力机制 (Self-Attention)


```
核心思想：每个位置关注序列中所有位置的信息

计算过程：
1. 线性变换得到 Q, K, V：
   Q = X × Wq    (Query：我要查什么)
   K = X × Wk    (Key：你能提供什么)
   V = X × Wv    (Value：实际内容)

2. 计算注意力分数：
   Attention(Q, K, V) = softmax(Q × K^T / √dk) × V

直觉理解：
- Q × K^T：计算每对位置的相关性
- / √dk：缩放，防止点积过大导致softmax梯度消失
- softmax：归一化为概率分布
- × V：按注意力权重加权求和

示例：
句子："猫 坐在 垫子 上"
"坐"的Q与"猫"的K点积高 → 关注主语
"坐"的Q与"垫子"的K点积高 → 关注宾语
```


## 多头注意力 (Multi-Head Attention)


```
多头：并行执行多组注意力，捕获不同类型的关联

MultiHead(Q, K, V) = Concat(head1, ..., headh) × WO
headi = Attention(Q × Wqi, K × Wki, V × Wvi)

示例（8个头可能学到的不同模式）：
Head 1：语法依赖（主语-谓语）
Head 2：指代关系（代词-名词）
Head 3：位置关系（相邻词）
Head 4：语义相似
...

位置编码 (Positional Encoding)：
- 自注意力是位置无关的！
- 需要注入位置信息
- 正弦位置编码：
  PE(pos, 2i) = sin(pos / 10000^(2i/d))
  PE(pos, 2i+1) = cos(pos / 10000^(2i/d))
- 或学习的位置嵌入
- RoPE：旋转位置编码（LLaMA等使用）
- ALiBi：注意力线性偏置
```


## Transformer架构


```
完整Transformer结构：
┌─────────────────────────────────────────────┐
│  Encoder（编码器，BERT使用）                  │
│  输入嵌入 + 位置编码                          │
│  × N层：                                     │
│  ├── Multi-Head Self-Attention               │
│  ├── Add & LayerNorm (残差连接+层归一化)      │
│  ├── Feed-Forward Network (FFN)              │
│  │   FFN(x) = ReLU(xW1 + b1)W2 + b2        │
│  └── Add & LayerNorm                         │
│                                             │
│  Decoder（解码器，GPT只用这个）               │
│  输出嵌入 + 位置编码                          │
│  × N层：                                     │
│  ├── Masked Multi-Head Self-Attention        │
│  │   （因果掩码，只看左边）                    │
│  ├── Add & LayerNorm                         │
│  ├── Cross-Attention（关注Encoder输出）       │
│  ├── Add & LayerNorm                         │
│  ├── FFN                                     │
│  └── Add & LayerNorm                         │
│  → Linear → Softmax → 输出概率               │
└─────────────────────────────────────────────┘

关键设计：
- 残差连接：缓解梯度消失，支持深层网络
- LayerNorm：稳定训练，加速收敛
- FFN：两层线性变换+激活函数，提供非线性
- 因果掩码：防止看到未来信息
```


## BERT与GPT架构对比


```
┌──────────┬────────────────┬────────────────┐
│          │ BERT           │ GPT            │
├──────────┼────────────────┼────────────────┤
│ 架构      │ Encoder-only   │ Decoder-only   │
│ 注意力    │ 双向(Bidirectional) │ 单向(因果) │
│ 预训练    │ MLM + NSP      │ 自回归LM       │
│ 微调      │ 添加分类头     │ Prompt/ICL     │
│ 擅长      │ 理解任务       │ 生成任务        │
│ 代表      │ BERT, RoBERTa  │ GPT, LLaMA     │
└──────────┴────────────────┴────────────────┘

复杂度分析：
Self-Attention: O(n² × d)
- n=序列长度，d=维度
- 长序列是主要瓶颈
- Flash Attention：优化GPU内存访问

FFN: O(n × d²)
- 在d>>n时，FFN是主要计算量
- MoE：只激活部分专家，减少FFN计算
```


> **Note:** Transformer已成为不仅限于NLP的通用架构，ViT(视觉)、Whisper(语音)都基于Transformer。


<!-- Converted from: 01_Transformer与注意力机制.html -->
