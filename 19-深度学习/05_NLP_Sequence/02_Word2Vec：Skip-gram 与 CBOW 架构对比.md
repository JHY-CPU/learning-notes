# 02_Word2Vec：Skip-gram 与 CBOW 架构对比

## 核心概念

- **Word2Vec 核心思想**：由 Mikolov 等人于 2013 年提出，核心假设是"一个词的含义由其上下文决定"（distributional hypothesis）。通过神经网络将每个词映射为稠密低维向量。
- **CBOW (Continuous Bag-of-Words)**：给定上下文词（周围词），预测中心词。目标是最大化给定上下文时中心词的条件概率。训练速度较快，对高频词效果更好。
- **Skip-gram**：给定中心词，预测其上下文词。目标是最大化给定中心词时上下文词的条件概率。对低频词和稀有词更友好，但训练速度较慢。
- **负采样 (Negative Sampling)**：替代原始的 softmax 输出层，通过采样少量负例（非目标词）进行二分类训练，大幅降低计算复杂度。
- **分层 Softmax (Hierarchical Softmax)**：利用霍夫曼树将 |V| 分类问题转化为 log|V| 次二分类，进一步加速训练。
- **嵌入向量的语义特性**：训练得到的词向量具有线性类比性质，如 vec("国王") - vec("男人") + vec("女人") ≈ vec("女王")。

## 数学推导

$$
\text{CBOW: } \max \frac{1}{T}\sum_{t=1}^{T}\log P(w_t | w_{t-k}:w_{t+k})
$$

$$
\text{Skip-gram: } \max \frac{1}{T}\sum_{t=1}^{T}\sum_{-k\leq j\leq k, j\neq 0}\log P(w_{t+j} | w_t)
$$

$$
\text{Negative Sampling 损失: } \log\sigma(v_{w_O}'^\top v_{w_I}) + \sum_{i=1}^{k}\mathbb{E}_{w_i\sim P_n(w)}[\log\sigma(-v_{w_i}'^\top v_{w_I})]
$$

其中 $v_w$ 是输入词向量，$v_w'$ 是输出词向量，$\sigma$ 是 sigmoid 函数，$P_n(w)$ 是噪声分布（通常为 Unigram 的 3/4 次方）。

## 直观理解

- **CBOW 像完形填空**：给你"我爱__编程"中的"爱"和"编程"，猜中间是什么。它整合了上下文信息，适合快速训练。
- **Skip-gram 像联想游戏**：给你"苹果"，让你联想周围的词（"水果"、"红色"、"iPhone"）。它能学到更丰富的语义关系，尤其是对低频词。
- **类比性质**：词向量空间中的向量运算对应语义关系，就像在语义地图上做矢量导航。

## 代码示例

```python
from gensim.models import Word2Vec

sentences = [
    ["猫", "喜欢", "追", "老鼠"],
    ["狗", "喜欢", "追", "猫"],
    ["狗", "是", "人类", "的", "朋友"],
    ["猫", "是", "宠物"]
]

# 训练 Skip-gram 模型
model_sg = Word2Vec(
    sentences,
    vector_size=100,    # 向量维度
    window=3,           # 上下文窗口大小
    sg=1,               # 1=Skip-gram, 0=CBOW
    negative=5,         # 负采样数量
    min_count=1,
    epochs=50
)

print("猫的向量:", model_sg.wv["猫"][:10], "...")
print("最接近'猫'的词:", model_sg.wv.most_similar("猫", topn=3))
print("类比: 猫-狗+人类 =", model_sg.wv.most_similar(
    positive=["猫", "人类"], negative=["狗"], topn=1
))
```

## 深度学习关联

- **预训练词嵌入的奠基**：Word2Vec 开创了无监督预训练词向量的范式，直接影响了后续 BERT、GPT 等预训练语言模型的发展。
- **与 Transformer 的关系**：Word2Vec 产生的静态词嵌入至今仍被用作 Transformer 模型 Embedding 层的初始化，但已逐渐被子词嵌入（Subword Embedding）取代。
- **对比学习的前身**：Skip-gram 的负采样本质上是噪声对比估计 (NCE) 的一种形式，与 SimCLR、CLIP 等现代对比学习方法共享理论基础。
