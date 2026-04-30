# 文本特征工程：TF-IDF 与 N-Gram

## 核心概念
- **词袋模型 (Bag of Words, BoW)**：将文本表示为词频向量，忽略词序和语法结构。每个文档是一个 $V$ 维向量（$V$ 是词典大小），每个维度对应一个词的计数。
- **TF-IDF**：Term Frequency - Inverse Document Frequency，对词频进行加权——一个词在文档中出现越多（TF 高）、在全体文档中出现越少（IDF 高），则权重越大。
- **N-Gram**：将连续的 $n$ 个词（或字符）作为一个特征单元。Unigram ($n=1$)、Bigram ($n=2$)、Trigram ($n=3$)。N-Gram 能捕捉局部词序信息但导致特征维度爆炸。
- **TF-IDF 的平滑**：$IDF(t) = \log \frac{1 + N}{1 + DF(t)} + 1$（sklearn 实现），避免除零且给罕见的 IDF 一个非零下界。
- **停用词 (Stop Words)**：高频但对语义贡献小的词（如"的"、"是"、"在"），通常在预处理阶段移除以减少噪声。
- **子线性 TF 缩放**：$TF(t, d) = 1 + \log(count_{t,d})$ 或使用原始计数，防止高频词的主导。

## 数学推导
**TF (词频)**：
$$
TF(t, d) = \frac{\text{词 } t \text{ 在文档 } d \text{ 中出现的次数}}{\text{文档 } d \text{ 中的总词数}}
$$
也可以是原始计数或 $1 + \log(\text{count})$。

**IDF (逆文档频率)**：
$$
IDF(t) = \log \frac{N}{DF(t)}
$$
其中 $N$ 是文档总数，$DF(t)$ 是包含词 $t$ 的文档数。

**TF-IDF 分数**：
$$
TFIDF(t, d) = TF(t, d) \times IDF(t)
$$

**与向量空间模型 (VSM) 的关联**：
文档 $d$ 的 TF-IDF 向量为 $\mathbf{v}_d \in \mathbb{R}^V$。两个文档的相似度用余弦相似度衡量：
$$
\cos(\mathbf{v}_a, \mathbf{v}_b) = \frac{\mathbf{v}_a \cdot \mathbf{v}_b}{\|\mathbf{v}_a\| \|\mathbf{v}_b\|}
$$

**N-Gram 特征**：
对于 $n=2$，文档 "I love NLP" 的 Bigram 特征为：
$$
\{\text{"I love"}, \text{"love NLP"}\}
$$

特征维度随 $n$ 增长：
$$
|V_n| \approx |V_1|^n \quad \text{(理论上)}
$$
实践中使用频率阈值筛选，通常只保留出现次数超过一定阈值的 N-Gram。

## 直观理解
- **TF-IDF 的"稀有就是重要"**：在一个关于"机器学习"的文章集中，"深度学习"这个词出现频率很高（TF 高），但几乎每篇文章都提到了它（DF 高，IDF 低），所以权重不会太高。而"胶囊网络"虽然只在某篇文章中出现（TF 中等），但只有这篇文章提到了它（DF 低，IDF 高），所以权重很高——TF-IDF 会自动识别出这篇文章的特色词汇。
- **N-Gram 与词的搭配**：Unigram "not" + "good" 的情感是中性的，但 Bigram "not good" 的情感是负面的。N-Gram 能捕捉这种词组合引发的语义变迁。但 N 越大，特征越稀疏——"I am not very happy" 的 5-gram 可能只在这篇文档中出现过。
- **词袋的"信息损失"**：词袋模型像用购物清单"苹果、香蕉、牛奶"表示一顿饭——知道有哪些食材，但不知道"先炒香蕉再放苹果"这样的顺序信息。同样，"A kills B" 和 "B kills A" 在词袋中完全一样。

## 代码示例
```python
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score

# 示例文档
corpus = [
    "I love machine learning and deep learning",
    "Natural language processing is fascinating",
    "Machine learning models need lots of data",
    "Deep learning is a subset of machine learning",
    "I love NLP and text processing",
]

# TF-IDF 向量化 (Unigram)
tfidf = TfidfVectorizer(ngram_range=(1, 1), stop_words='english')
X_tfidf = tfidf.fit_transform(corpus)
print(f"Unigram TF-IDF 特征数: {X_tfidf.shape[1]}")
print(f"词典: {tfidf.get_feature_names_out()}")

# 文档 0 的 TF-IDF 向量（稀疏矩阵转稠密看前几个词）
doc0_vec = X_tfidf[0].toarray()[0]
for word, score in zip(tfidf.get_feature_names_out(),
                       doc0_vec):
    if score > 0:
        print(f"  {word}: {score:.4f}")

# Bigram TF-IDF
tfidf_bi = TfidfVectorizer(ngram_range=(2, 2), max_features=20)
X_bi = tfidf_bi.fit_transform(corpus)
print(f"\nBigram TF-IDF 特征数: {X_bi.shape[1]}")
print(f"Bigram 词表: {tfidf_bi.get_feature_names_out()}")

# 使用 TF-IDF + 简单分类器
y_dummy = [0, 1, 0, 1, 0]
scores = cross_val_score(LogisticRegression(), X_tfidf, y_dummy, cv=3)
print(f"\nTF-IDF + LR CV 准确率: {scores.mean():.4f}")
```

## 深度学习关联
- **词向量嵌入 (Word Embeddings)**：Word2Vec、GloVe 等词嵌入是 TF-IDF 的深层替代——将词映射到稠密向量（如 300 维），捕捉词的语义信息（"国王" - "男人" + "女人" ≈ "女王"）。这是深度学习在 NLP 中的基石。
- **预训练语言模型**：BERT、GPT 等预训练模型使用上下文相关的词表示，同一词在不同语境中的向量不同（如"苹果"在"吃苹果"和"苹果公司"中不同）。这与 TF-IDF 的静态词权重形成鲜明对比。
- **Transformer 的注意力机制**：Transformer 的 Self-Attention 可以看作一种动态的"软"特征加权——模型根据上下文自适应地决定哪些词更重要，类似于 TF-IDF 的自适应加权但更强大灵活。
