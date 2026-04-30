# 01_词袋模型 (BoW) 与 TF-IDF 局限性

## 核心概念
- **词袋模型 (Bag-of-Words, BoW)**：将文本表示为一个无序的词频向量，忽略语法和词序信息。每个文档对应一个长度为 |V|（词汇表大小）的向量，每个维度表示对应词的出现次数或是否出现。
- **TF-IDF (Term Frequency-Inverse Document Frequency)**：在 BoW 基础上引入词权重，TF 衡量词在文档中的重要性，IDF 降低常见词的权重，提高罕见词的区分度。
- **稀疏表示问题**：BoW 和 TF-IDF 产生的向量极度稀疏，词汇表通常达数万甚至数十万，但每个文档只包含其中极少数词，导致存储和计算效率低下。
- **词序信息丢失**：BoW 完全忽略词语顺序，"猫追狗"和"狗追猫"会得到相同的表示，这是其最根本的语义缺陷。
- **语义鸿沟**：无法捕捉同义词（如"开心"和"快乐"）或近义词关系，词与词之间被视为独立，缺乏语义相似度度量。
- **OOV (Out-of-Vocabulary) 问题**：测试集中出现训练词汇表外的词时，模型无法处理，直接丢弃该词信息。

## 数学推导
$$
\text{BoW: } \mathbf{v}_d = [c(w_1, d), c(w_2, d), \ldots, c(w_{|V|}, d)]^\top
$$

$$
\text{TF-IDF: } \text{tfidf}(w, d) = \text{tf}(w, d) \times \text{idf}(w), \quad \text{idf}(w) = \log\frac{N}{\text{df}(w)}
$$

其中 $c(w, d)$ 是词 $w$ 在文档 $d$ 中出现次数，$N$ 是文档总数，$\text{df}(w)$ 是包含词 $w$ 的文档数。IDF 的平滑变体为 $\log\frac{N+1}{\text{df}(w)+1} + 1$，防止除零并避免零权重。

**TF-IDF 的直观含义**：一个词在特定文档中出现越频繁（TF 高），且在整个语料中出现越罕见（IDF 高），则其对区分该文档越重要。

## 直观理解
- **BoW 就像超市购物清单**：只记录买了什么（出现了哪些词）和数量（词频），完全不关心购物顺序。两张内容相同但顺序不同的清单被视为完全一样。
- **TF-IDF 像搜索引擎的关键词评估**："的"、"是"这类停用词几乎出现在所有文档中（IDF 接近 0），就像"的"在每篇文章中都出现，没有区分度。而"神经网络"只出现在特定文档中，一旦出现就高度相关。
- **稀疏性类比**：想象一个有 10 万个词的大型图书馆，但你每篇文章只提到其中 20-50 个词，其余 99950+ 个位置都是 0——这就是 BoW 的稀疏本质。

## 代码示例
```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

corpus = [
    "猫追狗",
    "狗追猫",
    "猫和狗是好朋友"
]

# BoW 表示
bow = CountVectorizer(token_pattern="(?u)\\b\\w+\\b")
X_bow = bow.fit_transform(corpus)
print("词汇表:", bow.get_feature_names_out())
print("BoW 矩阵:\n", X_bow.toarray())

# TF-IDF 表示
tfidf = TfidfVectorizer(token_pattern="(?u)\\b\\w+\\b")
X_tfidf = tfidf.fit_transform(corpus)
print("\nTF-IDF 矩阵:\n", X_tfidf.toarray())
```

## 深度学习关联
- **词嵌入 (Word Embeddings) 的动机**：BoW 的语义鸿沟直接催生了 Word2Vec、GloVe 等稠密向量表示方法，将离散稀疏表示转化为连续的稠密语义空间。
- **注意力机制中的"软"词权重**：TF-IDF 的加权思想在深度学习中演化为自注意力机制——模型动态地为每个词分配权重，而非依赖固定的 IDF 统计值。
- **现代 Embedding Layer**：神经网络中的 Embedding 层本质上是可学习的"稠密词袋"——每个词映射到一个低维连续向量，彻底解决了稀疏和语义孤立问题。
