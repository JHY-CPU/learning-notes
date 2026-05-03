# 03_GloVe：基于全局词频矩阵的分解

## 核心概念

- **GloVe (Global Vectors)**：由 Stanford 团队于 2014 年提出，结合了矩阵分解方法（如 LSA）的全局统计优势和 Word2Vec 的局部上下文窗口优势。
- **共现矩阵 (Co-occurrence Matrix)**：统计每个词在特定窗口大小内与其他词共同出现的次数。$X_{ij}$ 表示词 $j$ 出现在词 $i$ 上下文中的次数。
- **加权最小二乘目标**：GloVe 的目标是学习词向量，使得词向量内积 $w_i^\top \tilde{w}_j$ 逼近共现次数的对数 $\log X_{ij}$，同时引入权重函数 $f(X_{ij})$ 平衡高频和低频词。
- **全局 vs 局部**：Word2Vec 每次只利用一个窗口的局部信息，而 GloVe 直接对整个共现矩阵进行分解，利用全局统计信息。
- **向量偏置项**：引入 $b_i$ 和 $\tilde{b}_j$ 两个偏置项吸收词的独立频率信息，使内积专注于共现模式的建模。

## 数学推导

GloVe 的损失函数为：
$$
J = \sum_{i,j=1}^{|V|} f(X_{ij}) \left( w_i^\top \tilde{w}_j + b_i + \tilde{b}_j - \log X_{ij} \right)^2
$$

权重函数 $f$ 的设计满足三个性质：$f(0)=0$；非递减；对过大的 $X$ 截断以避免高频词主导：
$$
f(x) = \begin{cases}
(x/x_{\text{max}})^\alpha & \text{if } x < x_{\text{max}} \\
1 & \text{otherwise}
\end{cases}
$$

通常取 $x_{\text{max}} = 100$, $\alpha = 3/4$。最终词向量取 $w_i + \tilde{w}_i$ 作为最终表示。

**推导思路**：从 $w_i^\top \tilde{w}_j = \log P_{ij} - \log P_i$ 出发，其中 $P_{ij} = X_{ij}/X_i$ 是条件概率，$P_i = X_i / X_{\text{total}}$。GloVe 的核心洞察是词向量比值 $P_{ik}/P_{jk}$ 比原始概率本身更能编码语义信息。

## 直观理解

- **Word2Vec 像交朋友**：每个词通过与"周围邻居"的互动（局部窗口）来认识世界。
- **GloVe 像人口普查**：直接统计全局的"人际关系网"（整个语料的共现矩阵），然后用数学方法从中提取出主要特征。
- **比率比绝对值更有意义**：判断"冰"和"蒸汽"与"固体"的关系时，$P(固体|冰)/P(固体|蒸汽)$ 很大（冰和固体相关，蒸汽和固体不相关），这个比值比单独的概率更有区分力。GloVe 正是围绕这一发现建模。

## 代码示例

```python
from glove import Corpus, Glove
import numpy as np

sentences = [
    ["猫", "喜欢", "追", "老鼠"],
    ["狗", "喜欢", "追", "猫"],
    ["狗", "是", "人类", "的", "朋友"],
    ["猫", "是", "宠物"]
]

# 构建共现矩阵
corpus = Corpus()
corpus.fit(sentences, window=3)

# 训练 GloVe 模型
glove = Glove(no_components=50, learning_rate=0.05)
glove.fit(corpus.matrix, epochs=30, no_threads=1, verbose=True)
glove.add_dictionary(corpus.dictionary)

print("猫的向量:", glove.word_vectors[glove.dictionary['猫']][:10])
print("最接近'猫'的词:", glove.most_similar('猫', number=3))
```

## 深度学习关联

- **静态嵌入的巅峰**：GloVe 和 Word2Vec 代表了静态词嵌入的最高水平，但已被 BERT 等上下文动态嵌入取代。不过在资源受限场景中仍有广泛应用。
- **矩阵分解思想的延伸**：GloVe 的共现矩阵分解思想在后来的模型压缩（如 SVD 分解嵌入矩阵）和图神经网络中继续发挥作用。
- **加权策略的传承**：GloVe 中的加权函数 $f(X_{ij})$ 平衡高频/低频词的思想，影响了后续模型中的 log-uniform 采样和 importance sampling 策略。
