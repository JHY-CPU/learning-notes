# Embedding基础 - 特征工程深入


## 1. Embedding的核心概念


Embedding（嵌入）是将离散的高维标识（如用户ID、商品ID、词语）映射为低维稠密向量的过程。Embedding向量能够捕获实体之间的语义关系，相似的实体具有相近的向量表示。


### 为什么需要Embedding


| 对比维度 | One-Hot编码 | Embedding |
| --- | --- | --- |
| 维度 | 等于类别数（百万级） | 低维稠密（8~512维） |
| 稀疏性 | 极度稀疏 | 稠密 |
| 语义关系 | 无法表达，任意两个向量距离相等 | 相似实体距离近 |
| 参数量 | 特征维度×隐藏层 | 类别数×Embedding维度 |
| 可学习性 | 固定 | 通过训练学习 |


> **Example:** **直观理解：**
> 在推荐系统中，用户ID有1亿个。One-Hot编码需要1亿维向量，非常浪费。Embedding将每个用户映射为128维向量，参数量从1亿×128减少到1亿×128（Embedding层本质是一个查找表）。更重要的是，相似偏好的用户会学到相近的Embedding向量。


## 2. Word2Vec：CBOW与Skip-gram


Word2Vec（Google, 2013）是最早的词向量模型，基于分布式假设：词的含义由其上下文决定。


| 模型 | 输入 | 输出 | 特点 |
| --- | --- | --- | --- |
| CBOW | 上下文词（窗口内周围词） | 中心词 | 训练快，适合高频词 |
| Skip-gram | 中心词 | 上下文词 | 适合低频词，效果通常更好 |


$$
Skip-gram目标函数：
                maximize J = (1/T) Σ Σ log p(w_{t+j} | w_t)
                p(w_O | w_I) = exp(v'_{wO}ᵀ · v_{wI}) / Σ exp(v'_{w}ᵀ · v_{wI})
                其中 v_{wI} 为输入Embedding，v'_{wO} 为输出Embedding
$$


### 训练技巧


- **Negative Sampling：**
   不计算全词表softmax，而是采样负样本，大幅降低计算量
- **Hierarchical Softmax：**
   用哈夫曼树组织词表，将O(V)降为O(log V)
- **Subsampling：**
   高频词随机丢弃（如"的"、"the"），加速训练并提升低频词质量


## 3. Item2Vec与推荐系统Embedding


Item2Vec（Microsoft, 2016）将Word2Vec思想应用到推荐系统：将用户的行为序列视为"句子"，商品视为"词"。


> **Example:** **用户行为序列 → 文档：**
>
>
> 用户A行为：浏览手机 → 查看手机壳 → 购买耳机 → 浏览平板
>
>
> "文档"：[手机, 手机壳, 耳机, 平板]
>
>
> 通过Word2Vec训练，手机和手机壳的Embedding向量会接近


### 推荐系统中Embedding的应用


- **用户画像：**
   用户Embedding作为用户兴趣的向量化表示
- **召回：**
   基于用户Embedding和商品Embedding的向量相似度召回
- **排序：**
   Embedding作为排序模型的输入特征
- **特征交叉：**
   不同实体Embedding的内积/哈达玛积作为交叉特征


## 4. Embedding的初始化与训练


| 初始化方式 | 描述 | 适用场景 |
| --- | --- | --- |
| 随机初始化 | 从正态分布/均匀分布随机采样 | 从头训练，数据充足 |
| 预训练Embedding | 用Word2Vec等预训练，再fine-tune | 数据不足、加速收敛 |
| 迁移学习 | 从类似领域迁移Embedding | 新业务冷启动 |


### Embedding维度选择


- 经验公式：embedding_dim = 6 × (vocabulary_size)^(1/4)
- 用户ID/商品ID：通常64~256维
- 类别特征（城市、品类）：通常8~64维
- 维度过小无法充分表示，维度过大导致过拟合和计算开销


## 5. 冷启动问题


冷启动是Embedding面临的核心挑战：新用户/新商品没有行为数据，无法学习到有效的Embedding。


| 冷启动类型 | 挑战 | 解决方案 |
| --- | --- | --- |
| 用户冷启动 | 新用户无历史行为 | 基于画像（年龄、地域）初始化Embedding；使用热门推荐 |
| 物品冷启动 | 新商品无被交互记录 | 基于内容（文本、图像）生成Embedding；Lookalike |
| 系统冷启动 | 新系统无任何数据 | 引入外部知识、人工标注、迁移学习 |


> **Important:** **内容辅助Embedding：**
> 对于新物品，可以使用其文本描述（BERT）、图像特征（ResNet）等生成初始Embedding，而非随机初始化。这样新物品即使没有交互数据，也能与其他物品进行有意义的比较。


## 6. PyTorch nn.Embedding使用


```
import torch
import torch.nn as nn
import torch.optim as optim

# ========== 1. 基础Embedding使用 ==========
# 创建Embedding层：10000个词，每个词128维向量
embedding = nn.Embedding(num_embeddings=10000, embedding_dim=128)

# 输入：词索引（LongTensor）
input_ids = torch.LongTensor([1, 5, 3, 100, 9999])
embedded = embedding(input_ids)  # 形状: (5, 128)
print(f"Embedding形状: {embedded.shape}")

# ========== 2. 推荐系统中的多字段Embedding ==========
class RecommendationEmbedding(nn.Module):
    """推荐系统的多字段Embedding层"""
    def __init__(self):
        super().__init__()
        # 各字段的Embedding
        self.user_emb = nn.Embedding(1000000, 64)   # 100万用户，64维
        self.item_emb = nn.Embedding(500000, 64)     # 50万商品，64维
        self.city_emb = nn.Embedding(500, 16)        # 500个城市，16维
        self.category_emb = nn.Embedding(200, 16)    # 200个品类，16维
        self.gender_emb = nn.Embedding(3, 8)         # 性别：未知/男/女，8维

        # 连续特征的全连接层
        self.age_fc = nn.Linear(1, 8)                # 年龄连续值

        # 总维度：64+64+16+16+8+8 = 176
        self.output_dim = 176

    def forward(self, user_id, item_id, city, category, gender, age):
        u = self.user_emb(user_id)         # (batch, 64)
        i = self.item_emb(item_id)         # (batch, 64)
        c = self.city_emb(city)            # (batch, 16)
        cat = self.category_emb(category)  # (batch, 16)
        g = self.gender_emb(gender)        # (batch, 8)
        a = self.age_fc(age.unsqueeze(1))  # (batch, 8)

        # 拼接所有特征
        features = torch.cat([u, i, c, cat, g, a], dim=1)  # (batch, 176)
        return features

# ========== 3. 使用预训练Embedding ==========
# 加载预训练的Word2Vec向量
import numpy as np

def init_embedding_with_pretrained(embedding_layer, pretrained_vectors):
    """用预训练向量初始化Embedding层"""
    embedding_layer.weight.data.copy_(torch.from_numpy(pretrained_vectors))
    # 可选择冻结Embedding不参与训练
    # embedding_layer.weight.requires_grad = False

# 示例：随机生成预训练向量
vocab_size = 10000
embed_dim = 128
pretrained = np.random.randn(vocab_size, embed_dim).astype(np.float32)
init_embedding_with_pretrained(embedding, pretrained)
print("Embedding已用预训练向量初始化")

# ========== 4. Embedding的梯度更新 ==========
optimizer = optim.Adam(embedding.parameters(), lr=0.001)
criterion = nn.CrossEntropyLoss()

# 模拟训练
dummy_labels = torch.LongTensor([0, 1, 2, 3, 4])
logits = embedded @ torch.randn(128, 10)  # 简单线性分类
loss = criterion(logits, dummy_labels)
loss.backward()
optimizer.step()
print(f"训练损失: {loss.item():.4f}")
```


## 总结


- Embedding将离散ID映射为稠密向量，是推荐系统和NLP的基础技术
- Word2Vec（CBOW/Skip-gram）是最经典的Embedding训练方法
- Item2Vec将用户行为序列视为"文档"，为商品学习Embedding表示
- Embedding维度需要平衡表达能力和计算开销，经验公式：6*vocab^(1/4)
- 冷启动问题可通过内容特征辅助、迁移学习、相似用户/物品推荐解决
- PyTorch的nn.Embedding本质是一个查找表，通过反向传播学习最优表示


<!-- Converted from: 01_Embedding基础.html -->
