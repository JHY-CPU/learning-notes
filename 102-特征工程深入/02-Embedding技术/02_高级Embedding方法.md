# 高级Embedding方法 - 特征工程深入


## 1. 图Embedding方法总览


图Embedding（Graph Embedding）将图结构中的节点映射为低维向量，保留节点间的结构关系。在推荐系统中，用户-商品交互天然构成二部图，图Embedding能有效捕获高阶协同过滤信号。


| 方法 | 年份 | 核心思想 | 保留的信息 |
| --- | --- | --- | --- |
| DeepWalk | 2014 | 随机游走 + Word2Vec | 局部结构相似性 |
| Node2Vec | 2016 | 有偏随机游走（BFS+DFS） | 结构等价 + 同质性 |
| LINE | 2015 | 一阶+二阶邻近度 | 局部和全局邻近 |
| Struc2Vec | 2017 | 结构角色相似性 | 结构等价 |
| GraphSAGE | 2017 | GNN邻居聚合 | 局部邻域特征 |
| PinSage | 2018 | GraphSAGE工业级应用（Pinterest） | 大规模图学习 |


## 2. Node2Vec：有偏随机游走


Node2Vec是DeepWalk的改进版，通过调节BFS和DFS的平衡来捕获不同类型的节点关系。


### 随机游走策略


| 参数 | 效果 | 保留信息 |
| --- | --- | --- |
| p（返回参数） | p小：容易返回已访问节点 | BFS效果，局部结构 |
| q（进出参数） | q小：倾向于DFS（探索远距离节点） | DFS效果，全局结构 |
| p=1, q=1 | 均匀随机游走 | 等价于DeepWalk |


$$
转移概率：
                α(p,q,t,x) = 1/p  若 d_tx = 0（返回）
                α(p,q,t,x) = 1    若 d_tx = 1（邻居）
                α(p,q,t,x) = 1/q  若 d_tx = 2（远距离）
                其中 t 为前一个节点，x 为候选下一个节点，d_tx为距离
$$


```
import networkx as nx
import numpy as np
from gensim.models import Word2Vec

class Node2Vec:
    """简化版Node2Vec实现"""
    def __init__(self, graph, p=1, q=1, walk_length=80, num_walks=10):
        self.graph = graph
        self.p = p
        self.q = q
        self.walk_length = walk_length
        self.num_walks = num_walks

    def _alias_setup(self, probs):
        """Alias方法采样"""
        K = len(probs)
        q = np.zeros(K)
        J = np.zeros(K, dtype=int)
        smaller, larger = [], []
        for i, prob in enumerate(probs):
            q[i] = K * prob
            if q[i] < 1: smaller.append(i)
            else: larger.append(i)
        while smaller and larger:
            small = smaller.pop()
            large = larger.pop()
            J[small] = large
            q[large] -= (1 - q[small])
            if q[large] < 1: smaller.append(large)
            else: larger.append(large)
        return J, q

    def _biased_walk(self, start_node):
        """有偏随机游走"""
        walk = [start_node]
        neighbors = list(self.graph.neighbors(start_node))
        if not neighbors: return walk
        walk.append(np.random.choice(neighbors))

        for _ in range(self.walk_length - 2):
            curr = walk[-1]
            prev = walk[-2]
            neighbors = list(self.graph.neighbors(curr))
            if not neighbors: break

            # 计算转移概率
            probs = []
            for nbr in neighbors:
                if nbr == prev: probs.append(1.0/self.p)
                elif self.graph.has_edge(nbr, prev): probs.append(1.0)
                else: probs.append(1.0/self.q)

            # 归一化
            total = sum(probs)
            probs = [p/total for p in probs]
            J, q = self._alias_setup(probs)

            # Alias采样
            idx = np.random.randint(len(neighbors))
            if np.random.rand() < q[idx]:
                walk.append(neighbors[idx])
            else:
                walk.append(neighbors[J[idx]])
        return walk

    def train(self, embedding_dim=128):
        """训练Node2Vec"""
        walks = []
        nodes = list(self.graph.nodes())
        for _ in range(self.num_walks):
            np.random.shuffle(nodes)
            for node in nodes:
                walk = self._biased_walk(node)
                walks.append([str(n) for n in walk])

        # 用Word2Vec训练
        model = Word2Vec(walks, vector_size=embedding_dim,
                         window=10, min_count=0, sg=1, workers=4, epochs=5)
        embeddings = {int(n): model.wv[n] for n in model.wv.index_to_key}
        return embeddings
```


## 3. 多模态Embedding：CLIP


CLIP（Contrastive Language-Image Pre-training, OpenAI 2021）通过对比学习将文本和图像映射到同一向量空间，实现文本-图像对齐。


$$
对比学习损失（InfoNCE）：
                L = -log(exp(sim(I_i, T_i) / τ) / Σ_j exp(sim(I_i, T_j) / τ))
                其中 I_i 为图像Embedding，T_i 为对应文本Embedding，
                sim为余弦相似度，τ为温度参数
$$


### CLIP的应用


- **零样本图像分类：**
   无需训练，直接用文本描述分类图像
- **图文检索：**
   用文本搜索图片，或用图片搜索文本
- **推荐系统：**
   利用图像和文本Embedding增强商品表示
- **生成式AI：**
   Stable Diffusion等文生图模型的文本编码器


## 4. 向量检索技术


当Embedding规模达到百万甚至十亿级别时，精确的最近邻搜索变得不可行，需要近似最近邻（ANN）算法。


| 算法 | 原理 | 召回率 | 速度 | 内存 |
| --- | --- | --- | --- | --- |
| Faiss IVF | 倒排文件，聚类分区搜索 | 高 | 快 | 中 |
| Faiss PQ | 乘积量化，压缩向量 | 中 | 极快 | 低 |
| HNSW | 分层可导航小世界图 | 极高 | 快 | 高 |
| ScaNN | Google，各向异性量化 | 高 | 极快 | 中 |


```
import faiss
import numpy as np

# 创建测试数据
d = 128          # 向量维度
nb = 1000000     # 数据库大小
nq = 1000        # 查询数

np.random.seed(42)
database = np.random.randn(nb, d).astype(np.float32)
queries = np.random.randn(nq, d).astype(np.float32)

# ========== 1. 精确搜索（暴力搜索）==========
index_flat = faiss.IndexFlatL2(d)
index_flat.add(database)
D, I = index_flat.search(queries[:5], 10)  # Top-10
print(f"精确搜索完成，最近邻距离: {D[0][:3]}")

# ========== 2. IVF索引（倒排文件）==========
nlist = 1000  # 聚类中心数
quantizer = faiss.IndexFlatL2(d)
index_ivf = faiss.IndexIVFFlat(quantizer, d, nlist)
index_ivf.train(database[:100000])  # 用子集训练聚类
index_ivf.add(database)
index_ivf.nprobe = 10  # 搜索10个最近的聚类
D, I = index_ivf.search(queries[:5], 10)
print(f"IVF搜索完成")

# ========== 3. IVF-PQ索引（压缩版）==========
m = 16   # 子量化器数量（d必须整除m）
nbits = 8  # 每个子量化的位数
index_pq = faiss.IndexIVFPQ(quantizer, d, nlist, m, nbits)
index_pq.train(database[:100000])
index_pq.add(database)
index_pq.nprobe = 10
D, I = index_pq.search(queries[:5], 10)
print(f"IVF-PQ搜索完成，索引大小: {index_pq.ntotal}")

# ========== 4. GPU加速 ==========
# if faiss.get_num_gpus() > 0:
#     index_gpu = faiss.index_cpu_to_gpu(faiss.StandardGpuResources(), 0, index_ivf)
#     D, I = index_gpu.search(queries[:5], 10)
```


## 5. Embedding压缩


大规模推荐系统中，Embedding表可能占用数十GB内存，需要压缩技术来降低存储和计算成本。


| 压缩方法 | 原理 | 压缩比 | 精度损失 |
| --- | --- | --- | --- |
| PQ（乘积量化） | 将向量切分为多段，每段独立量化 | 8-32x | 较小 |
| 蒸馏压缩 | 用小Embedding蒸馏大Embedding | 2-8x | 很小 |
| Hash Embedding | 哈希到固定大小的表 | 可控 | 中等 |
| 混合精度 | 热门物品高精度，冷门物品低精度 | 2-4x | 很小 |
| 低秩分解 | Embedding矩阵分解为两个小矩阵 | 2-10x | 小 |


> **Note:** **混合精度策略：**
> 头部热门物品（被交互次数多）使用高维Embedding（128维），长尾冷门物品使用低维Embedding（16维）。这基于观察：热门物品有充足数据学习高质量Embedding，冷门物品数据少，高维Embedding反而容易过拟合。


## 6. 向量数据库


向量数据库是专为存储和检索高维向量设计的数据库系统，支持高效的ANN搜索、过滤和持久化。


| 数据库 | 类型 | 特点 | 适用场景 |
| --- | --- | --- | --- |
| Milvus | 开源/云 | 高性能、分布式、GPU加速 | 大规模生产环境 |
| Pinecone | 云服务 | 全托管、简单API、自动扩缩容 | 快速原型、中小规模 |
| Weaviate | 开源/云 | 内置向量化、GraphQL API | 语义搜索、知识图谱 |
| Qdrant | 开源/云 | Rust实现、高效过滤 | 需要过滤的向量搜索 |
| Chroma | 开源 | 轻量级、LLM生态友好 | RAG应用、开发测试 |
| pgvector | PostgreSQL扩展 | 复用PG生态、SQL查询 | 中小规模、已有PG |


```
# Milvus使用示例
from pymilvus import connections, Collection, FieldSchema, CollectionSchema, DataType

# 连接Milvus
connections.connect(host="localhost", port="19530")

# 定义Schema
fields = [
    FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
    FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=128),
    FieldSchema(name="category", dtype=DataType.VARCHAR, max_length=50),
]
schema = CollectionSchema(fields, description="商品Embedding集合")
collection = Collection("product_embeddings", schema)

# 插入数据
import numpy as np
ids = list(range(1000))
embeddings = np.random.randn(1000, 128).tolist()
categories = ["电子"] * 500 + ["服饰"] * 500
collection.insert([ids, embeddings, categories])

# 创建索引
index_params = {
    "index_type": "IVF_FLAT",
    "metric_type": "L2",
    "params": {"nlist": 128}
}
collection.create_index("embedding", index_params)

# 搜索
collection.load()
search_vectors = np.random.randn(1, 128).tolist()
results = collection.search(
    search_vectors, "embedding",
    param={"metric_type": "L2", "params": {"nprobe": 10}},
    limit=5,
    expr='category == "电子"'  # 过滤条件
)
for hits in results:
    for hit in hits:
        print(f"ID: {hit.id}, Distance: {hit.distance:.4f}")
```


## 总结


- 图Embedding（Node2Vec、DeepWalk）通过随机游走将图结构转化为向量表示
- Node2Vec通过p、q参数平衡BFS和DFS，控制局部/全局结构的保留
- CLIP通过对比学习实现文本-图像对齐，是多模态Embedding的里程碑
- 向量检索使用ANN算法（Faiss IVF/PQ、HNSW、ScaNN）实现大规模高效搜索
- Embedding压缩技术（PQ、蒸馏、混合精度）降低存储和计算成本
- 向量数据库（Milvus、Pinecone、Weaviate等）提供专业的向量存储和检索服务


<!-- Converted from: 02_高级Embedding方法.html -->
