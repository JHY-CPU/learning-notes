# HNSW详解


## HNSW 详解


向量检索HNSWANN


HNSW (Hierarchical Navigable Small World) 是当前最流行的近似最近邻搜索算法，兼具高召回率和低延迟。


## 理论基础：可导航小世界图


```
Small World (小世界网络)：
六度分隔理论：任意两个人之间平均只需6步就能建立联系
小世界图的特征：
- 平均路径长度短：O(log n)
- 聚类系数高：邻居之间互连紧密

Navigable Small World (可导航小世界)：
- 每个节点只知道自己的邻居
- 使用贪心策略可以高效找到目标节点
- 关键特征：long-range links（长距离连接）
- 节点不仅连接近邻，还随机连接一些远距离节点

贪心搜索过程：
从任意起始节点出发，每一步移动到距离目标最近的邻居
如果当前节点比所有邻居都更近 → 到达局部最优
Long-range links 帮助跳出局部最优

NSW (Navigable Small World) 图：
- 每个新节点加入时，连接到最近的M个已有节点
- 自然形成long-range links
- 但搜索可能陷入局部最优
- HNSW在此基础上引入分层结构来解决
```


## HNSW 分层结构


```
HNSW = 分层 + NSW图

核心思想：
- 将NSW图组织为多层
- 上层是下层的子集（稀疏）
- 上层提供long-range links，加速搜索
- 类似跳表 (Skip List) 的思路

层级分布：
┌─────────────────────────────────────────────┐
│  Layer 3:  ●─────────────────────────●      │
│            (极少节点，每层节点数约为下层的1/ef)│
│                                             │
│  Layer 2:  ●────────●──────────●────●      │
│            (较少节点)                         │
│                                             │
│  Layer 1:  ●────●──●───●──●───●──●──●     │
│            (更多节点)                         │
│                                             │
│  Layer 0:  ●●●●●●●●●●●●●●●●●●●●●●●●●     │
│            (所有节点都在Layer 0)              │
└─────────────────────────────────────────────┘

层级分配：
新节点的层级通过指数分布随机分配
layer = floor(-ln(rand()) × mL)
其中mL = 1/ln(M)，M是每层最大邻居数

例如M=16，mL = 1/ln(16) ≈ 0.36
Layer 0: 100% 的节点
Layer 1: 约36% 的节点
Layer 2: 约13% 的节点
Layer 3: 约5% 的节点
Layer 4: 约1.7% 的节点
...

节点数据结构：
struct HNSWNode {
    vector<float> data;      // 向量数据
    int level;                // 节点所在的最高层级
    vector<HNSWNode*> neighbors[MAX_LEVEL];
                              // 每层的邻居列表
};
```


## HNSW 图构建算法


```
构建过程（逐个插入节点）：

参数：
- M：每层每个节点的最大邻居数（默认16）
- efConstruction：构建时的候选列表大小（默认200）
- mL：层级因子 = 1/ln(M)

INSERT(hnsw, q, M, efConstruction):
┌─────────────────────────────────────────────┐
│  1. 随机确定新节点q的层级 L                 │
│                                             │
│  2. 初始化搜索入口点 ep = hnsw.entry_point  │
│                                             │
│  3. 从最高层到 Layer L+1：                  │
│     // 在这些层中只做贪心搜索，找到最近的节点│
│     ep = SEARCH_LAYER(q, ep, ef=1)         │
│                                             │
│  4. 从 Layer L 到 Layer 0：                 │
│     // 在这些层中做完整搜索                  │
│     candidates = SEARCH_LAYER(q, ep, efConstruction)│
│     neighbors = SELECT_NEIGHBORS(candidates, M)│
│     // 双向连接                              │
│     for each n in neighbors:                │
│       q.neighbors[l].add(n)                 │
│       n.neighbors[l].add(q)                 │
│       // 如果n的邻居超过M，修剪多余的       │
│       if len(n.neighbors[l]) > M:           │
│         n.neighbors[l] = SELECT_NEIGHBORS(n.neighbors[l], M)│
│     ep = candidates  // 下一层的入口点      │
│                                             │
│  5. 如果L > hnsw.max_level:                 │
│     hnsw.entry_point = q                    │
│     hnsw.max_level = L                      │
└─────────────────────────────────────────────┘

SEARCH_LAYER(q, ep, ef):
┌─────────────────────────────────────────────┐
│  // ef = 候选列表大小                       │
│  candidates = min-heap (按距离排序)         │
│  candidates.push(ep, dist(q, ep))          │
│  visited = {ep}                             │
│  results = max-heap (保留最近的ef个)        │
│  results.push(ep, dist(q, ep))             │
│                                             │
│  while candidates is not empty:             │
│    c = candidates.pop_nearest()             │
│    f = results.farthest()                   │
│    if dist(q, c) > dist(q, f):             │
│      break  // 剩余候选不可能更近           │
│    for each n in c.neighbors[l]:           │
│      if n not in visited:                   │
│        visited.add(n)                       │
│        f = results.farthest()               │
│        if dist(q, n) < dist(q, f) or len(results) < ef:│
│          candidates.push(n, dist(q, n))    │
│          results.push(n, dist(q, n))       │
│          if len(results) > ef:             │
│            results.pop_farthest()           │
│  return results                             │
└─────────────────────────────────────────────┘

邻居选择策略 (SELECT_NEIGHBORS)：
简单策略：选择最近的M个
高级策略（论文推荐）：
  1. 简单选择：按距离排序取前M个
  2. 稳健选择 (Robust Heuristic)：
     按距离排序候选
     依次添加，如果新候选与已选邻居太近则跳过
     这增加图的多样性，减少"死胡同"

构建复杂度：
- 插入一个节点：O(M × efConstruction × log n)
- 插入N个节点：O(N × M × efConstruction × log N)
- 100万向量，M=16，efConstruction=200：
  大约需要几分钟到几十分钟
```


## HNSW 搜索算法


```
SEARCH(hnsw, q, K, efSearch):
┌─────────────────────────────────────────────┐
│  1. ep = hnsw.entry_point                   │
│                                             │
│  2. 从最高层到 Layer 1：                     │
│     // 贪心搜索，只保留最近的1个             │
│     ep = SEARCH_LAYER(q, ep, ef=1)         │
│                                             │
│  3. 在 Layer 0：                             │
│     // 完整搜索，保留efSearch个候选          │
│     results = SEARCH_LAYER(q, ep, efSearch) │
│                                             │
│  4. 返回 results 中最近的K个                │
└─────────────────────────────────────────────┘

搜索过程可视化：
查询向量q，K=3，efSearch=10

Layer 3:  ●────────────────ep───────────────●
          贪心搜索：ep → 最近的节点A
Layer 2:  ●────────A──────────●───────●
          贪心搜索：A → 最近的节点B
Layer 1:  ●────B──●───●───●──●──●──●
          贪心搜索：B → 最近的节点C
Layer 0:  ●●●C●●●●●●●●●●●●●●●●●●●●●●●
          从C开始，efSearch=10的BFS搜索
          找到10个候选 → 返回最近的3个

关键参数的影响：
┌──────────────┬───────────────┬──────────────┐
│ 参数          │ 增大时         │ 减小时        │
├──────────────┼───────────────┼──────────────┤
│ M            │ 召回率↑        │ 内存↓        │
│              │ 构建时间↑      │ 搜索速度↑    │
│              │ 搜索时间↑      │ 召回率↓      │
├──────────────┼───────────────┼──────────────┤
│ efConstruction│ 图质量↑       │ 构建速度↑    │
│              │ 构建时间↑      │ 图质量↓      │
├──────────────┼───────────────┼──────────────┤
│ efSearch     │ 召回率↑       │ 搜索速度↑    │
│              │ 搜索时间↑     │ 召回率↓      │
└──────────────┴───────────────┴──────────────┘

典型参数选择：
- 高精度场景：M=32-64, efSearch=128-256
- 平衡场景：M=16, efSearch=64-128
- 高速度场景：M=8-16, efSearch=32-64

搜索复杂度：
- 时间：O(efSearch × log n)
- 空间：O(efSearch) 的候选列表

动态增删：
- 新增：直接插入，调用INSERT算法
- 删除：标记为已删除（lazy delete），定期重建
- 不支持原地更新，需删除后重新插入
```


## HNSW vs IVF 对比


```
┌────────────────┬─────────────────┬─────────────────┐
│ 特性            │ HNSW            │ IVF             │
├────────────────┼─────────────────┼─────────────────┤
│ 索引结构        │ 多层邻近图       │ 平面聚类+倒排   │
│ 查询复杂度      │ O(log n)        │ O(nprobe × n/k) │
│ 召回率          │ >99%            │ 95-99%          │
│ 内存占用        │ 高（图结构）     │ 中（聚类中心）  │
│ 构建速度        │ 慢（需逐个插入） │ 快（K-Means）   │
│ 支持动态增删    │ 支持（增量）     │ 需要重新训练    │
│ 磁盘友好        │ 否（全内存）     │ 可以（DiskANN） │
│ 适合数据规模    │ 百万-千万       │ 千万-亿级       │
│ GPU加速         │ 困难            │ 容易            │
│ 过滤能力        │ 弱              │ 强（标量过滤）  │
└────────────────┴─────────────────┴─────────────────┘

HNSW的优势场景：
- 对延迟和召回率要求极高
- 数据量在百万到千万级
- 需要频繁动态增删
- 内存充足

IVF的优势场景：
- 数据量超大（亿级+）
- 内存有限，需要磁盘索引
- 需要GPU加速
- 需要结合标量过滤

混合方案（实际系统常用）：
┌─────────────────────────────────────────────┐
│  IVF-HNSW：                                  │
│  用IVF做粗筛，每个簇内用HNSW做精排           │
│  兼顾IVF的扩展性和HNSW的高精度               │
│                                              │
│  DiskANN：                                   │
│  图索引存储在磁盘，SSD随机读取               │
│  用Vamana图（HNSW的磁盘优化版本）            │
│  内存只存压缩向量，磁盘存图结构              │
│  适合十亿级向量                              │
│                                              │
│  SPANN：                                     │
│  将向量分到多个"头"（聚类中心）              │
│  每个"头"的邻居列表存储在磁盘                │
│  查询时加载相关"头"的邻居到内存              │
└─────────────────────────────────────────────┘
```


> **Note:** HNSW是目前中小规模（千万级以内）向量检索的最佳选择，提供接近精确搜索的召回率和微秒级延迟。超大规模场景需要结合IVF或磁盘优化方案。


## 工程实现与代码示例


```
使用hnswlib（C++库，Python绑定）：

import hnswlib
import numpy as np

# 初始化
dim = 128
num_elements = 1000000
index = hnswlib.Index(space='cosine', dim=dim)

# 初始化索引（预分配内存）
index.init_index(max_elements=num_elements, ef_construction=200, M=16)

# 构建索引
data = np.random.randn(num_elements, dim).astype(np.float32)
index.add_items(data, num_threads=4)

# 设置查询参数
index.set_ef(64)  # efSearch=64

# 查询
query = np.random.randn(1, dim).astype(np.float32)
labels, distances = index.knn_query(query, k=10)

# 动态增删
new_data = np.random.randn(100, dim).astype(np.float32)
index.add_items(new_data)
index.mark_deleted(0)  # 删除第0个元素

# 保存和加载
index.save_index("hnsw_index.bin")
index.load_index("hnsw_index.bin", max_elements=num_elements)

使用Faiss（Facebook的向量检索库）：

import faiss

dim = 128
M = 16
efConstruction = 200

# 构建HNSW索引
index = faiss.IndexHNSWFlat(dim, M)
index.hnsw.efConstruction = efConstruction

# 添加数据
data = np.random.randn(1000000, dim).astype(np.float32)
index.add(data)

# 查询
index.hnsw.efSearch = 64
query = np.random.randn(10, dim).astype(np.float32)
distances, labels = index.search(query, k=10)

使用Qdrant（向量数据库）：

from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance

client = QdrantClient("localhost", port=6333)

# 创建集合（自动使用HNSW）
client.create_collection(
    collection_name="my_vectors",
    vectors_config=VectorParams(size=128, distance=Distance.COSINE),
    hnsw_config={"m": 16, "ef_construct": 200}
)

# 插入向量
client.upsert(collection_name="my_vectors", points=[...])

# 搜索
results = client.search(
    collection_name="my_vectors",
    query_vector=query_vec,
    limit=10,
    search_params={"hnsw_ef": 64}
)
```


<!-- Converted from: 02_HNSW详解.html -->
