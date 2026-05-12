# Neo4j 图算法

## 一、GDS（Graph Data Science）库概览

### 1.1 GDS 架构

Neo4j 的图算法通过 **GDS（Graph Data Science）** 库提供。GDS 是一个独立的插件，包含 50+ 种图算法。

```
Cypher 查询 → GDS API → Graph Projection (内存中的图副本) → 算法执行 → Stream / Write 回 Neo4j
```

**Graph Projection（图投影）**：GDS 不在原始图上直接运算，而是将图的指定部分投影到内存中，形成一个只读的高效数据结构。这允许：
- 只投影需要的节点标签和关系类型
- 多个算法共享同一个投影（节省创建时间）
- 不影响原始图的读写操作

### 1.2 执行模式

| 模式 | 语法 | 用途 | 性能 |
|------|------|------|------|
| **stream** | `gds.xxx.stream()` | 返回结果到客户端 | 最快（不写回） |
| **mutate** | `gds.xxx.mutate()` | 写入 GDS 内存投影 | 快（不影响 Neo4j） |
| **write** | `gds.xxx.write()` | 写回 Neo4j 数据库 | 最慢（含持久化开销） |
| **stats** | `gds.xxx.stats()` | 返回统计信息 | 中等 |

**最佳实践**：先用 stream 模式验证结果，确认无误后再用 write 模式持久化。

### 1.3 图投影操作

```cypher
// 创建命名图投影
CALL gds.graph.project(
  'social-graph',              // 投影名称
  'Person',                    // 节点标签
  'FRIENDS_WITH',              // 关系类型
  {
    nodeProperties: ['age'],    // 节点属性
    relationshipProperties: ['weight']  // 关系属性
  }
)

// 创建无向图投影
CALL gds.graph.project(
  'social-undirected',
  'Person',
  {
    FRIENDS_WITH: { orientation: 'UNDIRECTED' }
  }
)

// 带过滤的投影
CALL gds.graph.project.cypher(
  'filtered-graph',
  'MATCH (p:Person) WHERE p.active = true RETURN id(p) AS id',
  'MATCH (a:Person)-[r:FRIENDS_WITH]->(b:Person) WHERE r.weight > 0.5 RETURN id(a) AS source, id(b) AS target, r.weight AS weight'
)

// 查看投影信息
CALL gds.graph.list()

// 删除投影
CALL gds.graph.drop('social-graph')
```

---

## 二、路径算法

### 2.1 最短路径算法族

| 算法 | 加权 | 多条路径 | 适用场景 | 时间复杂度 |
|------|------|---------|---------|-----------|
| **Dijkstra** | 是 | 否 | 加权最短路径 | O((V+E)logV) |
| **A*** | 是 | 否 | 有启发函数的最短路径 | O((V+E)logV)，实践中更快 |
| **Yen's K-Shortest** | 是 | 是 (K条) | 多路径备份 | O(KV(E+VlogV)) |
| **BFS** | 否 | 否 | 无权最短路径 | O(V+E) |
| **Delta-Stepping** | 是 | 否 | 并行最短路径 | O(V+E)/P (P=并行度) |
| **All Pairs** | 否 | 是 | 所有节点对最短路径 | O(V(V+E)) |

### 2.2 Dijkstra 最短路径

```cypher
// 创建带权重的图投影
CALL gds.graph.project(
  'road-network',
  'Location',
  'ROAD',
  { relationshipProperties: ['distance'] }
)

// 单源单目标最短路径
CALL gds.shortestPath.dijkstra.stream({
  graphName: 'road-network',
  sourceNode: '北京',
  targetNode: '上海',
  relationshipWeightProperty: 'distance'
})
YIELD index, sourceNode, targetNode, totalCost, nodeIds, costs, path
RETURN gds.util.asNode(sourceNode).name AS 起点,
       gds.util.asNode(targetNode).name AS 终点,
       totalCost AS 总距离,
       [nodeId IN nodeIds | gds.util.asNode(nodeId).name] AS 路径,
       costs AS 累计距离

// 单源到所有节点的最短路径
CALL gds.shortestPath.dijkstra.stream({
  graphName: 'road-network',
  sourceNode: '北京',
  relationshipWeightProperty: 'distance'
})
YIELD targetNode, totalCost
WHERE totalCost < 2000
RETURN gds.util.asNode(targetNode).name AS 目标, totalCost AS 距离
ORDER BY distance
LIMIT 20
```

### 2.3 Yen's K 最短路径

```cypher
// 查找 K 条最短路径（用于路由备份）
CALL gds.shortestPath.yens.stream({
  graphName: 'road-network',
  sourceNode: '北京',
  targetNode: '上海',
  k: 3,                         // 找 3 条最短路径
  relationshipWeightProperty: 'distance'
})
YIELD index, totalCost, path
RETURN index AS 第几条,
       totalCost AS 总距离,
       [n IN nodes(path) | n.name] AS 路径
ORDER BY totalCost
```

### 2.4 A* 算法

A* 算法使用启发函数（通常是地理距离）来加速搜索。

```cypher
// A* 算法需要节点有坐标属性
CALL gds.shortestPath.astar.stream({
  graphName: 'road-network',
  sourceNode: '北京',
  targetNode: '上海',
  relationshipWeightProperty: 'distance',
  latitudeProperty: 'lat',
  longitudeProperty: 'lon'
})
YIELD totalCost, path
RETURN totalCost, [n IN nodes(path) | n.name]
```

---

## 三、中心性算法

中心性算法衡量节点在网络中的重要性。不同的中心性指标衡量不同维度的"重要"。

### 3.1 四种中心性对比

| 算法 | 衡量维度 | 通俗解释 | 典型应用 |
|------|---------|---------|---------|
| **Degree** | 连接数量 | 谁的朋友最多 | 社交网络活跃度 |
| **Betweenness** | 桥梁作用 | 谁是信息必经之路 | 关键节点/瓶颈检测 |
| **Closeness** | 到达效率 | 谁到所有人最近 | 信息传播效率 |
| **PageRank** | 影响力 | 谁被重要的人认可 | 网页排名、影响力传播 |

### 3.2 PageRank

```cypher
// PageRank 基础用法
CALL gds.pageRank.stream({
  graphName: 'social-graph',
  maxIterations: 20,            // 最大迭代次数
  dampingFactor: 0.85,          // 阻尼因子（默认 0.85）
  tolerance: 0.0000001          // 收敛阈值
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
RETURN node.name, score
ORDER BY score DESC
LIMIT 20

// 个性化 PageRank（从特定节点出发计算影响力）
CALL gds.pageRank.stream({
  graphName: 'social-graph',
  maxIterations: 50,
  dampingFactor: 0.85,
  sourceNodes: [gds.util.asNode('张三')]  // 从张三出发
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name, score
ORDER BY score DESC
LIMIT 10
```

**PageRank 原理**：
1. 每个节点初始分数 = 1/N（N = 节点总数）
2. 每轮迭代：节点的新分数 = (1-d)/N + d * Σ(邻居的出度分之一 * 邻居的分数)
3. d（damping factor）= 0.85，表示 85% 的概率沿着链接走，15% 的概率随机跳转
4. 通常 20 次迭代后收敛

**性能基准**（100 万节点、500 万关系）：
- PageRank 20 次迭代：~15-30 秒（stream 模式）
- 写回 Neo4j：额外 ~10 秒

### 3.3 Betweenness 中心性

```cypher
// 介数中心性（识别网络中的桥梁节点）
CALL gds.betweenness.stream({
  graphName: 'social-graph',
  samplingSize: 1000           // 采样大小（精确算法太慢时使用近似）
})
YIELD nodeId, score
WITH gds.util.asNode(nodeId) AS node, score
WHERE score > 0
RETURN node.name, score
ORDER BY score DESC
LIMIT 10

// 使用 write 模式持久化结果
CALL gds.betweenness.write({
  graphName: 'social-graph',
  writeProperty: 'betweenness'
})

// 介数中心性应用场景：
// 1. 识别关键节点：如果删除该节点，网络通信效率下降最多
// 2. 信息传播瓶颈：信息流动必须经过的节点
// 3. 社区边界：高介数中心性的节点通常位于社区边界
```

**性能分析**：精确的 Betweenness 计算需要对每个节点执行 BFS，复杂度为 O(V * E)。对于大型图：
- 1000 节点：~1 秒
- 10 万节点：~10 分钟
- 100 万节点：~数小时（必须使用采样）

```cypher
// 使用近似算法
CALL gds.betweenness.stream({
  graphName: 'social-graph',
  samplingSize: 500,           // 只随机选择 500 个源节点做 BFS
  strategy: 'random'
})
```

### 3.4 Degree 中心性

```cypher
// 度中心性
CALL gds.degree.stream({
  graphName: 'social-graph'
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, 
       score AS connections
ORDER BY score DESC
LIMIT 20

// 归一化度中心性（值在 0-1 之间）
CALL gds.degree.stream({
  graphName: 'social-graph',
  orientation: 'REVERSE'    // 入度（谁被最多人关注）
})
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name, score
ORDER BY score DESC
```

### 3.5 Closeness 中心性

```cypher
// 接近中心性
CALL gds.closeness.stream({
  graphName: 'social-graph'
})
YIELD nodeId, centrality
RETURN gds.util.asNode(nodeId).name, centrality
ORDER BY centrality DESC
LIMIT 10

// Harmonic Closeness（处理不连通图更稳健）
CALL gds.closeness.harmonic.stream({
  graphName: 'social-graph'
})
YIELD nodeId, centrality
RETURN gds.util.asNode(nodeId).name, centrality
ORDER BY centrality DESC
```

---

## 四、社区检测算法

社区检测识别图中紧密连接的子群体。

### 4.1 Louvain 算法

Louvain 是最常用的社区检测算法，基于模块度（Modularity）优化。

**原理**：
1. **阶段一**：每个节点初始为一个社区。遍历每个节点，将其移入使模块度增益最大的邻居社区。重复直到无增益。
2. **阶段二**：将每个社区合并为一个超级节点，社区间关系合并为加权关系。
3. 重复阶段一和二，直到模块度不再提升。

**模块度 Q 的定义**：
```
Q = (1/2m) * Σ[Aij - (ki*kj)/(2m)] * δ(ci, cj)
```
- Aij：节点 i 和 j 之间的边权重
- ki, kj：节点 i, j 的度
- δ(ci, cj)：i 和 j 是否在同一社区
- Q ∈ [-0.5, 1]，Q > 0.3 表示较好的社区结构

```cypher
// Louvain 社区检测
CALL gds.louvain.stream({
  graphName: 'social-graph',
  maxLevels: 10,               // 最大层级数
  maxIterations: 10,           // 每层级最大迭代
  tolerance: 0.0001            // 收敛阈值
})
YIELD nodeId, communityId, intermediateCommunityIds
WITH gds.util.asNode(nodeId) AS node, communityId
RETURN communityId, 
       count(node) AS 社区大小,
       collect(node.name)[..5] AS 代表成员
ORDER BY 社区大小 DESC
LIMIT 10

// 写回社区 ID
CALL gds.louvain.write({
  graphName: 'social-graph',
  writeProperty: 'community'
})

// 多层级 Louvain（保留中间层级社区结构）
CALL gds.louvain.stream({
  graphName: 'social-graph',
  includeIntermediateCommunities: true
})
YIELD nodeId, intermediateCommunityIds
RETURN gds.util.asNode(nodeId).name,
       intermediateCommunityIds[0] AS 大社区,
       intermediateCommunityIds[1] AS 子社区
```

### 4.2 Label Propagation

```cypher
// 标签传播算法（比 Louvain 更快，精度略低）
CALL gds.labelPropagation.stream({
  graphName: 'social-graph',
  maxIterations: 10,
  nodeWeightProperty: 'weight'  // 节点权重（可选）
})
YIELD nodeId, communityId
RETURN communityId, 
       count(*) AS size,
       collect(gds.util.asNode(nodeId).name)[..3] AS sample
ORDER BY size DESC
```

**Louvain vs Label Propagation 对比**：

| 特性 | Louvain | Label Propagation |
|------|---------|-------------------|
| 精度 | 高（优化模块度） | 中（贪心策略） |
| 速度 | 中 | 快（通常快 3-5 倍） |
| 确定性 | 确定性（相同输入→相同输出） | 非确定性（依赖初始化） |
| 层级结构 | 支持（多层级） | 不支持 |
| 适用 | 离线分析 | 实时/增量更新 |

### 4.3 弱连通分量（WCC）

```cypher
// 弱连通分量：找出图中互不连通的子图
CALL gds.wcc.stream({
  graphName: 'social-graph'
})
YIELD nodeId, componentId
RETURN componentId, 
       count(*) AS componentSize
ORDER BY componentSize DESC

// WCC 的实际用途：
// 1. 数据质量检查：预期全连通的图如果 WCC>1，说明有孤立子图
// 2. 并行处理：每个连通分量可以独立处理
// 3. 欺诈检测：异常的小连通分量可能是欺诈团伙
```

### 4.4 三角计数与聚类系数

```cypher
// 全局聚类系数（衡量网络的"小世界"特性）
CALL gds.triangleCount.stream({
  graphName: 'social-graph'
})
YIELD nodeId, triangleCount
WITH gds.util.asNode(nodeId) AS node, triangleCount
RETURN node.name, triangleCount
ORDER BY triangleCount DESC
LIMIT 10

// 局部聚类系数
CALL gds.localClusteringCoefficient.stream({
  graphName: 'social-graph'
})
YIELD nodeId, localClusteringCoefficient
RETURN gds.util.asNode(nodeId).name, localClusteringCoefficient
ORDER BY localClusteringCoefficient DESC
LIMIT 10
```

---

## 五、相似度算法

### 5.1 Jaccard 相似度

```cypher
// 手动计算 Jaccard 相似度（纯 Cypher）
MATCH (p1:Person)-[:LIKES]->(item)<-[:LIKES]-(p2:Person)
WHERE p1 <> p2
WITH p1, p2, count(item) AS intersection
MATCH (p1)-[:LIKES]->(i1)
WITH p1, p2, intersection, count(i1) AS set1
MATCH (p2)-[:LIKES]->(i2)
WITH p1, p2, intersection, set1, count(i2) AS set2
WITH p1, p2, intersection, set1, set2,
     toFloat(intersection) / (set1 + set2 - intersection) AS jaccard
WHERE jaccard > 0.3
RETURN p1.name, p2.name, jaccard
ORDER BY jaccard DESC

// 使用 GDS 计算（更高效）
CALL gds.nodeSimilarity.stream({
  graphName: 'social-graph',
  topK: 10,                    // 每个节点保留 topK 个最相似节点
  similarityCutoff: 0.1        // 相似度下限
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name AS person1,
       gds.util.asNode(node2).name AS person2,
       similarity
ORDER BY similarity DESC
```

### 5.2 Cosine 相似度

```cypher
// 余弦相似度（用于向量属性比较）
CALL gds.similarity.cosine.stream({
  graphName: 'user-vectors',
  nodeProperties: ['feature_1', 'feature_2', 'feature_3', 'feature_4', 'feature_5'],
  topK: 5
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name, gds.util.asNode(node2).name, similarity
ORDER BY similarity DESC
```

### 5.3 Pearson 相关性

```cypher
CALL gds.similarity.pearson.stream({
  graphName: 'user-ratings',
  nodeProperties: ['rating_movie1', 'rating_movie2', 'rating_movie3'],
  topK: 3
})
YIELD node1, node2, similarity
RETURN gds.util.asNode(node1).name, gds.util.asNode(node2).name, similarity
```

---

## 六、Python 集成实战

### 6.1 社交网络分析

```python
from neo4j import GraphDatabase

class SocialNetworkAnalyzer:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def setup_graph(self):
        """创建图投影"""
        with self.driver.session() as session:
            # 创建投影
            session.run("""
                CALL gds.graph.project(
                    'social',
                    'Person',
                    { FRIENDS_WITH: { orientation: 'UNDIRECTED' } }
                )
            """)
    
    def find_influencers(self, top_k=10):
        """找影响力最大的用户（综合 PageRank + Betweenness）"""
        with self.driver.session() as session:
            # PageRank
            pr_result = session.run("""
                CALL gds.pageRank.stream('social', {maxIterations: 20})
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS name, score AS pagerank
                ORDER BY score DESC LIMIT $k
            """, k=top_k * 3).data()
            
            # Betweenness
            bc_result = session.run("""
                CALL gds.betweenness.stream('social', {samplingSize: 500})
                YIELD nodeId, score
                RETURN gds.util.asNode(nodeId).name AS name, score AS betweenness
                ORDER BY score DESC LIMIT $k
            """, k=top_k * 3).data()
            
            # 合并评分
            scores = {}
            for i, r in enumerate(pr_result):
                scores[r['name']] = scores.get(r['name'], 0) + (1 - i / len(pr_result))
            for i, r in enumerate(bc_result):
                scores[r['name']] = scores.get(r['name'], 0) + (1 - i / len(bc_result))
            
            ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            return [{"name": name, "composite_score": score} 
                    for name, score in ranked[:top_k]]
    
    def detect_communities(self):
        """检测社交社区"""
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.louvain.write('social', {writeProperty: 'community'})
                YIELD communityCount, modularity
                RETURN communityCount, modularity
            """).single()
            
            # 获取每个社区的详情
            communities = session.run("""
                MATCH (p:Person)
                RETURN p.community AS community, 
                       count(p) AS size,
                       collect(p.name)[..5] AS members
                ORDER BY size DESC
            """).data()
            
            return {
                "total_communities": result["communityCount"],
                "modularity": result["modularity"],
                "communities": communities
            }
    
    def recommend_friends(self, user_name, top_k=5):
        """好友推荐：基于共同好友"""
        with self.driver.session() as session:
            result = session.run("""
                CALL gds.nodeSimilarity.stream('social', {topK: 100})
                YIELD node1, node2, similarity
                WITH gds.util.asNode(node1) AS p1, 
                     gds.util.asNode(node2) AS p2, 
                     similarity
                WHERE p1.name = $name
                  AND NOT (p1)-[:FRIENDS_WITH]-(p2)
                RETURN p2.name AS candidate, similarity
                ORDER BY similarity DESC LIMIT $k
            """, name=user_name, k=top_k).data()
            return result
    
    def cleanup(self):
        with self.driver.session() as session:
            session.run("CALL gds.graph.drop('social', false)")
        self.driver.close()

# 使用
analyzer = SocialNetworkAnalyzer("bolt://localhost:7687", "neo4j", "password")
influencers = analyzer.find_influencers()
communities = analyzer.detect_communities()
recommendations = analyzer.recommend_friends("张三")
analyzer.cleanup()
```

### 6.2 推荐系统

```python
class RecommendationEngine:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def collaborative_filtering(self, user_name, top_k=10):
        """协同过滤推荐"""
        with self.driver.session() as session:
            result = session.run("""
                // 找到与目标用户相似的用户
                MATCH (target:User {name: $name})-[:BOUGHT]->(p:Product)
                WITH target, collect(id(p)) AS target_products
                
                MATCH (target)-[:BOUGHT]->(common:Product)<-[:BOUGHT]-(other:User)
                WHERE other <> target
                WITH target, other, count(common) AS common_buys,
                     target_products
                
                // 找相似用户买了但目标用户没买的产品
                MATCH (other)-[:BOUGHT]->(rec:Product)
                WHERE NOT id(rec) IN target_products
                WITH rec, sum(common_buys) AS score
                RETURN rec.name AS product, rec.price AS price, score
                ORDER BY score DESC LIMIT $k
            """, name=user_name, k=top_k).data()
            return result
    
    def graph_based_recommend(self, user_name, top_k=10):
        """基于图算法的推荐（PersonalRank）"""
        with self.driver.session() as session:
            # 先创建投影
            session.run("""
                CALL gds.graph.project.cypher(
                    'reco-graph',
                    'MATCH (n) RETURN id(n) AS id, labels(n)[0] AS labels',
                    'MATCH (a)-[r]->(b) RETURN id(a) AS source, id(b) AS target'
                )
            """)
            
            # 个性化 PageRank
            result = session.run("""
                MATCH (u:User {name: $name})
                CALL gds.pageRank.stream('reco-graph', {
                    sourceNodes: [u],
                    maxIterations: 20
                })
                YIELD nodeId, score
                WITH gds.util.asNode(nodeId) AS node, score
                WHERE node:Product AND NOT ()-[:BOUGHT]->(node)<-[:BOUGHT]-(:User {name: $name})
                RETURN node.name AS product, score
                ORDER BY score DESC LIMIT $k
            """, name=user_name, k=top_k).data()
            
            session.run("CALL gds.graph.drop('reco-graph', false)")
            return result
```

---

## 七、性能分析与对比

### 7.1 算法性能基准

测试环境：100 万节点、1000 万关系、32GB 内存、8 核 CPU

| 算法 | 时间 | 内存占用 | 可扩展性 |
|------|------|---------|---------|
| **PageRank** (20 iter) | ~25s | ~2GB | 良好 |
| **Betweenness** (精确) | ~4h | ~8GB | 差 |
| **Betweenness** (采样 500) | ~3min | ~2GB | 中等 |
| **Degree** | ~5s | ~0.5GB | 优秀 |
| **Louvain** | ~40s | ~3GB | 良好 |
| **Label Propagation** | ~10s | ~1GB | 良好 |
| **WCC** | ~8s | ~1GB | 优秀 |
| **Node Similarity** | ~5min | ~4GB | 中等 |
| **Shortest Path (Dijkstra)** | ~2ms (单对) | ~0.1GB | 优秀 |

### 7.2 算法选择决策树

```
想解决什么问题？
├── 谁最重要/有影响力？
│   ├── 有向图（如网页链接） → PageRank
│   ├── 无向图，看连接数 → Degree 中心性
│   ├── 找信息瓶颈 → Betweenness 中心性
│   └── 找传播中心 → Closeness 中心性
├── 有哪些社区/群体？
│   ├── 需要层级结构 → Louvain
│   ├── 需要快速结果 → Label Propagation
│   └── 检查图是否连通 → WCC
├── 节点之间多相似？
│   ├── 基于共同邻居 → Node Similarity (Jaccard)
│   ├── 基于属性向量 → Cosine Similarity
│   └── 基于评分 → Pearson Correlation
├── 最短/最优路径？
│   ├── 无权重 → BFS / shortestPath
│   ├── 加权非负 → Dijkstra
│   ├── 加权 + 启发函数 → A*
│   └── 需要多条路径 → Yen's K-Shortest
└── 节点嵌入？
    └── 需要向量表示 → FastRP / Node2Vec
```

### 7.3 常见坑点

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| GDS 报 OOM | 图投影太大，超过堆内存 | 减少投影范围，或增加 `dbms.memory.heap.max_size` |
| Betweenness 运行极慢 | 精确算法 O(V*E) | 使用采样 `samplingSize: 500` |
| PageRank 不收敛 | 图中有悬挂节点（出度=0） | 增大 `maxIterations`，或先清理悬挂节点 |
| Louvain 社区数=1 | 图本身连通度很高 | 调低 `tolerance` 或检查数据 |
| Node Similarity 很慢 | 节点的共同邻居太多 | 设置 `similarityCutoff` 和 `topK` 过滤 |
| 投影创建失败 | 标签/类型不存在 | 检查 `MATCH (n:Label) RETURN count(n)` 确认数据存在 |
| 写回属性冲突 | 多个算法写入同一属性 | 使用不同的 `writeProperty` |
| 算法结果每次不同 | Label Propagation 非确定性 | 使用 Louvain 或设置随机种子 |
