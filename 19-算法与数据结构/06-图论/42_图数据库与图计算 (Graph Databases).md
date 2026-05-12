# Graph Databases

  图数据库使用图结构存储数据，如 Neo4j 的 Cypher 查询语言。

## 什么是图数据库

  图数据库以图的顶点和边为基本存储单元，适合处理高度关联的数据。与关系型数据库不同，图数据库中的关系是一等公民，查询多跳关系不需要 JOIN 操作，性能随跳数增加保持稳定。

## 关键性质

    - 属性图模型：顶点和边都可以有标签和属性
    - 原生图存储：数据以图结构存储，非关系型数据库的图扩展
    - 图遍历是核心操作，性能优于关系型 JOIN
    - 支持 Cypher、Gremlin 等图查询语言


```
// 图计算框架
// 1. BSP (Bulk Synchronous Parallel) — Pregel
//    - 超级步迭代计算
//    - 顶点间传递消息
//    - 典型应用：PageRank、最短路径
//
// 2. 图遍历框架 — Gremlin
//    - 图遍历查询语言
//    - 函数式风格
//
// 3. 属性图模型 — Neo4j
//    顶点有标签和属性
//    边有类型和属性
console.log('图数据库是 NoSQL 的重要分支');
```


## 主流产品与框架

    - **Neo4j：**最流行的图数据库，Cypher 查询语言
    - **JanusGraph：**分布式图数据库，支持多种后端存储
    - **Apache Spark GraphX：**大规模图计算框架
    - **NetworkX（Python）：**图分析库，适合研究和原型

## 复杂度分析

    - 图遍历查询：O(遍历的顶点和边数)，不受数据总量影响
    - 全图分析（如 PageRank）：需要分布式计算框架

## Cypher 查询示例

```cypher
-- Neo4j Cypher 查询语言

-- 查找某用户的朋友
MATCH (user:Person {name: "Alice"})-[:FRIEND]->(friend)
RETURN friend.name

-- 查找二度好友
MATCH (user:Person {name: "Alice"})-[:FRIEND*2]->(friendOfFriend)
WHERE friendOfFriend <> user
RETURN DISTINCT friendOfFriend.name

-- 查找最短路径
MATCH path = shortestPath(
  (a:Person {name: "Alice"})-[:FRIEND*]-(b:Person {name: "Bob"})
)
RETURN path

-- PageRank（使用 GDS 库）
CALL gds.pageRank.stream('social-network')
YIELD nodeId, score
RETURN gds.util.asNode(nodeId).name AS name, score
ORDER BY score DESC LIMIT 10
```

## 图计算框架详解

  ### Pregel (BSP 模型)
  - 超级步迭代：每个超级步中，每个顶点接收消息、计算、发送消息
  - 经典应用：PageRank、最短路径、连通分量
  - 代表实现：Apache Giraph、Spark GraphX

  ### 图遍历框架 (Gremlin)
  - 函数式图查询语言
  - 支持遍历、过滤、聚合操作
  - Apache TinkerPop 生态

## 选型指南

  | 需求 | 推荐方案 |
  | --- | --- |
  | 实时关系查询 | Neo4j |
  | 分布式图存储 | JanusGraph |
  | 大规模批处理 | Spark GraphX |
  | 快速原型 | NetworkX (Python) |
  | 图神经网络 | PyG / DGL |

## 常见陷阱

    - 图数据库不适合需要全表扫描的分析型查询
    - 选择属性图模型还是 RDF 三元组模型取决于应用场景
    - 大规模图的分布式存储和计算需要专门的工程方案
    - Neo4j 在单机上的数据量有上限，超大规模需用 JanusGraph 或 NebulaGraph
