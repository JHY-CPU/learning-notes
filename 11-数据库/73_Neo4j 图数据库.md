# Neo4j 图数据库


## 🕸️ Neo4j 图数据库


图数据库概念、Neo4j 模型（Node/Relationship/Property/Label）、Cypher 查询语言（MATCH/CREATE/WHERE/RETURN）、图遍历算法（最短路径/PageRank/社区发现）、社交关系建模、推荐系统、知识图谱。


## 图数据库概述


```
// ========== 图数据库 ==========
// 用图结构存储数据: 节点 (Node) + 关系 (Relationship)
// 关系是图数据库的核心优势

// ========== 为什么需要图数据库? ==========
// RDB 做多对多关系查询: N 张关联表 + 多次 JOIN
// 图数据库: 直接从节点出发遍历关系
//
// 例: "朋友的朋友的朋友"
// SQL: user JOIN friend JOIN friend JOIN friend (3 次 JOIN)
// Neo4j: MATCH (u:User)-[:FRIEND*3]->(f) (一行)

// ========== 核心概念 ==========
// ┌──────────────┬─────────────────────────────┐
// │ 图           │ 说明                       │
// ├──────────────┼─────────────────────────────┤
// │ Node (节点)  │ 实体 (人/商品/地点)        │
// │ Relationship│ 关系 (朋友/购买/位于)      │
// │ Property    │ 属性 (姓名/价格/坐标)      │
// │ Label       │ 标签 (User/Product/City)   │
// └──────────────┴─────────────────────────────┘

// ========== 适用场景 ==========
// ✅ 社交网络 (好友/关注/推荐)
// ✅ 推荐系统 (商品/内容/用户)
// ✅ 知识图谱 (实体关系)
// ✅ 欺诈检测 (异常环/团伙)
// ✅ 权限管理 (角色继承)
// ✅ 路线规划 (最短路径)

// ❌ 大量简单 CRUD (不如 RDB/NoSQL)
// ❌ 大范围聚合统计
// ❌ 作为主数据库
```


## Cypher 基础


```
// ========== Cypher 查询语言 ==========
// Neo4j 的图查询语言 (类 ASCII 艺术)

// ========== 节点语法 ==========
()              // 匿名节点
(u:User)        // 带标签
(u:User:Admin)  // 多标签
(u {name: "Alice"})  // 带属性
(u:User {name: "Alice", age: 30})

// ========== 关系语法 ==========
-->             // 有向关系
-[r:KNOWS]->    // 命名 + 类型
-[r:KNOWS {since: 2020}]->  // 带属性
-[*1..3]->      // 可变长度 (1 到 3 跳)
-[:FRIEND*]->   // 任意深度

// ========== CREATE ==========
// 创建节点:
CREATE (u:User {name: "Alice", age: 30})
CREATE (p:Product {name: "MacBook", price: 9999})

// 创建关系:
MATCH (u:User {name: "Alice"})
MATCH (p:Product {name: "MacBook"})
CREATE (u)-[:BOUGHT {date: "2024-01-01"}]->(p)

// 简化版 (一条语句):
CREATE (a:User {name: "Alice"})-[:KNOWS {since: 2020}]->(b:User {name: "Bob"})

// ========== 批量创建 ==========
UNWIND [
    {name: "Alice", age: 30},
    {name: "Bob", age: 25},
    {name: "Charlie", age: 35}
] AS user
CREATE (u:User {name: user.name, age: user.age})
```


## Cypher 查询


```
// ========== MATCH 查询 ==========
// 查找 Alice 的朋友:
MATCH (a:User {name: "Alice"})-[:KNOWS]->(friend:User)
RETURN friend.name, friend.age

// 查找 Alice 的朋友的朋友 (2 度):
MATCH (a:User {name: "Alice"})-[:KNOWS*2]->(fof:User)
RETURN DISTINCT fof.name

// 查找 Alice 的朋友买过的商品:
MATCH (a:User {name: "Alice"})-[:KNOWS]->(friend)-[:BOUGHT]->(product)
RETURN friend.name, product.name

// 最短路径:
MATCH p = SHORTEST 1 (a:User {name: "Alice"})-[:KNOWS*]-(b:User {name: "David"})
RETURN length(p) AS hops, [node IN nodes(p) | node.name] AS path

// ========== WHERE 过滤 ==========
MATCH (u:User)-[:BOUGHT]->(p:Product)
WHERE u.age > 25 AND p.price > 5000
RETURN u.name, p.name, p.price

// ========== 聚合 ==========
// 按商品统计购买人数:
MATCH (u:User)-[:BOUGHT]->(p:Product)
RETURN p.name, COUNT(u) AS buyers, AVG(u.age) AS avg_age
ORDER BY buyers DESC

// ========== DELETE ==========
// 删除节点和关系:
MATCH (u:User {name: "Alice"})
DETACH DELETE u    // 删除节点及其所有关系
```


## 图算法与建模


```
// ========== 图算法 (Neo4j GDS) ==========
// Graph Data Science 库

// ========== 1. 社区发现 ==========
// 标签传播 (Label Propagation):
CALL gds.labelPropagation.write({
  nodeProjection: 'User',
  relationshipProjection: 'KNOWS',
  writeProperty: 'community'
})

// 查询同一社区的成员:
MATCH (u:User)
WHERE u.community = 42
RETURN u.name

// ========== 2. 中心性 ==========
// PageRank (网页/用户影响力):
CALL gds.pageRank.write({
  nodeProjection: 'User',
  relationshipProjection: 'KNOWS',
  writeProperty: 'pagerank'
})

// 最有影响力的用户:
MATCH (u:User)
RETURN u.name, u.pagerank
ORDER BY u.pagerank DESC
LIMIT 10

// ========== 3. 路径发现 ==========
// 最短路径:
MATCH (a:User {name: "Alice"}), (b:User {name: "David"})
CALL gds.shortestPath.dijkstra({
  sourceNode: a,
  targetNode: b,
  relationshipProjection: 'KNOWS'
})
YIELD nodeIds, totalCost
RETURN nodeIds, totalCost

// ========== 社交网络建模 ==========
// ┌────────────────────────────────────┐
// │  (Alice) -[:KNOWS]-> (Bob)        │
// │    ↓                  ↓           │
// │  (Charlie) ←--[:KNOWS]--          │
// │    ↓                              │
// │  (David) -[:BOUGHT]-> MacBook     │
// └────────────────────────────────────┘

// 推荐: Alice 可能认识 David
// (因为 Alice→Charlie→David)
MATCH (a:User {name: "Alice"})-[:KNOWS*2]-(candidate:User)
WHERE NOT (a)-[:KNOWS]-(candidate)
RETURN candidate.name, COUNT(*) AS common_friends
ORDER BY common_friends DESC
```


## 知识图谱与最佳实践


```
// ========== 知识图谱 ==========
// 实体 - 关系 - 实体 三元组

// 创建一个知识图谱:
CREATE (e1:Entity {name: "秦始皇", type: "人物"})
CREATE (e2:Entity {name: "中国", type: "国家"})
CREATE (e3:Entity {name: "公元前221年", type: "年份"})
CREATE (e1)-[:事件 {type: "统一"}]->(e2)
CREATE (e2)-[:首都]->(c:City {name: "咸阳"})

// 查询知识图谱:
MATCH (p:Entity {type: "人物"})-[r]->(e:Entity)
RETURN p.name, type(r), e.name

// ========== 索引与约束 ==========
// 唯一约束:
CREATE CONSTRAINT unique_user_name
FOR (u:User) REQUIRE u.name IS UNIQUE

// 索引:
CREATE INDEX user_age_index FOR (u:User) ON (u.age)

// 全文索引:
CREATE FULLTEXT INDEX fulltext_product
FOR (p:Product) ON EACH [p.name, p.description]

// 全文搜索:
CALL db.index.fulltext.queryNodes('fulltext_product', 'iphone')
YIELD node, score
RETURN node.name, score

// ========== 最佳实践 ==========
// 1. 节点用 CamelCase 标签 (User, Product)
// 2. 关系用大写蛇形 (KNOWS, BOUGHT)
// 3. 属性用 snake_case
// 4. 避免过多属性在关系上
// 5. 用 WITH 管道传递结果
// 6. 用 PROFILE 分析查询性能
// 7. 合理使用索引
```


> **Note:** 💡 图数据库要点: 关系是一等公民, 遍历关系远比 RDB JOIN 高效; Cypher 语法直观 (ASCII 图); 适合社交/推荐/知识图谱/欺诈检测; 不擅长大量简单查询和聚合统计; 图算法 (PageRank/社区发现/最短路径) 提供高级分析能力; RDB + Neo4j 混合使用常见。


## 练习


<!-- Converted from: 73_Neo4j 图数据库.html -->
