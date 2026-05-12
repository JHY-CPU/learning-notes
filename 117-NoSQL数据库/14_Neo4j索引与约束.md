# Neo4j 索引与约束

## 一、索引底层原理

### 1.1 Neo4j 存储架构

Neo4j 的存储分为两层：

```
应用层 → [事务日志 (Transaction Log)] → [存储引擎]
                                            ├── 节点存储 (nodes.store)
                                            ├── 关系存储 (relationships.store)
                                            ├── 属性存储 (properties.store)
                                            └── 索引 (Lucene / Native B-Tree)
```

- **节点存储**：每个节点 15 字节（Neo4j 4.x），包含：第一个关系 ID、第一个属性 ID、标签信息
- **关系存储**：每个关系 34 字节，包含：起始节点 ID、终止节点 ID、关系类型、双向链表指针（prev/next for start node, prev/next for end node）
- **属性存储**：链表结构，每个属性包含 key ID 和 value（短值内联，长值引用外部存储）

**图遍历的核心优势**：从一个节点出发查找其关系，只需读取节点记录 → 获取 first_rel → 沿双向链表遍历关系。**时间复杂度 = O(该节点的关系数)**，与图的总大小无关。这就是图数据库在关系密集查询上碾压关系型数据库的根本原因。

### 1.2 B-Tree 索引（RANGE 索引）

Neo4j 默认使用 Lucene B-Tree 索引（Neo4j 4.x 及以下）或自研 Native B-Tree（Neo4j 5.x+）。

**Lucene B-Tree 原理**：
- B-Tree 是多路平衡搜索树，每个节点存储多个 key 和指向子节点的指针
- 叶子节点存储 `(key → [nodeID list])` 的映射
- 查询复杂度 O(log n)，n 是索引条目数

**Native B-Tree（Neo4j 5.x）**：
- 完全自研，不再依赖 Lucene
- 支持复合索引的原生优化
- 索引直接存储 nodeID → 存储位置的映射，跳过 Lucene 的间接层
- 性能比 Lucene B-Tree 提升约 20-30%

```
Native B-Tree 结构:
              [20, 50]
             /    |    \
        [5,12]  [25,35] [60,80]
        / | \    / | \    / | \
      ...     ...       ...
叶子节点: [key → InternalIdMapping] 
其中 InternalIdMapping = entityId → entityIdToken (存储层引用)
```

### 1.3 索引类型详解

| 索引类型 | 引擎 | 适用查询 | 支持版本 |
|---------|------|---------|---------|
| **RANGE** | Native B-Tree | 等值 (`=`)、范围 (`<`, `>`, `IN`)、STARTS WITH | 5.x 默认 |
| **TEXT** | Lucene Full-Text | 全文搜索、模糊匹配、CONTAINS | 3.x+ |
| **POINT** | Native Point | 地理空间查询（距离、包含） | 3.4+ |
| **LOOKUP** | 内置 | 按标签/关系类型查找节点 | 4.0+ |
| **VECTOR** | Native Vector | 向量相似度搜索（ANN） | 5.11+ |
| **FULLTEXT** | Lucene Full-Text | 跨多个属性的全文搜索 | 3.5+ |

---

## 二、RANGE 索引

### 2.1 单属性索引

```cypher
// 创建索引（异步，默认方式）
CREATE INDEX person_name FOR (p:Person) ON (p.name)

// 创建索引并指定名称
CREATE INDEX person_name_idx FOR (p:Person) ON (p.name)

// 查看所有索引
SHOW INDEXES

// 查看索引详细信息
SHOW INDEX YIELD name, type, labelsOrTypes, properties, state, populationPercent

// 删除索引
DROP INDEX person_name_idx

// 强制使用特定索引
MATCH (p:Person)
USING INDEX p:Person(name)
WHERE p.name = "张三"
RETURN p
```

### 2.2 复合索引

复合索引覆盖多个属性，查询时必须按索引定义的顺序使用属性。

```cypher
// 创建复合索引
CREATE INDEX person_name_age FOR (p:Person) ON (p.name, p.age)

// 能利用此索引的查询
MATCH (p:Person) WHERE p.name = "张三" AND p.age > 25 RETURN p             // OK
MATCH (p:Person) WHERE p.name = "张三" RETURN p                             // OK (前缀)
MATCH (p:Person) WHERE p.name = "张三" AND p.age = 28 RETURN p             // OK

// 不能利用此索引的查询
MATCH (p:Person) WHERE p.age > 25 RETURN p                                  // 不能跳过第一个属性
MATCH (p:Person) WHERE p.age = 28 AND p.name = "张三" RETURN p              // 顺序不对（某些版本优化器可能自动调整）
```

**复合索引的内部结构**：复合索引的 key 是多个属性值的拼接（如 `"张三|28"`）。B-Tree 按字典序排列这些拼接 key。因此前缀查询能利用索引，非前缀查询不能。

### 2.3 索引选择性

索引的选择性 = 不同值的数量 / 总记录数。选择性越高，索引越有效。

```cypher
// 查看属性的基数
MATCH (p:Person)
RETURN count(DISTINCT p.gender) AS gender_cardinality,
       count(DISTINCT p.name) AS name_cardinality,
       count(*) AS total

// 结果示例:
// gender_cardinality: 2   (选择性极低，索引无意义)
// name_cardinality:   50000 (选择性高，索引有效)
// total:              100000

// 性别字段的索引几乎无用，因为每个值匹配 50% 的数据
// 名字字段的索引非常有效，因为每个值匹配 ~0.002% 的数据
```

**建议**：
- 选择性 < 0.01（1%）的属性不要建索引（如布尔值、性别）
- 选择性 > 0.1（10%）的属性建索引收益显著
- 对于选择性低但需要精确查询的场景，考虑复合索引（如 `(gender, name)`）

### 2.4 索引状态管理

```cypher
// 查看索引状态
SHOW INDEXES YIELD name, state, populationPercent, failureMessage

// 状态说明:
// ONLINE - 索引正常
// POPULATING - 正在构建（异步）
// FAILED - 构建失败
// ONLINE+FAILED_POPULATING - 构建失败但旧数据仍可用

// 强制重建索引
DROP INDEX person_name IF EXISTS;
CREATE INDEX person_name FOR (p:Person) ON (p.name);

// 等待所有索引就绪
CALL db.awaitIndexes(300)  // 最多等 300 秒
```

---

## 三、全文搜索索引

### 3.1 FULLTEXT 索引

全文搜索索引基于 Apache Lucene，支持分词、停用词、同义词等特性。

```cypher
// 创建全文索引（跨多个标签和属性）
CREATE FULLTEXT INDEX article_search FOR (a:Article) ON EACH [a.title, a.content]
CREATE FULLTEXT INDEX product_search FOR (p:Product) ON EACH [p.name, p.description]

// 全文查询
CALL db.index.fulltext.queryNodes("article_search", "机器学习 AND 深度学习")
YIELD node, score
WHERE score > 1.0
RETURN node.title, node.content, score
ORDER BY score DESC
LIMIT 10

// 支持的查询语法（Lucene Query Parser 语法）:
// "机器学习"          — 精确词匹配
// "机器 深度学习"     — OR (默认)
// "机器 AND 深度学习" — AND
// "机器 NOT 监督"     — 排除
// "机器~"             — 模糊匹配（编辑距离）
// "机*"               — 通配符前缀
// "title:机器"        — 指定字段
// title:"机器学习"    — 短语匹配

// 带评分阈值的查询
CALL db.index.fulltext.queryNodes("article_search", "Python 机器学习")
YIELD node, score
WHERE score > 0.5
RETURN node.title, score

// 全文索引搜索关系
CREATE FULLTEXT INDEX rel_search FOR ()-[r:REVIEWED]-() ON EACH [r.comment]
CALL db.index.fulltext.queryRelationships("rel_search", "很好 推荐")
YIELD relationship, score
RETURN relationship.comment, score
```

### 3.2 TEXT 索引（字符串索引）

TEXT 索引用于 `CONTAINS` 和 `ENDS WITH` 查询，比 RANGE 索引更灵活但更慢。

```cypher
// 创建 TEXT 索引
CREATE TEXT INDEX person_email_text FOR (p:Person) ON (p.email)

// CONTAINS 查询（利用 TEXT 索引）
MATCH (p:Person)
WHERE p.email CONTAINS "gmail"
RETURN p.name, p.email

// ENDS WITH 查询
MATCH (p:Person)
WHERE p.email ENDS WITH ".com"
RETURN p.name, p.email

// RANGE 索引 vs TEXT 索引对比
// RANGE: 支持 =, <, >, IN, STARTS WITH
// TEXT:  支持 CONTAINS, ENDS WITH（RANGE 索引不支持）
```

**常见坑点**：
- 不要对同一属性同时创建 RANGE 和 TEXT 索引，Neo4j 会根据查询类型自动选择
- `CONTAINS` 查询即使有 TEXT 索引，仍然比 `=` 查询慢很多（需要遍历后缀树）
- 全文索引需要额外的 Lucene 存储空间（约为原始数据的 1-2 倍）

---

## 四、向量索引

### 4.1 向量索引基础（Neo4j 5.11+）

向量索引用于嵌入向量的近似最近邻（ANN）搜索，适用于推荐系统、语义搜索等场景。

```cypher
// 创建向量索引
CREATE VECTOR INDEX movie_embeddings FOR (m:Movie) ON (m.embedding)
OPTIONS {
  indexConfig: {
    `vector.dimensions`: 1536,         // OpenAI ada-002 嵌入维度
    `vector.similarity_function`: 'cosine'  // 余弦相似度
  }
}

// 查看向量索引
SHOW INDEXES YIELD name, type, labelsOrTypes, properties, options
WHERE type = "VECTOR"

// 向量相似度搜索
WITH [0.1, 0.2, -0.3, ...] AS query_embedding  // 1536 维
CALL db.index.vector.queryNodes(
  'movie_embeddings',   // 索引名
  10,                   // topK
  query_embedding       // 查询向量
) YIELD node, score
RETURN node.title, node.genres, score
ORDER BY score DESC
```

### 4.2 向量索引与 Python 集成

```python
from neo4j import GraphDatabase
import openai

def get_embedding(text):
    """获取 OpenAI 嵌入向量"""
    response = openai.embeddings.create(
        model="text-embedding-ada-002",
        input=text
    )
    return response.data[0].embedding

class VectorSearch:
    def __init__(self, uri, user, password):
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
    
    def create_movie_with_embedding(self, title, description, genres):
        embedding = get_embedding(f"{title}: {description}")
        with self.driver.session() as session:
            session.execute_write(
                lambda tx: tx.run(
                    "CREATE (m:Movie {title: $title, description: $desc, "
                    "genres: $genres, embedding: $embedding})",
                    title=title, desc=description,
                    genres=genres, embedding=embedding
                )
            )
    
    def semantic_search(self, query, top_k=10):
        """语义搜索电影"""
        query_embedding = get_embedding(query)
        with self.driver.session() as session:
            result = session.run(
                "CALL db.index.vector.queryNodes('movie_embeddings', $k, $vec) "
                "YIELD node, score "
                "RETURN node.title AS title, node.genres AS genres, score "
                "ORDER BY score DESC",
                k=top_k, vec=query_embedding
            )
            return [dict(r) for r in result]
    
    def hybrid_search(self, query, genre_filter=None, top_k=10):
        """混合搜索：向量相似度 + 属性过滤"""
        query_embedding = get_embedding(query)
        with self.driver.session() as session:
            result = session.run(
                "CALL db.index.vector.queryNodes('movie_embeddings', $k * 3, $vec) "
                "YIELD node, score "
                "WHERE ($genre IS NULL OR $genre IN node.genres) "
                "RETURN node.title AS title, score "
                "ORDER BY score DESC LIMIT $k",
                k=top_k, vec=query_embedding, genre=genre_filter
            )
            return [dict(r) for r in result]
```

### 4.3 向量索引性能

| 向量维度 | 索引构建时间 (100万向量) | 查询延迟 (top-10) | 召回率 |
|---------|----------------------|------------------|--------|
| 128 | ~2 min | ~2ms | ~95% |
| 768 (BERT) | ~8 min | ~5ms | ~95% |
| 1536 (OpenAI) | ~15 min | ~8ms | ~95% |
| 3072 (OpenAI large) | ~30 min | ~15ms | ~95% |

**底层实现**：Neo4j 向量索引使用 **Hierarchical Navigable Small World (HNSW)** 图算法。构建时将向量组织为多层图结构（类似 Skip List），查询时从顶层开始逐层下降，在每层贪心地走向最近邻。HNSW 的时间复杂度接近 O(log n)。

---

## 五、约束详解

### 5.1 约束类型

| 约束类型 | 语法 | 作用 | 底层实现 |
|---------|------|------|---------|
| **唯一性** | `IS UNIQUE` | 属性值唯一 | 隐式创建唯一索引 |
| **存在性** | `IS NOT NULL` | 属性必须存在 | 写入时检查 |
| **节点键** | `IS NODE KEY` | 唯一 + 存在 | 唯一索引 + 存在性检查 |
| **属性类型** | `IS :: TYPENAME` | 值必须是指定类型 | 写入时检查 |

### 5.2 唯一性约束

```cypher
// 单属性唯一性
CREATE CONSTRAINT unique_person_name FOR (p:Person) REQUIRE p.name IS UNIQUE

// 复合唯一性（两个属性的组合唯一）
CREATE CONSTRAINT unique_person_name_city FOR (p:Person) 
REQUIRE (p.name, p.city) IS UNIQUE

// 测试约束
CREATE (p:Person {name: "张三", city: "北京"})  // OK
CREATE (p:Person {name: "张三", city: "北京"})  // 失败: ConstraintViolation
CREATE (p:Person {name: "张三", city: "上海"})  // OK (不同组合)

// 查看约束
SHOW CONSTRAINTS

// 删除约束
DROP CONSTRAINT unique_person_name

// 带索引选项的约束
CREATE CONSTRAINT unique_email FOR (p:Person) REQUIRE p.email IS UNIQUE
OPTIONS {indexProvider: 'native-btree-1.0'}
```

**底层机制**：唯一性约束在底层创建一个唯一 B-Tree 索引。每次写入时，先在 B-Tree 中查找，如果 key 已存在则拒绝写入。这与关系型数据库的 UNIQUE INDEX 实现原理一致。

### 5.3 存在性约束

```cypher
// 节点属性必须存在
CREATE CONSTRAINT person_name_exists FOR (p:Person) REQUIRE p.name IS NOT NULL

// 节点属性必须存在 + 必须有特定标签
// (Neo4j 4.4+ 支持 EXISTS 语法)
CREATE CONSTRAINT person_required FOR (p:Person) REQUIRE p.name IS NOT NULL

// 关系属性存在性
CREATE CONSTRAINT rel_since_exists FOR ()-[r:FRIENDS_WITH]-() 
REQUIRE r.since IS NOT NULL

// 测试
CREATE (p:Person)                          // 失败: name 不存在
CREATE (p:Person {name: "张三"})           // OK
CREATE (p:Person {name: null})             // 失败: null 不满足 IS NOT NULL
```

### 5.4 节点键（NODE KEY）

节点键 = 唯一性 + 存在性，是数据完整性的最强保证。

```cypher
// 单属性节点键
CREATE CONSTRAINT person_key FOR (p:Person) REQUIRE p.email IS NODE KEY

// 复合节点键
CREATE CONSTRAINT person_composite_key FOR (p:Person) 
REQUIRE (p.tenant_id, p.user_id) IS NODE KEY

// 节点键的效果:
// 1. email 必须存在（IS NOT NULL）
// 2. email 必须唯一（IS UNIQUE）
// 3. 合并为一个约束 + 一个唯一索引
```

### 5.5 属性类型约束（Neo4j 5.x）

```cypher
// 确保属性类型
CREATE CONSTRAINT age_is_integer FOR (p:Person) REQUIRE p.age IS :: INTEGER
CREATE CONSTRAINT price_is_float FOR (p:Product) REQUIRE p.price IS :: FLOAT
CREATE CONSTRAINT name_is_string FOR (p:Person) REQUIRE p.name IS :: STRING NOT NULL
CREATE CONSTRAINT tags_is_list FOR (p:Post) REQUIRE p.tags IS :: LIST<STRING>

// 支持的类型:
// INTEGER, FLOAT, STRING, BOOLEAN, DATE, DATETIME, 
// POINT, DURATION, LIST<inner_type>

// 组合约束
CREATE CONSTRAINT person_valid FOR (p:Person) 
REQUIRE p.email IS :: STRING NOT NULL
```

---

## 六、索引与约束的最佳实践

### 6.1 索引策略决策树

```
需要查询什么？
├── 精确匹配 / 范围查询 → RANGE 索引
│   ├── 单属性 → CREATE INDEX FOR (n:Label) ON (n.prop)
│   └── 多属性组合 → CREATE INDEX FOR (n:Label) ON (n.prop1, n.prop2)
├── 全文搜索 / 模糊匹配
│   ├── CONTAINS / ENDS WITH → TEXT 索引
│   └── 跨字段搜索 → FULLTEXT 索引
├── 地理位置 → POINT 索引
├── 向量相似度 → VECTOR 索引
└── 嵌入属性 (高维向量) → VECTOR 索引
```

### 6.2 紦束策略

```
数据完整性要求？
├── ID 必须唯一且存在 → NODE KEY 约束
├── 需要唯一但可为空 → UNIQUE 约束
├── 必须有值但不一定唯一 → EXISTS 约束
├── 类型必须正确 → 类型约束 (5.x)
└── 无特殊要求 → 不加约束（减少写入开销）
```

### 6.3 生产环境配置示例

```cypher
// ========== 用户系统 ==========
// 主键：邮箱必须唯一且存在
CREATE CONSTRAINT user_email_key FOR (u:User) REQUIRE u.email IS NODE KEY
// 用户名索引（用于搜索）
CREATE INDEX user_name FOR (u:User) ON (u.name)
// 创建时间索引（用于范围查询）
CREATE INDEX user_created FOR (u:User) ON (u.created_at)

// ========== 商品系统 ==========
// SKU 唯一
CREATE CONSTRAINT product_sku_key FOR (p:Product) REQUIRE p.sku IS NODE KEY
// 分类 + 价格复合索引（用于按分类+价格筛选）
CREATE INDEX product_cat_price FOR (p:Product) ON (p.category, p.price)
// 商品名全文搜索
CREATE FULLTEXT INDEX product_search FOR (p:Product) ON EACH [p.name, p.description]

// ========== 社交系统 ==========
// 好友关系的 since 属性必须存在
CREATE CONSTRAINT friend_since FOR ()-[r:FRIENDS_WITH]-() REQUIRE r.since IS NOT NULL
// 用户标签索引
CREATE INDEX user_tag FOR (u:User) ON (u.tag)
```

### 6.4 性能基准

在 100 万节点、500 万关系的图上测试：

| 操作 | 无索引 | 有索引 | 提升倍数 |
|------|--------|--------|---------|
| 按 name 查找节点 | ~120ms (全标签扫描) | ~1ms (B-Tree) | 120x |
| 按 name+age 复合查找 | ~150ms | ~1.5ms | 100x |
| 全文搜索 (Lucene) | N/A | ~10ms | — |
| 向量相似度 top-10 | 全量扫描 ~30s | ~8ms (HNSW) | 3750x |
| 唯一性约束检查 | ~0ms | ~0.5ms (额外开销) | — |

### 6.5 常见坑点

| 问题 | 原因 | 解决方案 |
|------|------|---------|
| 索引创建后状态一直是 POPULATING | 数据量大，构建慢 | 等待完成，监控 `populationPercent` |
| 复合索引只对前缀属性生效 | B-Tree 按拼接 key 排序 | 设计查询时考虑索引顺序 |
| 全文搜索不返回结果 | 停用词过滤 | 检查 Lucene 分词器配置 |
| 约束创建失败 | 已有数据违反约束 | 先清理数据再创建约束 |
| 索引不被使用 | 查询条件与索引不匹配 | 用 EXPLAIN 检查执行计划 |
| 写入性能下降 | 约束检查 + 索引更新 | 批量导入时先删约束/索引，导入完再创建 |
| 向量索引维度不匹配 | 查询向量与索引维度不一致 | 确保 `vector.dimensions` 配置正确 |
