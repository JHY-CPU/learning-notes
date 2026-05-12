# MongoDB 索引与查询优化

## 1. 索引概述

索引是 MongoDB 查询性能优化的核心机制。没有索引时，MongoDB 执行**全集合扫描（Collection Scan）**，逐个检查每个文档；使用索引后，可以快速定位目标数据。

### 索引的工作原理

MongoDB 使用 **B-Tree** 索引结构（WiredTiger 引擎），与 MySQL 的 B+Tree 类似：

```
全集合扫描: O(n) → 遍历所有文档
索引扫描:   O(log n) → 通过 B-Tree 快速定位
```

## 2. 单字段索引

```python
from pymongo import MongoClient, ASCENDING, DESCENDING

client = MongoClient("mongodb://localhost:27017/")
db = client["ecommerce"]
orders = db["orders"]

# 创建单字段升序索引
orders.create_index([("user_id", ASCENDING)])

# 创建单字段降序索引
orders.create_index([("created_at", DESCENDING)])

# 创建唯一索引
users = db["users"]
users.create_index([("email", ASCENDING)], unique=True)

# 创建稀疏索引（只索引包含该字段的文档）
users.create_index([("phone", ASCENDING)], sparse=True)

# 创建 TTL 索引（自动过期删除）
# 3600秒后自动删除文档
sessions = db["sessions"]
sessions.create_index([("expires_at", ASCENDING)], expireAfterSeconds=3600)
```

### 查看集合索引

```python
# 列出所有索引
for index in orders.list_indexes():
    print(f"索引名: {index['name']}, 键: {index['key']}")

# 获取索引信息统计
stats = orders.index_information()
print(stats)
```

## 3. 复合索引

复合索引包含多个字段，字段顺序至关重要。

```python
# 创建复合索引
orders.create_index([
    ("user_id", ASCENDING),
    ("status", ASCENDING),
    ("created_at", DESCENDING)
])
```

### 复合索引的最左前缀原则

以上述复合索引为例，可以高效支持以下查询：

```python
# ✓ 可以使用索引
orders.find({"user_id": "123"})
orders.find({"user_id": "123", "status": "paid"})
orders.find({"user_id": "123", "status": "paid"}).sort("created_at", -1)

# ✗ 无法有效使用索引（跳过了 user_id）
orders.find({"status": "paid"})
orders.find({"created_at": {"$gt": date}})

# ✓ 范围查询后的排序字段也能用到索引
orders.find({"user_id": "123", "created_at": {"$gt": date}}).sort("status", 1)
```

### 复合索引排序规则

```python
# 索引: {"user_id": 1, "created_at": -1}
# ✓ 排序方向匹配索引
orders.find({"user_id": "123"}).sort([("created_at", -1)])

# ✗ 排序方向不匹配，需要内存排序
orders.find({"user_id": "123"}).sort([("created_at", 1)])
```

## 4. 文本索引

文本索引支持对字符串内容进行全文搜索。

```python
# 创建文本索引（每个集合只能有一个）
articles = db["articles"]
articles.create_index([("title", "text"), ("content", "text")])

# 全文搜索
results = articles.find({"$text": {"$search": "MongoDB 数据库"}})

# 带权重的文本索引（标题权重高于内容）
articles.drop_index("title_text_content_text")
articles.create_index(
    [("title", "text"), ("content", "text")],
    weights={"title": 10, "content": 1},
    default_language="chinese",
    language_override="language"
)

# 搜索并按相关性分数排序
results = articles.find(
    {"$text": {"$search": "MongoDB"}},
    {"score": {"$meta": "textScore"}}
).sort([("score", {"$meta": "textScore"})])
```

## 5. 多键索引

对数组字段自动创建多键索引，数组中每个元素都会被索引。

```python
# products 集合中 tags 是数组字段
products = db["products"]
products.create_index([("tags", ASCENDING)])

# 查询包含特定标签的产品
products.find({"tags": "electronics"})
products.find({"tags": {"$all": ["electronics", "sale"]}})
```

## 6. explain() 查询分析

`explain()` 是分析查询性能的核心工具。

```python
# 查看查询执行计划
result = orders.find({"user_id": "123", "status": "paid"}).explain()

# executionStats 模式（推荐）
result = orders.find({"user_id": "123"}).explain("executionStats")

print(f"查询阶段: {result['queryPlanner']['winningPlan']['stage']}")
print(f"扫描文档数: {result['executionStats']['totalDocsExamined']}")
print(f"返回文档数: {result['executionStats']['nReturned']}")
print(f"执行时间: {result['executionStats']['executionTimeMillis']}ms")
```

### explain 输出关键指标

```python
# 判断是否使用了索引
explain_result = orders.find({"user_id": "123"}).explain("executionStats")
plan = explain_result["executionStats"]

# 关键指标解读
print(f"""
查询分析结果:
  扫描文档数 (totalDocsExamined): {plan['totalDocsExamined']}
  返回文档数 (nReturned):         {plan['nReturned']}
  执行时间 (executionTimeMillis): {plan['executionTimeMillis']}ms
  扫描阶段 (stage):               {explain_result['queryPlanner']['winningPlan']['stage']}
""")

# 判断查询效率
if plan['totalDocsExamined'] > plan['nReturned'] * 10:
    print("⚠️ 警告: 扫描文档数远大于返回文档数，考虑添加索引")
```

### 常见执行阶段

| Stage | 含义 |
|-------|------|
| `COLLSCAN` | 全集合扫描（无索引） |
| `IXSCAN` | 索引扫描 |
| `FETCH` | 根据索引获取完整文档 |
| `PROJECTION_COVERED` | 覆盖查询（不需要获取文档） |
| `SORT` | 内存排序（应避免） |
| `LIMIT` | 限制返回数量 |

## 7. 覆盖查询（Covered Query）

当查询的所有字段都包含在索引中，且不需要返回 `_id 以外的字段`，MongoDB 不需要读取文档，直接从索引返回结果。

```python
# 创建复合索引
users.create_index([("name", ASCENDING), ("age", ASCENDING)])

# 覆盖查询：查询条件和返回字段都在索引中
result = users.find(
    {"name": "张三"},
    {"_id": 0, "name": 1, "age": 1}  # 只返回索引包含的字段
).explain("executionStats")

# 检查是否为覆盖查询
stage = result["queryPlanner"]["winningPlan"]["inputStage"]["stage"]
print(f"执行阶段: {stage}")  # 如果是 IXSCAN 且没有 FETCH，就是覆盖查询
```

## 8. 索引管理与统计

```python
# 索引使用统计
stats = db.command("aggregate", "orders", pipeline=[
    {"$indexStats": {}}
])
for stat in stats["cursor"]["firstBatch"]:
    print(f"索引: {stat['name']}, 使用次数: {stat['accesses']['ops']}")

# 重建索引
orders.reindex()

# 删除指定索引
orders.drop_index("user_id_1")

# 删除除 _id 外的所有索引
orders.drop_indexes()

# 索引大小
index_stats = orders.index_information()
total_size = sum(
    info.get("totalIndexSize", 0)
    for info in index_stats.values()
)
print(f"总索引大小: {total_size / 1024 / 1024:.2f} MB")
```

## 9. 索引最佳实践

### 索引设计原则

```python
# 原则1: ESR 原则 (Equality → Sort → Range)
# 索引字段顺序应按 等值查询 → 排序 → 范围查询 排列
orders.create_index([
    ("user_id", ASCENDING),    # Equality
    ("status", ASCENDING),     # Equality
    ("created_at", ASCENDING), # Sort / Range
])

# 原则2: 选择性高的字段放前面
# user_id 比 status 选择性高
users.create_index([
    ("user_id", ASCENDING),   # 高选择性
    ("status", ASCENDING),    # 低选择性
])

# 原则3: 索引覆盖常用查询
# 分析最频繁的查询模式，针对性创建索引
```

### 索引监控脚本

```python
def analyze_slow_queries(db, collection_name, threshold_ms=100):
    """分析慢查询并建议索引"""
    collection = db[collection_name]

    # 获取 profiler 数据
    slow_queries = db["system.profile"].find({
        "op": {"$in": ["query", "update", "remove"]},
        "millis": {"$gt": threshold_ms}
    }).sort("millis", -1).limit(20)

    for query in slow_queries:
        print(f"慢查询: {query.get('command', query.get('query'))}")
        print(f"耗时: {query['millis']}ms")
        print(f"扫描: {query.get('docsExamined', 'N/A')} 文档")
        print(f"返回: {query.get('nreturned', 'N/A')} 文档")
        print("---")

# 启用 profiler
db.command("profile", 2, slowms=100)
```

### 常见索引问题排查

```python
# 问题1: 排序内存溢出
# 解决: 确保 sort 字段在索引中
try:
    result = orders.find().sort("created_at", -1).limit(100)
except Exception as e:
    print(f"排序错误: {e}")
    # 创建索引解决
    orders.create_index([("created_at", DESCENDING)])

# 问题2: 索引过多影响写入性能
# 每个写操作需要更新所有相关索引
print(f"当前索引数量: {len(list(orders.list_indexes()))}")

# 问题3: 未使用的索引浪费空间
# 使用 $indexStats 查找并清理未使用的索引
```
