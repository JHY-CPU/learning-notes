# MongoDB 概述与架构

## 1. 文档模型与 BSON

MongoDB 是面向文档的 NoSQL 数据库，数据以 BSON（Binary JSON）格式存储。与传统关系型数据库的行和列不同，MongoDB 使用**集合（Collection）**和**文档（Document）**组织数据。

### 文档结构

文档是 MongoDB 的基本数据单元，类似于 JSON 对象：

```json
{
    "_id": ObjectId("64a1b2c3d4e5f6a7b8c9d0e1"),
    "name": "张三",
    "age": 28,
    "address": {
        "city": "北京",
        "district": "海淀区"
    },
    "hobbies": ["编程", "阅读", "跑步"],
    "created_at": ISODate("2024-01-15T08:30:00Z")
}
```

### BSON 数据类型

BSON 在 JSON 基础上扩展了更多数据类型：

| 类型 | 说明 | 示例 |
|------|------|------|
| String | UTF-8 字符串 | `"hello"` |
| Number | 整数和浮点数 | `42`, `3.14` |
| ObjectId | 12字节唯一标识符 | `ObjectId("...")` |
| Date | 日期时间 | `ISODate("2024-01-01")` |
| Boolean | 布尔值 | `true`, `false` |
| Array | 数组 | `[1, 2, 3]` |
| Object | 嵌套文档 | `{"key": "value"}` |
| Binary | 二进制数据 | `BinData(0, "...")` |
| Null | 空值 | `null` |
| Decimal128 | 高精度小数 | `NumberDecimal("9.99")` |

### 与关系型数据库的对比

```
关系型数据库        MongoDB
─────────────     ─────────────
Database    →     Database
Table       →     Collection
Row         →     Document
Column      →     Field
JOIN        →     Embedded Document / $lookup
PRIMARY KEY →     _id
```

### 使用 Python PyMongo 连接

```python
from pymongo import MongoClient

# 连接 MongoDB
client = MongoClient("mongodb://localhost:27017/")

# 选择数据库和集合
db = client["myapp"]
collection = db["users"]

# 插入文档验证 BSON 类型
from datetime import datetime
from bson import ObjectId

doc = {
    "_id": ObjectId(),
    "name": "张三",
    "age": 28,
    "created_at": datetime.utcnow(),
    "scores": [95, 87, 92]
}
result = collection.insert_one(doc)
print(f"插入的文档ID: {result.inserted_id}")
```

## 2. 副本集架构（Replica Set）

副本集是 MongoDB 的高可用方案，由多个 MongoDB 实例组成，自动进行主从切换。

### 副本集组成

```
┌─────────────────────────────────────────────┐
│              Replica Set                     │
│                                             │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  │
│  │ Primary  │  │Secondary │  │Secondary │  │
│  │  (读写)   │→│  (只读)   │→│  (只读)   │  │
│  └──────────┘  └──────────┘  └──────────┘  │
│       ↑              ↑             ↑        │
│       └──────────────┴─────────────┘        │
│            数据复制 (Oplog)                   │
└─────────────────────────────────────────────┘
```

- **Primary**：接收所有写操作，记录到 Oplog
- **Secondary**：从 Primary 异步复制 Oplog，保持数据同步
- **Arbiter**：不存储数据，仅参与选举投票（奇数节点时不需要）

### 副本集配置

```python
from pymongo import MongoClient

# 连接副本集
client = MongoClient(
    "mongodb://host1:27017,host2:27017,host3:27017/"
    "?replicaSet=myReplicaSet"
)

# 检查副本集状态
status = client.admin.command("replSetGetStatus")
print(f"副本集名称: {status['set']}")
for member in status["members"]:
    print(f"  {member['name']} - 状态: {member['stateStr']}")
```

### 选举机制

当 Primary 节点不可用时（心跳超时 10 秒），Secondary 节点发起选举：
1. 每个节点向其他节点发送选举请求
2. 获得**多数派**投票的节点成为新 Primary
3. 3 节点副本集需要至少 2 票才能当选

## 3. 分片架构（Sharding）

分片是 MongoDB 的水平扩展方案，将数据分散到多个服务器上。

### 分片架构组件

```
┌─────────────────────────────────────────────────────┐
│                   Application                        │
│                      ↓                               │
│  ┌─────────────────────────────────────────────┐    │
│  │              mongos (路由)                    │    │
│  └─────────────────────────────────────────────┘    │
│           ↓                    ↓                      │
│  ┌──────────────┐    ┌──────────────────────┐       │
│  │ Config Server │    │    Shard Cluster      │       │
│  │  (元数据存储)  │    │  ┌───────┐ ┌───────┐ │       │
│  └──────────────┘    │  │Shard 1│ │Shard 2│ │       │
│                      │  │(RS)   │ │(RS)   │ │       │
│                      │  └───────┘ └───────┘ │       │
│                      └──────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

- **mongos**：查询路由，接收客户端请求并转发到正确的分片
- **Config Server**：存储集群元数据（分片键范围、数据分布）
- **Shard**：实际存储数据的分片，每个分片建议使用副本集

### 分片键选择

分片键的选择直接影响数据分布和查询性能：

```python
# 启用分片
client.admin.command("enableSharding", "myapp")

# 对集合进行分片，选择 hashed 分片键
client.admin.command("shardCollection", "myapp.users", key={"user_id": "hashed"})

# 范围分片键
client.admin.command("shardCollection", "myapp.orders", key={"created_at": 1})
```

**分片键选择原则：**

| 策略 | 优点 | 缺点 |
|------|------|------|
| Hashed 分片 | 数据分布均匀 | 范围查询效率低 |
| 范围分片 | 范围查询高效 | 可能产生热点 |
| 复合分片键 | 灵活控制分布 | 设计复杂度高 |

### 片键选择最佳实践

- **基数要高**：片键值的种类越多越好
- **分布要均匀**：避免数据集中在少数分片
- **避免单调递增**：如自增 ID 会导致写热点
- **查询要带片键**：定向路由到特定分片，避免全集群扫描

## 4. 存储引擎

MongoDB 支持多种存储引擎：

- **WiredTiger**（默认）：支持文档级锁、压缩、内存缓存
- **In-Memory**：所有数据存储在内存中，适用于低延迟场景
- **Encrypted**：支持静态数据加密

```python
# 查看存储引擎信息
server_status = client.admin.command("serverStatus")
print(f"存储引擎: {server_status['storageEngine']['name']}")
```

WiredTiger 的核心特性：
- 使用 B+ 树索引
- 支持 Snappy 和 Zlib 压缩
- 检查点机制保证数据持久化
- 默认使用 WiredTiger 缓存（约 50% 可用内存）
