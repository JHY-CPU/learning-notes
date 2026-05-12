# MongoDB 副本集与分片

## 1. 副本集详解

### 副本集的核心概念

副本集（Replica Set）是 MongoDB 提供数据冗余和高可用的基础架构。一个副本集由多个 MongoDB 实例组成，其中一个为 Primary 节点，其余为 Secondary 节点。

```
写操作 → Primary ──Oplog复制──→ Secondary 1
                            └──→ Secondary 2
读操作 ← 可配置读偏好
```

### Oplog（操作日志）

Oplog 是副本集数据同步的核心。Primary 节点的所有写操作都会被记录到 Oplog 中，Secondary 节点持续从 Oplog 中读取并重放操作。

```python
from pymongo import MongoClient

client = MongoClient("mongodb://localhost:27017/")
db = client["local"]

# 查看 Oplog 信息
oplog_stats = db.oplog.rs.stats()
print(f"Oplog 大小: {oplog_stats['size'] / 1024 / 1024:.2f} MB")
print(f"Oplog 范围: {db.oplog.rs.find().sort('$natural', 1).limit(1)[0]['ts']}")
print(f"           → {db.oplog.rs.find().sort('$natural', -1).limit(1)[0]['ts']}")
```

### 副本集配置

```python
from pymongo import MongoClient

# 连接副本集
client = MongoClient(
    "mongodb://node1:27017,node2:27017,node3:27017/"
    "?replicaSet=rs0"
    "&readPreference=secondaryPreferred"
    "&retryWrites=true"
    "&w=majority"
)

# 查看副本集状态
status = client.admin.command("replSetGetStatus")
print(f"副本集: {status['set']}")
print(f"当前节点: {status['myState']}")  # 1=PRIMARY, 2=SECONDARY

for member in status["members"]:
    print(f"  {member['name']}")
    print(f"    状态: {member['stateStr']}")
    print(f"    健康: {member['health']}")
    print(f"    延迟: {member.get('optimeDate', 'N/A')}")
```

### 初始化副本集（开发环境）

```python
# 初始化副本集配置
config = {
    "_id": "rs0",
    "members": [
        {"_id": 0, "host": "node1:27017", "priority": 2},   # 优先成为 Primary
        {"_id": 1, "host": "node2:27017", "priority": 1},
        {"_id": 2, "host": "node3:27017", "priority": 1},
    ]
}

# 仅在首次初始化时执行
# client.admin.command("replSetInitiate", config)

# 添加新成员
# client.admin.command("replSetAdd", {"host": "node4:27017"})

# 移除成员
# client.admin.command("replSetRemove", "node4:27017")
```

## 2. 读偏好（Read Preference）

读偏好决定了读操作路由到哪个节点。

```python
from pymongo import MongoClient, ReadPreference

# 不同的读偏好设置
client_primary = MongoClient(
    "mongodb://localhost:27017/",
    readPreference=ReadPreference.PRIMARY  # 只从 Primary 读（默认）
)

client_secondary = MongoClient(
    "mongodb://localhost:27017/",
    readPreference=ReadPreference.SECONDARY  # 只从 Secondary 读
)

client_preferred = MongoClient(
    "mongodb://localhost:27017/",
    readPreference=ReadPreference.SECONDARY_PREFERRED  # 优先 Secondary
)

client_nearest = MongoClient(
    "mongodb://localhost:27017/",
    readPreference=ReadPreference.NEAREST  # 延迟最低的节点
)
```

### 读偏好对比

| 读偏好 | 说明 | 适用场景 |
|--------|------|----------|
| `PRIMARY` | 只从主节点读 | 强一致性要求 |
| `SECONDARY` | 只从从节点读 | 读写分离、报表查询 |
| `PRIMARY_PREFERRED` | 优先主节点 | 读多写少 |
| `SECONDARY_PREFERRED` | 优先从节点 | 对一致性要求不高的读 |
| `NEAREST` | 网络延迟最低 | 地理分布部署 |

### 写关注（Write Concern）

```python
# w: 写关注级别
# w=1: 主节点确认
# w="majority": 多数节点确认
# j=true: 写入日志确认

client_safe = MongoClient(
    "mongodb://localhost:27017/",
    w="majority",       # 多数节点确认写入
    j=True,             # 等待 journal 刷盘
    wtimeout=5000       # 超时5秒
)

# 在操作级别指定写关注
result = db.users.insert_one(
    {"name": "test"},
    write_concern=WriteConcern(w="majority", j=True)
)
```

### 读关注（Read Concern）

```python
from pymongo import ReadConcern

# local: 返回节点当前数据（可能未被多数节点确认）
# majority: 返回已被多数节点确认的数据
# linearizable: 线性一致性读

# 会话级别设置
with client.start_session() as session:
    doc = db.users.find_one(
        {"name": "张三"},
        read_concern=ReadConcern("majority"),
        session=session
    )
```

## 3. 故障转移与选举

### 选举触发条件

```
触发选举的场景:
1. 副本集初始化时
2. Secondary 检测不到 Primary 心跳（默认10秒超时）
3. Primary 降级（如优先级变化）
4. 人工执行 rs.stepDown()
```

### 选举优先级

```python
# 修改成员优先级
config = client.admin.command("replSetGetConfig")
config["config"]["members"][1]["priority"] = 3  # 提高优先级
client.admin.command("replSetReconfig", config["config"])

# priority=0 的节点不会成为 Primary
# 适用于只读节点、备份节点

# hidden=true 的节点对客户端不可见
# 适用于专用备份和报表查询
```

## 4. 分片架构详解

### 分片的工作原理

```
客户端 → mongos 路由 → 查询 Config Server 获取元数据
                      → 路由到目标 Shard
                      → 返回合并结果
```

### 启用分片

```python
from pymongo import MongoClient

# 连接 mongos
client = MongoClient("mongodb://mongos1:27017,mongos2:27017/")

# 启用数据库分片
client.admin.command("enableSharding", "ecommerce")

# 对集合进行分片
# 方式1: Hashed 分片键
client.admin.command(
    "shardCollection", "ecommerce.users",
    key={"_id": "hashed"}
)

# 方式2: 范围分片键
client.admin.command(
    "shardCollection", "ecommerce.orders",
    key={"user_id": 1, "created_at": 1}
)
```

### 查看分片状态

```python
# 查看集群分片信息
shards = client.admin.command("listShards")
for shard in shards["shards"]:
    print(f"分片: {shard['_id']}, 主机: {shard['host']}")

# 查看集合的分片分布
status = client.admin.command("shardCollection", "ecommerce.orders")
chunk_status = client.config.chunks.find({"ns": "ecommerce.orders"})
for chunk in chunk_status:
    print(f"分片: {chunk['shard']}, 范围: {chunk['min']} → {chunk['max']}")
```

## 5. 分片键选择策略

### 分片键的类型

```python
# 1. Hashed 分片键 - 均匀分布，适合点查询
client.admin.command(
    "shardCollection", "db.collection",
    key={"user_id": "hashed"}
)

# 2. 范围分片键 - 支持范围查询
client.admin.command(
    "shardCollection", "db.collection",
    key={"created_at": 1}
)

# 3. 区域分片键 - 地理亲和性
# 将特定范围的数据路由到特定分片
```

### 分片键选择的考量因素

```python
"""
分片键选择检查清单:

1. 基数（Cardinality）
   ✓ 高基数: user_id, order_id
   ✗ 低基数: status (只有几个值), gender

2. 写分布（Write Distribution）
   ✓ 随机分布: hashed user_id
   ✗ 单调递增: 自增 ID, 精确到秒的时间戳

3. 查询隔离（Query Isolation）
   ✓ 查询包含片键: 可路由到单一分片
   ✗ 查询不包含片键: 需要广播到所有分片

4. 组合分片键
   {region: 1, user_id: 1} → 支持按地区查询
   {status: 1, created_at: 1} → 支持状态+时间范围查询
"""
```

## 6. Chunk 管理

### Chunk 分裂与迁移

```python
# Chunk 默认大小为 64MB
# 查看和修改 Chunk 大小
config_db = client.config
settings = config_db.settings.find_one({"_id": "chunksize"})
print(f"当前 Chunk 大小: {settings['value']} MB")

# 修改 Chunk 大小
config_db.settings.update_one(
    {"_id": "chunksize"},
    {"$set": {"value": 128}}  # 改为 128MB
)

# 手动分裂 Chunk
client.admin.command("split", "ecommerce.orders", middle={
    "user_id": "user_5000",
    "created_at": "2024-06-01"
})

# 手动迁移 Chunk
client.admin.command(
    "moveChunk",
    "ecommerce.orders",
    find={"user_id": "user_1000"},
    to="shard02"
)
```

### 分片集合的查询

```python
# 包含片键的查询 - 定向路由（高效）
orders = client.ecommerce.orders
result = orders.find({"user_id": "user_123"})

# 不包含片键的查询 - 广播查询（需要查所有分片）
result = orders.find({"status": "completed"})

# 使用 explain 查看路由信息
explain = orders.find({"user_id": "user_123"}).explain()
print(f"查询路由: {explain['queryPlanner']['winningPlan']['shards']}")
```

## 7. 分片集群的监控

```python
def check_shard_balance(client):
    """检查分片数据均衡性"""
    config_db = client.config

    # 获取各分片的数据量
    shards_info = config_db.shards.find()
    for shard in shards_info:
        shard_conn = MongoClient(shard["host"])
        stats = shard_conn.admin.command("dbStats", "ecommerce")
        print(f"分片 {shard['_id']}:")
        print(f"  数据大小: {stats['dataSize'] / 1024 / 1024:.2f} MB")
        print(f"  文档数量: {stats['objects']}")

# 检查 balancer 状态
balancer = client.admin.command("balancerStatus")
print(f"Balancer 运行状态: {balancer['mode']}")

# 启停 balancer
# client.admin.command("balancerStart")
# client.admin.command("balancerStop")
```
