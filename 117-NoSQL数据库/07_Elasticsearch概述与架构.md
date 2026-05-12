# Elasticsearch 概述与架构

## 1. Elasticsearch 简介

Elasticsearch 是一个分布式搜索和分析引擎，基于 Apache Lucene 构建。它提供全文搜索、结构化搜索、分析和向量搜索能力，是 ELK 技术栈的核心组件。

### 核心特性

- **近实时搜索（Near Real-Time）**：文档写入后约 1 秒即可被搜索到
- **分布式架构**：自动分片和复制，支持水平扩展
- **全文搜索**：基于倒排索引，支持复杂的文本分析和相关性评分
- **RESTful API**：所有操作通过 HTTP REST API 完成，使用 JSON 格式

### Python 连接示例

```python
from elasticsearch import Elasticsearch

# 连接 Elasticsearch
es = Elasticsearch(
    "http://localhost:9200",
    basic_auth=("elastic", "your_password"),
    request_timeout=30
)

# 检查连接
info = es.info()
print(f"集群名称: {info['cluster_name']}")
print(f"版本: {info['version']['number']}")
```

## 2. 倒排索引原理

倒排索引（Inverted Index）是 Elasticsearch 搜索能力的核心，它建立了从词项到文档的映射关系。

### 正排索引 vs 倒排索引

正排索引以文档为单位，记录每个文档包含哪些词：

```
正排索引（文档 → 词项）:
  文档1: "Elasticsearch 是 分布式 搜索 引擎"
  文档2: "MongoDB 是 NoSQL 数据库"
```

倒排索引以词项为单位，记录每个词出现在哪些文档中：

```
倒排索引（词项 → 文档列表）:
  "Elasticsearch" → [文档1]
  "分布式"        → [文档1]
  "搜索"          → [文档1]
  "MongoDB"       → [文档2]
  "NoSQL"         → [文档2]
  "是"            → [文档1, 文档2]
```

### 倒排索引的数据结构

每个词项在倒排索引中存储一个 Posting List（倒排记录表），包含：

- **DocID**：包含该词项的文档编号
- **Term Frequency（TF）**：词项在文档中的出现次数
- **Position**：词项在文档中的位置（支持短语查询）
- **Offset**：词项的字符偏移量（支持高亮）

```
Term Dictionary (词典，使用 FST 数据结构)
├── "elasticsearch" → Posting List: [doc1(TF=2), doc5(TF=1)]
├── "搜索"          → Posting List: [doc1(TF=3), doc2(TF=1), doc5(TF=2)]
├── "引擎"          → Posting List: [doc1(TF=1), doc3(TF=1)]
└── "数据库"        → Posting List: [doc2(TF=2), doc4(TF=1)]

FST (Finite State Transducer): 有限状态转换器
  优点: 内存占用小，查找速度快，支持前缀匹配
```

### 中文分词对倒排索引的影响

```python
# 使用 IK 分词器的效果对比
# 原始文本: "Elasticsearch是一个分布式搜索引擎"

# standard 分词器（按空格和标点分词）:
# ["elasticsearch是一个分布式搜索引擎"]  → 几乎无法搜索

# ik_max_word 分词器:
# ["elasticsearch", "是", "一个", "一", "个", "分布式", "分布", "式",
#  "搜索引擎", "搜索", "引擎"]

# ik_smart 分词器:
# ["elasticsearch", "是", "一个", "分布式", "搜索引擎"]
```

## 3. 集群架构

### 集群组成

```
┌──────────────────────────────────────────────────────┐
│                    ES Cluster                         │
│                                                      │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  │
│  │   Node 1    │  │   Node 2    │  │   Node 3    │  │
│  │  (Master)   │  │   (Data)    │  │   (Data)    │  │
│  │             │  │             │  │             │  │
│  │ Index:      │  │ Index:      │  │ Index:      │  │
│  │ products_P0 │  │ products_P1 │  │ products_P2 │  │
│  │ orders_R0   │  │ orders_P0   │  │ orders_P1   │  │
│  │ users_P0    │  │ users_R1    │  │ users_R0    │  │
│  └─────────────┘  └─────────────┘  └─────────────┘  │
└──────────────────────────────────────────────────────┘
P0 = Primary Shard 0, R0 = Replica Shard 0
```

### 节点角色

| 角色 | 说明 | 配置方式 |
|------|------|----------|
| **Master-eligible** | 有资格参与 Master 选举 | node.roles: [master] |
| **Data** | 存储数据，执行 CRUD 和搜索 | node.roles: [data] |
| **Data hot/warm/cold/frozen** | 数据分层存储 | node.roles: [data_hot] |
| **Ingest** | 执行预处理管道 | node.roles: [ingest] |
| **ML** | 机器学习任务 | node.roles: [ml] |
| **Coordinating** | 仅路由请求，不存储数据 | 不设置任何角色 |

### Master 选举机制

Master 选举使用改进的 Bully 算法：
- 需要超过半数的 master-eligible 节点投票才能当选
- 3 个 master 节点最多容忍 1 个故障
- 推荐使用奇数个 master 节点（3 或 5 个）

当网络分区时，可能出现多个 Master（脑裂问题）。ES 7+ 使用 cluster.initial_master_nodes 配置来避免此问题。

## 4. 分片（Shard）与副本（Replica）

### 分片的工作原理

```python
# 创建索引时指定分片数
es.indices.create(
    index="products",
    body={
        "settings": {
            "number_of_shards": 3,        # 主分片数（创建后不可减少）
            "number_of_replicas": 1        # 每个主分片的副本数
        }
    }
)
```

分片分布示意：

```
索引 products (3 shards, 1 replica)

Shard 0 (Primary)  → Node 1
Shard 0 (Replica)  → Node 2
Shard 1 (Primary)  → Node 2
Shard 1 (Replica)  → Node 3
Shard 2 (Primary)  → Node 3
Shard 2 (Replica)  → Node 1

每个文档根据 _routing (默认为 _id) 哈希决定落入哪个分片:
  shard = hash(_routing) % number_of_primary_shards
```

### 分片数选择建议

- 每个分片建议存储 10GB ~ 50GB 数据
- 分片数 = 数据总量 / 目标分片大小
- 分片数设置后不可减少，需提前规划
- 过多分片会消耗资源（每个分片约占用一定内存）

```python
# 示例: 500GB 数据，目标 25GB/分片
# 建议分片数: 500 / 25 = 20 个分片
# 如果有 4 个数据节点: 每节点 5 个分片
```

## 5. 段（Segment）与近实时搜索

### 段的写入流程

```
文档写入 → 内存 Buffer → Refresh → 新 Segment (可搜索)
                      → Translog (事务日志，保证数据持久化)
                      → Flush → 磁盘持久化 + 清空 Translog
```

```python
# 手动刷新（使数据立即可搜索）
es.indices.refresh(index="products")

# 写入时等待刷新
es.index(
    index="products",
    body={"name": "测试商品", "price": 99.9},
    refresh="wait_for"  # 等待刷新完成后返回
)

# 关闭自动刷新（批量导入时提高性能）
es.indices.put_settings(
    index="products",
    body={"index": {"refresh_interval": "-1"}}
)
# 导入完成后恢复
es.indices.put_settings(
    index="products",
    body={"index": {"refresh_interval": "1s"}}
)
```

### 段合并（Segment Merge）

段合并将多个小段合并为更少的大段，作用包括：
1. 减少段数量，降低搜索时需要合并的结果集大小
2. 删除已标记删除的文档，释放磁盘空间
3. 减少文件句柄占用

```python
# 手动触发段合并（只读索引推荐）
es.indices.forcemerge(
    index="products",
    max_num_segments=1  # 强制合并为1个段
)
```

## 6. 集群健康与监控

```python
# 查看集群健康状态
health = es.cluster.health()
print(f"状态: {health['status']}")       # green / yellow / red
print(f"节点数: {health['number_of_nodes']}")
print(f"数据节点: {health['number_of_data_nodes']}")
print(f"活跃主分片: {health['active_primary_shards']}")
print(f"活跃分片: {health['active_shards']}")
print(f"未分配分片: {health['unassigned_shards']}")
```

状态说明：
- **green**：所有主分片和副本分片都正常分配
- **yellow**：所有主分片正常，但部分副本分片未分配（常见于单节点部署）
- **red**：部分主分片不可用（数据可能丢失）

```python
# 诊断分片未分配的原因
explain = es.cluster.allocation_explain(
    body={
        "index": "products",
        "shard": 0,
        "primary": False
    }
)
print(f"原因: {explain['allocate_explanation']}")
```
