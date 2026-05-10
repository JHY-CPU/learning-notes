# Hive与HBase


## Hive与HBase


大数据HiveHBase


Hive提供SQL-on-Hadoop能力，HBase提供实时随机读写的列式存储。


## Hive - SQL on Hadoop


Hive将SQL查询转换为MapReduce/Tez/Spark任务，让熟悉SQL的用户能够分析HDFS上的数据。


```
Hive 架构：
┌─────────────────────────────────────┐
│  HiveQL (类SQL查询语言)              │
│         ↓                           │
│  编译器 (Compiler)                   │
│  ├── 解析器 (Parser)：SQL → AST      │
│  ├── 语义分析器：AST → 逻辑计划       │
│  └── 优化器：逻辑计划 → 物理计划      │
│         ↓                           │
│  执行引擎                            │
│  ├── MapReduce（默认，批处理）        │
│  ├── Tez（DAG优化，更高效）           │
│  └── Spark（内存计算，更快）          │
│         ↓                           │
│  HDFS / HBase / S3                  │
└─────────────────────────────────────┘

Hive 核心概念：
- 表(Table)：对应HDFS上的一个目录
- 分区(Partition)：按列值分目录，加速查询
- 桶(Bucket)：在分区内进一步哈希分文件
- 外部表(External Table)：删除表不删数据
- SerDe：序列化/反序列化器
```


## Hive SQL示例


```
-- 创建分区表（内部表）
CREATE TABLE user_behavior (
    user_id     BIGINT,
    item_id     BIGINT,
    category_id BIGINT,
    behavior    STRING,
    ts          TIMESTAMP
)
PARTITIONED BY (dt STRING)       -- 按日期分区
CLUSTERED BY (user_id) INTO 32 BUCKETS  -- 分桶
STORED AS ORC;                  -- 列式存储格式

-- 加载数据
LOAD DATA INPATH '/data/user_behavior/2024-01-01'
    OVERWRITE INTO TABLE user_behavior PARTITION (dt='2024-01-01');

-- 分析查询（转化为MapReduce/Tez任务）
SELECT behavior, COUNT(*) as cnt
FROM user_behavior
WHERE dt = '2024-01-01'
GROUP BY behavior
ORDER BY cnt DESC;

-- 分区裁剪：只扫描需要的分区，大幅减少数据量
-- WHERE dt = '2024-01-01' 只读取对应分区目录
```


> **Note:** Hive适合离线批处理分析，不适合实时查询。查询延迟通常在秒到分钟级。


## HBase - 列族式NoSQL数据库


HBase基于Google BigTable论文，构建在HDFS之上，提供实时随机读写能力。


```
HBase 数据模型：
┌─────────────────────────────────────────────┐
│  RowKey → Column Family → Column → Value    │
│                                             │
│  表(Table)                                   │
│  └── 行(Row)：由RowKey唯一标识                │
│      └── 列族(Column Family)：物理存储单元     │
│          └── 列(Column/Qualifier)            │
│              └── 单元格(Cell)：值+时间戳       │
│                                             │
│  示例：                                      │
│  RowKey    info:name   info:age   score:math │
│  ┌────────┬──────────┬─────────┬───────────┐│
│  │ user01 │ "Alice"  │ "25"    │ "95"      ││
│  │ user02 │ "Bob"    │ "30"    │ "88"      ││
│  └────────┴──────────┴─────────┴───────────┘│
│                                             │
│  特点：                                      │
│  - 稀疏：每个行可以有不同的列                  │
│  - 多版本：每个单元格可存储多个版本             │
│  - 自动分片：按RowKey范围自动分区(Region)      │
└─────────────────────────────────────────────┘
```


## HBase架构组件


```
HBase 架构：
┌───────────────────────────────────────────┐
│  HMaster（主节点）                         │
│  ├── 管理RegionServer                      │
│  ├── Region分配和负载均衡                  │
│  ├── 处理DDL操作（建表/删除表）             │
│  └── 管理元数据（Meta表）                  │
│                                           │
│  RegionServer（从节点）                    │
│  ├── 管理多个Region                        │
│  ├── 处理数据读写请求                      │
│  └── 处理Region的分裂和合并                │
│                                           │
│  Region：                                  │
│  ├── 表按RowKey范围水平切分                 │
│  ├── 每个Region存储一段连续的RowKey        │
│  └── 当数据量超过阈值时自动分裂            │
│                                           │
│  Store：                                   │
│  └── 每个列族对应一个Store                  │
│      ├── MemStore（内存写缓冲）             │
│      └── StoreFile/HFile（磁盘文件）        │
│                                           │
│  底层依赖：                                │
│  - ZooKeeper：集群协调、Master选举         │
│  - HDFS：实际数据存储                      │
└───────────────────────────────────────────┘

写入流程：
1. 写入WAL（Write-Ahead Log）防数据丢失
2. 写入MemStore（内存）
3. MemStore达到阈值后flush为HFile
4. 多个HFile定期进行Compaction合并
```


## CAP权衡与选型对比


```
Hive vs HBase vs 其他系统对比：

| 特性        | Hive        | HBase       | Elasticsearch |
|------------|-------------|-------------|---------------|
| 查询类型    | 批处理分析   | 实时点查     | 全文搜索       |
| 延迟       | 秒~分钟     | 毫秒级      | 毫秒级         |
| 数据模型    | 结构化表     | 列族NoSQL   | 文档型         |
| 数据规模    | PB级       | TB~PB级    | TB级          |
| 一致性     | 最终一致    | 强一致      | 最终一致       |
| CAP        | AP(批处理)  | CP          | AP            |

CAP定理回顾：
- Consistency（一致性）：所有节点看到相同数据
- Availability（可用性）：每个请求都能收到响应
- Partition Tolerance（分区容错）：网络分区时系统继续工作
- 分布式系统只能同时满足其中两个

HBase选择CP：保证一致性，可能牺牲部分可用性
Hive批处理天然容忍延迟，关注的是吞吐量而非可用性
```


> **Note:** 选型建议：离线分析用Hive，实时点查用HBase，全文搜索用Elasticsearch。实际项目中常组合使用。


<!-- Converted from: 02_Hive与HBase.html -->
