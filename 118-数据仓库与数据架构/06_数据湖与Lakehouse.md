# 数据湖与Lakehouse架构

## 概念对比：数据湖 vs 数据仓库 vs Lakehouse

### 数据湖（Data Lake）

数据湖是一个以原始格式存储大量结构化和非结构化数据的系统。数据在被使用之前不需要预先定义Schema（Schema-on-Read）。

**核心特征**：
- 存储原始数据，保留所有细节
- 支持结构化、半结构化和非结构化数据
- 低成本存储（通常基于对象存储如S3、HDFS）
- 适合探索性分析和机器学习

**常见问题**：缺乏事务支持、数据质量难以保障、容易变成"数据沼泽"。

### 数据仓库（Data Warehouse）

数据仓库是面向主题的、集成的、相对稳定的、反映历史变化的数据集合，用于支持管理决策（Schema-on-Write）。

**核心特征**：
- Schema在写入时定义
- 强一致性和ACID事务支持
- 优化的查询性能
- 适合BI报表和标准分析

### Lakehouse架构

Lakehouse是一种将数据湖的灵活性与数据仓库的管理能力相结合的新型架构。通过开放表格式（Open Table Format）在低成本对象存储上实现ACID事务。

```text
+--------------------------------------------------+
|                  应用层 (Applications)              |
|   BI报表  |  数据科学  |  机器学习  |  实时分析      |
+--------------------------------------------------+
|              统一元数据层 (Unified Metadata)         |
|         Delta Lake / Iceberg / Hudi               |
+--------------------------------------------------+
|              开放数据格式 (Open Format)              |
|              Parquet / ORC / Avro                 |
+--------------------------------------------------+
|              云对象存储 (Object Storage)             |
|           S3 / ADLS / GCS / HDFS                  |
+--------------------------------------------------+
```

## 三大开放表格式对比

| 特性 | Delta Lake | Apache Iceberg | Apache Hudi |
|------|-----------|---------------|-------------|
| 创始公司 | Databricks | Netflix | Uber |
| ACID事务 | 支持 | 支持 | 支持 |
| Schema演化 | 支持 | 支持 | 支持 |
| 时间旅行 | 支持 | 支持 | 支持 |
| 增量读取 | CDF (Change Data Feed) | 增量扫描 | 增量查询 |
| 并发写入 | 乐观并发 | 乐观并发 | 基于文件锁 |
| 引擎支持 | Spark优先 | 引擎无关 | Spark/Flink |
| 小文件合并 | 自动优化 | 合并后写入 | 自动Compaction |

## Delta Lake 实践

### 基本操作

```sql
-- 创建Delta表
CREATE TABLE events (
    event_id BIGINT,
    user_id BIGINT,
    event_type STRING,
    event_time TIMESTAMP,
    properties MAP<STRING, STRING>
)
USING delta
PARTITIONED BY (date(event_time))
LOCATION 's3://data-lake/events/'
TBLPROPERTIES (
    'delta.autoOptimize.optimizeWrite' = 'true',
    'delta.autoOptimize.autoCompact' = 'true'
);

-- 写入数据
INSERT INTO events VALUES
(1, 1001, 'click', '2024-01-15 10:30:00', MAP('page', 'home')),
(2, 1002, 'purchase', '2024-01-15 11:00:00', MAP('item', 'laptop', 'amount', '999'));

-- 时间旅行查询：查看某个时间点的数据快照
SELECT * FROM events VERSION AS OF 5;
SELECT * FROM events TIMESTAMP AS OF '2024-01-15 12:00:00';

-- 查看表历史
DESCRIBE HISTORY events LIMIT 10;
```

### Schema演化

```sql
-- 添加新列（自动Schema演化）
ALTER TABLE events ADD COLUMNS (
    session_id STRING AFTER user_id,
    device_type STRING COMMENT '设备类型：mobile/desktop/tablet'
);

-- 启用Schema合并写入
SET spark.databricks.delta.schema.autoMerge.enabled = true;

-- 写入包含新列的数据，旧数据新列值为NULL
INSERT INTO events (event_id, user_id, session_id, device_type, event_type, event_time)
VALUES (3, 1003, 'sess_abc', 'mobile', 'click', '2024-01-16 09:00:00');
```

### Change Data Feed (CDF)

```sql
-- 启用CDF
ALTER TABLE events SET TBLPROPERTIES (
    'delta.enableChangeDataFeed' = 'true'
);

-- 读取变更数据
SELECT * table_changes('events', '2024-01-15 00:00:00', '2024-01-16 00:00:00');
```

## Apache Iceberg 实践

### 表创建与管理

```sql
-- 创建Iceberg表
CREATE TABLE catalog.db.user_events (
    event_id BIGINT,
    user_id BIGINT,
    event_type STRING,
    event_time TIMESTAMP,
    country STRING
)
USING iceberg
PARTITIONED BY (
    country,                        -- 按国家分区
    hours(event_time)               -- 按小时子分区
)
TBLPROPERTIES (
    'write.metadata.delete-after-commit.enabled' = 'true',
    'history.expire.max-snapshot-age-ms' = '432000000'  -- 5天
);

-- 隐式分区：数据写入时自动计算分区值
INSERT INTO user_events VALUES
(1, 1001, 'login', TIMESTAMP '2024-01-15 10:30:00', 'CN');

-- 分区演化：无需重写数据即可修改分区策略
ALTER TABLE catalog.db.user_events ADD PARTITION FIELD days(event_time);
ALTER TABLE catalog.db.user_events DROP PARTITION FIELD country;

-- 表维护：过期快照清理
CALL catalog.system.expire_snapshots(
    table => 'db.user_events',
    older_than => TIMESTAMP '2024-01-10 00:00:00',
    retain_last => 5
);
```

### 增量读取

```sql
-- 读取自某个快照以来的增量变更
SELECT * FROM catalog.db.user_events
INCREMENTAL_FROM_SNAPSHOT 100
WHERE event_time > '2024-01-15';

-- 合并小文件（Compaction）
CALL catalog.system.rewrite_data_files(
    table => 'db.user_events',
    strategy => 'sort',
    sort_order => 'event_time'
);
```

## Python操作示例

```python
from pyspark.sql import SparkSession
from delta import configure_spark_with_delta_pip

# 初始化Spark with Delta Lake
builder = SparkSession.builder \
    .appName("LakehouseDemo") \
    .config("spark.sql.extensions", "io.delta.sql.DeltaSparkSessionExtension") \
    .config("spark.sql.catalog.spark_catalog", "org.apache.spark.sql.delta.catalog.DeltaCatalog") \
    .config("spark.databricks.delta.retentionDurationCheck.enabled", "false")

spark = configure_spark_with_delta_pip(builder).getOrCreate()

# 写入Delta表
df = spark.createDataFrame([
    (1, 1001, "login", "2024-01-15 10:00:00"),
    (2, 1002, "purchase", "2024-01-15 11:00:00"),
], ["event_id", "user_id", "event_type", "event_time"])

df.write.format("delta") \
    .mode("append") \
    .partitionBy("event_type") \
    .save("/data/lake/events")

# MERGE操作（Upsert）
from delta.tables import DeltaTable

deltaTable = DeltaTable.forPath(spark, "/data/lake/events")

updates = spark.createDataFrame([
    (2, 1002, "purchase", "2024-01-15 11:30:00"),  -- 更新
    (3, 1003, "click", "2024-01-15 12:00:00"),     -- 新增
], ["event_id", "user_id", "event_type", "event_time"])

deltaTable.alias("target").merge(
    updates.alias("source"),
    "target.event_id = source.event_id"
).whenMatchedUpdateAll() \
 .whenNotMatchedInsertAll() \
 .execute()

# 读取历史版本
df_v1 = spark.read.format("delta").option("versionAsOf", 1).load("/data/lake/events")
df_ts = spark.read.format("delta").option("timestampAsOf", "2024-01-15").load("/data/lake/events")

# 优化表（Z-Ordering）
from delta.tables import DeltaTable
DeltaTable.forPath(spark, "/data/lake/events").optimize().executeZOrderBy("user_id")
```

## 选型建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| Databricks生态 | Delta Lake | 原生支持，功能最完整 |
| 多引擎场景(Flink+Spark+Trino) | Iceberg | 引擎无关，社区活跃 |
| 需要行级更新/删除 | Hudi | Copy-on-Write + Merge-on-Read模型 |
| 纯Hive迁移 | Iceberg | 与Hive兼容性最好 |
| 云原生数据湖 | Delta Lake / Iceberg | 主流云厂商支持 |

## 总结

Lakehouse架构通过开放表格式在对象存储上实现了数据仓库级别的管理能力，同时保留了数据湖的灵活性。选择哪种表格式主要取决于技术栈、团队经验和具体业务需求。Iceberg凭借其引擎无关性和活跃的社区发展势头强劲，而Delta Lake在Databricks生态中具有天然优势。
