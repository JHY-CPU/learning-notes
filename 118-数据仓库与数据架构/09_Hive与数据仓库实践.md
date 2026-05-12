# Hive与数据仓库实践

## Hive基础架构

Apache Hive是构建在Hadoop之上的数据仓库基础设施，提供类SQL查询语言(HiveQL)来查询存储在HDFS中的大规模数据。Hive将SQL查询转换为MapReduce、TeZ或Spark任务执行。

```text
+-------------------------------+
|         Hive CLI / JDBC        |
+-------------------------------+
|         HiveServer2            |
+-------------------------------+
|       查询编译器 & 优化器        |
|  (Parser -> Planner -> Optimizer) |
+-------------------------------+
|    执行引擎 (MapReduce/TeZ/Spark)  |
+-------------------------------+
|          HDFS / S3             |
+-------------------------------+
```

## 内部表 vs 外部表

### 内部表(Managed Table)

Hive管理表的数据和元数据，删除表时数据和元数据都会被删除。

```sql
-- 创建内部表
CREATE TABLE IF NOT EXISTS ods_user (
    user_id     BIGINT,
    username    STRING,
    email       STRING,
    phone       STRING,
    created_at  TIMESTAMP,
    status      STRING
)
COMMENT '用户基础信息表 - 内部表'
ROW FORMAT DELIMITED
FIELDS TERMINATED BY '\t'
LINES TERMINATED BY '\n'
STORED AS TEXTFILE
LOCATION '/user/hive/warehouse/ods.db/ods_user'
TBLPROPERTIES ('creator' = 'data_team', 'created_at' = '2024-01-15');

-- 从HDFS加载数据
LOAD DATA INPATH '/data/raw/user_data.tsv' INTO TABLE ods_user;

-- 删除内部表（数据和元数据都删除）
DROP TABLE ods_user;
```

### 外部表(External Table)

外部表的数据不由Hive管理，删除表时只删除元数据，数据文件保留在HDFS上。

```sql
-- 创建外部表
CREATE EXTERNAL TABLE IF NOT EXISTS ods_order_ext (
    order_id      BIGINT,
    user_id       BIGINT,
    product_id    BIGINT,
    quantity      INT,
    unit_price    DECIMAL(10, 2),
    total_amount  DECIMAL(12, 2),
    order_time    TIMESTAMP,
    order_date    STRING
)
COMMENT '订单数据 - 外部表'
ROW FORMAT SERDE 'org.apache.hadoop.hive.serde2.JsonSerDe'
STORED AS TEXTFILE
LOCATION '/data/external/orders/'
TBLPROPERTIES ('external.table.purge' = 'true');

-- 删除外部表（仅删除元数据，数据保留）
DROP TABLE ods_order_ext;
```

**选择建议**：ODS层使用外部表（源数据需保留），DWD及以上层使用内部表（由Hive管理生命周期）。

## 分区表（Partitioning）

分区是Hive最重要的性能优化手段，将数据按某个列的值分散存储在不同子目录中。

### 静态分区

```sql
-- 创建分区表（按日期分区）
CREATE TABLE dwd_user_events (
    event_id    BIGINT,
    user_id     BIGINT,
    event_type  STRING,
    event_time  TIMESTAMP,
    page_url    STRING,
    ref_url     STRING,
    device_type STRING,
    ip_address  STRING
)
PARTITIONED BY (dt STRING)
STORED AS ORC
TBLPROPERTIES ('orc.compress' = 'SNAPPY');

-- 静态分区插入
INSERT OVERWRITE TABLE dwd_user_events PARTITION (dt = '2024-01-15')
SELECT
    event_id, user_id, event_type, event_time,
    page_url, ref_url, device_type, ip_address
FROM ods_user_events
WHERE DATE(event_time) = '2024-01-15';
```

### 动态分区

```sql
-- 启用动态分区
SET hive.exec.dynamic.partition = true;
SET hive.exec.dynamic.partition.mode = nonstrict;
SET hive.exec.max.dynamic.partitions = 10000;
SET hive.exec.max.dynamic.partitions.pernode = 10000;

-- 动态分区插入（按event_type自动创建分区）
INSERT OVERWRITE TABLE dwd_user_events PARTITION (dt)
SELECT
    event_id, user_id, event_type, event_time,
    page_url, ref_url, device_type, ip_address,
    DATE(event_time) AS dt   -- 最后一列作为分区列
FROM ods_user_events;
```

### 多级分区

```sql
-- 二级分区表：按日期和国家分区
CREATE TABLE dwd_order_detail (
    order_id     BIGINT,
    user_id      BIGINT,
    product_id   BIGINT,
    amount       DECIMAL(12, 2),
    order_time   TIMESTAMP
)
PARTITIONED BY (dt STRING, country STRING)
STORED AS ORC;

-- 写入二级分区
INSERT OVERWRITE TABLE dwd_order_detail PARTITION (dt = '2024-01-15', country = 'CN')
SELECT order_id, user_id, product_id, amount, order_time
FROM ods_order
WHERE DATE(order_time) = '2024-01-15' AND country = 'CN';

-- 查询时使用分区裁剪
SELECT * FROM dwd_order_detail
WHERE dt = '2024-01-15' AND country = 'CN';
```

### 分区管理

```sql
-- 查看分区
SHOW PARTITIONS dwd_user_events;
SHOW PARTITIONS dwd_user_events PARTITION(dt='2024-01-15');

-- 添加分区
ALTER TABLE dwd_user_events ADD PARTITION (dt = '2024-01-16')
LOCATION '/user/hive/warehouse/dwd.db/dwd_user_events/dt=2024-01-16';

-- 删除分区
ALTER TABLE dwd_user_events DROP PARTITION (dt < '2024-01-01');

-- 修复分区（元数据与HDFS同步）
MSCK REPAIR TABLE dwd_user_events;
```

## 分桶表（Bucketing）

分桶将数据按某个列的哈希值分散到固定数量的文件中，适合采样和Join优化。

```sql
-- 创建分桶表
CREATE TABLE dwd_user_profile (
    user_id     BIGINT,
    age         INT,
    gender      STRING,
    city        STRING,
    register_dt DATE
)
CLUSTERED BY (user_id) INTO 32 BUCKETS
STORED AS ORC
TBLPROPERTIES ('transactional' = 'true');

-- 启用分桶
SET hive.enforce.bucketing = true;

-- 写入分桶表
INSERT OVERWRITE TABLE dwd_user_profile
SELECT user_id, age, gender, city, register_date
FROM ods_user;

-- 分桶采样（只读取1/32的数据）
SELECT * FROM dwd_user_profile TABLESAMPLE(BUCKET 1 OUT OF 32 ON user_id);
SELECT * FROM dwd_user_profile TABLESAMPLE(BUCKET 3 OUT OF 16 ON user_id);
```

## HiveQL常用操作

### DDL操作

```sql
-- 创建数据库
CREATE DATABASE IF NOT EXISTS dws
COMMENT '数据汇总层'
LOCATION '/user/hive/warehouse/dws.db'
WITH DBPROPERTIES ('owner' = 'data_team');

-- 查看表详情
DESCRIBE FORMATTED dwd_user_events;

-- 修改表
ALTER TABLE dwd_user_events RENAME TO dwd_user_events_v2;
ALTER TABLE dwd_user_events ADD COLUMNS (session_id STRING COMMENT '会话ID');
ALTER TABLE dwd_user_events CHANGE COLUMN event_time event_time TIMESTAMP COMMENT '事件发生时间';

-- 表属性修改
ALTER TABLE dwd_user_events SET TBLPROPERTIES ('comment' = '用户事件明细表V2');
```

### DML操作（事务表）

```sql
-- 创建支持ACID的事务表
CREATE TABLE dws_user_summary (
    user_id         BIGINT,
    total_orders    INT,
    total_amount    DECIMAL(14, 2),
    last_order_time TIMESTAMP
)
CLUSTERED BY (user_id) INTO 16 BUCKETS
STORED AS ORC
TBLPROPERTIES ('transactional' = 'true');

-- INSERT
INSERT INTO dws_user_summary VALUES (1001, 5, 299.99, '2024-01-15 10:00:00');

-- UPDATE
UPDATE dws_user_summary
SET total_orders = total_orders + 1,
    total_amount = total_amount + 99.99,
    last_order_time = '2024-01-16 14:30:00'
WHERE user_id = 1001;

-- DELETE
DELETE FROM dws_user_summary WHERE user_id = 1002;

-- MERGE (UPSERT)
MERGE INTO dws_user_summary AS target
USING (
    SELECT user_id, COUNT(*) AS new_orders, SUM(amount) AS new_amount, MAX(order_time) AS latest_time
    FROM dwd_order
    WHERE dt = '2024-01-16'
    GROUP BY user_id
) AS source
ON target.user_id = source.user_id
WHEN MATCHED THEN UPDATE SET
    total_orders = target.total_orders + source.new_orders,
    total_amount = target.total_amount + source.new_amount,
    last_order_time = source.latest_time
WHEN NOT MATCHED THEN INSERT VALUES
    (source.user_id, source.new_orders, source.new_amount, source.latest_time);
```

### 窗口函数

```sql
-- 用户订单排名与累计计算
SELECT
    user_id,
    order_id,
    amount,
    order_time,
    -- 排名函数
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY order_time) AS order_seq,
    RANK() OVER (PARTITION BY user_id ORDER BY amount DESC) AS amount_rank,
    -- 聚合窗口函数
    SUM(amount) OVER (PARTITION BY user_id ORDER BY order_time
                      ROWS BETWEEN UNBOUNDED PRECEDING AND CURRENT ROW) AS running_total,
    AVG(amount) OVER (PARTITION BY user_id ORDER BY order_time
                      ROWS BETWEEN 2 PRECEDING AND CURRENT ROW) AS moving_avg_3,
    -- 偏移函数
    LAG(amount, 1) OVER (PARTITION BY user_id ORDER BY order_time) AS prev_amount,
    LEAD(order_time, 1) OVER (PARTITION BY user_id ORDER BY order_time) AS next_order_time,
    FIRST_VALUE(amount) OVER (PARTITION BY user_id ORDER BY order_time) AS first_order_amount
FROM dwd_order
WHERE dt >= '2024-01-01';
```

### 常用聚合查询

```sql
-- DWS层：每日用户活跃统计
INSERT OVERWRITE TABLE dws_user_daily_active PARTITION (dt = '2024-01-15')
SELECT
    user_id,
    MIN(event_time) AS first_active_time,
    MAX(event_time) AS last_active_time,
    COUNT(DISTINCT event_id) AS event_count,
    COUNT(DISTINCT CASE WHEN event_type = 'click' THEN event_id END) AS click_count,
    COUNT(DISTINCT CASE WHEN event_type = 'purchase' THEN event_id END) AS purchase_count,
    COUNT(DISTINCT session_id) AS session_count
FROM dwd_user_events
WHERE dt = '2024-01-15'
GROUP BY user_id;

-- 使用GROUPING SETS多维聚合
SELECT
    COALESCE(country, 'ALL') AS country,
    COALESCE(city, 'ALL') AS city,
    COALESCE(category, 'ALL') AS category,
    SUM(amount) AS total_amount,
    COUNT(DISTINCT user_id) AS user_count
FROM dwd_order
WHERE dt = '2024-01-15'
GROUP BY country, city, category
GROUPING SETS (
    (country, city, category),
    (country, city),
    (country),
    ()
);
```

## Hive性能优化

### 1. 文件格式选择

```sql
-- ORC格式（推荐，列式存储，压缩率高）
CREATE TABLE optimized_table (...)
STORED AS ORC
TBLPROPERTIES (
    'orc.compress' = 'SNAPPY',          -- 压缩算法：SNAPPY/ZLIB/ZSTD
    'orc.bloom.filter.columns' = 'user_id',  -- 布隆过滤器
    'orc.bloom.filter.fpp' = '0.05'
);

-- Parquet格式（兼容性好）
CREATE TABLE parquet_table (...)
STORED AS PARQUET
TBLPROPERTIES ('parquet.compression' = 'SNAPPY');
```

### 2. Join优化

```sql
-- Map Join（小表广播，适合小表 < 25MB）
SET hive.auto.convert.join = true;
SET hive.mapjoin.smalltable.filesize = 25000000;

SELECT /*+ MAPJOIN(b) */
    a.user_id, a.amount, b.username
FROM dwd_order a
JOIN dim_user b ON a.user_id = b.user_id;

-- Bucket Map Join（分桶表之间的Join）
SET hive.optimize.bucketmapjoin = true;
SELECT /*+ MAPJOIN(b) */
    a.user_id, a.order_id, b.profile
FROM dwd_order a
JOIN dwd_user_profile b ON a.user_id = b.user_id;

-- Sort Merge Bucket Join（SMB Join）
SET hive.auto.convert.sortmerge.join = true;
SET hive.optimize.bucketmapjoin.sortedmerge = true;
SELECT a.user_id, a.order_id, b.profile
FROM dwd_order a
JOIN dwd_user_profile b ON a.user_id = b.user_id;
```

### 3. 数据倾斜处理

```sql
-- 方法1：开启倾斜Join优化
SET hive.optimize.skewjoin = true;
SET hive.skewjoin.key = 100000;

-- 方法2：加盐打散（手动处理）
WITH skewed_data AS (
    SELECT
        IF(user_id = 0, CONCAT('0_', CAST(CEIL(RAND() * 10) AS STRING)), CAST(user_id AS STRING)) AS salted_key,
        order_id, amount
    FROM dwd_order
    WHERE dt = '2024-01-15'
)
SELECT
    REGEXP_EXTRACT(salted_key, '^(\\d+)', 1) AS user_id,
    SUM(amount) AS total_amount
FROM skewed_data
GROUP BY REGEXP_EXTRACT(salted_key, '^(\\d+)', 1);

-- 方法3：两阶段聚合
-- 第一阶段：局部聚合
SELECT user_id, SUM(amount) AS partial_sum, COUNT(*) AS partial_count
FROM dwd_order WHERE dt = '2024-01-15'
GROUP BY user_id;
-- 第二阶段：全局聚合
SELECT user_id, SUM(partial_sum) AS total_amount, SUM(partial_count) AS total_count
FROM partial_result
GROUP BY user_id;
```

### 4. 参数调优

```sql
-- 并行执行
SET hive.exec.parallel = true;
SET hive.exec.parallel.thread.number = 8;

-- 内存配置
SET mapreduce.map.memory.mb = 4096;
SET mapreduce.reduce.memory.mb = 8192;
SET hive.tez.container.size = 4096;

-- 小文件合并
SET hive.merge.mapfiles = true;
SET hive.merge.mapredfiles = true;
SET hive.merge.size.per.task = 256000000;
SET hive.merge.smallfiles.avgsize = 128000000;

-- 推测执行
SET mapreduce.map.speculative = true;
SET mapreduce.reduce.speculative = true;
```

## 数据仓库分层架构示例

```sql
-- ========== ODS层：原始数据层 ==========
CREATE EXTERNAL TABLE ods_mysql_users (
    id          BIGINT,
    username    STRING,
    email       STRING,
    phone       STRING,
    status      TINYINT,
    created_at  TIMESTAMP,
    updated_at  TIMESTAMP
)
PARTITIONED BY (dt STRING)
STORED AS ORC
LOCATION '/data/ods/mysql_users/';

-- ========== DWD层：明细数据层 ==========
CREATE TABLE dwd_user (
    user_id         BIGINT,
    username        STRING,
    email           STRING,
    phone           STRING,
    status          STRING,      -- 转换为可读状态
    register_date   DATE,
    is_new_user     INT,         -- 是否新用户(当日注册)
    etl_time        TIMESTAMP
)
PARTITIONED BY (dt STRING)
STORED AS ORC;

INSERT OVERWRITE TABLE dwd_user PARTITION (dt = '2024-01-15')
SELECT
    id AS user_id,
    username,
    email,
    phone,
    CASE status WHEN 1 THEN 'active' WHEN 0 THEN 'inactive' ELSE 'unknown' END AS status,
    DATE(created_at) AS register_date,
    IF(DATE(created_at) = '2024-01-15', 1, 0) AS is_new_user,
    CURRENT_TIMESTAMP() AS etl_time
FROM ods_mysql_users
WHERE dt = '2024-01-15';

-- ========== DWS层：汇总数据层 ==========
CREATE TABLE dws_user_behavior_daily (
    user_id         BIGINT,
    login_count     INT,
    click_count     INT,
    purchase_count  INT,
    purchase_amount DECIMAL(12, 2),
    active_days     INT
)
PARTITIONED BY (dt STRING)
STORED AS ORC;

-- ========== ADS层：应用数据层 ==========
CREATE TABLE ads_daily_report (
    report_date     DATE,
    total_users     BIGINT,
    active_users    BIGINT,
    new_users       BIGINT,
    total_orders    BIGINT,
    total_amount    DECIMAL(16, 2),
    avg_amount      DECIMAL(12, 2)
)
STORED AS ORC;
```

## 最佳实践

1. **优先使用ORC格式**：列式存储 + Snappy压缩，查询性能和存储效率最优
2. **合理设计分区键**：通常按日期(dt)分区，避免产生过多小分区
3. **分区裁剪**：查询时务必带上分区条件，避免全表扫描
4. **小文件治理**：定期运行Compaction，合并小文件到128MB-256MB
5. **Map Join优先**：小表Join时使用Map Join避免Shuffle
6. **外部表存原始数据**：ODS层用外部表，防止误删原始数据
