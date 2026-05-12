# ETL与ELT流程

## ETL vs ELT

### ETL（Extract-Transform-Load）

ETL是传统的数据集成方式，先从源系统抽取数据，在中间服务器上进行转换，最后加载到目标数据仓库。

```text
+----------+     +----------+     +----------+
|  源系统   |--->|  ETL引擎  |--->|  数仓    |
| (Extract)|     |(Transform)|     | (Load)  |
+----------+     +----------+     +----------+
```

**适用场景**：
- 传统数据仓库（如Teradata、Oracle DW）
- 数据量较小，需要复杂清洗逻辑
- 对数据隐私敏感，需要在加载前脱敏

### ELT（Extract-Load-Transform）

ELT是现代数据架构的主流方式，先将原始数据加载到目标系统（数据湖/云数仓），再利用目标系统的计算能力进行转换。

```text
+----------+     +----------+     +----------+
|  源系统   |--->| 数据湖/仓 |--->| 转换计算  |
| (Extract)|     |  (Load)  |     |(Transform)|
+----------+     +----------+     +----------+
```

**适用场景**：
- 云数据仓库（Snowflake、BigQuery、Redshift）
- 数据量大，需要弹性计算能力
- 数据湖架构，保留原始数据

### 对比总结

| 维度 | ETL | ELT |
|------|-----|-----|
| 处理顺序 | 先转换后加载 | 先加载后转换 |
| 计算位置 | 中间ETL服务器 | 目标数据仓库 |
| 数据保留 | 通常只保留转换后数据 | 保留原始数据 |
| 扩展性 | 受ETL服务器限制 | 弹性扩展 |
| 适用架构 | 传统数仓 | 云数仓/数据湖 |
| 代表工具 | Informatica, DataStage | dbt, Spark SQL |

## 数据抽取（Extract）

### 全量抽取

每次抽取源表全部数据，适合数据量小或无时间戳的表。

```python
# 全量抽取示例：Python + SQLAlchemy
from sqlalchemy import create_engine
import pandas as pd

source_engine = create_engine("mysql+pymysql://user:pass@source-host:3306/ecommerce")
target_engine = create_engine("hive://hive-server:10000/ods")

# 全量读取
df = pd.read_sql("SELECT * FROM dim_category", source_engine)

# 写入目标（覆盖）
df.to_sql("ods_dim_category", target_engine, schema="ods", if_exists="replace", index=False)
print(f"全量抽取完成，共 {len(df)} 条记录")
```

### 增量抽取

只抽取自上次抽取以来新增或修改的数据，适合数据量大且有时间戳或自增ID的表。

```python
# 增量抽取示例：基于时间戳
from datetime import datetime
import pandas as pd

def incremental_extract(source_conn, target_conn, table_name, watermark_col="update_time"):
    """
    增量抽取：基于水位线(Watermark)的时间戳抽取
    """
    # 1. 获取上次抽取的水位线
    last_watermark = target_conn.execute(
        f"SELECT MAX({watermark_col}) FROM ods_{table_name}"
    ).scalar()

    if last_watermark is None:
        last_watermark = "1970-01-01 00:00:00"

    # 2. 从源系统抽取增量数据
    sql = f"""
        SELECT * FROM {table_name}
        WHERE {watermark_col} > '{last_watermark}'
        ORDER BY {watermark_col}
    """
    df = pd.read_sql(sql, source_conn)

    if df.empty:
        print(f"表 {table_name} 无增量数据")
        return

    # 3. 写入目标（追加模式）
    df.to_sql(f"ods_{table_name}", target_conn, if_exists="append", index=False)

    # 4. 更新水位线
    new_watermark = df[watermark_col].max()
    print(f"增量抽取完成: {len(df)} 条, 新水位线: {new_watermark}")
    return new_watermark
```

### 基于CDC的抽取

CDC（Change Data Capture）通过捕获数据库的变更日志实现近乎实时的数据同步。

```python
# Debezium CDC消费者示例
from confluent_kafka import Consumer
import json

consumer = Consumer({
    'bootstrap.servers': 'kafka:9092',
    'group.id': 'cdc-consumer',
    'auto.offset.reset': 'earliest'
})
consumer.subscribe(['cdc.ecommerce.orders'])

def process_cdc_event(event):
    """处理CDC事件"""
    payload = event.get("payload", {})
    op = payload.get("op")  # c=创建, u=更新, d=删除, r=全量快照

    if op in ('c', 'r'):  # INSERT或快照
        return {"action": "INSERT", "data": payload.get("after")}
    elif op == 'u':  # UPDATE
        return {"action": "UPDATE", "data": payload.get("after")}
    elif op == 'd':  # DELETE
        return {"action": "DELETE", "data": payload.get("before")}

while True:
    msg = consumer.poll(1.0)
    if msg is None or msg.error():
        continue
    event = json.loads(msg.value().decode('utf-8'))
    result = process_cdc_event(event)
    print(f"CDC事件: {result}")
```

## 数据转换（Transform）

### 常见转换类型

```sql
-- 1. 数据清洗：去重、空值处理、格式统一
SELECT
    DISTINCT                             -- 去重
    COALESCE(user_name, '未知') AS user_name,  -- 空值处理
    TRIM(email) AS email,                -- 去除空格
    LOWER(status) AS status,             -- 统一小写
    REGEXP_REPLACE(phone, '[^0-9]', '') AS phone  -- 正则清洗
FROM ods_user_raw;

-- 2. 数据标准化：编码映射、枚举转换
SELECT
    order_id,
    CASE order_status
        WHEN 0 THEN 'pending'
        WHEN 1 THEN 'paid'
        WHEN 2 THEN 'shipped'
        WHEN 3 THEN 'completed'
        WHEN 4 THEN 'cancelled'
        ELSE 'unknown'
    END AS order_status_std,
    CASE pay_channel
        WHEN 'alipay' THEN '支付宝'
        WHEN 'wechat' THEN '微信支付'
        WHEN 'unionpay' THEN '银联'
        ELSE pay_channel
    END AS pay_channel_name
FROM ods_order;

-- 3. 数据聚合：多维汇总
SELECT
    user_id,
    DATE(order_time) AS order_date,
    COUNT(*) AS order_count,
    SUM(amount) AS total_amount,
    AVG(amount) AS avg_amount,
    MAX(amount) AS max_amount
FROM dwd_order
GROUP BY user_id, DATE(order_time);

-- 4. 数据关联：多表Join
SELECT
    o.order_id,
    u.user_name,
    u.city,
    p.product_name,
    p.category,
    o.amount,
    o.order_time
FROM dwd_order o
JOIN dim_user u ON o.user_id = u.user_id
JOIN dim_product p ON o.product_id = p.product_id;

-- 5. 行转列（PIVOT）
SELECT
    user_id,
    SUM(CASE WHEN event_type = 'click' THEN 1 ELSE 0 END) AS click_count,
    SUM(CASE WHEN event_type = 'purchase' THEN 1 ELSE 0 END) AS purchase_count,
    SUM(CASE WHEN event_type = 'view' THEN 1 ELSE 0 END) AS view_count
FROM dwd_user_event
GROUP BY user_id;
```

## 数据加载（Load）

### 加载策略

| 策略 | 说明 | SQL示例 | 适用场景 |
|------|------|---------|---------|
| 全量覆盖 | 覆盖目标表所有数据 | `INSERT OVERWRITE` | 维度表全量刷新 |
| 追加 | 向目标表追加新数据 | `INSERT INTO` | 增量事实表 |
| Upsert | 存在则更新，不存在则插入 | `MERGE INTO` | 缓慢变化维度 |
| 分区覆盖 | 覆盖指定分区 | `INSERT OVERWRITE PARTITION` | 按日分区的事实表 |

### MERGE操作（Upsert）

```sql
-- Hive/Spark MERGE：处理缓慢变化维度
MERGE INTO dim_user AS target
USING (
    SELECT user_id, user_name, email, phone, address, update_time
    FROM ods_user_update
    WHERE dt = '2024-01-15'
) AS source
ON target.user_id = source.user_id

-- 匹配到且数据有变化则更新（SCD Type 1）
WHEN MATCHED AND target.update_time < source.update_time THEN UPDATE SET
    user_name = source.user_name,
    email = source.email,
    phone = source.phone,
    address = source.address,
    update_time = source.update_time

-- 未匹配则插入新记录
WHEN NOT MATCHED THEN INSERT (
    user_id, user_name, email, phone, address, update_time, create_time
) VALUES (
    source.user_id, source.user_name, source.email, source.phone,
    source.address, source.update_time, CURRENT_TIMESTAMP()
);
```

## CDC (Change Data Capture) 深入

### CDC技术对比

| 技术 | 原理 | 延迟 | 侵入性 | 代表工具 |
|------|------|------|--------|---------|
| 基于触发器 | 数据库触发器 | 低 | 高 | 自研 |
| 基于时间戳 | 查询时间戳字段 | 高 | 中 | 自研 |
| 基于快照 | 全量对比差异 | 很高 | 低 | Sqoop |
| 基于日志 | 解析Binlog/WAL | 低 | 低 | Debezium, Canal, Maxwell |

### Debezium CDC Pipeline

```yaml
# docker-compose-debezium.yml
version: "3.8"
services:
  kafka:
    image: confluentinc/cp-kafka:7.5.0
    ports:
      - "9092:9092"
    environment:
      KAFKA_BROKER_ID: 1
      KAFKA_OFFSETS_TOPIC_REPLICATION_FACTOR: 1

  connect:
    image: debezium/connect:2.4
    ports:
      - "8083:8083"
    environment:
      BOOTSTRAP_SERVERS: kafka:9092
      GROUP_ID: debezium-connect
      CONFIG_STORAGE_TOPIC: connect-configs
      OFFSET_STORAGE_TOPIC: connect-offsets
      STATUS_STORAGE_TOPIC: connect-status
    depends_on:
      - kafka
```

```bash
# 注册MySQL CDC Connector
curl -X POST http://localhost:8083/connectors \
  -H "Content-Type: application/json" \
  -d '{
    "name": "mysql-cdc-connector",
    "config": {
      "connector.class": "io.debezium.connector.mysql.MySqlConnector",
      "tasks.max": "1",
      "database.hostname": "mysql",
      "database.port": "3306",
      "database.user": "debezium",
      "database.password": "dbz",
      "database.server.id": "184054",
      "topic.prefix": "mysql-prod",
      "database.include.list": "ecommerce",
      "schema.history.internal.kafka.bootstrap.servers": "kafka:9092",
      "schema.history.internal.kafka.topic": "schema-changes"
    }
  }'
```

## ETL调度与编排

### 调度层级

```text
+--------------------------+
| 编排层 (Orchestration)   |
| Airflow / Dagster       |
+--------------------------+
| 调度层 (Scheduling)      |
| Cron / Airflow Scheduler|
+--------------------------+
| 执行层 (Execution)       |
| Spark / Hive / dbt      |
+--------------------------+
```

### ETL任务依赖管理

```python
# Airflow DAG定义ETL依赖
from airflow import DAG
from airflow.providers.apache.spark.operators.spark_submit import SparkSubmitOperator
from airflow.providers.common.sql.operators.sql import SQLCheckOperator
from datetime import datetime, timedelta

default_args = {
    "owner": "data_team",
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
}

with DAG(
    "daily_etl_pipeline",
    default_args=default_args,
    schedule_interval="0 2 * * *",  # 每天凌晨2点
    start_date=datetime(2024, 1, 1),
    catchup=False,
) as dag:

    # 任务1：从MySQL抽取到ODS
    extract_orders = SparkSubmitOperator(
        task_id="extract_orders",
        application="/jobs/extract/mysql_to_ods_order.py",
        conf={"spark.executor.memory": "4g"},
    )

    # 任务2：数据质量检查
    quality_check = SQLCheckOperator(
        task_id="quality_check",
        conn_id="hive_conn",
        sql="""
            SELECT COUNT(1) > 0 AS has_data
            FROM ods_order
            WHERE dt = '{{ ds }}'
        """,
    )

    # 任务3：ODS到DWD转换
    transform_to_dwd = SparkSubmitOperator(
        task_id="transform_to_dwd",
        application="/jobs/transform/ods_to_dwd_order.py",
    )

    # 任务4：DWD到DWS汇总
    aggregate_to_dws = SparkSubmitOperator(
        task_id="aggregate_to_dws",
        application="/jobs/aggregate/dwd_to_dws_user_trade.py",
    )

    # 定义依赖链
    extract_orders >> quality_check >> transform_to_dwd >> aggregate_to_dws
```

## 最佳实践

1. **优先ELT**：在云原生场景下优先选择ELT模式，利用目标系统计算能力
2. **保留原始数据**：ODS层保留原始数据，支持数据回溯和重新加工
3. **增量优先**：优先采用增量抽取，减少数据传输量和计算资源
4. **CDC优于轮询**：对于需要近实时同步的场景，使用CDC代替定时轮询
5. **幂等设计**：ETL任务必须支持重复执行而不产生重复数据
6. **数据质量前置**：在ETL入口处加入数据质量检查，不合格数据隔离处理
