# Apache Airflow 工作流调度

## Airflow概述

Apache Airflow是一个用Python编写的工作流编排平台，通过有向无环图（DAG）定义和调度任务。Airflow广泛应用于数据工程领域的ETL/ELT任务编排。

### 核心架构

```text
+--------------------------------------------------+
|                  Web Server (UI)                  |
+--------------------------------------------------+
|                  Scheduler                        |
|         (解析DAG, 调度任务到Executor)              |
+--------------------------------------------------+
|                  Executor                         |
|    LocalExecutor | CeleryExecutor | K8sExecutor  |
+--------------------------------------------------+
|                  Workers                          |
+--------------------------------------------------+
|              Metadata Database                    |
|              (PostgreSQL / MySQL)                 |
+--------------------------------------------------+
+--------------------------------------------------+
|              DAG Files Directory                  |
|              (dags/)                              |
+--------------------------------------------------+
```

## DAG定义

DAG（Directed Acyclic Graph，有向无环图）是Airflow的核心概念，定义了一组任务及其依赖关系。

### 基础DAG定义

```python
# dags/daily_etl.py
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
from datetime import datetime, timedelta

# 默认参数
default_args = {
    "owner": "data_team",
    "depends_on_past": False,
    "email": ["data-alerts@company.com"],
    "email_on_failure": True,
    "email_on_retry": False,
    "retries": 3,
    "retry_delay": timedelta(minutes=5),
    "retry_exponential_backoff": True,
    "max_retry_delay": timedelta(minutes=30),
    "execution_timeout": timedelta(hours=1),
    "sla": timedelta(hours=2),
}

# 定义DAG
with DAG(
    dag_id="daily_order_etl",
    default_args=default_args,
    description="每日订单数据ETL流水线",
    schedule_interval="0 2 * * *",        # 每天凌晨2点
    start_date=datetime(2024, 1, 1),
    end_date=None,
    catchup=False,                         # 不补跑历史
    max_active_runs=1,                     # 同时只允许一个运行实例
    tags=["etl", "orders", "daily"],
    doc_md="""
    # 每日订单ETL流水线

    此DAG执行每日订单数据的ETL流程：
    1. 从业务库抽取订单数据到ODS层
    2. 数据清洗转换到DWD层
    3. 聚合汇总到DWS层
    4. 生成ADS报表数据
    """,
) as dag:

    def extract_orders(**context):
        """从业务库抽取订单数据"""
        from airflow.providers.mysql.hooks.mysql import MySqlHook
        from airflow.providers.apache.hive.hooks.hive import HiveHook

        # 获取执行日期
        ds = context["ds"]  # YYYY-MM-DD

        mysql_hook = MySqlHook(mysql_conn_id="mysql_business")
        hive_hook = HiveHook(hive_cli_conn_id="hive_default")

        # 抽取数据
        df = mysql_hook.get_pandas_df(
            sql=f"""
                SELECT id, user_id, product_id, quantity, total_amount,
                       status, create_time, update_time
                FROM orders
                WHERE DATE(update_time) = '{ds}' OR DATE(create_time) = '{ds}'
            """
        )

        # 推送到Hive
        df.to_csv(f"/tmp/orders_{ds}.csv", index=False)
        hive_hook.run_cli(
            f"LOAD DATA LOCAL INPATH '/tmp/orders_{ds}.csv' "
            f"INTO TABLE ods.orders PARTITION (dt='{ds}')"
        )

        # 通过XCom传递记录数
        return len(df)

    def transform_orders(**context):
        """数据清洗转换"""
        ds = context["ds"]
        # Pull上游任务的结果
        record_count = context["ti"].xcom_pull(task_ids="extract_orders")
        print(f"待处理记录数: {record_count}")

        # 执行HiveQL转换
        from airflow.providers.apache.hive.hooks.hive import HiveHook
        hive_hook = HiveHook(hive_cli_conn_id="hive_default")
        hive_hook.run_cli(f"""
            INSERT OVERWRITE TABLE dwd.dwd_order PARTITION (dt='{ds}')
            SELECT
                id AS order_id,
                user_id,
                product_id,
                quantity,
                total_amount,
                CASE status
                    WHEN 0 THEN 'pending'
                    WHEN 1 THEN 'paid'
                    WHEN 2 THEN 'shipped'
                    WHEN 3 THEN 'completed'
                    WHEN 4 THEN 'cancelled'
                    ELSE 'unknown'
                END AS order_status,
                create_time,
                update_time
            FROM ods.orders
            WHERE dt = '{ds}'
        """)

    def aggregate_orders(**context):
        """聚合汇总"""
        ds = context["ds"]
        from airflow.providers.apache.hive.hooks.hive import HiveHook
        hive_hook = HiveHook(hive_cli_conn_id="hive_default")
        hive_hook.run_cli(f"""
            INSERT OVERWRITE TABLE dws.dws_user_trade_1d PARTITION (dt='{ds}')
            SELECT
                user_id,
                COUNT(DISTINCT order_id) AS order_count,
                SUM(total_amount) AS total_amount,
                MIN(create_time) AS first_order_time,
                MAX(create_time) AS last_order_time
            FROM dwd.dwd_order
            WHERE dt = '{ds}' AND order_status IN ('paid','shipped','completed')
            GROUP BY user_id
        """)

    def generate_report(**context):
        """生成报表"""
        ds = context["ds"]
        from airflow.providers.apache.hive.hooks.hive import HiveHook
        hive_hook = HiveHook(hive_cli_conn_id="hive_default")
        hive_hook.run_cli(f"""
            INSERT OVERWRITE TABLE ads.ads_daily_trade_report
            SELECT
                '{ds}' AS report_date,
                COUNT(DISTINCT user_id) AS paying_users,
                SUM(order_count) AS total_orders,
                SUM(total_amount) AS total_amount
            FROM dws.dws_user_trade_1d
            WHERE dt = '{ds}'
        """)

    def send_notification(**context):
        """发送完成通知"""
        ds = context["ds"]
        print(f"ETL完成通知: {ds} 的订单数据处理完毕")

    # 定义任务
    extract = PythonOperator(
        task_id="extract_orders",
        python_callable=extract_orders,
        pool="mysql_pool",          # 使用连接池限制并发
    )

    transform = PythonOperator(
        task_id="transform_orders",
        python_callable=transform_orders,
    )

    aggregate = PythonOperator(
        task_id="aggregate_orders",
        python_callable=aggregate_orders,
    )

    report = PythonOperator(
        task_id="generate_report",
        python_callable=generate_report,
    )

    # 数据质量检查
    quality_check = BashOperator(
        task_id="quality_check",
        bash_command="""
            hive -e "
                SELECT IF(COUNT(1) > 0, 'PASS', 'FAIL')
                FROM dws.dws_user_trade_1d
                WHERE dt = '{{ ds }}'
            "
        """,
    )

    notify = PythonOperator(
        task_id="send_notification",
        python_callable=send_notification,
        trigger_rule="all_done",   # 无论上游成功或失败都执行
    )

    # 定义依赖关系
    extract >> transform >> aggregate >> quality_check >> report >> notify
```

## Operators（操作符）

### 常用Operators

```python
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.bash import BashOperator
from airflow.operators.empty import EmptyOperator
from airflow.operators.trigger_dagrun import TriggerDagRunOperator
from airflow.providers.common.sql.operators.sql import SQLExecuteQueryOperator
from airflow.providers.http.operators.http import SimpleHttpOperator
from airflow.providers.slack.operators.slack_webhook import SlackWebhookOperator
from airflow.utils.task_group import TaskGroup

with DAG("operator_examples", schedule_interval="@daily",
         start_date=datetime(2024, 1, 1)) as dag:

    # 1. BashOperator：执行Shell命令
    run_spark_job = BashOperator(
        task_id="run_spark",
        bash_command="""
            spark-submit \
                --master yarn \
                --deploy-mode cluster \
                --num-executors 10 \
                /jobs/etl/spark_order_etl.py --date {{ ds }}
        """,
    )

    # 2. PythonOperator：执行Python函数
    def process_data(date, **context):
        import pandas as pd
        df = pd.read_parquet(f"/data/processed/{date}/orders.parquet")
        summary = df.groupby("user_id")["amount"].sum().reset_index()
        summary.to_parquet(f"/data/summary/{date}/user_summary.parquet")
        return f"处理完成，共 {len(summary)} 个用户"

    process_task = PythonOperator(
        task_id="process_data",
        python_callable=process_data,
        op_kwargs={"date": "{{ ds }}"},
    )

    # 3. SQLExecuteQueryOperator：执行SQL
    create_table = SQLExecuteQueryOperator(
        task_id="create_table",
        conn_id="hive_default",
        sql="""
            CREATE TABLE IF NOT EXISTS dws.user_daily_summary (
                user_id BIGINT,
                total_amount DECIMAL(12,2),
                stat_date STRING
            )
            PARTITIONED BY (dt STRING)
            STORED AS ORC
        """,
    )

    # 4. BranchPythonOperator：条件分支
    def check_data_volume(**context):
        record_count = context["ti"].xcom_pull(task_ids="extract_orders")
        if record_count > 100000:
            return "process_large_batch"
        else:
            return "process_small_batch"

    branch = BranchPythonOperator(
        task_id="branch_decision",
        python_callable=check_data_volume,
    )

    large_batch = EmptyOperator(task_id="process_large_batch")
    small_batch = EmptyOperator(task_id="process_small_batch")
    join = EmptyOperator(task_id="join", trigger_rule="none_failed")

    # 5. TriggerDagRunOperator：触发其他DAG
    trigger_downstream = TriggerDagRunOperator(
        task_id="trigger_report_dag",
        trigger_dag_id="daily_report_generation",
        conf={"date": "{{ ds }}"},
        wait_for_completion=False,
    )

    # 6. TaskGroup：任务分组
    with TaskGroup(group_id="data_quality") as quality_group:
        check_nulls = SQLExecuteQueryOperator(
            task_id="check_null_values",
            conn_id="hive_default",
            sql="SELECT COUNT(1) FROM dws.user_daily_summary WHERE user_id IS NULL AND dt='{{ ds }}'",
        )
        check_duplicates = SQLExecuteQueryOperator(
            task_id="check_duplicates",
            conn_id="hive_default",
            sql="""
                SELECT user_id, COUNT(1) as cnt
                FROM dws.user_daily_summary WHERE dt='{{ ds }}'
                GROUP BY user_id HAVING COUNT(1) > 1
            """,
        )

    # 依赖关系
    create_table >> run_spark_job >> process_task >> branch
    branch >> [large_batch, small_batch] >> join
    join >> quality_group >> trigger_downstream
```

## Sensors（传感器）

Sensors用于等待某个条件满足后再继续执行。

```python
from airflow.sensors.external_task import ExternalTaskSensor
from airflow.sensors.time_sensor import TimeSensor
from airflow.sensors.filesystem import FileSensor
from airflow.providers.http.sensors.http import HttpSensor

with DAG("sensor_examples", schedule_interval="@daily",
         start_date=datetime(2024, 1, 1)) as dag:

    # 1. 等待外部DAG完成
    wait_for_upstream = ExternalTaskSensor(
        task_id="wait_for_user_etl",
        external_dag_id="daily_user_etl",
        external_task_id="generate_report",
        timeout=3600,
        poke_interval=60,
    )

    # 2. 等待文件出现
    wait_for_file = FileSensor(
        task_id="wait_for_data_file",
        filepath="/data/landing/orders/{{ ds }}/orders.csv",
        fs_conn_id="fs_default",
        poke_interval=30,
        timeout=1800,
    )

    # 3. 等待HTTP端点可用
    wait_for_api = HttpSensor(
        task_id="wait_for_api_ready",
        http_conn_id="data_api",
        endpoint="/api/v1/health",
        response_check=lambda response: response.json().get("status") == "healthy",
        poke_interval=30,
        timeout=300,
    )

    # 4. 自定义Python Sensor
    from airflow.sensors.python import PythonSensor

    def check_kafka_lag(**context):
        """检查Kafka消费者延迟"""
        from kafka import KafkaAdminClient
        admin = KafkaAdminClient(bootstrap_servers="kafka:9092")
        # 获取消费者组延迟
        # 返回True表示条件满足，可以继续
        return True

    wait_kafka = PythonSensor(
        task_id="wait_kafka_catch_up",
        python_callable=check_kafka_lag,
        poke_interval=60,
        timeout=1800,
    )

    wait_for_upstream >> wait_for_file >> wait_for_api >> wait_kafka
```

## XCom（跨任务通信）

XCom允许任务之间传递少量数据。

```python
def extract_and_push(**context):
    """提取数据并通过XCom传递"""
    df = pd.read_sql("SELECT COUNT(1) as cnt FROM orders", conn)
    record_count = int(df["cnt"].iloc[0])

    # Push到XCom
    context["ti"].xcom_push(key="record_count", value=record_count)
    context["ti"].xcom_push(key="extract_status", value="success")

    # 返回值自动推送到XCom（key=return_value）
    return {"count": record_count, "table": "orders"}

def consume_xcom(**context):
    """消费XCom数据"""
    ti = context["ti"]

    # Pull XCom
    record_count = ti.xcom_pull(task_ids="extract", key="record_count")
    result = ti.xcom_pull(task_ids="extract", key="return_value")

    print(f"记录数: {record_count}")
    print(f"完整结果: {result}")
```

## 调度与时区

### 调度表达式

```python
# Cron表达式
schedule_interval="0 2 * * *"      # 每天02:00
schedule_interval="0 */4 * * *"    # 每4小时
schedule_interval="0 0 * * 1"      # 每周一00:00
schedule_interval="0 0 1 * *"      # 每月1日00:00

# 预设表达式
schedule_interval="@daily"         # 每天
schedule_interval="@hourly"        # 每小时
schedule_interval="@weekly"        # 每周
schedule_interval="@monthly"       # 每月
schedule_interval="@once"          # 只执行一次
schedule_interval=None             # 不自动调度，只能手动触发
```

### 时区配置

```python
# airflow.cfg
# [core]
# default_timezone = Asia/Shanghai

from pendulum import datetime as pendulum_datetime

with DAG(
    dag_id="timezone_example",
    schedule_interval="0 2 * * *",
    start_date=pendulum_datetime(2024, 1, 1, tz="Asia/Shanghai"),
) as dag:
    pass
```

## 部署配置

### Docker Compose部署

```yaml
# docker-compose.yml
version: "3.8"
services:
  airflow-webserver:
    image: apache/airflow:2.8.0
    command: webserver
    ports:
      - "8080:8080"
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
      AIRFLOW__CORE__FERNET_KEY: ''
      AIRFLOW__CORE__DAGS_ARE_PAUSED_AT_CREATION: 'true'
      AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres
      - redis

  airflow-scheduler:
    image: apache/airflow:2.8.0
    command: scheduler
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins
    depends_on:
      - postgres
      - redis

  airflow-worker:
    image: apache/airflow:2.8.0
    command: celery worker
    environment:
      AIRFLOW__CORE__EXECUTOR: CeleryExecutor
      AIRFLOW__DATABASE__SQL_ALCHEMY_CONN: postgresql+psycopg2://airflow:airflow@postgres/airflow
      AIRFLOW__CELERY__BROKER_URL: redis://:@redis:6379/0
    volumes:
      - ./dags:/opt/airflow/dags
      - ./logs:/opt/airflow/logs
      - ./plugins:/opt/airflow/plugins

  postgres:
    image: postgres:15
    environment:
      POSTGRES_USER: airflow
      POSTGRES_PASSWORD: airflow
      POSTGRES_DB: airflow
    volumes:
      - postgres_data:/var/lib/postgresql/data

  redis:
    image: redis:7-alpine

volumes:
  postgres_data:
```

## 最佳实践

1. **幂等性**：每个任务应支持重复执行而不产生副作用
2. **原子性**：每个任务完成一个独立的工作单元
3. **失败重试**：合理配置retry策略，处理瞬时故障
4. **连接池**：使用Pool限制对外部系统的并发访问
5. **参数化**：使用DAG参数和模板变量，避免硬编码
6. **监控告警**：配置email_on_failure和SLA告警
7. **DAG文件简洁**：业务逻辑放在独立的Python模块中，DAG只做编排
