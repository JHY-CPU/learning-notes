# dbt数据转换

## dbt概述

dbt (data build tool) 是一个开源的数据转换工具，让数据分析师和工程师使用SQL编写数据模型，通过软件工程最佳实践（版本控制、测试、文档）来管理数据转换流程。

**核心理念**：dbt不做数据抽取和加载，只负责T（Transform）阶段，将数据从staging表转换为可用于分析的模型。

### dbt工作原理

```text
+-------------------+     +-------------------+     +-------------------+
|  Source Data       |     |  dbt Models       |     |  Transformed Data |
|  (Staging Tables)  |---->|  (.sql + .yml)    |---->|  (Marts/Tables)   |
+-------------------+     +-------------------+     +-------------------+
                                   |
                          +-------------------+
                          |  dbt编译 & 执行     |
                          |  - 解析依赖关系     |
                          |  - 生成DAG          |
                          |  - 执行SQL          |
                          +-------------------+
```

## 项目结构

```text
my_dbt_project/
├── dbt_project.yml          # 项目配置
├── profiles.yml             # 数据库连接配置
├── models/
│   ├── staging/             # Staging层（1:1映射源表）
│   │   ├── stg_orders.sql
│   │   ├── stg_users.sql
│   │   └── stg_orders.yml
│   ├── intermediate/        # 中间层（业务逻辑）
│   │   ├── int_user_orders.sql
│   │   └── int_order_items.sql
│   └── marts/               # Mart层（最终输出）
│       ├── finance/
│       │   ├── fct_revenue.sql
│       │   └── _finance__docs.md
│       └── marketing/
│           ├── fct_user_acquisition.sql
│           └── dim_campaign.sql
├── seeds/                   # 静态CSV数据
│   ├── country_codes.csv
│   └── currency_rates.csv
├── tests/                   # 自定义测试
│   ├── assert_no_future_orders.sql
│   └── assert_positive_revenue.sql
├── macros/                  # 可复用宏
│   └── cents_to_dollars.sql
└── snapshots/               # 快照（SCD Type 2）
    └── snap_users.sql
```

## 核心概念

### Sources（数据源）

Sources定义外部数据来源，让dbt追踪源数据的血缘关系。

```yaml
# models/staging/_sources.yml
version: 2

sources:
  - name: raw_ecommerce
    description: "电商平台原始数据（来自MySQL CDC）"
    database: raw
    schema: ecommerce
    freshness:
      warn_after: {count: 12, period: hour}
      error_after: {count: 24, period: hour}
    loaded_at_field: _cdc_timestamp
    tables:
      - name: orders
        description: "订单主表"
        columns:
          - name: id
            description: "订单唯一ID"
            tests:
              - unique
              - not_null
      - name: order_items
        description: "订单商品明细"
      - name: users
        description: "用户表"
```

### Models（模型）

dbt模型就是SQL SELECT语句，dbt将其编译为CREATE TABLE AS或CREATE VIEW。

#### Staging Model

```sql
-- models/staging/stg_orders.sql
-- Staging层：1:1映射源表，做基础类型转换和字段重命名

{{ config(
    materialized='view',
    tags=['staging', 'ecommerce']
) }}

WITH source AS (
    SELECT * FROM {{ source('raw_ecommerce', 'orders') }}
),

renamed AS (
    SELECT
        id              AS order_id,
        user_id,
        status          AS order_status,
        total_amount    AS order_total,
        currency        AS currency_code,
        payment_method,
        shipping_address,
        created_at      AS order_created_at,
        updated_at      AS order_updated_at,
        _cdc_timestamp  AS _loaded_at
    FROM source
)

SELECT * FROM renamed
WHERE order_id IS NOT NULL
```

#### Intermediate Model

```sql
-- models/intermediate/int_user_orders.sql
-- 中间层：关联多表，计算业务指标

{{ config(
    materialized='ephemeral',  -- CTE，不物化为表
    tags=['intermediate']
) }}

WITH orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

order_items AS (
    SELECT * FROM {{ ref('stg_order_items') }}
),

users AS (
    SELECT * FROM {{ ref('stg_users') }}
),

user_order_summary AS (
    SELECT
        u.user_id,
        u.user_name,
        u.email,
        u.registered_at,
        COUNT(DISTINCT o.order_id) AS total_orders,
        SUM(o.order_total) AS lifetime_value,
        MIN(o.order_created_at) AS first_order_at,
        MAX(o.order_created_at) AS last_order_at,
        DATEDIFF('day', MIN(o.order_created_at), MAX(o.order_created_at)) AS customer_lifespan_days
    FROM users u
    LEFT JOIN orders o ON u.user_id = o.user_id
    GROUP BY 1, 2, 3, 4
)

SELECT * FROM user_order_summary
```

#### Mart Model（事实表）

```sql
-- models/marts/finance/fct_revenue.sql
-- Mart层：最终输出的事实表

{{ config(
    materialized='table',
    partition_by={
        "field": "order_date",
        "data_type": "date",
        "granularity": "day"
    },
    cluster_by=["product_category"],
    tags=['finance', 'daily'],
    pre_hook="SET query_tag = 'fct_revenue'",
    post_hook=[
        "GRANT SELECT ON {{ this }} TO ROLE analyst_role"
    ]
) }}

WITH order_items AS (
    SELECT * FROM {{ ref('stg_order_items') }}
),

orders AS (
    SELECT * FROM {{ ref('stg_orders') }}
),

products AS (
    SELECT * FROM {{ ref('dim_products') }}
),

revenue AS (
    SELECT
        DATE(o.order_created_at)        AS order_date,
        o.order_id,
        o.user_id,
        o.order_status,
        oi.product_id,
        p.product_name,
        p.product_category,
        p.product_subcategory,
        oi.quantity,
        oi.unit_price,
        oi.quantity * oi.unit_price     AS gross_revenue,
        COALESCE(oi.discount_amount, 0) AS discount_amount,
        oi.quantity * oi.unit_price - COALESCE(oi.discount_amount, 0) AS net_revenue,
        o.currency_code,
        o.payment_method
    FROM order_items oi
    JOIN orders o ON oi.order_id = o.order_id
    JOIN products p ON oi.product_id = p.product_id
    WHERE o.order_status NOT IN ('cancelled', 'refunded')
)

SELECT * FROM revenue
```

#### Mart Model（维度表）

```sql
-- models/marts/marketing/dim_campaign.sql

{{ config(
    materialized='table',
    unique_key='campaign_id'
) }}

SELECT
    c.campaign_id,
    c.campaign_name,
    c.campaign_type,
    c.channel,
    c.start_date,
    c.end_date,
    c.budget,
    c.target_audience,
    -- 聚合指标
    COUNT(DISTINCT o.user_id) AS attributed_users,
    COUNT(DISTINCT o.order_id) AS attributed_orders,
    SUM(o.order_total) AS attributed_revenue,
    SUM(o.order_total) / NULLIF(c.budget, 0) AS roas  -- 广告投资回报率
FROM {{ ref('stg_campaigns') }} c
LEFT JOIN {{ ref('fct_revenue') }} o
    ON DATE(o.order_date) BETWEEN c.start_date AND c.end_date
    AND o.attribution_campaign_id = c.campaign_id
GROUP BY 1, 2, 3, 4, 5, 6, 7, 8
```

## 物化策略（Materializations）

| 策略 | 说明 | 适用场景 | 性能 |
|------|------|---------|------|
| `view` | 创建视图 | Staging层，轻量级查询 | 查询时计算 |
| `table` | 创建物理表 | Mart层，频繁查询 | 预计算，快 |
| `incremental` | 增量更新 | 大表，每日新增 | 折中 |
| `ephemeral` | CTE | 中间层，临时计算 | 不存储 |

### 增量模型（Incremental）

```sql
-- models/marts/events/fct_events.sql
-- 增量模型：只处理新增数据

{{ config(
    materialized='incremental',
    unique_key='event_id',
    incremental_strategy='merge',  -- merge / append / delete+insert
    partition_by={
        "field": "event_date",
        "data_type": "date",
        "granularity": "day"
    }
) }}

WITH events AS (
    SELECT
        event_id,
        user_id,
        event_type,
        event_time,
        DATE(event_time) AS event_date,
        properties
    FROM {{ source('raw_events', 'user_events') }}

    {% if is_incremental() %}
    -- 增量逻辑：只处理上次运行之后的新数据
    WHERE event_time > (SELECT MAX(event_time) FROM {{ this }})
    {% endif %}
)

SELECT * FROM events

{% if is_incremental() %}
-- 去重：防止重复数据
QUALIFY ROW_NUMBER() OVER (PARTITION BY event_id ORDER BY event_time DESC) = 1
{% endif %}
```

### 增量策略配置

```sql
-- BigQuery增量策略
{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='order_id',
    partition_by={"field": "order_date", "data_type": "date"},
    cluster_by=["user_id", "order_status"]
) }}

-- Snowflake增量策略
{{ config(
    materialized='incremental',
    incremental_strategy='delete+insert',
    unique_key='order_id'
) }}

-- Spark/Databricks增量策略
{{ config(
    materialized='incremental',
    incremental_strategy='merge',
    unique_key='order_id',
    file_format='delta'
) }}
```

## Jinja模板

dbt使用Jinja模板引擎实现SQL的动态化和复用。

### 变量与配置

```yaml
# dbt_project.yml
vars:
  start_date: "2024-01-01"
  enable_new_logic: true
  currency_rates_table: "raw.fxrates"
```

```sql
-- 使用变量
SELECT *
FROM {{ ref('fct_events') }}
WHERE event_date >= '{{ var("start_date") }}'

{% if var("enable_new_logic") %}
    -- 新逻辑
    , NEW_FUNCTION(event_data) AS processed_data
{% else %}
    -- 旧逻辑
    , event_data AS processed_data
{% endif %}
```

### 宏（Macros）

```sql
-- macros/finance/cents_to_dollars.sql
-- 将分为单位的金额转为元
{% macro cents_to_dollars(column_name, decimal_places=2) %}
    ROUND(CAST({{ column_name }} AS DECIMAL(18, {{ decimal_places }})) / 100, {{ decimal_places }})
{% endmacro %}

-- 使用宏
SELECT
    order_id,
    {{ cents_to_dollars('total_amount_cents') }} AS total_amount_dollars,
    {{ cents_to_dollars('tax_cents', 4) }} AS tax_dollars
FROM {{ ref('stg_orders') }}
```

```sql
-- macros/utils/generate_schema_name.sql
-- 自定义schema生成逻辑
{% macro generate_schema_name(custom_schema_name, node) %}
    {%- if custom_schema_name is none -%}
        {{ target.schema }}
    {%- else -%}
        {{ target.schema }}_{{ custom_schema_name | trim }}
    {%- endif -%}
{% endmacro %}
```

```sql
-- macros/utils/union_tables.sql
-- 联合多张结构相同的表
{% macro union_tables(table_prefix, database, schema_pattern) %}
    {% set tables = adapter.get_relations_in_schema(database, schema_pattern) %}

    {% for table in tables %}
        SELECT *, '{{ table.name }}' AS _source_table
        FROM {{ database }}.{{ schema_pattern }}.{{ table.name }}
        {% if not loop.last %}UNION ALL{% endif %}
    {% endfor %}
{% endmacro %}
```

## 测试（Tests）

### Schema Tests（声明式）

```yaml
# models/marts/finance/_finance__models.yml
version: 2

models:
  - name: fct_revenue
    description: "收入事实表，记录每一笔交易的收入明细"
    columns:
      - name: order_id
        description: "订单ID"
        tests:
          - not_null
          - relationships:
              to: ref('dim_orders')
              field: order_id

      - name: net_revenue
        description: "净收入（扣除折扣）"
        tests:
          - not_null
          - dbt_utils.expression_is_true:
              expression: ">= 0"

    # 模型级别测试
    tests:
      - dbt_utils.unique_combination_of_columns:
          combination_of_columns:
            - order_id
            - product_id
```

### 自定义Data Tests

```sql
-- tests/assert_no_negative_revenue.sql
-- 自定义测试：收入不能为负数

SELECT *
FROM {{ ref('fct_revenue') }}
WHERE net_revenue < 0

-- tests/assert_order_dates_valid.sql
-- 自定义测试：订单日期不能在未来

SELECT *
FROM {{ ref('fct_revenue') }}
WHERE order_date > CURRENT_DATE
```

## Snapshots（SCD Type 2）

Snapshots实现缓慢变化维度Type 2，追踪历史变化。

```sql
-- snapshots/snap_users.sql

{% snapshot snap_users %}

{{
    config(
        target_schema='snapshots',
        unique_key='user_id',
        strategy='timestamp',        -- timestamp / check
        updated_at='updated_at',     -- 用于判断变化的列
        invalidate_hard_deletes=True  -- 软删除处理
    )
}}

SELECT
    user_id,
    user_name,
    email,
    membership_level,
    city,
    updated_at
FROM {{ source('raw_ecommerce', 'users') }}

{% endsnapshot %}
```

snapshot生成的表结构：

| 列名 | 说明 |
|------|------|
| dbt_scd_id | 快照记录唯一ID |
| dbt_updated_at | 记录更新时间 |
| dbt_valid_from | 有效开始时间 |
| dbt_valid_to | 有效结束时间（NULL表示当前有效） |
| dbt_is_deleted | 是否已删除 |

```sql
-- 查询快照：获取特定日期的用户状态
SELECT *
FROM {{ ref('snap_users') }}
WHERE user_id = 1001
    AND dbt_valid_from <= '2024-06-01'
    AND (dbt_valid_to > '2024-06-01' OR dbt_valid_to IS NULL);
```

## Seeds

Seeds是项目中的CSV文件，通过`dbt seed`加载到数据库，适合小型维度表和映射表。

```csv
# seeds/country_codes.csv
country_code,country_name,region,continent
CN,中国,东亚,亚洲
US,美国,北美,北美洲
JP,日本,东亚,亚洲
DE,德国,西欧,欧洲
```

```sql
-- 在模型中引用seed
SELECT
    o.order_id,
    o.amount,
    c.country_name,
    c.region
FROM {{ ref('stg_orders') }} o
JOIN {{ ref('country_codes') }} c ON o.country_code = c.country_code
```

## dbt命令

```bash
# 安装依赖
dbt deps

# 运行所有模型
dbt run

# 运行指定模型及其下游
dbt run --select fct_revenue+

# 运行指定tag的模型
dbt run --select tag:daily

# 运行增量模型
dbt run --full-refresh  # 全量刷新

# 运行测试
dbt test
dbt test --select fct_revenue  # 测试指定模型

# 生成文档
dbt docs generate
dbt docs serve  # 启动文档网站

# 加载seeds
dbt seed

# 运行snapshot
dbt snapshot

# 编译（不执行，查看生成的SQL）
dbt compile --select stg_orders
```

## 最佳实践

1. **分层架构**：Staging -> Intermediate -> Marts，每层职责清晰
2. **Sources定义**：所有外部表通过sources引用，追踪血缘
3. **测试先行**：每个模型至少有not_null和unique测试
4. **文档完善**：每个模型和字段都有description
5. **增量优先**：大表使用incremental物化策略
6. **命名规范**：stg\_前缀表示staging，fct\_表示事实表，dim\_表示维度表
