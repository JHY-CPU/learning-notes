# 数据仓库与数据架构 笔记模块

本模块系统梳理数据仓库与数据架构的核心理论、主流工具及工程实践，适合数据工程师、数据架构师以及希望深入理解大数据体系的学习者。

## 目录

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | `01_数据仓库概述.md` | 数据仓库定义、OLTP vs OLAP、Inmon vs Kimball、架构分层 |
| 02 | `02_维度建模理论.md` | 星型/雪花模型、事实表/维度表、缓慢变化维度(SCD) |
| 03 | `03_ETL与ELT流程.md` | ETL vs ELT、抽取模式、转换规则、加载策略、CDC |
| 04 | `04_Apache_Airflow工作流.md` | DAG定义、Operators、Sensors、XCom、调度与Python示例 |
| 05 | `05_dbt数据转换.md` | dbt模型、Jinja模板、测试、文档、增量模型与SQL示例 |
| 06 | `06_数据湖与Lakehouse.md` | 数据湖/仓库/Lakehouse对比、Delta Lake、Iceberg、Hudi |
| 07 | `07_数据质量与治理.md` | 数据质量维度、Great Expectations、数据血缘、元数据管理 |
| 08 | `08_实时数据架构.md` | Lambda架构、Kappa架构、Kafka+Flink实时ETL |
| 09 | `09_Hive与数据仓库实践.md` | Hive分区/分桶、内部/外部表、HiveQL、优化策略 |
| 10 | `10_Flink实时计算.md` | DataStream API、窗口、状态管理、精确一次语义 |
| 11 | `11_数据可视化与BI.md` | 仪表盘设计、Superset/Metabase、图表选型原则 |
| 12 | `12_数据仓库分层设计.md` | ODS/DWD/DWS/ADS分层、命名规范、开发标准 |

## 技术栈覆盖

- **存储**: Hive, Delta Lake, Apache Iceberg, Apache Hudi
- **计算**: Spark, Flink, Presto/Trino
- **调度**: Apache Airflow, dbt
- **流处理**: Kafka, Flink, Spark Structured Streaming
- **数据质量**: Great Expectations, dbt tests, Deequ
- **可视化**: Apache Superset, Metabase, Grafana
- **治理**: Apache Atlas, DataHub, Amundsen

## 学习路径建议

1. **基础理论** (01-02): 建立数据仓库与维度建模的概念基础
2. **数据处理** (03-05): 掌握ETL/ELT流程和主流工具(Airflow, dbt)
3. **存储架构** (06, 09): 理解数据湖、Lakehouse及Hive实践
4. **实时计算** (08, 10): 学习实时数据架构与Flink编程
5. **质量与治理** (07): 数据质量保障与治理体系
6. **应用层** (11-12): 数据可视化与分层设计规范
