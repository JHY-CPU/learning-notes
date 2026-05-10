# SQL 数据库选型


## 🔍 SQL 数据库选型


MySQL vs PostgreSQL vs SQLite 深入对比、选型决策因素、云数据库方案 (RDS/Aurora/Cloud SQL)、NewSQL 与分布式数据库 (TiDB/CockroachDB)、迁移策略。


## 主流数据库对比


```
// ========== MySQL vs PostgreSQL vs SQLite ==========

// 特性            MySQL 8.0         PostgreSQL 16       SQLite
// ──────────────  ───────────────  ─────────────────  ──────────────
// 开源许可        GPL               MIT                  Public Domain
// ACID 事务       InnoDB ✅         ✅                   ✅
// 并发控制        MVCC              MVCC                读写锁
// JSON            支持 (一般)        JSONB (强大)         JSON1 扩展
// 全文搜索        InnoDB FTS        tsvector/tsquery     FTS5
// 窗口函数        8.0+ ✅           ✅                  3.25+
// CTE/递归        8.0+              ✅                  3.8.3+
// GIS             ✅                PostGIS (极强)        ❌
// 物化视图        ❌                ✅                   ❌
// 分区            RANGE/LIST/HASH   RANGE/LIST/HASH     ❌
// 并行查询        8.0.17+           ✅                   ❌
// 扩展性          Replication/集群  Streaming Replication  单文件
// 存储引擎        InnoDB/MyISAM    (单引擎)              (单引擎)
// 默认端口        3306              5432                  -
// 内存占用        高                中                    极低
// 启动速度        慢                中                    极快

// ========== 性能对比 ==========
// 简单查询 (OLTP):      MySQL > PostgreSQL (5-10% 优势)
// 复杂查询 (OLAP):      PostgreSQL > MySQL (20-50% 优势)
// 写入并发:             MySQL ≈ PostgreSQL
// 大表处理:             PostgreSQL 略优
// JSON 查询:            PostgreSQL JSONB >> MySQL JSON
// 全文搜索:             PostgreSQL >> MySQL
```


## 选型决策因素


```
// ========== 选型决策树 ==========

// 1. 项目规模
//    小型/嵌入式:   SQLite
//    中小型 Web:    MySQL 或 PostgreSQL
//    大型复杂:      PostgreSQL 或 TiDB
//    超大规模:      CockroachDB/YugabyteDB

// 2. 数据结构
//    关系型强:      PostgreSQL (复杂查询)
//    简单 CRUD:     MySQL
//    半结构化:      PostgreSQL JSONB
//    地理空间:      PostgreSQL + PostGIS

// 3. 团队技能
//    熟悉 LAMP:     MySQL (PHP/WordPress)
//    熟悉 Python:    PostgreSQL / MySQL
//    熟悉 .NET:      SQL Server
//    云原生:        PostgreSQL (Aurora/Cloud SQL)

// 4. 部署方式
//    自建:          MySQL / PostgreSQL
//    云托管:        RDS / Cloud SQL / Azure DB
//    容器化:        Kubernetes + Operator
//    Serverless:    Aurora Serverless / Planetscale

// ========== 选型建议 ==========
// 用 MySQL 当:
// - LAMP/LEMP 技术栈
// - WordPress/Magento
// - 简单 CRUD 应用
// - 需要成熟的复制方案
// - 团队熟悉 MySQL

// 用 PostgreSQL 当:
// - 复杂查询/分析
// - JSON/全文搜索需求
// - GIS 地理数据
// - 金融/合规 (ACID 更强)
// - 需要高级功能 (物化视图/窗口函数)
```


## 云数据库方案


```
// ========== 云数据库对比 ==========

// AWS:
// - RDS MySQL/PostgreSQL: 托管数据库
// - Aurora MySQL/PostgreSQL: 高性能兼容, 自动扩缩
// - DynamoDB: NoSQL (非 SQL)

// 阿里云:
// - RDS MySQL: 最流行
// - PolarDB: 兼容 MySQL, 性能更强
// - AnalyticDB: 分析型

// GCP:
// - Cloud SQL: MySQL/PostgreSQL/SQL Server
// - Cloud Spanner: 全球分布式, 强一致
// - BigQuery: 分析型 (数据仓库)

// ========== 托管 vs 自建 ==========
// 托管 (RDS/Cloud SQL):
//   优点: 自动备份, 运维少, 监控完善
//   缺点: 贵, 定制少, 可能有供应商锁定
//   适合: 大多数项目

// 自建:
//   优点: 便宜(大流量), 完全控制, 灵活
//   缺点: 运维成本高, 需要 DBA
//   适合: 大规模 (节省成本) 或有特殊需求

// ========== NewSQL / 分布式 ==========
// 当单机数据库不够时:

// TiDB: 兼容 MySQL 协议, 水平扩展
// CockroachDB: 兼容 PostgreSQL 协议, 全球部署
// YugabyteDB: 兼容 PostgreSQL, 高性能
// Vitess: MySQL 集群中间件 (YouTube)
// ProxySQL: MySQL 中间件 (读写分离/分片)

// 选型:
// 需要 MySQL 兼容 + 水平扩展 → TiDB / Vitess
// 需要 PostgreSQL 兼容 + 全球部署 → CockroachDB
```


## 数据库迁移策略


```
// ========== 迁移策略 ==========

// 1. 同数据库升级
//    MySQL 5.7 → 8.0:
//    使用 mysql_upgrade 或逻辑升级

// 2. 异构数据库迁移
//    MySQL → PostgreSQL:
//    工具: pgloader, AWS DMS
//    挑战: 数据类型差异, 存储过程重写, 语法差异

// 3. 迁移到云
//    AWS DMS (Database Migration Service)
//    持续同步, 切换时停机短

// 4. 分库分表
//    应用层: ShardingSphere, MyCat
//    代理层: ProxySQL, Vitess
//    数据库: TiDB (自动分片)

// ========== 常见迁移陷阱 ==========
// 1. 数据类型差异
//    MySQL TINYINT(1) ↔ PostgreSQL BOOLEAN
//    MySQL DATETIME ↔ PostgreSQL TIMESTAMP
//    MySQL TEXT ↔ PostgreSQL TEXT

// 2. 函数差异
//    MySQL DATE_FORMAT ↔ PostgreSQL TO_CHAR
//    MySQL IFNULL ↔ PostgreSQL COALESCE
//    MySQL GROUP_CONCAT ↔ PostgreSQL STRING_AGG

// 3. 排序规则
//    MySQL 默认 utf8_general_ci (大小写不敏感)
//    PostgreSQL 默认 (大小写敏感)

// 4. 自增 ID
//    MySQL AUTO_INCREMENT ↔ PostgreSQL SERIAL
//    迁移后自增值不同!

// ========== 建议 ==========
// 新项目: PostgreSQL (功能强,省心)
// 生态依赖: MySQL (WordPress, LAMP)
// 小工具: SQLite
// 超大规模: TiDB / CockroachDB
// 云上: 优先使用托管服务
```


> **Note:** 💡 选型要点: 简单 CRUD + LAMP → MySQL; 复杂查询 + JSON + GIS → PostgreSQL; 小/嵌入式 → SQLite; 超大规模 → 分布式数据库; 云上优先托管服务; 迁移注意数据类型和函数差异。


## 练习


<!-- Converted from: 33_SQL 数据库选型.html -->
