# NewSQL 与分布式数据库


## 🌐 NewSQL 与分布式数据库


NewSQL 概念、TiDB (MySQL 兼容)、CockroachDB (PostgreSQL 兼容)、Vitess MySQL 集群、分布式数据库 vs 传统数据库、云原生数据库趋势。


## NewSQL 概念


```
// ========== NewSQL ==========
// 兼具 SQL 的关系模型和 NoSQL 的水平扩展能力

// ========== 为什么需要 NewSQL? ==========
// SQL 的问题: 难以水平扩展 (分库分表复杂)
// NoSQL 的问题: 不支持/弱支持 SQL, 事务弱
// NewSQL: SQL + 分布式 + ACID

// ========== NewSQL 核心特性 ==========
// 1. SQL 兼容 (MySQL/PostgreSQL 协议)
// 2. 水平扩展 (自动分片)
// 3. 分布式 ACID 事务
// 4. 高可用 (自动故障转移)
// 5. 强一致 (CP 系统)

// ========== 代表产品 ==========
// ┌──────────────┬──────────────┬──────────────┐
// │ 产品          │ 兼容         │ 特点         │
// ├──────────────┼──────────────┼──────────────┤
// │ TiDB         │ MySQL        │ 自动分片     │
// │ CockroachDB  │ PostgreSQL   │ 全球部署     │
// │ YugabyteDB   │ PostgreSQL   │ 高性能       │
// │ Google Spanner│ 自研        │ 全球一致     │
// │ Vitess       │ MySQL 中间件 │ 集群方案     │
// └──────────────┴──────────────┴──────────────┘

// ========== 何时使用 NewSQL ==========
// 1. MySQL/PostgreSQL 单机不够用
// 2. 需要 SQL + 事务 + 水平扩展
// 3. 不想用分库分表中间件
// 4. 需要强一致性
```


## TiDB (MySQL 兼容)


```
// ========== TiDB ==========
// PingCAP 开源的分布式 NewSQL 数据库
// 兼容 MySQL 5.7 协议

// ========== 架构 ==========
// ┌─────────────────────────────────┐
// │          TiDB Server            │
// │  (SQL 层, 无状态, 水平扩展)     │
// ├─────────────────────────────────┤
// │          PD Server              │
// │  (调度层, 存储元数据/调度)       │
// ├─────────────────────────────────┤
// │     TiKV (存储层, 分布式 KV)     │
// │     Raft 复制 + 自动分片        │
// └─────────────────────────────────┘

// 分层架构:
// - TiDB: MySQL 协议兼容, SQL 解析
// - PD: 元数据管理, 调度
// - TiKV: 数据存储 (Raft 共识)
// - TiFlash: 列存储 (分析)

// ========== 特点 ==========
// ✅ MySQL 兼容 (可无缝迁移)
// ✅ 水平扩展 (加 TiDB/TiKV 节点)
// ✅ 分布式事务 (Percolator 模型)
// ✅ HTAP (TiFlash 列存)
// ✅ 在线 DDL (ALTER 不锁表)

// ❌ 写入延迟稍高 (Raft 复制)
// ❌ 复杂 JOIN 不如单机 PG
// ❌ 运维复杂

// ========== 连接 ==========
// 像 MySQL 一样连接:
mysql -h 127.0.0.1 -P 4000 -u root

// 创建表 (和 MySQL 几乎一样):
CREATE TABLE users (
    id INT PRIMARY KEY,
    name VARCHAR(50)
);

// ========== 适用场景 ==========
// - MySQL 不够用需要水平扩展
// - 金融级分布式事务
// - 混合负载 (OLTP + OLAP)
```


## CockroachDB (PostgreSQL 兼容)


```
// ========== CockroachDB ==========
// Cockroach Labs 开源的分布式 SQL 数据库
// 兼容 PostgreSQL 协议

// ========== 架构 ==========
// ┌────────────────────────────────────┐
// │    CockroachDB 节点 (对等架构)      │
// │  ├── SQL 层 (PG 协议兼容)          │
// │  ├── 事务层 (分布式 ACID)          │
// │  ├── 复制层 (Raft 共识)            │
// │  └── 存储层 (KV 引擎)             │
// ├────────────────────────────────────┤
// │   所有节点对等, 无特殊角色          │
// │   每个节点都是 SQL 入口             │
// └────────────────────────────────────┘

// ========== 特点 ==========
// ✅ PostgreSQL 兼容
// ✅ 全球部署 (多区域)
// ✅ 强一致性 (CP)
// ✅ 自动故障修复 (自愈)
// ✅ 在线 Schema 变更
// ✅ 对等架构 (无单点)

// ❌ 写入延迟 (跨区域 Raft)
// ❌ 不支持某些 PG 特性
// ❌ 单机性能不如 PostgreSQL

// ========== 连接 ==========
cockroach sql --insecure --host=localhost:26257

// CREATE TABLE (PG 语法):
CREATE TABLE users (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name STRING,
    email STRING UNIQUE
);

// ========== 适用场景 ==========
// - 全球部署 (多区域)
// - 需要强一致 + 高可用
// - 灾难恢复 (多活)
// - 替代 PostgreSQL 单机瓶颈
```


## Vitess & 数据库未来趋势


```
// ========== Vitess ==========
// YouTube 开源的 MySQL 集群方案
// 不是数据库, 是数据库中间件

// ========== 架构 ==========
// ┌────────────────────────────────┐
// │     应用 (连接 VTGate)         │
// ├────────────────────────────────┤
// │  VTGate (代理, 无状态)         │
// │  路由/连接池/查询重写          │
// ├────────────────────────────────┤
// │  VTTablet (每个 MySQL 实例)    │
// │  管理/备份/复制                │
// ├────────────────────────────────┤
// │  Topology (元数据: etcd/zk)    │
// └────────────────────────────────┘

// 功能: 分片, 读写分离, 连接池, 故障转移
// 适合: 已有 MySQL 需要水平扩展

// ========== 数据库未来趋势 ==========
// 1. 云原生 (Kubernetes 部署)
//    数据库 Operator 自动运维
//    Serverless 数据库 (按需付费)

// 2. HTAP (混合事务分析处理)
//    同一数据库同时支撑 OLTP 和 OLAP
//    代表: TiFlash, ClickHouse

// 3. AI + 数据库
//    自动调优 (索引/查询)
//    自然语言查询
//    向量搜索 (pgvector)

// 4. 多模型
//    一库支持多种数据模型
//    代表: PostgreSQL (JSON/GIS/全文/向量)
//          ArangoDB (文档/图/键值)

// ========== 选型建议 ==========
// 小/中: PostgreSQL / MySQL
// 大 (SQL): TiDB / CockroachDB
// 大 (NoSQL): MongoDB / Cassandra
// 超大规模: Spanner / ClickHouse
// 已有 MySQL: Vitess / TiDB
```


> **Note:** 💡 NewSQL 要点: SQL + 水平扩展 + 分布式事务; TiDB 兼容 MySQL; CockroachDB 兼容 PostgreSQL; Raft 共识算法保证一致性; 适合单机不够用又需要 SQL 的场景; 运维复杂度比单机高; 云原生/HTAP 是未来趋势。


## 练习


<!-- Converted from: 68_NewSQL 与分布式数据库.html -->
