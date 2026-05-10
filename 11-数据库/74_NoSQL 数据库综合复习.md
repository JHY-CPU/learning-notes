# NoSQL 数据库综合复习


## 🗄️ NoSQL 数据库综合复习


NoSQL 四大家族全景对比、35 文件速查索引、MongoDB/Redis/Cassandra/ES/Neo4j/InfluxDB 核心要点、多语言持久化架构、选型决策树、实践项目清单、下一步学习指引。


## NoSQL 全景对比


```
// ========== NoSQL 四大家族 ==========
// ┌──────────┬──────────────┬───────────────┬──────────────────┐
// │ 类型      │ 代表产品      │ 数据模型       │ 适用场景          │
// ├──────────┼──────────────┼───────────────┼──────────────────┤
// │ 文档     │ MongoDB      │ JSON/BSON 文档 │ 通用/日志/CMS    │
// │ 键值     │ Redis        │ Key-Value     │ 缓存/计数器/会话  │
// │ 列族     │ Cassandra    │ 宽列 + 行键   │ 时序/写入密集     │
// │ 图       │ Neo4j        │ 节点 + 关系   │ 社交/推荐/图谱   │
// └──────────┴──────────────┴───────────────┴──────────────────┘
//
// 其他类型:
// 时序: InfluxDB, TimescaleDB, Prometheus
// 搜索: Elasticsearch (基于 Lucene)
// NewSQL: TiDB, CockroachDB (SQL + 水平扩展)

// ========== CAP 分类 ==========
// CP 系统: MongoDB (默认), HBase, Redis (Cluster), TiDB
// AP 系统: Cassandra, DynamoDB, CouchDB
// CA 系统: 单机数据库 (MySQL/PostgreSQL 非分布式)

// ========== BASE vs ACID ==========
// ACID: 强一致, 适合金融/订单
// BASE: 最终一致, 适合高并发/大规模
```


## 35 个 NoSQL 文件速查


```
// ========== 35 文件索引 ==========
// 661  NoSQL 概述        — 分类/CAP/BASE/PACELC
// 662  MongoDB 文档模型   — BSON/嵌入vs引用/ObjectId
// 663  插入与查询         — insert/find/运算符/投影
// 664  更新与删除         — update/delete/数组运算符
// 665  索引               — 单字段/复合/ESR/文本/TTL
// 666  聚合管道           — $match/$group/$lookup/$bucket
// 667  副本集             — Primary/Secondary/选举/写关注
// 668  分片集群           — 分片键/hash槽/块迁移
// 669  Mongoose ODM      — Schema/验证/hooks/populate
// 670  数据建模           — 电商/博客/时序建模模式
// 671  运维与监控         — mongodump/mongostat/慢查询
// 672  Redis 介绍安装     — 安装/CLI/redis.conf 配置
// 673  String与Key       — SET/GET/INCR/过期/位图
// 674  List与Set         — 队列/集合运算/去重
// 675  Hash与ZSet        — 对象存储/排行榜/延迟队列
// 676  持久化RDB与AOF    — 快照/追加日志/混合模式
// 677  事务与Pipeline    — MULTI/WATCH/批量优化
// 678  发布订阅与Stream  — PubSub/Stream/消费组
// 679  缓存模式与过期     — 穿透/击穿/雪崩/分布式锁
// 680  Lua脚本           — EVAL/原子操作/限流
// 681  客户端与编程       — ioredis/redis-py/Spring/go-redis
// 682  高可用哨兵集群     — 主从/哨兵/Cluster 分片
// 683  实际应用场景       — 排行榜/限流/Geo/UV
// 684  安全与运维         — 密码/ACL/bigkey/调优
// 685  模块与扩展         — RediSearch/RedisJSON/RedisBloom
// 686  NoSQL对比与选型    — MongoDB/PG/Redis 用例
// 687  NoSQL面试题        — 高频面试 Q&A
// 688  运维与监控         — 备份/监控/Prometheus/Grafana
// 689  NewSQL分布式       — TiDB/CockroachDB/Vitess
// 690  综合复习           — 全景图/速查/Next
// 691  Cassandra          — 列族/CQL/分区键/Gossip
// 692  Elasticsearch      — 倒排索引/Query DSL/聚合
// 693  时序数据库          — InfluxDB/TimescaleDB/Prometheus
// 694  Neo4j图数据库      — Cypher/图算法/知识图谱
// 695  综合复习(本文件)    — 四大家族/选型/实践
```


## 数据库核心要点速记


```
// ========== MongoDB 核心 ==========
// 1. 文档模型: BSON, 无 Schema, 嵌入 vs 引用
// 2. 聚合管道: $match→$group→$sort→$lookup
// 3. 索引 ESR: E=Equality, S=Sort, R=Range
// 4. 副本集: w:majority + j:true 最高安全
// 5. 分片键: 高基数 + 查询频繁, 选后不可改
// 6. Mongoose: .lean() 提升 4-10x 读性能

// ========== Redis 核心 ==========
// 1. 5 结构: String/List/Set/Hash/ZSet
// 2. 缓存问题: 穿透(空值/布隆) 击穿(锁) 雪崩(随机TTL)
// 3. 持久化: 推荐 AOF+RDB 混合
// 4. 高可用: 数据<内存用哨兵, >内存用集群
// 5. Lua: 原子脚本, 分布式锁/限流
// 6. Pipeline: 批量减少 RTT

// ========== Cassandra 核心 ==========
// 1. 列族模型: Keyspace→Table→Row Key→Column
// 2. CQL: 查询必须包含分区键!
// 3. AP 系统: 最终一致, 写入极快
// 4. 去中心化: 所有节点对等, Gossip 协议
// 5. Compaction: STCS/LCS/TWCS

// ========== Elasticsearch 核心 ==========
// 1. 倒排索引: Term→Document 映射
// 2. 分词: text 分词 vs keyword 精确
// 3. Query DSL: bool (must/filter/should/must_not)
// 4. 聚合: terms + avg/percentile/stats
// 5. 分片: 部署前确定, 单分片 20-50GB

// ========== Neo4j 核心 ==========
// 1. 图模型: Node + Relationship = 一等公民
// 2. Cypher: ASCII 图语法
// 3. 图算法: PageRank/社区发现/最短路径
// 4. 适合: 社交/推荐/知识图谱/欺诈检测

// ========== 时序数据库核心 ==========
// 1. 模型: Tag(索引) + Field(数值) + Time
// 2. 保留策略: 自动过期, 降采样
// 3. InfluxDB: 专业时序, Flux
// 4. TimescaleDB: PG 扩展, SQL 支持
```


## 选型决策树


```
// ========== NoSQL 选型决策 ==========
//
// 1. 需要全文搜索?
//    ├── 是 → Elasticsearch
//    └── 否 → 2
//
// 2. 需要图关系分析/社交推荐?
//    ├── 是 → Neo4j
//    └── 否 → 3
//
// 3. 需要时序数据 (监控/IoT)?
//    ├── 是 → InfluxDB / TimescaleDB
//    └── 否 → 4
//
// 4. 写入密集或海量数据?
//    ├── 是 → Cassandra (AP) / MongoDB (CP)
//    └── 否 → 5
//
// 5. 需要缓存/计数器/队列?
//    ├── 是 → Redis
//    └── 否 → 6
//
// 6. 灵活 Schema + 快速迭代?
//    ├── 是 → MongoDB
//    └── 否 → PostgreSQL/MySQL 可能足够

// ========== 混合架构推荐 ==========
// ┌────────────────────────────────────────┐
// │ 小型项目:                               │
// │   PostgreSQL (一切) + Redis (缓存)     │
// ├────────────────────────────────────────┤
// │ 中型项目:                               │
// │   PostgreSQL (业务) +                  │
// │   Redis (缓存/队列) +                  │
// │   MongoDB (日志/CMS)                   │
// ├────────────────────────────────────────┤
// │ 大型项目:                               │
// │   PostgreSQL (订单/用户) +             │
// │   Redis (缓存/会话/限流) +             │
// │   MongoDB (日志/非结构) +             │
// │   Elasticsearch (搜索) +               │
// │   Cassandra/InfluxDB (时序) +          │
// │   Neo4j (推荐/图谱)                    │
// └────────────────────────────────────────┘
```


## 实践项目清单


```
// ========== 动手实践 ==========
//
// Level 1: 基础 (1-2 天每个)
// 1. MongoDB: 写一个博客 CRUD (Node.js + Mongoose)
// 2. Redis: 实现缓存 + 计数器 (ioredis)
// 3. Elasticsearch: 商品搜索 + 聚合
//
// Level 2: 中级 (3-5 天每个)
// 4. MongoDB 聚合: 电商订单报表 ($lookup + $group)
// 5. Redis 分布式锁 + 限流 (Lua 脚本)
// 6. Redis 排行榜 + 实时统计 (ZSet + HyperLogLog)
// 7. ES + Kibana: 日志分析仪表盘
//
// Level 3: 高级 (1-2 周每个)
// 8. 混合架构: Express + PostgreSQL + Redis + MongoDB
// 9. 社交推荐: Neo4j 好友推荐 + PageRank
// 10. 监控系统: Prometheus + InfluxDB + Grafana
//
// ========== Docker 快速启动 ==========
// MongoDB:
docker run -d --name mongo -p 27017:27017 mongo:7

// Redis:
docker run -d --name redis -p 6379:6379 redis:7

// Elasticsearch:
docker run -d --name es -p 9200:9200 -e "discovery.type=single-node" elasticsearch:8

// Cassandra:
docker run -d --name cassandra -p 9042:9042 cassandra:4

// InfluxDB:
docker run -d --name influx -p 8086:8086 influxdb:2

// Neo4j:
docker run -d --name neo4j -p 7474:7474 -p 7687:7687 neo4j:5
```


> **Note:** 💡 NoSQL 不是替代 SQL, 而是补充。现代架构通常混合使用多种数据库, 各取所长。核心原则: 用 SQL 处理关系和事务, 用 NoSQL 处理大规模/高性能/灵活场景。35 个文件覆盖了主流 NoSQL 数据库, 下一阶段进入 Node.js 后端深入 (696-745), 结合 Express + MongoDB + Redis 构建实战项目。


## 练习


<!-- Converted from: 74_NoSQL 数据库综合复习.html -->
