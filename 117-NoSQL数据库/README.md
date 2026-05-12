# 117 - NoSQL 数据库学习笔记

本模块系统性地整理了三种主流 NoSQL 数据库的核心知识，涵盖架构原理、操作语法、性能优化和高级特性。

## 目录结构

### MongoDB（01-06）

| 文件 | 主题 |
|------|------|
| `01_MongoDB概述与架构.md` | 文档模型、BSON、副本集与分片架构 |
| `02_MongoDB_CRUD操作.md` | 增删改查、批量操作、Python PyMongo 示例 |
| `03_MongoDB索引与查询优化.md` | 单字段/复合/文本索引、explain()、覆盖查询 |
| `04_MongoDB聚合管道.md` | `$match/$group/$project/$sort/$lookup` 等阶段 |
| `05_MongoDB副本集与分片.md` | 副本集配置、读偏好、分片键选择 |
| `06_MongoDB事务与变更流.md` | 多文档事务、Change Streams、Resume Token |

### Elasticsearch（07-11）

| 文件 | 主题 |
|------|------|
| `07_Elasticsearch概述与架构.md` | 倒排索引原理、集群/节点/分片/段 |
| `08_ES索引与文档操作.md` | 索引管理、文档 CRUD、Bulk API、Mapping |
| `09_ES查询DSL.md` | match/term/range/bool 查询、聚合 |
| `10_ES分词与Analyzer.md` | Standard/IK/拼音分词器、自定义分析器 |
| `11_ES高级功能.md` | Scroll、Reindex、Alias、Snapshot |

### Neo4j（12-15）

| 文件 | 主题 |
|------|------|
| `12_Neo4j图数据库概述.md` | 属性图模型、与关系型数据库对比 |
| `13_Cypher查询语言.md` | CREATE/MATCH/WHERE/MERGE、模式匹配 |
| `14_Neo4j索引与约束.md` | 节点/关系索引、唯一性约束、全文搜索 |
| `15_Neo4j图算法.md` | PageRank、最短路径、社区检测、中心性 |

## 环境依赖

```bash
pip install pymongo elasticsearch neo4j
```

## 学习建议

- **MongoDB** 部分建议按顺序学习，聚合管道和索引优化是日常开发重点
- **Elasticsearch** 部分可结合实际搜索需求选择性阅读
- **Neo4j** 部分重点关注 Cypher 语法和图算法的应用场景
