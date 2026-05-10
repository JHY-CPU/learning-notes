# MongoDB深入 - NoSQL深入

*深入理解 WiredTiger 存储引擎、文档模型设计、聚合管道、Change Streams 及分片策略*

WiredTiger 核心特性

| 特性 | 说明 |
| --- | --- |
| 并发控制 | 文档级锁（Document-Level Locking），不同文档的操作互不阻塞 |
| 事务支持 | 支持多文档 ACID 事务（MongoDB 4.0+） |
| 数据压缩 | 默认 Snappy 压缩，可选 Zlib/Zstd，节省 50-80% 存储空间 |
| Checkpoint | 每 60 秒或 2GB 数据写入时创建一致性快照 |
| Journal（WAL） | 预写日志保证持久性，默认每 100ms 刷盘一次 |
| 内存缓存 | 默认使用 50%（减 1GB）系统内存作为缓存 |

内嵌文档 vs 引用（范式化）对比

| 维度 | 内嵌（Embedded） | 引用（Referenced） |
| --- | --- | --- |
| 数据模型 | 子文档直接嵌入父文档 | 存储 ObjectId 引用，查询时关联 |
| 读取性能 | 一次查询获取所有数据 | 需要多次查询或 $lookup |
| 更新性能 | 需更新整个文档 | 可独立更新 |
| 文档大小 | 可能超过 16MB 限制 | 不受影响 |
| 数据一致性 | 天然一致（同一文档） | 需要应用层保证 |
| 适用场景 | 一对少、数据一起访问 | 一对多、多对多、频繁独立更新 |

聚合管道核心阶段

| 阶段 | 作用 | SQL 等价 |
| --- | --- | --- |
| $match | 过滤文档 | WHERE |
| $project | 选择/计算字段 | SELECT |
| $group | 分组聚合 | GROUP BY |
| $sort | 排序 | ORDER BY |
| $limit / $skip | 分页 | LIMIT / OFFSET |
| $unwind | 展开数组 | 类似 LATERAL JOIN |
| $lookup | 关联查询（类似 JOIN） | LEFT JOIN |
| $addFields | 添加计算字段 | SELECT expr AS alias |
| $facet | 多管道并行执行 | 多个子查询 UNION |
| $bucket | 分桶统计 | CASE WHEN 分组 |

分片策略对比

| 策略 | 原理 | 优点 | 缺点 |
| --- | --- | --- | --- |
| Hashed Sharding | 对分片键计算哈希值 | 数据均匀分布 | 不支持范围查询高效执行 |
| Ranged Sharding | 按分片键值范围分片 | 范围查询高效 | 可能产生热点分片 |
| Zone Sharding | 按区域亲和性分配 | 数据本地化 | 配置复杂 |


<!-- Converted from: 01_MongoDB深入.html -->
