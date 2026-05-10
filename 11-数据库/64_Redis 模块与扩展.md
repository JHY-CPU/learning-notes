# Redis 模块与扩展


## 🧩 Redis 模块与扩展


Redis 模块系统 (Redis Modules)、RediSearch 全文搜索、RedisJSON JSON 操作、RedisGraph 图数据库、RedisTimeSeries 时序、RedisBloom 布隆过滤器。


## Redis 模块介绍


```
// ========== Redis 模块 ==========
// 扩展 Redis 功能的插件系统 (Redis 4.0+)

// ========== 模块加载 ==========
// 配置文件加载:
loadmodule /path/to/module.so

// 运行时加载:
MODULE LOAD /path/to/module.so

// 查看已加载模块:
MODULE LIST

// 卸载模块:
MODULE UNLOAD module_name

// ========== 官方模块 ==========
// Redis 官方维护的模块:
// RediSearch: 全文搜索
// RedisJSON: JSON 操作
// RedisGraph: 图数据库
// RedisTimeSeries: 时序数据
// RedisBloom: 布隆过滤器/Count-min Sketch
// RedisAI: 深度学习模型推理

// ========== 第三方模块 ==========
// RedisCell: 限流
// Redis-ML: 机器学习
// ReJSON: JSON (已被 RedisJSON 替代)
// ZSet 扩展
```


## RediSearch 全文搜索


```
// ========== RediSearch ==========
// 基于 Redis 的全文搜索模块

// ========== 创建索引 ==========
FT.CREATE idx:products ON HASH PREFIX 1 product:
    SCHEMA name TEXT WEIGHT 5.0
    description TEXT
    price NUMERIC
    tags TAG

// 说明:
// ON HASH: 基于 Hash 数据类型
// PREFIX 1 product:: 监视 product: 开头的 key
// TEXT: 全文搜索字段 (可分词)
// TAG: 精确匹配字段
// NUMERIC: 数字范围查询

// ========== 搜索 ==========
// 全文搜索:
FT.SEARCH idx:products "mongodb"
FT.SEARCH idx:products "mongodb database"
FT.SEARCH idx:products "%mongo%"   // 模糊搜索

// 组合条件:
FT.SEARCH idx:products "@name:redis @price:[0 100]"

// 分页:
FT.SEARCH idx:products "redis" LIMIT 0 10

// ========== 聚合 ==========
FT.AGGREGATE idx:products "*"
    GROUPBY 1 @tags
    REDUCE COUNT 0 AS num

// ========== 自动补全 ==========
FT.SUGADD autocomplete "redis database" 100
FT.SUGGET autocomplete "red"  // 返回建议

// ========== 管理 ==========
FT.INFO idx:products         // 索引信息
FT.DROPINDEX idx:products    // 删除索引

// ========== 与传统全文搜索对比 ==========
// 性能: Redis 内存搜索, 比 Elasticsearch 快 10x
// 功能: 不如 ES 丰富 (无聚合/分析)
// 适合: 轻量全文搜索, 自动补全
```


## RedisJSON 与 RedisGraph


```
// ========== RedisJSON ==========
// 原生 JSON 操作 (不需要序列化/反序列化)

// 设置 JSON:
JSON.SET user:1 $ '{"name":"Alice","age":28,"address":{"city":"Beijing"}}'

// 获取:
JSON.GET user:1 $             // 全部
JSON.GET user:1 $.name        // "Alice"
JSON.GET user:1 $.address.city  // "Beijing"

// 更新:
JSON.SET user:1 $.age 29
JSON.NUMINCRBY user:1 $.age 1   // age +1

// 数组操作:
JSON.ARRAPPEND user:1 $.tags '"tech"'
JSON.ARRPOP user:1 $.tags

// 删除:
JSON.DEL user:1 $.address

// ========== RedisGraph ==========
// 图数据库, 使用 Cypher 查询语言

// 创建节点:
GRAPH.QUERY social "CREATE (:Person {name:'Alice', age:28})"
GRAPH.QUERY social "CREATE (:Person {name:'Bob', age:25})"

// 创建关系:
GRAPH.QUERY social
    "MATCH (a:Person {name:'Alice'}), (b:Person {name:'Bob'})
     CREATE (a)-[:KNOWS {since:2020}]->(b)"

// 查询:
GRAPH.QUERY social
    "MATCH (a:Person)-[:KNOWS]->(b:Person)
     WHERE a.name = 'Alice'
     RETURN b.name, b.age"

// ========== RedisTimeSeries ==========
// 时序数据, 支持聚合和降采样

// 创建时序:
TS.CREATE sensor:temp RETENTION 86400000   // 保留 1 天
TS.CREATE sensor:humidity LABELS type "humidity"

// 添加数据点:
TS.ADD sensor:temp 1696118400 23.5
TS.MADD sensor:temp 1696118500 24.1 sensor:humidity 1696118500 60

// 查询 (聚合):
TS.RANGE sensor:temp 1696118400 1696204800
    AGGREGATION AVG 60000   // 每分钟平均
```


## RedisBloom 与 RedisCell


```
// ========== RedisBloom ==========
// 概率性数据结构模块

// ========== 布隆过滤器 ==========
// 添加元素:
BF.ADD bloom:users "alice@test.com"
BF.MADD bloom:users "bob@test.com" "carol@test.com"

// 检查存在:
BF.EXISTS bloom:users "alice@test.com"   // 1 (可能存在)
BF.EXISTS bloom:users "unknown"           // 0 (不存在)

// 自定义参数:
BF.RESERVE bloom:users 0.01 1000000
// 0.01 = 1% 错误率
// 1000000 = 预计元素数
// 错误率越低, 占用内存越大

// ========== Cuckoo Filter ==========
// 类似布隆过滤器, 支持删除
CF.ADD cf:users "alice@test.com"
CF.EXISTS cf:users "alice@test.com"  // 1
CF.DEL cf:users "alice@test.com"     // 支持删除!

// ========== Count-Min Sketch ==========
// 频率估计
CMS.INITBYPROB cms:events 0.01 0.01
CMS.INCR cms:events "page:42"        // 出现次数 +1
CMS.QUERY cms:events "page:42"       // 估计出现次数

// ========== RedisCell (第三方限流) ==========
// 令牌桶限流
CL.THROTTLE api:user:100 10 5 60
// 10: 爆发限制 (burst)
// 5: 每 60 秒 5 个请求
// 返回: [0(允许)/1(拒绝), 总限制, 剩余, 重试秒数, 重置秒数]

// ========== 模块选择指南 ==========
// 全文搜索: RediSearch
// JSON 操作: RedisJSON
// 图数据: RedisGraph
// 时序数据: RedisTimeSeries
// 存在判断: RedisBloom
// 限流: RedisCell 或 Lua
```


> **Note:** 💡 模块要点: Redis 模块扩展新功能; RediSearch 做轻量全文搜索; RedisJSON 原生 JSON 操作; RedisTimeSeries 处理时序; RedisBloom 做存在判断与去重; 模块加载 loadmodule; 官方模块比第三方更稳定。


## 练习


<!-- Converted from: 64_Redis 模块与扩展.html -->
