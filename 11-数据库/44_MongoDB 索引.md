# MongoDB 索引


## 📑 MongoDB 索引


单字段/复合/多键/文本/哈希索引、索引属性 (唯一/稀疏/TTL)、索引创建与管理、explain() 查询分析、索引选择与优化。


## 索引类型


```
// ========== 索引类型 ==========

// ========== 1. 单字段索引 ==========
// 默认 _id 有唯一索引

// 创建单字段索引
db.users.createIndex({ age: 1 })       // 升序
db.users.createIndex({ name: -1 })     // 降序

// ========== 2. 复合索引 ==========
// 多个字段的组合索引

// 创建复合索引
db.users.createIndex({ age: 1, name: -1 })

// 复合索引最左前缀:
// 有效查询: age / age+name
// 无效: name (缺少前导字段 age)

// ESRegex 索引使用:
// { age: { $gte: 25 } }       → 使用 age 索引
// { age: 25, name: "Alice" }  → 使用 age+name 复合索引 (最佳)

// ========== 3. 多键索引 (数组字段) ==========
// 对数组字段自动创建多键索引

db.articles.createIndex({ tags: 1 })

// 每个数组元素都指向原文档
// 注意: 一个复合索引只能有一个数组字段

// ========== 4. 文本索引 ==========
// 支持全文搜索

db.articles.createIndex(
    { title: "text", body: "text" },
    { weights: { title: 10, body: 3 } }  // 权重
)

// 使用文本索引
db.articles.find(
    { $text: { $search: "mongodb database" } }
)

// 文本搜索排序
db.articles.find(
    { $text: { $search: "mongodb" } },
    { score: { $meta: "textScore" } }
).sort({ score: { $meta: "textScore" } })

// ========== 5. 哈希索引 ==========
// 用于散列分片

db.users.createIndex({ _id: "hashed" })
```


## 索引属性


```
// ========== 索引属性 ==========

// ========== 唯一索引 ==========
// 保证字段值唯一

db.users.createIndex({ email: 1 }, { unique: true })
// 插入重复 email 会报错

// 复合唯一索引
db.users.createIndex(
    { first_name: 1, last_name: 1 },
    { unique: true }
)

// ========== 稀疏索引 ==========
// 只索引有该字段的文档
db.users.createIndex(
    { phone: 1 },
    { sparse: true }
)
// 适合: 字段在部分文档中存在

// ========== TTL 索引 ==========
// 自动过期删除 (适合日志/会话)

db.logs.createIndex(
    { created_at: 1 },
    { expireAfterSeconds: 86400 }  // 24 小时后删除
)

// TTL 限制:
// - 只能用于日期字段
// - 不能是复合索引
// - 后台 60 秒检查一次

// ========== 部分索引 ==========
// 只索引满足条件的文档 (3.2+)

db.users.createIndex(
    { age: 1 },
    { partialFilterExpression: { age: { $gte: 18 } } }
)
// 节省空间, 只索引成年人

// ========== 隐藏索引 ==========
// 评估删除索引的影响 (4.4+)

db.users.hideIndex("age_1")          // 隐藏索引
db.users.unhideIndex("age_1")        // 取消隐藏
// 隐藏后查询不会使用该索引, 但索引仍在维护
```


## 索引管理


```
// ========== 索引管理命令 ==========

// 创建索引
db.users.createIndex(
    { email: 1 },
    { name: "idx_email", unique: true }  // 自定义名称
)

// 后台创建 (不阻塞)
db.users.createIndex(
    { age: 1 },
    { background: true }  // MongoDB 4.2+ 默认后台
)

// 查看集合索引
db.users.getIndexes()

// 输出示例:
// [
//   { "v": 2, "key": { "_id": 1 }, "name": "_id_" },
//   { "v": 2, "key": { "age": 1 }, "name": "age_1" },
//   { "v": 2, "key": { "email": 1 }, "name": "idx_email", "unique": true }
// ]

// 索引大小
db.users.totalIndexSize()

// 删除单索引
db.users.dropIndex("age_1")
db.users.dropIndex({ age: 1 })

// 删除所有索引 (除了 _id)
db.users.dropIndexes()

// ========== explain() 分析 ==========

// 查看查询计划
db.users.find({ age: { $gte: 25 } }).explain("executionStats")

// 关键字段:
// stage: IXSCAN (走索引) / COLLSCAN (全表扫描)
// nReturned: 实际返回行数
// totalDocsExamined: 扫描文档数
// totalKeysExamined: 扫描索引键数
// executionTimeMillis: 执行时间

// 理想: totalDocsExamined ≈ nReturned (精确过滤)

// ========== hint() 强制索引 ==========

// 强制使用指定索引
db.users.find({ age: { $gte: 25 } }).hint({ age: 1 })

// 验证索引是否被使用
db.users.find({ age: { $gte: 25 } }).explain()
// stage 应为 IXSCAN
```


## 索引优化指南


```
// ========== 索引优化原则 ==========

// ========== ESR 原则 (复合索引字段顺序) ==========
// E — Equality (等值查询)
// S — Sort (排序)
// R — Range (范围查询)

// 查询: db.users.find({ age: 25 }).sort({ name: 1 })
//        .find({ age: { $gte: 20, $lte: 30 } })

// 最佳索引: { age: 1, name: 1 }
// E: age=25 精确匹配
// S: name 排序
// R: age 范围匹配 (但 E 优先)

// ========== 常见性能指标 ==========

// 1. 选择性 (Selectivity)
//    唯一值越多, 选择性越好
//    email > name > gender
//    选择性高的字段放前面

// 2. 覆盖查询 (Covered Query)
//    索引包含查询所有字段
//    不需要回表读取文档

// 创建覆盖索引
db.users.createIndex(
    { email: 1, name: 1, age: 1 }
)

// 覆盖查询:
db.users.find(
    { email: "alice@test.com" },
    { _id: 0, name: 1, age: 1 }
)
// 只从索引读取, 不访问文档

// ========== 索引注意事项 ==========
// 1. 不是越多越好 (写性能下降)
// 2. 监控慢查询, 针对性索引
// 3. 复合索引字段顺序影响大
// 4. 避免索引数组字段过多
// 5. 大集合创建索引用后台模式
// 6. 定期检查未使用的索引
```


> **Note:** 💡 索引要点: 单字段索引基础; 复合索引 ESR 顺序; 文本索引做搜索; TTL 索引自动过期; 唯一索引保证数据完整性; explain() 分析查询计划; 选择性高的字段放索引前部; 覆盖索引避免回表; 索引不是越多越好。


## 练习


<!-- Converted from: 44_MongoDB 索引.html -->
