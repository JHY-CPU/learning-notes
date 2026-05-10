# MongoDB 运维与监控


## 🔧 MongoDB 运维与监控


备份与恢复 (mongodump/mongorestore)、监控命令 (serverStatus/currentOp)、慢查询分析、内存与存储管理、MongoDB Atlas 云服务。


## 备份与恢复


```
// ========== 备份 ==========

// ========== mongodump (逻辑备份) ==========

// 备份整个数据库
mongodump --uri="mongodb://localhost:27017" --out=/backup/mongo/

// 备份指定数据库
mongodump --db=mydb --out=/backup/mongo/

// 备份指定集合
mongodump --db=mydb --collection=users --out=/backup/mongo/

// 压缩备份
mongodump --db=mydb --archive=/backup/mydb.gz --gzip

// 带认证
mongodump --host=localhost --port=27017 \
    --username=admin --password=secret \
    --db=mydb --out=/backup/mongo/

// ========== mongorestore (恢复) ==========

// 恢复整个备份
mongorestore --uri="mongodb://localhost:27017" /backup/mongo/

// 恢复到指定数据库
mongorestore --db=mydb --drop /backup/mongo/mydb/
// --drop: 恢复前删除已存在的集合

// 从归档恢复
mongorestore --archive=/backup/mydb.gz --gzip

// ========== 副本集备份 ==========
// 在 Secondary 上备份 (不影响 Primary)
// 先停止 Secondary 的 balancer

// ========== Atlas 备份 ==========
// MongoDB Atlas 提供自动快照备份
// - 连续备份 (PITR): 恢复到任意时间点
// - 快照调度: 每 6/12/24 小时
```


## 监控命令


```
// ========== 监控命令 ==========

// ========== serverStatus ==========
db.serverStatus()

// 关键指标:
// connections: {
//   current: 42,           // 当前连接数
//   available: 958,        // 可用连接数
//   totalCreated: 5000     // 累计创建
// }

// opcounters: {            // 操作计数
//   insert: 10000,
//   query: 50000,
//   update: 2000,
//   delete: 500,
//   command: 80000
// }

// network: {
//   bytesIn: 1024000,
//   bytesOut: 5120000,
//   numRequests: 150000
// }

// ========== currentOp (查看当前操作) ==========
db.currentOp()

// 过滤长时间运行的操作 (> 5 秒)
db.currentOp({
    "active": true,
    "secs_running": { "$gt": 5 }
})

// ========== 终止操作 ==========
db.killOp(opid)

// ========== 慢查询监控 ==========

// 1. 开启慢查询日志
db.setProfilingLevel(1, { slowms: 100 })
// 0 = 关闭, 1 = 记录慢查询, 2 = 记录所有

// 2. 查看慢查询
db.system.profile.find().sort({ ts: -1 }).limit(10)

// 3. 慢查询示例:
// {
//   "op": "query",
//   "ns": "mydb.orders",
//   "query": { "status": "pending" },
//   "nreturned": 10,
//   "nscanned": 50000,          // 扫描太多文档!
//   "millis": 2500,             // 耗时 2.5 秒
//   "planSummary": "COLLSCAN"  // 全表扫描!
// }

// ========== mongostat ==========
// 命令行工具: mongostat --host=localhost:27017
// 输出: insert/s, query/s, update/s, delete/s
//      dirty (脏数据), used (内存使用)
```


## 内存与存储


```
// ========== MongoDB 内存管理 ==========

// MongoDB 使用内存映射文件 (MMAPv1 / WiredTiger)
// WiredTiger 内部缓存:
// - 默认: 50% (RAM - 1GB), 或 256MB (小于 1GB)
// - 推荐: 60-80%

// 查看缓存使用:
db.serverStatus().wiredTiger.cache

// 关键指标:
// "bytes currently in the cache": 缓存数据量
// "tracked dirty bytes in the cache": 脏数据
// "pages requested from the cache": 缓存请求
// "pages read into cache": 从磁盘读入

// 缓存命中率:
// 理想: > 95%
// 公式: 1 - (pages read into cache / pages requested from cache)

// 配置缓存大小 (配置文件):
// wiredTiger:
//   engineConfig:
//     cacheSizeGB: 8

// ========== 存储统计 ==========

// 数据库统计
db.stats()
// {
//   "dataSize": "10 GB",       // 实际数据大小
//   "storageSize": "8 GB",     // 磁盘占用
//   "indexSize": "2 GB",       // 索引大小
//   "totalSize": "10 GB"       // 总大小
// }

// 集合统计
db.orders.stats()

// ========== 存储引擎 ==========
// WiredTiger (MongoDB 3.2+ 默认)
// - 文档级并发
// - 压缩: snappy (默认), zlib, zstd
// - 检查点 + journal 日志

// ========== 碎片整理 ==========
// 删除大量数据后, 磁盘空间不立即释放

// 整理碎片 (需要大量磁盘空间):
db.runCommand({ compact: "orders" })

// 重建索引:
db.orders.reIndex()
```


## 安全与 Atlas


```
// ========== MongoDB 安全 ==========

// ========== 用户管理 ==========
use admin

// 创建管理用户
db.createUser({
    user: "admin",
    pwd: "secret",
    roles: [ "root" ]
})

// 创建应用用户 (最小权限)
db.createUser({
    user: "app_user",
    pwd: "app_secret",
    roles: [
        { role: "readWrite", db: "mydb" }
    ]
})

// 认证登录:
// mongosh -u app_user -p --authenticationDatabase mydb

// ========== 启用认证 ==========
// mongod.conf:
// security:
//   authorization: enabled

// ========== 网络与加密 ==========
// net:
//   bindIp: 127.0.0.1      # 只监听本地
//   port: 27017
//   tls:
//     mode: requireTLS
//     certificateKeyFile: /etc/mongo/server.pem

// ========== MongoDB Atlas ==========
// 官方云数据库服务

// 特性:
// - 自动备份 (PITR)
// - 自动扩缩
// - 多区域部署
// - 监控告警
// - 自动 TLS/SSL

// 层级:
// - M0 (免费): 512MB, 共享
// - M2/M5 (入门): 几美元/月
// - M10+ (生产): 专用实例
// - Serverless: 按需付费

// 连接:
// mongodb+srv://user:pass@cluster.mongodb.net/mydb
```


> **Note:** 💡 运维要点: mongodump/mongorestore 定期备份; serverStatus/currentOp 监控; system.profile 查慢查询; WiredTiger 缓存命中率 >95%; 启用认证和 TLS; 压缩和 compact 管理存储; Atlas 简化运维。


## 练习


<!-- Converted from: 50_MongoDB 运维与监控.html -->
