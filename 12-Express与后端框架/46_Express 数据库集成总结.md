# Express 数据库集成总结


## 🗄️ Express 数据库集成总结


数据库选型对比 (SQL vs NoSQL)、Node.js 数据库客户端全景、连接池管理、事务策略、迁移方案、缓存策略、生产最佳实践、综合架构决策树。


## 数据库选型对比


```
// ========== 数据库选型决策 ==========

// ┌──────────────┬──────────────┬──────────────┬──────────────┬──────────────┐
// │              │  PostgreSQL  │   MongoDB    │    Redis     │    Prisma    │
// ├──────────────┼──────────────┼──────────────┼──────────────┼──────────────┤
// │ 类型         │  关系型      │  文档型      │  键值对      │  ORM 抽象层   │
// │ 数据模型     │  表 + 行     │  集合 + 文档  │  String/Hash  │  Schema       │
// │ 事务         │  ACID 完整   │  副本集支持   │  MULTI/EXEC  │  $transaction │
// │ 迁移         │  Knex/Prisma │  Mongoose     │  —           │  内置         │
// │ 适合场景     │  强一致/复杂查询│ 灵活/快速迭代  │  缓存/会话/队列│  类型安全     │
// │ Node 客户端  │  pg          │  mongoose     │  ioredis     │  @prisma/client│
// │ 查询构建     │  Knex/raw SQL│  MongoDriver  │  命令        │  prisma.xxx   │
// └──────────────┴──────────────┴──────────────┴──────────────┴──────────────┘

// ========== 选型指南 ==========
// 你的数据是什么样的?
// ├── 结构化, 关联复杂 → PostgreSQL + Prisma/Knex
// ├── 非结构化, 快速迭代 → MongoDB + Mongoose
// └── 临时数据, 高速访问 → Redis

// 你需要什么?
// ├── 强事务保证 → PostgreSQL
// ├── 灵活 Schema → MongoDB
// ├── 高速缓存 → Redis
// ├── 类型安全 → Prisma
// └── 最大性能 → raw SQL / pg

// ========== 混合架构 ==========
// 大多数应用同时使用多个数据库:
//
// PostgreSQL (主数据库)
//   ├── 用户, 订单, 产品 (结构化)
//   ├── ACID 事务
//   └── 复杂查询/报表
//
// Redis (缓存/实时)
//   ├── API 缓存 (Cache Aside)
//   ├── Session 存储
//   ├── 排行榜/计数器
//   └── 消息队列
//
// MongoDB (可选)
//   ├── 日志/事件 (非结构化)
//   ├── 产品目录 (灵活 Schema)
//   └── 快速原型
```


## Node.js 数据库客户全景


```
// ========== Node.js 数据库生态 ==========

// ┌─────────────────────────────────────────────────────┐
// │                    Node.js 数据库                     │
// ├─────────┬──────────┬──────────┬──────────────────────┤
// │ 类型     │ 客户端    │ 查询构建  │ ORM                  │
// ├─────────┼──────────┼──────────┼──────────────────────┤
// │ PostgreSQL│ pg      │ Knex     │ Prisma, Sequelize,   │
// │         │          │          │ TypeORM, MikroORM    │
// ├─────────┼──────────┼──────────┼──────────────────────┤
// │ MySQL   │ mysql2   │ Knex     │ Prisma, Sequelize     │
// ├─────────┼──────────┼──────────┼──────────────────────┤
// │ MongoDB │ mongodb  │ —        │ Mongoose, Prisma      │
// ├─────────┼──────────┼──────────┼──────────────────────┤
// │ Redis   │ ioredis  │ —        │ —                     │
// └─────────┴──────────┴──────────┴──────────────────────┘

// ========== 各层选择对比 ==========

// Raw SQL (pg + 手写 SQL)
// ✅ 性能最佳, 完全控制
// ✅ 复杂查询灵活
// ❌ 需手动防注入 (参数化)
// ❌ 无迁移工具

// 查询构建器 (Knex)
// ✅ 自动参数化防注入
// ✅ 迁移 + 种子内置
// ✅ 多数据库切换
// ❌ 复杂查询不如 raw SQL 灵活

// ORM (Prisma/Mongoose)
// ✅ 类型安全 (Prisma)
// ✅ 关联查询方便
// ✅ 自动迁移
// ❌ 性能开销
// ❌ 学习曲线

// ========== 推荐组合 ==========
// 小项目: Prisma (快速开发)
// 中项目: Knex (灵活 + 迁移)
// 大项目: pg + Knex (性能 + 灵活)
// 微服务: Prisma (类型安全)
// 原型: Mongoose (灵活 Schema)
```


## 连接池与事务最佳实践


```
// ========== 连接池管理 ==========

// ========== 通用连接池配置 ==========
const poolConfig = {
    // PostgreSQL (pg)
    pg: { max: 20, min: 2, idleTimeoutMillis: 30000 },

    // MongoDB (mongoose)
    mongo: { maxPoolSize: 10, minPoolSize: 2, serverSelectionTimeoutMS: 5000 },

    // Redis (ioredis)
    redis: { maxRetriesPerRequest: 3, retryStrategy: (t) => Math.min(t * 200, 2000) },
};

// ========== 连接池监控 ==========
// 定期检查连接池健康
setInterval(async () => {
    // PostgreSQL
    const pgStats = {
        total: pgPool.totalCount,
        idle: pgPool.idleCount,
        waiting: pgPool.waitingCount,
    };

    // MongoDB
    const mongoConn = mongoose.connection.readyState;
    // 0=断开, 1=已连, 2=连接中, 3=断开中

    // Redis
    const redisStatus = redis.status;  // 'ready' | 'connecting' | 'reconnecting'

    logger.info('Connection pool stats', { pgStats, mongoConn, redisStatus });
}, 60000);

// ========== 事务策略对比 ==========
// PostgreSQL: ACID 事务, SAVEPOINT 部分回滚
// MongoDB:    副本集事务, 4.0+ 支持多文档事务
// Redis:      MULTI/EXEC 乐观锁, WATCH 监视

// ========== 数据库迁移策略 ==========
// Knex:     migrations/ 目录, up/down 方法
// Prisma:   prisma/migrations/, migrate dev/deploy
// Mongoose: 无内置迁移 (用 migrate-mongo 或 manual)

// ========== 综合架构示例 ==========
// app.js - 数据库初始化
async function initializeDatabases() {
    // 1. PostgreSQL
    await pgPool.connect();
    logger.info('PostgreSQL connected');

    // 2. MongoDB
    await mongoose.connect(config.mongo.uri);
    logger.info('MongoDB connected');

    // 3. Redis
    await redis.connect();
    logger.info('Redis connected');

    // 4. 运行迁移
    await knex.migrate.latest();
    logger.info('Migrations completed');
}

// ========== 优雅关闭 ==========
async function gracefulShutdown() {
    logger.info('Closing database connections...');

    await Promise.all([
        pgPool.end(),
        mongoose.connection.close(),
        redis.quit(),
    ]);

    logger.info('All connections closed');
    process.exit(0);
}

process.on('SIGTERM', gracefulShutdown);
process.on('SIGINT', gracefulShutdown);
```


## 生产 Checklist


```
// ========== 数据库生产 Checklist ==========

// ✅ 1. 连接池大小合理配置
//    PostgreSQL: max=20 (根据服务器配置)
//    MongoDB: maxPoolSize=10
//    Redis: 单连接即可

// ✅ 2. 连接超时设置
//    connectionTimeoutMillis: 5000
//    serverSelectionTimeoutMS: 5000

// ✅ 3. 重试策略
//    不要无限重试, 设置最大次数
//    指数退避 (exponential backoff)

// ✅ 4. SSL/TLS 加密 (生产)
//    PostgreSQL: ssl: { rejectUnauthorized: false }
//    MongoDB: mongodb+srv://... (自动 TLS)
//    Redis: tls: {}

// ✅ 5. 连接监控与告警
//    连接池使用率
//    慢查询日志
//    连接错误通知

// ✅ 6. 备份策略
//    PostgreSQL: pg_dump / WAL 归档
//    MongoDB: mongodump / Atlas 备份
//    Redis: RDB + AOF 持久化 + 备份

// ✅ 7. 索引策略
//    频繁查询字段建索引
//    复合索引最左前缀
//    监控慢查询添加索引

// ✅ 8. 读写分离 (高并发)
//    PostgreSQL: 主从 + pgpool
//    MongoDB: 副本集 (primary/ secondary)
//    Redis: 哨兵 / Cluster

// ✅ 9. 连接字符串管理
//    环境变量注入
//    不提交到 git
//    定期轮换密码

// ✅ 10. 迁移自动化
//    CI/CD 中自动运行迁移
//    回滚策略
//    迁移测试 (staging 先跑)
```


> **Note:** 💡 数据库集成要点: 选型决策树 (结构化→PG, 灵活→Mongo, 缓存→Redis); 混合架构 (PG + Redis 最常见); 连接池监控; 事务策略按数据库选择; 迁移版本控制; 生产 Checklist (SSL/备份/索引/读写分离); 优雅关闭所有连接; 根据项目规模选择客户端 (raw SQL / Knex / ORM)。


## 练习


<!-- Converted from: 46_Express 数据库集成总结.html -->
