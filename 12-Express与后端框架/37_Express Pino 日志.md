# Express Pino 日志


## ⚡ Express Pino 日志


Pino 高性能日志、pino-http Express 集成、pino-pretty 格式化、传输 (Transport/File/轮转)、多流输出、子 Logger、性能对比 Winston vs Pino。


## Pino 基础


```
// ========== Pino 日志 ==========
// 最快的 Node.js 日志库
// 比 Winston 快 5-10 倍

// 安装:
// npm install pino pino-http pino-pretty

const pino = require('pino');

// ========== 创建 Logger ==========
const logger = pino({
    // 级别: fatal, error, warn, info, debug, trace
    level: process.env.LOG_LEVEL || 'info',

    // 生产用默认 JSON, 开发用 pretty
    transport: process.env.NODE_ENV !== 'production'
        ? {
            target: 'pino-pretty',
            options: {
                colorize: true,
                translateTime: 'SYS:yyyy-mm-dd HH:MM:ss',
                ignore: 'pid,hostname',
            },
        }
        : undefined,

    // 红色警戒
    redact: {
        paths: ['password', 'secret', 'token', 'headers.authorization', 'req.headers.cookie'],
        censor: '***REDACTED***',
    },
});

// ========== 使用 ==========
logger.info('Server started');
logger.warn({ diskUsage: '85%' }, 'Disk space low');
logger.error({ err }, 'Database connection failed');
logger.debug({ query: 'SELECT ...', duration: 15 }, 'Query executed');

// 结构化日志 (JSON 默认)
// {"level":30,"time":1712345678901,"pid":1234,"hostname":"myhost","msg":"Server started"}
// level: 60=fatal, 50=error, 40=warn, 30=info, 20=debug, 10=trace

// ========== 子 Logger ==========
const dbLogger = logger.child({ service: 'database' });
const authLogger = logger.child({ service: 'auth' });
const httpLogger = logger.child({ service: 'http' });

dbLogger.info('Connected to MongoDB');    // 自动包含 { service: 'database' }
authLogger.warn('Login failed', { email: 'test@test.com' });

// ========== 级别方法 ==========
logger.fatal('Fatal error');      // 进程应退出
logger.error('Error message');     // 错误
logger.warn('Warning');            // 警告
logger.info('Info message');       // 普通信息
logger.debug('Debug info');        // 调试
logger.trace('Trace detail');      // 追踪

// 检查级别是否启用:
if (logger.isLevelEnabled('debug')) {
    logger.debug('Expensive debug log');
}
```


## pino-http Express 集成


```
// ========== pino-http ==========
// Express 请求日志中间件

const pinoHttp = require('pino-http');

// ========== 基础使用 ==========
app.use(pinoHttp({
    logger,  // 复用之前创建的 logger

    // 自动生成 req.id (trace ID)
    genReqId: (req) => req.headers['x-request-id'] || crypto.randomUUID(),

    // 自定义日志级别
    autoLogging: {
        ignore: (req) => req.url === '/health',  // 忽略健康检查
        ignorePaths: ['/favicon.ico'],
    },

    // 自定义序列化
    serializers: {
        req: (req) => ({
            method: req.method,
            url: req.url,
            ip: req.remoteAddress,
        }),
        res: (res) => ({
            statusCode: res.statusCode,
        }),
    },

    // 自定义属性
    customProps: (req) => ({
        userId: req.user?.sub || 'anonymous',
        traceId: req.id,
    }),
}));

// 在路由中使用 req.log:
app.get('/users', async (req, res) => {
    req.log.info('Fetching users');
    // req.log 是请求级别的 logger (自动包含 requestId)
    const users = await User.find();
    res.json(users);
});

// ========== 请求日志输出 ==========
// 每次请求自动输出:
// [2024-04-01 12:00:00] INFO: req completed
//     reqId: "abc-123"
//     req: { method: "GET", url: "/users" }
//     res: { statusCode: 200 }
//     responseTime: 15
//     userId: "u123"

// ========== 完整配置 ==========
const httpLogger = pinoHttp({
    logger,
    genReqId: (req) => req.headers['x-trace-id'] || uuidv4(),

    // 成功/失败不同级别
    autoLogging: {
        ignorePaths: ['/health', '/metrics'],
        getLevel: (res) => {
            if (res.statusCode >= 500) return 'error';
            if (res.statusCode >= 400) return 'warn';
            return 'info';
        },
    },

    // 自定义成功/失败消息
    customSuccessMessage: (req, res) => `${req.method} ${req.url} ${res.statusCode}`,
    customErrorMessage: (req, res, err) => `${req.method} ${req.url} FAILED: ${err.message}`,
});

app.use(httpLogger);

// ========== 错误处理中记录 ==========
app.use((err, req, res, next) => {
    req.log.error({ err, body: req.body }, 'Request error');
    next(err);
});
```


## 多流与文件输出


```
// ========== 多 Stream Transport ==========
// Pino 的 Transport 架构: 子进程处理

const { Writable } = require('stream');
const fs = require('fs');
const pinoms = require('pino-multi-stream');
// npm install pino-multi-stream

// ========== 1. 基本文件输出 ==========
const logger = pino({
    level: 'info',
}, pino.destination('./logs/app.log'));  // 写入文件

// ========== 2. 多流输出 ==========
// 错误日志到文件, 所有日志到控制台

const streams = [
    { stream: process.stdout },                                // 所有级别到控制台
    { level: 'error', stream: pino.destination('logs/error.log') },  // error 到文件
    { level: 'info', stream: pino.destination('logs/combined.log') },
];

const logger = pino({
    level: 'trace',
    redact: ['password', 'token'],
}, pino.multistream(streams));

// ========== 3. 日志轮转 ==========
// Pino 级别较低, 用外部工具轮转

// 方案 A: 使用 pino-roll
// npm install pino-roll
const pinoroll = require('pino-roll');

const logger = pino({
    level: 'info',
}, pinoroll({
    file: 'logs/app.log',
    frequency: 'daily',     // 每天轮转
    size: '10M',            // 或按大小
    maxFiles: 14,
    dateFormat: 'YYYY-MM-DD',
}));

// 方案 B: 系统 logrotate
// /etc/logrotate.d/myapp:
// /var/log/myapp/*.log {
//   daily
//   rotate 14
//   compress
//   delaycompress
//   missingok
//   notifempty
//   copytruncate   # 不影响 Pino 的写入
// }

// ========== 4. 自定义 Transport ==========
// Pino v7+ 的 transport 是子进程

const transport = pino.transport({
    targets: [
        { target: 'pino-pretty', options: { colorize: true }, level: 'info' },
        { target: 'pino/file', options: { destination: './logs/app.log' }, level: 'debug' },
        { target: 'pino/file', options: { destination: './logs/errors.log' }, level: 'error' },
    ],
});

const logger = pino(transport);

// ========== 5. 外部 Transport ==========
// pino-datadog, pino-elasticsearch, pino-sentry

const transport = pino.transport({
    target: 'pino-datadog',
    options: {
        ddClientConf: {
            authMethods: {
                apiKey: process.env.DATADOG_API_KEY,
            },
        },
        ddServerConf: { site: 'datadoghq.com' },
    },
});
```


## Winston vs Pino


```
// ========== Winston vs Pino 对比 ==========

// ┌──────────────┬───────────────────┬────────────────────┐
// │              │      Winston      │       Pino         │
// ├──────────────┼───────────────────┼────────────────────┤
// │ 性能         │ 较慢 (中间件多)   │ 最快 (原生 JSON)   │
// │ 吞吐量       │ ~50k msg/s        │ ~200k msg/s        │
// │ 格式         │ 灵活 (自定义组合) │ 原生 JSON          │
// │ Transport    │ 内置丰富          │ 子进程架构          │
// │ 易用性       │ 上手简单          │ 需了解 transport   │
// │ 社区         │ 更成熟            │ 快速增长           │
// │ 文件轮转     │ 内置 (winston-daily-rotate-file) │ 外挂 (pino-roll/logrotate) │
// │ 子 Logger    │ 支持 (child)      │ 支持 (child)        │
// │ 请求日志     │ Morgan 集成       │ pino-http 原生     │
// │ 生产推荐     │ 通用场景          │ 高性能场景          │
// └──────────────┴───────────────────┴────────────────────┘

// ========== 性能对比 ==========
// Pino: 5x-10x faster than Winston
// 原因:
// 1. Pino 直接输出 JSON (无复杂格式化)
// 2. Transport 在子进程处理 (不阻塞主线程)
// 3. 最小化对象创建
// 4. 流式写入

// ========== 选型建议 ==========
// 小项目: Winston (简单直接)
// 高并发: Pino (性能敏感)
// 微服务: Pino (轻量 JSON)
// 已有 ELK: Pino (原生 JSON)
// 需要轮转: Winston (内置)
// 需要复杂格式: Winston

// ========== 生产 Pino 配置示例 ==========
const pino = require('pino');
const config = require('./config');

const logger = pino({
    level: config.log.level,
    redact: {
        paths: ['password', 'token', 'secret', 'req.headers.cookie', 'req.headers.authorization'],
        censor: '***',
    },
    serializers: {
        err: pino.stdSerializers.err,
        req: pino.stdSerializers.req,
        res: pino.stdSerializers.res,
    },
    base: {
        service: config.serviceName,
        environment: config.env,
        nodeVersion: process.version,
    },
    timestamp: pino.stdTimeFunctions.isoTime,  // ISO 8601
});

module.exports = logger;
```


> **Note:** 💡 Pino 要点: 最快 Node.js 日志 (原生 JSON); pino-http 替代 Morgan; pino-pretty 开发格式化; 子 Logger 自动携带上下文; redact 脱敏敏感字段; multistream 多目标输出; pino-roll 或 logrotate 轮转; 子进程 Transport 不阻塞主线程; Winston 适合通用, Pino 适合高性能。


## 练习


<!-- Converted from: 37_Express Pino 日志.html -->
