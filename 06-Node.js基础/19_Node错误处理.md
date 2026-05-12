# Node错误处理


## Node 错误处理


异步错误、uncaughtException/unhandledRejection、错误码、日志。


## Node 错误处理机制


```
// ========== 同步错误 ==========
try {
    JSON.parse('invalid');
} catch (err) {
    console.error('解析错误:', err.message);
}

// ========== 异步错误 (回调) ==========
fs.readFile('/notfound', (err, data) => {
    if (err) return console.error('读取失败:', err);
});

// ========== Promise 错误 ==========
Promise.reject(new Error('失败'))
    .catch(err => console.error(err));

// ========== async/await 错误 ==========
async function read() {
    try {
        await fs.promises.readFile('/notfound');
    } catch (err) {
        console.error(err.code); // 'ENOENT'
    }
}

// ========== 全局未捕获 ==========
process.on('uncaughtException', (err) => {
    console.error('未捕获异常:', err);
    process.exit(1);
});

process.on('unhandledRejection', (reason) => {
    console.error('未处理拒绝:', reason);
});

// ========== 错误码 ==========
// ENOENT — 文件不存在
// EACCES — 权限不足
// EEXIST — 文件已存在
// ENOTDIR — 不是目录
// ECONNREFUSED — 连接被拒
// ETIMEDOUT — 连接超时
```


## 演示：错误处理

点击按钮查看


## 自定义错误类

```javascript
// ========== 自定义错误层次 ==========
class AppError extends Error {
    constructor(message, statusCode = 500, code = 'INTERNAL_ERROR') {
        super(message);
        this.name = this.constructor.name;
        this.statusCode = statusCode;
        this.code = code;
        this.isOperational = true; // 标识为可预期的错误
        Error.captureStackTrace(this, this.constructor);
    }
}

class NotFoundError extends AppError {
    constructor(resource = 'Resource') {
        super(`${resource} not found`, 404, 'NOT_FOUND');
    }
}

class ValidationError extends AppError {
    constructor(fields) {
        super('Validation failed', 400, 'VALIDATION_ERROR');
        this.fields = fields; // { email: '格式不正确', name: '不能为空' }
    }
}

class UnauthorizedError extends AppError {
    constructor(message = 'Unauthorized') {
        super(message, 401, 'UNAUTHORIZED');
    }
}

class ConflictError extends AppError {
    constructor(message = 'Resource already exists') {
        super(message, 409, 'CONFLICT');
    }
}

// ========== 使用 ==========
async function getUser(id) {
    const user = await db.users.findById(id);
    if (!user) throw new NotFoundError('User');
    return user;
}

async function createUser(data) {
    if (!data.email) throw new ValidationError({ email: '邮箱不能为空' });
    const existing = await db.users.findByEmail(data.email);
    if (existing) throw new ConflictError('邮箱已注册');
    return db.users.create(data);
}
```

## 全局错误处理中间件

```javascript
// ========== Express 错误处理中间件 ==========
// 4 个参数的中间件自动成为错误处理器
function errorHandler(err, req, res, next) {
    // 记录错误
    console.error(`[${new Date().toISOString()}] ${err.stack || err.message}`);

    // 可预期的错误
    if (err.isOperational) {
        res.status(err.statusCode).json({
            error: {
                code: err.code,
                message: err.message,
                ...(err.fields && { fields: err.fields }),
            }
        });
        return;
    }

    // 未知错误 — 不暴露内部信息
    res.status(500).json({
        error: {
            code: 'INTERNAL_ERROR',
            message: process.env.NODE_ENV === 'production'
                ? '服务器内部错误'
                : err.message,
        }
    });
}

// 404 处理
function notFoundHandler(req, res) {
    res.status(404).json({
        error: { code: 'NOT_FOUND', message: `路由 ${req.method} ${req.url} 不存在` }
    });
}

// 应用
const express = require('express');
const app = express();
app.use(notFoundHandler);
app.use(errorHandler);
```

## 日志与监控集成

```javascript
// ========== 结构化日志 ==========
const winston = require('winston');

const logger = winston.createLogger({
    level: process.env.LOG_LEVEL || 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/combined.log' }),
        new winston.transports.Console({
            format: winston.format.combine(
                winston.format.colorize(),
                winston.format.simple()
            )
        }),
    ],
});

// ========== 集成到错误处理 ==========
function logError(err, context = {}) {
    const logData = {
        message: err.message,
        stack: err.stack,
        code: err.code,
        statusCode: err.statusCode,
        isOperational: err.isOperational,
        ...context,
    };

    if (err.isOperational) {
        logger.warn(logData);
    } else {
        logger.error(logData);
    }
}

// ========== 进程级错误捕获 ==========
process.on('uncaughtException', (err) => {
    logger.error('未捕获异常:', { error: err.message, stack: err.stack });
    // 给日志系统时间写入
    setTimeout(() => process.exit(1), 1000);
});

process.on('unhandledRejection', (reason, promise) => {
    logger.error('未处理的 Promise 拒绝:', {
        reason: reason instanceof Error ? reason.message : reason,
        stack: reason instanceof Error ? reason.stack : undefined,
    });
});
```

<!-- Converted from: 19_Node错误处理.html -->
