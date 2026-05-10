# Express 中间件机制


## 🔗 Express 中间件机制


中间件概念与执行流程、应用级 vs 路由级中间件、内置中间件 (express.json/static/urlencoded)、第三方中间件 (morgan/cors/helmet)、错误处理中间件、异步中间件错误捕获、中间件顺序最佳实践。


## 中间件概念


```
// ========== 中间件 ==========
// 函数, 访问 req/res, 执行逻辑, 调用 next()

// ========== 中间件签名 ==========
function middleware(req, res, next) {
    // 1. 执行逻辑 (日志/认证/解析)
    // 2. 修改 req/res (添加属性/数据)
    // 3. 结束响应 (res.send/json) 或
    // 4. 调用 next() 传递到下一个
}

// ========== 中间件执行流程 ==========
// ┌────────────────────────────────────────┐
// │  请求进入                                │
// │    ↓                                     │
// │  中间件 1 (日志)                         │
// │    ↓ next()                              │
// │  中间件 2 (解析 JSON)                    │
// │    ↓ next()                              │
// │  中间件 3 (CORS)                         │
// │    ↓ next()                              │
// │  路由处理函数 (业务逻辑)                 │
// │    ↓                                     │
// │  响应返回客户端                          │
// └────────────────────────────────────────┘

// ========== 创建自定义中间件 ==========
// 请求计时器:
app.use((req, res, next) => {
    const start = Date.now();
    res.on('finish', () => {
        const duration = Date.now() - start;
        console.log(`${req.method} ${req.path} ${duration}ms`);
    });
    next();
});

// 请求日志:
app.use((req, res, next) => {
    console.log(`[${new Date().toISOString()}] ${req.method} ${req.url}`);
    next();
});
```


## 中间件类型


```
// ========== 1. 应用级中间件 ==========
// 绑定到 app 实例

// 全局 (所有请求):
app.use(middleware);

// 路径前缀 (匹配 /api 开头的请求):
app.use('/api', middleware);

// 条件路径 (多个路径):
app.use(['/admin', '/dashboard'], authMiddleware);

// ========== 2. 路由级中间件 ==========
// 绑定到 Router 实例

const router = express.Router();

// 所有 /users 请求:
router.use(authMiddleware);

// 特定路由:
router.get('/:id', validateId, getUser);

// ========== 3. 内置中间件 ==========
// Express 自带:
express.json()                    // 解析 JSON body
express.urlencoded({ extended: true }) // 解析表单
express.static('public')          // 静态文件
express.Router()                  // 路由模块

// ========== 4. 第三方中间件 ==========
const morgan = require('morgan');     // 请求日志
const cors = require('cors');         // 跨域
const helmet = require('helmet');     // 安全头
const compression = require('compression'); // 压缩
const cookieParser = require('cookie-parser');
const rateLimit = require('express-rate-limit');

// 使用:
app.use(morgan('combined'));          // 日志格式
app.use(cors({ origin: 'https://example.com' }));
app.use(helmet());
app.use(compression());
app.use(cookieParser());
```


## 错误处理中间件


```
// ========== 错误处理中间件 ==========
// 4 个参数: (err, req, res, next)
// 必须放在所有路由和其他中间件之后!

app.use((err, req, res, next) => {
    console.error('Error:', err);

    // 区分已知和未知错误
    const status = err.status || 500;
    const message = err.message || 'Internal Server Error';

    // 生产环境不暴露错误详情
    const error = process.env.NODE_ENV === 'production'
        ? { message: 'Internal Server Error' }
        : { message, stack: err.stack };

    res.status(status).json({
        code: status,
        message: error.message,
        ...(error.stack && { stack: error.stack })
    });
});

// ========== 自定义错误类 ==========
class AppError extends Error {
    constructor(message, status = 400) {
        super(message);
        this.status = status;
        this.isOperational = true;  // 可预知的错误
    }
}

class NotFoundError extends AppError {
    constructor(message = 'Resource not found') {
        super(message, 404);
    }
}

class ValidationError extends AppError {
    constructor(message = 'Validation failed') {
        super(message, 422);
    }
}

// 使用:
app.get('/users/:id', async (req, res, next) => {
    try {
        const user = await User.findById(req.params.id);
        if (!user) throw new NotFoundError('User not found');
        res.json(user);
    } catch (err) {
        next(err);  // 传给错误处理中间件
    }
});

// ========== 404 处理 ==========
// 必须放在所有路由之后!
app.use((req, res, next) => {
    res.status(404).json({
        code: 404,
        message: `Route ${req.method} ${req.url} not found`
    });
});
```


## 异步中间件与最佳实践


```
// ========== 异步中间件 ==========
// Express 5 原生支持 async/await
// Express 4 需要手动捕获错误

// 方案 1: try/catch + next:
app.get('/users/:id', async (req, res, next) => {
    try {
        const user = await User.findById(req.params.id);
        res.json(user);
    } catch (err) {
        next(err);
    }
});

// 方案 2: 封装 asyncHandler:
function asyncHandler(fn) {
    return (req, res, next) => {
        Promise.resolve(fn(req, res, next)).catch(next);
    };
}

// 使用:
app.get('/users/:id', asyncHandler(async (req, res) => {
    const user = await User.findById(req.params.id);
    if (!user) throw new NotFoundError();
    res.json(user);
}));

// ========== 中间件顺序 (重要!) ==========
// 正确的顺序决定了应用行为:

const express = require('express');
const app = express();

// 1. 安全与解析 (最先)
app.use(helmet());
app.use(cors());
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

// 2. 日志
app.use(morgan('dev'));

// 3. 静态文件
app.use(express.static('public'));

// 4. 请求级中间件 (认证/限流)
app.use('/api', rateLimit({ windowMs: 15 * 60 * 1000, max: 100 }));
app.use('/api', authMiddleware);

// 5. 路由
app.use('/api/users', usersRouter);
app.use('/api/posts', postsRouter);

// 6. 404 处理
app.use((req, res) => {
    res.status(404).json({ message: 'Not Found' });
});

// 7. 全局错误处理 (最后!)
app.use((err, req, res, next) => {
    console.error(err);
    res.status(err.status || 500).json({
        message: err.message || 'Internal Error'
    });
});
```


> **Note:** 💡 中间件是 Express 的核心, 本质是函数链。顺序决定行为: 安全→解析→日志→静态→认证→路由→404→错误。错误处理中间件必须有 4 个参数。异步中间件必须捕获错误传给 next()。第三方中间件 morgan/cors/helmet 是生产必备。


## 练习


<!-- Converted from: 3_Express 中间件机制.html -->
