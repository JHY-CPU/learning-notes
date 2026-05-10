# Koa.js 中间件与常用中间件


## 🔗 Koa.js 中间件与常用中间件


Koa 中间件模式、常用中间件 (koa-body/koa-cors/koa-static/koa-session)、自定义中间件、组合中间件、中间件执行顺序控制、JWT 认证中间件、静态文件服务。


## 常用中间件


```
// ========== Koa 常用中间件 ==========
// Koa 生态以中间件形式提供功能

const Koa = require('koa');
const app = new Koa();

// ========== 1. koa-body (请求体解析) ==========
const { koaBody } = require('koa-body');

app.use(koaBody({
    multipart: true,
    json: true,
    jsonLimit: '1mb',
    formLimit: '1mb',
    textLimit: '1mb',
    encoding: 'utf-8',
    formidable: {
        maxFileSize: 10 * 1024 * 1024,  // 10MB
        keepExtensions: true,
    },
}));

app.use(async (ctx) => {
    // 普通字段: ctx.request.body.fieldName
    // 文件: ctx.request.files.fileName
    console.log(ctx.request.body);
});

// ========== 2. @koa/cors (跨域) ==========
const cors = require('@koa/cors');

app.use(cors({
    origin: (ctx) => {
        const allowed = ['http://localhost:5173', 'https://myapp.com'];
        const origin = ctx.get('Origin');
        return allowed.includes(origin) ? origin : false;
    },
    credentials: true,
    allowMethods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    allowHeaders: ['Content-Type', 'Authorization'],
    maxAge: 86400,
}));

// ========== 3. koa-static (静态文件) ==========
const serve = require('koa-static');
const path = require('path');

app.use(serve(path.join(__dirname, 'public'), {
    maxage: 365 * 24 * 60 * 60 * 1000,  // 1年缓存
    hidden: false,       // 隐藏文件不提供
    gzip: true,
    defer: false,        // true = 先处理后端路由
}));

// ========== 4. koa-session ==========
const session = require('koa-session');

app.keys = ['secret-key-1', 'secret-key-2'];  // 用于签名

app.use(session({
    key: 'koa:sess',           // cookie key
    maxAge: 86400000,          // 1天
    autoCommit: true,
    overwrite: true,
    httpOnly: true,
    signed: true,
    rolling: false,            // 每次响应刷新
    renew: false,              // 快过期时刷新
}, app));

app.use(async (ctx) => {
    if (!ctx.session.views) {
        ctx.session.views = 0;
    }
    ctx.session.views++;
    ctx.body = { views: ctx.session.views };
});

// ========== 5. koa-helmet (安全头) ==========
// npm install koa-helmet
// const helmet = require('koa-helmet');
// app.use(helmet());

// ========== 6. koa-logger (日志) ==========
// npm install koa-logger
// const logger = require('koa-logger');
// app.use(logger());

// ========== 中间件使用顺序 ==========
// 1. 错误处理 (最外层)
// 2. CORS
// 3. 日志
// 4. Session
// 5. 请求体解析
// 6. 认证
// 7. 路由
// 8. 静态文件 (最后)

app.use(errorHandler);
app.use(cors());
app.use(logger());
app.use(session(app));
app.use(koaBody());
app.use(authMiddleware);
app.use(router.routes());
app.use(serve('public'));
```


## 自定义中间件


```
// ========== 自定义 Koa 中间件 ==========

// ========== 1. 请求计时 ==========
function responseTime() {
    return async (ctx, next) => {
        const start = Date.now();
        await next();
        const ms = Date.now() - start;
        ctx.set('X-Response-Time', `${ms}ms`);
    };
}

app.use(responseTime());

// ========== 2. 请求 ID ==========
const { v4: uuidv4 } = require('uuid');

function requestId() {
    return async (ctx, next) => {
        ctx.state.requestId = ctx.get('X-Request-Id') || uuidv4();
        ctx.set('X-Request-Id', ctx.state.requestId);
        await next();
    };
}

app.use(requestId());

// ========== 3. JWT 认证中间件 ==========
const jwt = require('jsonwebtoken');

function authenticate({ optional = false } = {}) {
    return async (ctx, next) => {
        const authHeader = ctx.get('Authorization');
        if (!authHeader || !authHeader.startsWith('Bearer ')) {
            if (optional) {
                ctx.state.user = null;
                return next();
            }
            ctx.throw(401, 'Authentication required');
        }

        try {
            const token = authHeader.split(' ')[1];
            const decoded = jwt.verify(token, process.env.JWT_SECRET);
            ctx.state.user = decoded;
            await next();
        } catch (err) {
            ctx.throw(401, 'Invalid or expired token');
        }
    };
}

// 使用:
// router.get('/profile', authenticate(), async (ctx) => {
//     ctx.body = ctx.state.user;
// });
//
// router.get('/public', authenticate({ optional: true }), handler);

// ========== 4. 角色授权 ==========
function authorize(...roles) {
    return async (ctx, next) => {
        const user = ctx.state.user;
        if (!user) ctx.throw(401);
        if (!roles.includes(user.role)) ctx.throw(403, 'Insufficient permissions');
        await next();
    };
}

// router.delete('/users/:id', authenticate(), authorize('admin'), handler);

// ========== 5. 限流 ==========
const rateLimitStore = new Map();

function rateLimit({ windowMs = 60000, max = 100 } = {}) {
    return async (ctx, next) => {
        const key = ctx.ip;
        const now = Date.now();

        if (!rateLimitStore.has(key)) {
            rateLimitStore.set(key, []);
        }

        const timestamps = rateLimitStore.get(key).filter(t => now - t < windowMs);
        timestamps.push(now);

        if (timestamps.length > max) {
            ctx.set('Retry-After', Math.ceil(windowMs / 1000).toString());
            ctx.throw(429, 'Too many requests');
        }

        rateLimitStore.set(key, timestamps);
        await next();
    };
}

// ========== 6. 条件中间件 ==========
function unless(paths, middleware) {
    return async (ctx, next) => {
        const skip = paths.some(p => ctx.path.match(p));
        if (skip) return next();
        await middleware(ctx, next);
    };
}

// app.use(unless(['/health', '/metrics'], rateLimit()));
```


## 中间件组合


```
// ========== 中间件组合模式 ==========

// ========== 1. compose 函数 ==========
// Koa 内部使用 koa-compose 组合中间件
const compose = require('koa-compose');

// 组合多个中间件为一个
const stack = [];

stack.push(async (ctx, next) => {
    console.log('Middleware 1 before');
    await next();
    console.log('Middleware 1 after');
});

stack.push(async (ctx, next) => {
    console.log('Middleware 2 before');
    await next();
    console.log('Middleware 2 after');
});

const composed = compose(stack);

app.use(composed);

// ========== 2. 路由级中间件 ==========
const Router = require('@koa/router');
const router = new Router();

// 路由级中间件 (只有匹配此路由时才执行)
router.get('/admin', authorize('admin'), async (ctx) => {
    ctx.body = { secret: 'admin data' };
});

// 路由前缀级中间件
const adminRouter = new Router({ prefix: '/admin' });
adminRouter.use(authorize('admin'));
adminRouter.get('/users', listUsers);
adminRouter.get('/settings', getSettings);

// ========== 3. 中间件工厂 ==========
// 生成可配置中间件
function createRateLimiter({ windowMs, max, message }) {
    return async (ctx, next) => {
        // ... 限流逻辑
        await next();
    };
}

const strictLimiter = createRateLimiter({ windowMs: 60000, max: 10, message: 'Too fast!' });
const looseLimiter = createRateLimiter({ windowMs: 60000, max: 100 });

router.post('/auth/login', strictLimiter, loginHandler);
router.get('/api/users', looseLimiter, listUsers);

// ========== 4. Koa 中间件 = Express 中间件 ==========
// 很多 Express 中间件可以通过 koa-connect 适配
// npm install koa-connect

const { use } = require('koa-connect');
const morgan = require('morgan');

// app.use(use(morgan('combined')));

// ========== 5. 中间件顺序控制 ==========
// 可以通过条件提前返回
app.use(async (ctx, next) => {
    if (ctx.path === '/health') {
        ctx.body = { status: 'ok' };
        return;  // 不调用 next(), 短路
    }
    await next();
});
```


## Koa 静态文件与模板


```
// ========== Koa 静态文件与模板 ==========

// ========== 静态文件服务 ==========
const serve = require('koa-static');
const mount = require('koa-mount');  // 路由前缀

// 整个目录
app.use(serve('public'));

// 路由前缀
app.use(mount('/static', serve('public/static')));

// 多个目录
app.use(mount('/uploads', serve('uploads')));
app.use(mount('/assets', serve('assets')));

// ========== 模板引擎 ==========
// Koa 无内置模板, 选用你喜欢的

// koa-ejs:
// npm install koa-ejs
const render = require('koa-ejs');
const path = require('path');

render(app, {
    root: path.join(__dirname, 'views'),
    layout: 'layout',
    viewExt: 'ejs',
    cache: process.env.NODE_ENV === 'production',
    debug: process.env.NODE_ENV === 'development',
});

app.use(async (ctx) => {
    await ctx.render('index', {
        title: 'Koa App',
        user: ctx.state.user,
    });
});

// ========== 综合示例 ==========
const Koa = require('koa');
const Router = require('@koa/router');
const { koaBody } = require('koa-body');
const cors = require('@koa/cors');
const serve = require('koa-static');

const app = new Koa();
const router = new Router({ prefix: '/api/v1' });

// 中间件
app.use(cors());
app.use(koaBody());
app.use(serve('public'));

// 路由
router.get('/users', async (ctx) => {
    ctx.body = await User.find();
});

router.post('/users', async (ctx) => {
    const user = await User.create(ctx.request.body);
    ctx.status = 201;
    ctx.body = user;
});

app.use(router.routes());
app.use(router.allowedMethods());

app.listen(3000);
```


> **Note:** 💡 Koa 中间件要点: 洋葱模型 await next() 控制流程; koa-body 解析 JSON/表单/文件; @koa/cors 跨域; koa-static 静态文件; koa-session 会话; 自定义 JWT/限流/日志中间件; 条件中间件 unless; koa-compose 组合; 路由级中间件; koa-connect 兼容 Express 中间件。


## 练习


<!-- Converted from: 48_Koa.js 中间件与常用中间件.html -->
