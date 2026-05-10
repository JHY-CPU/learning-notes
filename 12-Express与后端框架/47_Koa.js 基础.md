# Koa.js 基础


## 🚀 Koa.js 基础


Koa.js 简介、洋葱模型 (Onion Model)、async 中间件、Context (ctx) 对象、请求/响应处理、错误处理、与 Express 核心区别。


## Koa.js 简介


```
// ========== Koa.js ==========
// 由 Express 原班人马打造的更轻量框架
// 利用 async/await 彻底解决回调问题

// 核心特点:
// 1. 洋葱模型中间件 (Koa 的灵魂)
// 2. 只有中间件核心, 不捆绑路由/模板
// 3. 原生 async/await
// 4. 更优雅的 Context 设计
// 5. 更轻量 (Express 依赖多)

// 安装:
// npm install koa
// npm install @koa/router    (路由)
// npm install koa-body       (请求体解析)
// npm install koa-cors       (跨域)

// ========== Hello Koa ==========
const Koa = require('koa');
const app = new Koa();

// 中间件
app.use(async (ctx) => {
    ctx.body = 'Hello Koa!';
});

app.listen(3000);

// ========== async 中间件 ==========
// Koa 中间件必须返回 Promise
// async 函数自动返回 Promise

app.use(async (ctx, next) => {
    const start = Date.now();
    await next();  // 等待后续中间件完成
    const ms = Date.now() - start;
    ctx.set('X-Response-Time', `${ms}ms`);
});

// ========== 洋葱模型执行顺序 ==========
// 请求 → middleware1(前) → middleware2(前) → handler → middleware2(后) → middleware1(后) → 响应
//
// app.use(async (ctx, next) => {
//   console.log('1-进入');
//   await next();
//   console.log('1-退出');
// });
//
// app.use(async (ctx, next) => {
//   console.log('2-进入');
//   await next();
//   console.log('2-退出');
// });
//
// app.use(async (ctx) => {
//   console.log('3-处理');
//   ctx.body = 'Hello';
// });
//
// 输出: 1-进入 → 2-进入 → 3-处理 → 2-退出 → 1-退出
```


## Context (ctx) 对象


```
// ========== Koa Context ==========
// ctx 封装了 request 和 response
// 是 Koa 最核心的设计

app.use(async (ctx) => {
    // ========== 请求相关 ==========
    ctx.method;           // GET/POST
    ctx.url;              // /users?id=1
    ctx.originalUrl;      // 原始 URL
    ctx.path;             // /users
    ctx.querystring;      // id=1
    ctx.query;            // { id: '1' }
    ctx.headers;          // 请求头对象
    ctx.get('Content-Type');  // 获取请求头

    ctx.host;             // localhost:3000
    ctx.hostname;         // localhost
    ctx.protocol;         // http/https
    ctx.secure;           // 是否 HTTPS
    ctx.ip;               // 客户端 IP
    ctx.ips;              // 代理 IP 列表

    // ========== 响应相关 ==========
    ctx.body = 'Hello';         // 响应体 (字符串/Buffer/Stream/JSON)
    ctx.body = { id: 1 };       // 自动 JSON
    ctx.body = fs.createReadStream('file.txt');  // 流

    ctx.status = 201;           // 状态码
    ctx.status = 404;

    ctx.set('X-Custom', 'value');  // 设置响应头
    ctx.set({
        'X-Request-Id': '123',
        'Cache-Control': 'no-cache',
    });

    ctx.type = 'text/html';     // Content-Type
    ctx.type = 'application/json';

    ctx.redirect('/login');     // 重定向
    ctx.redirect('https://example.com');

    ctx.attachment('file.pdf');  // 下载附件
    ctx.attachment('photo.jpg', { type: 'image/jpeg' });
});

// ========== ctx.state (推荐的状态传递) ==========
// 中间件间传递数据
app.use(async (ctx, next) => {
    ctx.state.user = { id: 1, name: 'Alice' };
    ctx.state.requestId = 'abc-123';
    await next();
});

app.use(async (ctx) => {
    // 后面的中间件可以访问
    ctx.body = ctx.state.user;
});

// ========== ctx.throw (错误抛出) ==========
app.use(async (ctx) => {
    if (!ctx.query.token) {
        ctx.throw(401, 'Authentication required');
        // 相当于:
        // const err = new Error('Authentication required');
        // err.status = 401;
        // err.expose = true;
        // throw err;
    }
});

// ========== 请求体获取 ==========
// Koa 不自带 body parser, 需要 koa-body
// npm install koa-body

const { koaBody } = require('koa-body');

app.use(koaBody({
    multipart: true,          // 支持文件上传
    jsonLimit: '1mb',
    formLimit: '1mb',
    textLimit: '1mb',
}));

app.use(async (ctx) => {
    ctx.body = ctx.request.body;  // 解析后的请求体
    // ctx.request.files  (文件上传)
});
```


## 路由处理


```
// ========== Koa 路由 ==========
// Koa 无内置路由, 使用 @koa/router

// 安装:
// npm install @koa/router

const Router = require('@koa/router');
const router = new Router();

// ========== 基础路由 ==========
router.get('/', (ctx) => {
    ctx.body = { message: 'Home page' };
});

router.get('/users', (ctx) => {
    ctx.body = { users: ['Alice', 'Bob'] };
});

router.get('/users/:id', (ctx) => {
    ctx.body = { userId: ctx.params.id };
});

router.post('/users', (ctx) => {
    ctx.body = { created: ctx.request.body };
});

router.put('/users/:id', (ctx) => {
    ctx.body = { updated: ctx.params.id };
});

router.del('/users/:id', (ctx) => {  // delete 是关键字, 用 del
    ctx.status = 204;
});

// ========== 路由前缀 ==========
const userRouter = new Router({ prefix: '/users' });

userRouter.get('/', listUsers);
userRouter.post('/', createUser);
userRouter.get('/:id', getUser);
userRouter.put('/:id', updateUser);
userRouter.del('/:id', deleteUser);

// ========== 注册路由 ==========
app.use(router.routes());           // 路由中间件
app.use(router.allowedMethods());    // 405 Method Not Allowed

// allowedMethods 自动处理 OPTIONS 和 405

// ========== 路由嵌套 ==========
// 拆分路由文件
// routes/users.js:
const Router = require('@koa/router');
const router = new Router();

router.get('/', async (ctx) => {
    ctx.body = await User.find();
});

module.exports = router;

// app.js:
const userRouter = require('./routes/users');
app.use(userRouter.routes());

// ========== 路由参数 ==========
router.get('/posts/:category/:id', (ctx) => {
    ctx.body = {
        category: ctx.params.category,
        id: ctx.params.id,
        query: ctx.query,
    };
});

// 命名路由 (用于 URL 生成)
router.get('user', '/users/:id', (ctx) => {
    ctx.body = ctx.url;  // 当前 URL
});
// router.url('user', { id: 1 })  →  '/users/1'
```


## 错误处理


```
// ========== Koa 错误处理 ==========
// Koa 错误处理比 Express 更简洁

// ========== 全局错误处理 ==========
// 错误中间件要放在最前面
app.use(async (ctx, next) => {
    try {
        await next();
    } catch (err) {
        // 统一错误处理
        ctx.status = err.status || err.statusCode || 500;
        ctx.body = {
            success: false,
            code: err.code || 'INTERNAL_ERROR',
            message: err.expose ? err.message : 'Internal Server Error',
            ...(process.env.NODE_ENV === 'development' && { stack: err.stack }),
        };

        // 记录错误
        ctx.app.emit('error', err, ctx);
    }
});

// 错误事件监听
app.on('error', (err, ctx) => {
    logger.error('Server error', {
        error: err.message,
        stack: err.stack,
        url: ctx?.url,
        method: ctx?.method,
    });
});

// ========== 404 处理 ==========
app.use(async (ctx) => {
    if (ctx.status === 404 && !ctx.body) {
        ctx.status = 404;
        ctx.body = { success: false, message: 'Not Found' };
    }
});

// 或放在最后:
app.use(async (ctx, next) => {
    await next();
    if (ctx.status === 404 && !ctx.body) {
        ctx.throw(404, 'Route not found');
    }
});

// ========== 自定义错误类 ==========
class AppError extends Error {
    constructor(message, status = 500, code = 'INTERNAL_ERROR') {
        super(message);
        this.status = status;
        this.code = code;
        this.expose = true;  // 客户端可见
    }
}

class NotFoundError extends AppError {
    constructor(message = 'Resource not found') {
        super(message, 404, 'NOT_FOUND');
    }
}

class BadRequestError extends AppError {
    constructor(message = 'Bad request') {
        super(message, 400, 'BAD_REQUEST');
    }
}

class UnauthorizedError extends AppError {
    constructor(message = 'Unauthorized') {
        super(message, 401, 'UNAUTHORIZED');
    }
}

// 使用:
// router.get('/users/:id', async (ctx) => {
//     const user = await User.findById(ctx.params.id);
//     if (!user) throw new NotFoundError('User not found');
//     ctx.body = user;
// });
```


> **Note:** 💡 Koa 要点: 洋葱模型中间件 (await next()); async/await 原生支持; ctx 封装 req/res; ctx.state 传递数据; ctx.throw 抛错误; @koa/router 路由; 全局 try/catch 错误处理; 更轻量 (无内置中间件); koa-body 解析请求体; allowedMethods 自动 405。


## 练习


<!-- Converted from: 47_Koa.js 基础.html -->
