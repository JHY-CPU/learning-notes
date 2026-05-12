# HTTP路由


## HTTP 路由


URL 解析、路由分发、RESTful 风格、请求体解析。


## 路由实现


```
// ========== 简易路由 ==========
const http = require('http');

const server = http.createServer((req, res) => {
    const { method, url } = req;
    const urlObj = new URL(url, `http://${req.headers.host}`);
    const pathname = urlObj.pathname;

    // 路由分发
    if (pathname === '/api/users' && method === 'GET') {
        // GET /api/users
    } else if (pathname === '/api/users' && method === 'POST') {
        // POST /api/users
    } else if (pathname.match(/^\/api\/users\/(\d+)$/) && method === 'GET') {
        // GET /api/users/:id
    } else {
        res.statusCode = 404;
        res.end('Not Found');
    }
});

// ========== Router 类 ==========
class Router {
    constructor() {
        this.routes = [];
    }
    get(path, handler) { this.routes.push({ method: 'GET', path, handler }); }
    post(path, handler) { this.routes.push({ method: 'POST', path, handler }); }
    put(path, handler) { this.routes.push({ method: 'PUT', path, handler }); }
    delete(path, handler) { this.routes.push({ method: 'DELETE', path, handler }); }

    match(method, url) {
        for (const route of this.routes) {
            if (route.method !== method) continue;
            const params = this._matchPath(route.path, url);
            if (params !== null) return { handler: route.handler, params };
        }
        return null;
    }
}

// ========== RESTful 路由设计 ==========
// GET    /api/users        — 列表
// POST   /api/users        — 创建
// GET    /api/users/:id    — 详情
// PUT    /api/users/:id    — 更新
// DELETE /api/users/:id    — 删除
// GET    /api/users/:id/posts — 子资源
```


## 演示：路由系统

点击按钮查看


## 完整 Router 类实现

```javascript
// ========== 功能完整的 Router ==========
class Router {
    constructor() {
        this.routes = [];
    }

    // 注册路由
    get(path, ...handlers) { this._add('GET', path, handlers); }
    post(path, ...handlers) { this._add('POST', path, handlers); }
    put(path, ...handlers) { this._add('PUT', path, handlers); }
    delete(path, ...handlers) { this._add('DELETE', path, handlers); }
    patch(path, ...handlers) { this._add('PATCH', path, handlers); }

    // 支持所有方法
    all(path, ...handlers) {
        ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'].forEach(method => {
            this._add(method, path, handlers);
        });
    }

    _add(method, path, handlers) {
        // 支持中间件数组
        handlers.forEach(handler => {
            this.routes.push({ method, path: this._pathToRegex(path), handler, originalPath: path });
        });
    }

    // 路径转正则: /users/:id → /^\/users\/([^/]+)$/
    _pathToRegex(path) {
        const pattern = path
            .replace(/:(\w+)/g, '(?<$1>[^/]+)')
            .replace(/\//g, '\\/');
        return new RegExp(`^${pattern}$`);
    }

    // 匹配路由
    match(method, url) {
        for (const route of this.routes) {
            if (route.method !== method) continue;
            const match = url.match(route.path);
            if (match) {
                return {
                    handler: route.handler,
                    params: match.groups || {},
                };
            }
        }
        return null;
    }

    // 执行匹配的路由处理器链
    async handle(req, res) {
        const parsedUrl = new URL(req.url, `http://${req.headers.host}`);
        const pathname = parsedUrl.pathname;
        const matched = this.match(req.method, pathname);

        if (!matched) {
            res.statusCode = 404;
            res.end(JSON.stringify({ error: 'Not Found' }));
            return;
        }

        // 注入 params 到 req
        req.params = matched.params;
        req.query = Object.fromEntries(parsedUrl.searchParams);

        // 串行执行处理器链 (中间件模式)
        let index = 0;
        const next = async () => {
            if (index < matched.handler.length) {
                await matched.handler[index++](req, res, next);
            }
        };
        await next();
    }
}

// ========== 使用示例 ==========
const http = require('http');
const router = new Router();

// 日志中间件
function logger(req, res, next) {
    console.log(`${req.method} ${req.url}`);
    next();
}

router.get('/api/users', logger, (req, res) => {
    res.end(JSON.stringify([{ id: 1, name: 'Alice' }]));
});

router.get('/api/users/:id', (req, res) => {
    res.end(JSON.stringify({ id: req.params.id, name: 'Alice' }));
});

router.post('/api/users', async (req, res) => {
    const body = await parseBody(req);
    res.statusCode = 201;
    res.end(JSON.stringify({ created: body }));
});

const server = http.createServer((req, res) => {
    res.setHeader('Content-Type', 'application/json');
    router.handle(req, res);
});
server.listen(3000);
```

## 查询参数与 URL 解析

```javascript
// ========== URL 和查询参数解析 ==========
const { URL, URLSearchParams } = require('url');

// 解析带查询参数的 URL
const myUrl = new URL('https://example.com/api/users?page=1&limit=20&sort=name');
console.log(myUrl.pathname);   // '/api/users'
console.log(myUrl.search);     // '?page=1&limit=20&sort=name'

// 获取参数
const params = myUrl.searchParams;
console.log(params.get('page'));    // '1'
console.log(params.get('limit'));   // '20'
console.log(params.has('sort'));    // true

// 遍历参数
for (const [key, value] of params) {
    console.log(`${key}: ${value}`);
}

// 构建 URL
const api = new URL('https://api.example.com/search');
api.searchParams.set('q', 'node.js');
api.searchParams.set('lang', 'zh');
console.log(api.href); // 'https://api.example.com/search?q=node.js&lang=zh'
```

## RESTful API 设计规范

```
// ========== RESTful 命名规范 ==========
// 资源用名词复数，方法表示操作
//
// GET    /api/users          获取用户列表
// POST   /api/users          创建新用户
// GET    /api/users/:id      获取指定用户
// PUT    /api/users/:id      完整更新用户
// PATCH  /api/users/:id      部分更新用户
// DELETE /api/users/:id      删除用户
//
// GET    /api/users/:id/posts  获取用户的帖子（嵌套资源）
//
// 查询参数:
// GET /api/users?page=1&limit=20&sort=-createdAt
// GET /api/users?filter[role]=admin&search=alice
//
// 状态码:
// 200 OK          — 成功
// 201 Created     — 创建成功
// 204 No Content  — 删除成功
// 400 Bad Request — 请求参数错误
// 401 Unauthorized — 未认证
// 403 Forbidden   — 无权限
// 404 Not Found   — 资源不存在
// 500 Server Error — 服务器错误
```

<!-- Converted from: 13_HTTP路由.html -->
