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


<!-- Converted from: 13_HTTP路由.html -->
