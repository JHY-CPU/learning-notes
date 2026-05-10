# Express CORS 跨域


## 🌐 Express CORS 跨域


同源策略、CORS 机制 (简单请求/预检请求)、cors 中间件配置 (origin/methods/credentials)、自定义 CORS 中间件、多环境 CORS 策略、Cookie 跨域 (credentials + withCredentials)。


## CORS 概念


```
// ========== CORS ==========
// Cross-Origin Resource Sharing (跨域资源共享)
// 浏览器安全策略, 阻止不同源的请求

// ========== 同源策略 ==========
// 协议 + 域名 + 端口 完全相同才算同源
// http://localhost:3000/api 和 http://localhost:5173 不同源!
//   不同端口 (3000 vs 5173) → 跨域

// ========== 简单请求 ==========
// 满足所有条件:
// - GET/HEAD/POST
// - 只有简单头 (Accept/Accept-Language/Content-Language/Content-Type)
// - Content-Type 为 application/x-www-form-urlencoded
//   multipart/form-data 或 text/plain

// 浏览器直接发送请求, 检查响应头 Access-Control-Allow-Origin

// ========== 预检请求 (Preflight) ==========
// 不满足简单条件时, 先发 OPTIONS 请求
// 浏览器先 OPTIONS 询问服务器是否允许
// 服务器回复允许后, 再发实际请求

// 触发条件:
// - 自定义头 (Authorization, X-Requested-With)
// - PUT/DELETE/PATCH 等方法
// - Content-Type: application/json
// - 带 Cookie/凭据

// OPTIONS 响应:
// HTTP/1.1 204 No Content
// Access-Control-Allow-Origin: http://localhost:5173
// Access-Control-Allow-Methods: GET, POST, PUT, DELETE
// Access-Control-Allow-Headers: Content-Type, Authorization
// Access-Control-Max-Age: 86400
```


## cors 中间件


```
// ========== cors 中间件 ==========
// 最流行的 Express CORS 解决方案

npm install cors

// ========== 全部允许 (开发环境) ==========
const cors = require('cors');
app.use(cors());
// 允许所有来源 (生产不安全)

// ========== 指定源 ==========
app.use(cors({
    origin: 'http://localhost:5173'     // 只允许此源
}));

// ========== 多源白名单 ==========
const allowedOrigins = [
    'http://localhost:5173',
    'http://localhost:3000',
    'https://myapp.com',
    'https://admin.myapp.com',
];

app.use(cors({
    origin: (origin, callback) => {
        // 允许无 origin 的请求 (Postman/curl)
        if (!origin) return callback(null, true);

        if (allowedOrigins.includes(origin)) {
            callback(null, true);
        } else {
            callback(new Error('Not allowed by CORS'));
        }
    },
    methods: ['GET', 'POST', 'PUT', 'DELETE', 'PATCH'],
    allowedHeaders: ['Content-Type', 'Authorization'],
    credentials: true,                      // 允许 Cookie
    maxAge: 86400,                          // 预检缓存 1 天
}));

// ========== 按路由配置 ==========
// 公共 API 允许所有:
app.get('/api/public', cors(), handler);

// 私有 API 限制:
app.use('/api/private', cors({
    origin: 'https://admin.myapp.com',
    credentials: true,
}));

// ========== 动态源 (根据请求) ==========
const corsOptions = {
    origin: (origin, callback) => {
        const whitelist = getWhitelistForRequest(origin);
        callback(null, whitelist);
    },
    credentials: true,
};
app.use(cors(corsOptions));
```


## 带 Cookie 的跨域


```
// ========== 跨域 Cookie ==========
// 前后端分离时, 需要在跨域请求中带 Cookie

// ========== 后端配置 ==========
app.use(cors({
    origin: 'http://localhost:5173',    // 必须指定具体源!
    credentials: true,                  // 允许 Cookie
}));

// 也可以手动设置 (自定义中间件):
app.use((req, res, next) => {
    res.setHeader('Access-Control-Allow-Origin', 'http://localhost:5173');
    res.setHeader('Access-Control-Allow-Methods', 'GET, POST, PUT, DELETE');
    res.setHeader('Access-Control-Allow-Headers', 'Content-Type, Authorization');
    res.setHeader('Access-Control-Allow-Credentials', 'true');

    // 预检请求直接返回
    if (req.method === 'OPTIONS') {
        return res.status(204).end();
    }

    next();
});

// ========== 前端配置 (Axios) ==========
// axios 需要 withCredentials:
axios.get('http://localhost:3000/api/users', {
    withCredentials: true
});

// ========== 常见问题 ==========
// 1. credentials: true 时 origin 不能为 "*"
//    → 必须指定具体源

// 2. 预检请求 OPTIONS 不携带 Cookie
//    → 正常, 预检不需要 Cookie

// 3. 302 重定向会丢失自定义头
//    → 避免 CORS 请求中的重定向

// 4. 多级路径 Cookie
//    → 设置 sameSite: 'none' + secure: true

// ========== 生产建议 ==========
// 开发环境: app.use(cors())
// 生产环境: 白名单 specific origins
// 微服务: API 网关统一处理 CORS
// Nginx 反向代理时: Nginx 层处理 CORS
```


## Nginx CORS 配置


```
// ========== Nginx 处理 CORS ==========
// 生产常用: Nginx 反向代理 + CORS

/*
nginx.conf:

server {
    listen 80;
    server_name api.myapp.com;

    location / {
        # CORS 头
        add_header Access-Control-Allow-Origin $http_origin always;
        add_header Access-Control-Allow-Methods "GET, POST, PUT, DELETE, OPTIONS" always;
        add_header Access-Control-Allow-Headers "Content-Type, Authorization" always;
        add_header Access-Control-Allow-Credentials "true" always;
        add_header Access-Control-Max-Age "86400" always;

        # 预检请求
        if ($request_method = OPTIONS) {
            return 204;
        }

        proxy_pass http://localhost:3000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
    }
}
*/

// ========== CORS 错误排查 ==========
// 1. 检查浏览器 Network 面板
// 2. 预检成功? → 看 OPTIONS 响应头
// 3. 实际请求报错? → 看实际响应头
// 4. 常见错误:
//    "No 'Access-Control-Allow-Origin' header"
//      → 服务器没返回 CORS 头
//    "The value of the 'Access-Control-Allow-Credentials' header is ''"
//      → credentials 不匹配
//    "Response to preflight doesn't pass access control check"
//      → OPTIONS 处理有问题
```


> **Note:** 💡 CORS 要点: 浏览器安全策略, 非服务器安全机制; 简单请求直接发, 复杂请求先 OPTIONS; credentials:true 时 origin 不能为 *; 开发用 cors(), 生产用白名单; 预检请求直接 204 返回; Nginx 层处理 CORS 更高效; 同源策略只影响浏览器, API/curl 不受限。


## 练习


<!-- Converted from: 7_Express CORS 跨域.html -->
