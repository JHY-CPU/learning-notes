# Express 请求与响应


## 📨 Express 请求与响应


请求对象 req (params/query/body/headers/cookies/ip)、响应对象 res (send/json/status/redirect/attachment)、响应头设置、链式响应、流式响应、全局 res.locals。


## 请求对象 req


```
// ========== req 对象 ==========
// 请求对象, 包含所有请求信息

// ========== 路径参数 ==========
app.get('/users/:userId/posts/:postId', (req, res) => {
    req.params.userId;   // URL 参数
    req.params.postId;
    req.params[0];       // 通配符匹配
});

// ========== 查询参数 ==========
app.get('/search', (req, res) => {
    req.query.q;          // ?q=express
    req.query.page;       // &page=1
    req.query.limit;      // &limit=10
    // 无参数时返回 {}
});

// ========== 请求体 ==========
// 需要 body parser 中间件
app.use(express.json());
app.use(express.urlencoded({ extended: true }));

app.post('/users', (req, res) => {
    req.body;             // 解析后的请求体
    req.body.name;
    req.body.email;
});

// ========== 请求头 ==========
req.headers;                // 所有请求头 (对象)
req.get('Content-Type');    // 获取特定请求头
req.get('User-Agent');
req.get('Authorization');

// ========== 其他属性 ==========
req.ip;                     // 客户端 IP (需 trust proxy)
req.ips;                    // 代理 IP 列表
req.hostname;               // 主机名
req.protocol;               // http 或 https
req.method;                 // GET/POST/PUT/DELETE
req.path;                   // 路径 (/users/123)
req.url;                    // 完整 URL (/users/123?page=1)
req.originalUrl;            // 原始 URL
req.baseUrl;                // 路由前缀 (/api/v1)
req.secure;                 // 是否 HTTPS
req.accepts('json');        // 客户端接受的响应类型
req.xhr;                    // 是否为 AJAX 请求
```


## 响应对象 res


```
// ========== res 对象 ==========
// 响应对象, 控制发送给客户端的内容

// ========== 发送响应 ==========
res.send('Hello');               // 发送字符串
res.send(Buffer.from('hi'));    // 发送 Buffer
res.send({ user: 'Alice' });    // 发送对象 (自动 JSON)
res.send([1, 2, 3]);            // 发送数组 (自动 JSON)

res.json({ user: 'Alice' });    // 明确 JSON 响应
res.json(200, { msg: 'ok' });   // 带状态码 (旧式)
res.jsonp({ user: 'Alice' });   // JSONP 支持

// ========== 状态码 ==========
res.status(200);                 // 设置状态码
res.status(201).json({ id: 1 });// 链式: 201 Created
res.status(204).send();          // 204 No Content
res.sendStatus(404);             // 快捷: 404
res.sendStatus(500);             // 快捷: 500

// ========== 重定向 ==========
res.redirect('/login');              // 302 重定向
res.redirect(301, '/new-page');      // 301 永久重定向
res.redirect('https://example.com'); // 外部重定向
res.redirect('back');                // 回到上一页

// ========== 文件下载 ==========
res.download('path/to/file.pdf');       // 下载文件
res.download('report.pdf', '报表.pdf'); // 自定义文件名

res.attachment('photo.jpg');            // 设置 Content-Disposition
res.attachment();                       // 自动推断类型

// ========== 响应头 ==========
res.set('X-Custom-Header', 'value');    // 设置响应头
res.set({
    'X-Powered-By': 'Express',
    'X-RateLimit-Remaining': 100
});
res.type('json');                       // Content-Type: application/json
res.type('html');                       // Content-Type: text/html
res.links({
    next: '/users?page=2',
    prev: '/users?page=1'
});
```


## 高级响应技术


```
// ========== 流式响应 ==========
// 大文件或实时数据用流:

const fs = require('fs');
const { createReadStream } = require('fs');

app.get('/download/:file', (req, res) => {
    const stream = createReadStream(`./files/${req.params.file}`);
    stream.pipe(res);                    // 管道到响应
});

// ========== 分块传输 ==========
// 适用于 Server-Sent Events:
app.get('/events', (req, res) => {
    res.writeHead(200, {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive'
    });

    // 定时发送事件
    const timer = setInterval(() => {
        res.write(`data: ${JSON.stringify({ time: Date.now() })}\n\n`);
    }, 1000);

    // 客户端断开清理
    req.on('close', () => {
        clearInterval(timer);
    });
});

// ========== 条件响应 ==========
app.get('/users/:id', async (req, res) => {
    const user = await getUser(req.params.id);

    // 设置 ETag (内容哈希)
    const etag = crypto.createHash('md5')
        .update(JSON.stringify(user))
        .digest('hex');

    // 客户端 If-None-Match 匹配 → 304
    if (req.get('If-None-Match') === etag) {
        return res.status(304).end();
    }

    res.set('ETag', etag);
    res.json(user);
});

// ========== res.locals ==========
// 在中间件间传递数据 (请求级别)
app.use((req, res, next) => {
    res.locals.user = req.session?.user;
    res.locals.title = 'My App';
    next();
});

// 后续路由可以访问:
app.get('/', (req, res) => {
    console.log(res.locals.user);   // 模板中也可用
    res.render('index');
});

// ========== req 扩展 ==========
// 中间件可以向 req 添加自定义属性:
app.use((req, res, next) => {
    req.user = { id: 1, role: 'admin' };
    req.isAuthenticated = () => !!req.user;
    next();
});
```


## 常见模式


```
// ========== 统一响应格式 ==========
// 封装成功/失败响应:

// middleware/response.js:
function success(res, data, message = 'ok', status = 200) {
    return res.status(status).json({
        code: 0,
        message,
        data,
        timestamp: Date.now()
    });
}

function fail(res, message = 'error', code = -1, status = 400) {
    return res.status(status).json({
        code,
        message,
        timestamp: Date.now()
    });
}

// 添加到 res 对象:
app.use((req, res, next) => {
    res.success = (data, msg) => success(res, data, msg);
    res.fail = (msg, code) => fail(res, msg, code);
    next();
});

// 使用:
app.get('/users/:id', async (req, res) => {
    try {
        const user = await getUser(req.params.id);
        if (!user) return res.fail('User not found', 404);
        res.success(user);
    } catch (err) {
        res.fail(err.message, 500);
    }
});

// ========== 分页响应 ==========
app.get('/users', async (req, res) => {
    const page = parseInt(req.query.page) || 1;
    const limit = parseInt(req.query.limit) || 20;
    const skip = (page - 1) * limit;

    const [users, total] = await Promise.all([
        User.find().skip(skip).limit(limit),
        User.countDocuments()
    ]);

    res.json({
        data: users,
        pagination: {
            page,
            limit,
            total,
            totalPages: Math.ceil(total / limit),
            hasNext: page * limit < total,
            hasPrev: page > 1
        }
    });
});
```


> **Note:** 💡 req/res 要点: req.params 路径参数; req.query 查询参数; req.body 请求体(需解析); res.json() 返回 JSON; res.status() 设置状态码; res.set() 设置响应头; res.download() 文件下载; 流式传输大文件; 统一响应格式让 API 更规范。生产注意加请求验证和错误处理。


## 练习


<!-- Converted from: 2_Express 请求与响应.html -->
