# Express 压缩与 Cookie


## 🗜️ Express 压缩与 Cookie


compression 中间件 (gzip/brotli/动态压缩)、响应压缩最佳实践、cookie-parser (签名Cookie/JSON Cookie)、Cookie 选项 (httpOnly/secure/sameSite)、Session Cookie vs 持久 Cookie。


## compression 压缩


```
// ========== compression ==========
// 使用 gzip/deflate 压缩响应体 (减少传输大小)

npm install compression

// ========== 基本使用 ==========
const compression = require('compression');
app.use(compression());

// ========== 配置选项 ==========
app.use(compression({
    // 压缩阈值 (默认 1kb)
    threshold: 1024,                  // 小于 1kb 不压缩

    // 过滤函数 (哪些响应要压缩)
    filter: (req, res) => {
        // 不压缩图片 (已压缩格式)
        if (req.path.match(/\.(jpg|png|gif|webp|mp4)$/)) {
            return false;
        }
        // 使用默认过滤 (text/html, application/json 等)
        return compression.filter(req, res);
    },

    // 压缩级别 (1-9, 默认 6)
    level: 6,                         // 平衡速度与压缩比

    // 内存占用 (级别越高占用越大)
    memLevel: 8,                      // 内存级别 1-9

    // 配合 Brotli (Node 11.7+)
    brotli: {
        enabled: true,
        quality: 4,                   // 0-11, 默认 4
    }
}));

// ========== 按路由压缩 ==========
// 只为 API 响应启用压缩:
app.use('/api', compression());

// 只为大响应启用:
app.get('/api/report', compression({ threshold: 512 }), (req, res) => {
    // 大报告
});

// ========== 压缩效果 ==========
// JSON 响应:   10KB → 2KB  (80% 减少)
// HTML 页面:   50KB → 10KB (80% 减少)
// CSS/JS:      100KB → 20KB (80% 减少)
// 图片:        已压缩, 不启用

// ========== 注意事项 ==========
// 1. 压缩消耗 CPU → 高并发需权衡
// 2. Nginx 层已经压缩的话, Express 不需要再压缩
// 3. 小响应 (< 1KB) 不值得压缩
// 4. 图片/视频不要重复压缩
// 5. 流式响应可能不支持压缩
```


## cookie-parser


```
// ========== cookie-parser ==========
// 解析 Cookie 头, 填充 req.cookies

npm install cookie-parser

// ========== 基本使用 ==========
const cookieParser = require('cookie-parser');
app.use(cookieParser());

// req.cookies → { key: value }

// ========== 签名 Cookie ==========
// 防止 Cookie 被篡改
app.use(cookieParser('my-secret-key'));

// 设置签名 Cookie:
res.cookie('session', 'abc123', {
    signed: true,        // 签名
    httpOnly: true,
    maxAge: 24 * 60 * 60 * 1000,  // 1 天
});

// 读取签名 Cookie:
req.signedCookies.session   // → 'abc123'
// 如果被篡改 → false

// ========== JSON Cookie ==========
res.cookie('cart', JSON.stringify({ items: [1, 2, 3] }));

// 或直接传对象 (cookie-parser 自动处理):
res.cookie('preferences', { theme: 'dark', lang: 'zh-CN' });

// ========== Cookie 选项 ==========
res.cookie('token', 'jwt-xxx', {
    // 安全选项
    httpOnly: true,            // 禁止 JS 访问 (防 XSS)
    secure: true,              // 仅 HTTPS 传输
    sameSite: 'strict',        // 'strict' | 'lax' | 'none'

    // 生命周期
    maxAge: 7 * 24 * 60 * 60 * 1000,  // 7 天后过期 (毫秒)
    expires: new Date('2025-12-31'),  // 绝对过期时间
    // maxAge 和 expires 选一个

    // 路径/域名
    path: '/',                 // 作用路径
    domain: '.example.com',    // 作用域名
});

// ========== 删除 Cookie ==========
res.clearCookie('token');
res.clearCookie('session', { path: '/admin' });
```


## Session vs Cookie


```
// ========== Cookie 类型 ==========
// ┌──────────────┬───────────────────┬──────────────────┐
// │ 特性          │ Session Cookie    │ 持久 Cookie      │
// ├──────────────┼───────────────────┼──────────────────┤
// │ 存储位置      │ 内存 (不持久)    │ 硬盘             │
// │ 生命周期      │ 浏览器关闭即删   │ 直到 maxAge      │
// │ 设置方法      │ res.cookie(无maxAge) │ res.cookie(有maxAge) │
// └──────────────┴───────────────────┴──────────────────┘

// Session Cookie (关闭浏览器就删):
res.cookie('sessionId', 'abc', { httpOnly: true });

// 持久 Cookie (7 天后过期):
res.cookie('rememberMe', 'token', {
    httpOnly: true,
    maxAge: 7 * 24 * 60 * 60 * 1000,
});

// ========== SameSite 详解 ==========
// 'strict':   仅同站请求带 Cookie (最安全)
//    也不带 Cookie
//
// 'lax':      同站 + 顶级导航带 Cookie (默认)
//    会带, / 不带
//
// 'none':     所有请求都带 (必须 secure: true)
//   用于跨站请求 (第三方嵌入)

// ========== 最佳实践 ==========
// Cookie vs localStorage vs sessionStorage:
// ┌──────────┬──────────────────────────────┐
// │ Cookie   │ HTTP 自动发送, 可 httpOnly    │
// │ localStorage │ JS 访问, 不自动发送       │
// │ sessionStorage│ 标签页隔离              │
// └──────────┴──────────────────────────────┘

// Cookie 安全原则:
// 1. 敏感数据用 signed cookie
// 2. httpOnly 防 XSS 读取
// 3. secure (HTTPS only)
// 4. sameSite: 'strict' 或 'lax'
// 5. 不要存敏感数据 (用 session ID 存服务端)
// 6. Cookie 大小限制 4KB
```


> **Note:** 💡 compression 要点: gzip 减少 80% 传输大小; 小响应不压缩; 图片/视频跳过; Nginx 反向代理时在 Nginx 层压缩更高效。Cookie 要点: cookie-parser 解析 Cookie; signed cookie 防篡改; httpOnly+secure+sameSite 三件套保护; Session Cookie 关闭即删; SameSite strict 最安全但影响用户体验; Cookie 4KB 限制不能存大量数据。


## 练习


<!-- Converted from: 10_Express 压缩与Cookie.html -->
