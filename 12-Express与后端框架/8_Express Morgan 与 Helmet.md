# Express Morgan 与 Helmet


## 📋 Express Morgan 与 Helmet


Morgan 请求日志 (格式/自定义 token/日志轮转)、Helmet 安全头 (CSP/HSTS/X-Frame-Options/XSS 过滤)、内容安全策略配置、生产日志最佳实践、日志文件分割。


## Morgan 请求日志


```
// ========== Morgan ==========
// HTTP 请求日志中间件

npm install morgan

// ========== 预定义格式 ==========
const morgan = require('morgan');

app.use(morgan('dev'));
// :method :url :status :response-time ms - :res[content-length]

// 输出: GET /api/users 200 12.345 ms - 123

// 其他格式:
app.use(morgan('combined'));    // Apache 标准格式 (生产)
app.use(morgan('common'));      // Apache common 格式
app.use(morgan('short'));       // 短格式
app.use(morgan('tiny'));        // 最小格式

// combined 示例:
// ::1 - - [01/Jan/2024:12:00:00 +0000] "GET /api/users HTTP/1.1" 200 123 "-" "Mozilla/5.0"

// ========== 自定义格式 ==========
morgan.format('custom', ':method :url :status :res[content-length] - :response-time ms - :date[iso]');

app.use(morgan('custom'));

// ========== 自定义 Token ==========
// 添加自定义 token:
morgan.token('user', (req, res) => {
    return req.user?.id || 'anonymous';
});

morgan.token('query-params', (req) => {
    return JSON.stringify(req.query);
});

// 使用自定义 token:
app.use(morgan(':method :url :status :user - :response-time ms'));
```


## Morgan 日志写入文件


```
// ========== 日志流 ==========
// 默认输出到 stdout, 也可写入文件

const fs = require('fs');
const path = require('path');

// ========== 写入单个文件 ==========
const logStream = fs.createWriteStream(
    path.join(__dirname, 'logs', 'access.log'),
    { flags: 'a' }   // append 模式
);

app.use(morgan('combined', { stream: logStream }));

// ========== 日志轮转 (文件分割) ==========
// 使用 rotating-file-stream 按日期分割

npm install rotating-file-stream

const rfs = require('rotating-file-stream');

// 每天生成一个新日志文件:
const accessLogStream = rfs.createStream('access.log', {
    interval: '1d',              // 每天轮转
    path: path.join(__dirname, 'logs'),
    maxFiles: 30,                // 保留 30 天
    compress: 'gzip',            // 压缩旧日志
});

app.use(morgan('combined', { stream: accessLogStream }));

// ========== 环境区分 ==========
if (process.env.NODE_ENV === 'production') {
    // 生产: 写入文件, combined 格式
    app.use(morgan('combined', { stream: accessLogStream }));
} else {
    // 开发: 控制台, dev 格式
    app.use(morgan('dev'));
}

// ========== 跳过特定请求 ==========
app.use(morgan('dev', {
    skip: (req, res) => {
        // 跳过健康检查
        return req.url === '/health' || req.url === '/metrics';
        // 或跳过成功请求 (只记录错误)
        // return res.statusCode < 400;
    }
}));
```


## Helmet 安全头


```
// ========== Helmet ==========
// 设置安全相关的 HTTP 头
// 保护 Express 应用免受常见 Web 漏洞攻击

npm install helmet

// ========== 基本使用 ==========
const helmet = require('helmet');

app.use(helmet());
// 设置多个安全头 (默认全部启用)

// ========== 设置的安全头 ==========
// ┌──────────────────────────┬──────────────────────────────┐
// │ 头                        │ 作用                        │
// ├──────────────────────────┼──────────────────────────────┤
// │ Content-Security-Policy  │ 防止 XSS (限制资源来源)      │
// │ X-Content-Type-Options   │ 防止 MIME 类型嗅探           │
// │ X-Frame-Options          │ 防止点击劫持 (iframe)        │
// │ Strict-Transport-Security│ 强制 HTTPS                   │
// │ X-XSS-Protection         │ 浏览器 XSS 过滤              │
// │ Referrer-Policy          │ 控制 Referer 头              │
// │ Permissions-Policy       │ 限制 API (摄像头/麦克风)     │
// │ Expect-CT                │ 证书透明度                   │
// └──────────────────────────┴──────────────────────────────┘

// ========== 自定义配置 ==========
app.use(helmet({
    // 内容安全策略
    contentSecurityPolicy: {
        directives: {
            defaultSrc: ["'self'"],
            scriptSrc: ["'self'", "'unsafe-inline'", "cdn.example.com"],
            styleSrc: ["'self'", "'unsafe-inline'"],
            imgSrc: ["'self'", "data:", "images.example.com"],
            fontSrc: ["'self'", "fonts.googleapis.com"],
            connectSrc: ["'self'", "api.example.com"],
            frameSrc: ["'none'"],
            objectSrc: ["'none'"],
        },
    },

    // HSTS (HTTP 严格传输安全)
    strictTransportSecurity: {
        maxAge: 31536000,           // 1 年
        includeSubDomains: true,
        preload: true,
    },

    // 允许 iframe (如需嵌入)
    frameguard: { action: 'deny' },   // deny / sameorigin

    // 自定义
    referrerPolicy: { policy: 'strict-origin-when-cross-origin' },
}));

// ========== 禁用特定头 ==========
// 某些情况下需要禁用:
app.use(helmet({
    contentSecurityPolicy: false,   // 如果前端框架处理 CSP
    crossOriginEmbedderPolicy: false, // 加载第三方资源时
}));
```


## 最佳实践


```
// ========== Morgan + Helmet 组合 ==========
const express = require('express');
const helmet = require('helmet');
const morgan = require('morgan');
const app = express();

// 1. 安全头 (最先)
app.use(helmet());

// 2. 请求日志 (尽早)
if (app.get('env') === 'production') {
    app.use(morgan('combined', { stream: accessLogStream }));
} else {
    app.use(morgan('dev'));
}

// 3. 其他中间件
app.use(cors());
app.use(express.json());

// ========== Helmet 检查清单 ==========
// ✅ X-Frame-Options: DENY (防点击劫持)
// ✅ X-Content-Type-Options: nosniff (防 MIME 嗅探)
// ✅ Strict-Transport-Security: max-age=31536000 (强制 HTTPS)
// ✅ X-XSS-Protection: 1; mode=block (XSS 过滤)
// ✅ Content-Security-Policy: (防 XSS)
// ✅ Referrer-Policy: strict-origin-when-cross-origin
// ✅ Permissions-Policy: (限制 API 权限)
// ✅ 禁用 X-Powered-By: Express (信息泄露)

// ========== CSP 常见配置 ==========
// 严格:
// default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self'

// 宽松 (含 CDN 和内联):
// default-src 'self'; script-src 'self' cdn.jsdelivr.net 'unsafe-inline'
// style-src 'self' cdn.jsdelivr.net 'unsafe-inline'
// img-src 'self' data: https:

// 无 CSP (开发/兼容):
// app.use(helmet({ contentSecurityPolicy: false }))

// ========== 验证安全头 ==========
// 1. curl -I http://localhost:3000 查看响应头
// 2. https://securityheaders.com 在线检查
// 3. 浏览器的 Network 面板
// 4. CSP Evaluator 检查配置
```


> **Note:** 💡 Morgan 要点: dev 格式开发用, combined 格式生产用; 日志写入文件并轮转; 自定义 token 添加额外信息; 生产跳过健康检查日志。Helmet 要点: 一键设置多个安全头; CSP 防止 XSS 最有效但配置复杂; HSTS 强制 HTTPS; 禁用 X-Powered-By 避免信息泄露; 安全头可通过 curl 验证。


## 练习


<!-- Converted from: 8_Express Morgan 与 Helmet.html -->
