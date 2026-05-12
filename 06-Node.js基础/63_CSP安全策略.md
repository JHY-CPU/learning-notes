# CSP安全策略


## CSP 安全策略


Content-Security-Policy 指令、script-src/style-src/img-src、report-only。


## CSP API


```
// ========== HTTP 头 ==========
Content-Security-Policy: default-src 'self';
    script-src 'self' https://cdn.example.com;
    style-src 'self' 'unsafe-inline';
    img-src 'self' data: https:;
    connect-src 'self' https://api.example.com;
    font-src 'self' https://fonts.gstatic.com;
    frame-src 'none';
    object-src 'none';

// ========== meta 标签 ==========


// ========== 报告模式 ==========
Content-Security-Policy-Report-Only: default-src 'self';
    report-uri /csp-report;
```


## 演示：CSP

点击按钮查看


## 什么是 CSP

CSP（Content Security Policy，内容安全策略）通过HTTP头或meta标签告诉浏览器哪些资源可以加载和执行，是防御XSS最有效的手段之一。

## 核心指令

| 指令 | 作用 |
|------|------|
| `default-src` | 所有资源的默认策略 |
| `script-src` | JS脚本来源 |
| `style-src` | CSS样式来源 |
| `img-src` | 图片来源 |
| `connect-src` | fetch/XHR/WebSocket目标 |
| `font-src` | 字体来源 |
| `frame-src` | iframe来源 |
| `object-src` | 插件来源 |

## 关键值

- `'self'`：同源资源
- `'unsafe-inline'`：允许内联脚本/样式（尽量避免）
- `'unsafe-eval'`：允许eval（尽量避免）
- `'nonce-<random>'`：带随机数的内联脚本白名单
- `'sha256-<hash>'`：按内容哈希白名单

## 部署建议

1. 先用 `Content-Security-Policy-Report-Only` 报告模式测试，不阻断请求
2. 逐步收紧策略，去掉 `'unsafe-inline'`
3. 使用 nonce 方式替代 inline 脚本
4. 设置 `report-uri` 收集违规报告
5. 生产环境必须设置 `object-src 'none'` 和 `base-uri 'none'`

## CSP 实战配置

```javascript
// ========== Express 中间件 ==========
const helmet = require('helmet');

app.use(helmet.contentSecurityPolicy({
    directives: {
        defaultSrc: ["'self'"],
        scriptSrc: ["'self'", "'nonce-{random}'", "https://cdn.jsdelivr.net"],
        styleSrc: ["'self'", "'unsafe-inline'"], // 逐步移除
        imgSrc: ["'self'", "data:", "https:"],
        connectSrc: ["'self'", "https://api.example.com"],
        fontSrc: ["'self'", "https://fonts.gstatic.com"],
        frameSrc: ["'none'"],
        objectSrc: ["'none'"],
        baseUri: ["'self'"],
        formAction: ["'self'"],
        upgradeInsecureRequests: [],
    },
    reportOnly: process.env.NODE_ENV !== 'production',
}));

// ========== Nonce 生成 ==========
const crypto = require('crypto');

app.use((req, res, next) => {
    res.locals.nonce = crypto.randomBytes(16).toString('base64');
    next();
});

// HTML 中使用:
// <script nonce="<%= nonce %>">...</script>
```

## CSP 违规报告

```javascript
// ========== 收集 CSP 违规报告 ==========
app.post('/csp-report', express.json({ type: 'application/csp-report' }), (req, res) => {
    const report = req.body['csp-report'];
    console.warn('CSP 违规:', {
        documentUri: report['document-uri'],
        violatedDirective: report['violated-directive'],
        blockedUri: report['blocked-uri'],
        sourceFile: report['source-file'],
        lineNumber: report['line-number'],
    });
    // 生产环境: 发送到日志服务
    res.status(204).end();
});

// ========== 渐进式部署 ==========
// 阶段1: Report-Only 模式，只收集不阻断
// Content-Security-Policy-Report-Only: default-src 'self'; report-uri /csp-report

// 阶段2: 分析报告，调整策略
// 阶段3: 正式启用 CSP
// Content-Security-Policy: default-src 'self'; script-src 'self' 'nonce-xxx'

// 阶段4: 移除 unsafe-inline / unsafe-eval
```

## 常见问题解决方案

```javascript
// ========== 内联脚本替换 ==========
// 问题: CSP 阻止 inline script
// 方案1: 提取到外部文件
// 方案2: 使用 nonce

// 问题: eval() 被阻止 (某些库依赖)
// 方案: 找到不使用 eval 的替代方案
// 如: JSON.parse 替代 eval('(' + str + ')')

// 问题: 第三方脚本不兼容
// 方案: 为其单独设置白名单
// script-src 'self' https://third-party.com

// ========== CSP 对 SPA 的特殊考虑 ==========
// SPA 路由需要允许 inline 配置
// 使用 hash 或 nonce 替代 unsafe-inline
// hash: script-src 'sha256-{base64hash}'
// <script integrity="sha256-..."></script>
```

<!-- Converted from: 63_CSP安全策略.html -->
