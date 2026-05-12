# CORS深入


## CORS 深入


简单/复杂请求区分、预检请求缓存、withCredentials、通配符限制。


## CORS API


```
// ========== 简单请求 ==========
// 方法: GET / HEAD / POST
// Content-Type: application/x-www-form-urlencoded
//              multipart/form-data
//              text/plain
// 无自定义头

// ========== 复杂请求 ==========
// 方法: PUT / DELETE / PATCH
// Content-Type: application/json
// 自定义头
// → 先发 OPTIONS 预检

// ========== 服务端 CORS 头 ==========
Access-Control-Allow-Origin: https://example.com
Access-Control-Allow-Methods: GET, POST, PUT
Access-Control-Allow-Headers: Content-Type, Authorization
Access-Control-Max-Age: 86400
Access-Control-Allow-Credentials: true
Access-Control-Expose-Headers: X-Total-Count
```


## 演示：CORS

点击按钮查看


## 什么是 CORS

CORS（Cross-Origin Resource Sharing，跨域资源共享）是浏览器的安全机制，限制页面向不同源的服务器发起请求。由服务端通过响应头声明允许的跨域访问。

## 简单请求 vs 复杂请求

**简单请求**（直接发送）：GET/HEAD/POST + 特定Content-Type + 无自定义头。

**复杂请求**（先发预检）：PUT/DELETE/PATCH、`application/json`、自定义Header等。浏览器先发 `OPTIONS` 预检请求，通过后才发实际请求。

## 关键响应头

- `Access-Control-Allow-Origin`：允许的来源（不能用通配符配合Credentials）
- `Access-Control-Allow-Credentials`：是否允许携带Cookie
- `Access-Control-Max-Age`：预检结果缓存时间（秒）
- `Access-Control-Expose-Headers`：允许前端读取的额外响应头

## 常见问题

- **Credentials冲突**：`Allow-Credentials: true` 时 `Allow-Origin` 不能为 `*`，必须指定具体域名
- **预检缓存**：`Max-Age` 设置合理值减少OPTIONS请求
- **自定义Header**：需要在 `Allow-Headers` 中声明

## Node.js 实现

```js
app.use((req, res, next) => {
    res.header('Access-Control-Allow-Origin', req.headers.origin);
    res.header('Access-Control-Allow-Credentials', 'true');
    res.header('Access-Control-Allow-Methods', 'GET,POST,PUT,DELETE');
    res.header('Access-Control-Allow-Headers', 'Content-Type,Authorization');
    if (req.method === 'OPTIONS') return res.sendStatus(204);
    next();
});
```

<!-- Converted from: 64_CORS深入.html -->
