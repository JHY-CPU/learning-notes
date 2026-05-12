# HTTP服务器


## HTTP 服务器


http.createServer、req/res 对象、状态码、响应头、静态文件。


## HTTP 服务器 API


```
// ========== 创建 HTTP 服务器 ==========
const http = require('http');

const server = http.createServer((req, res) => {
    // req — IncomingMessage (可读流)
    // res — ServerResponse (可写流)

    res.statusCode = 200;
    res.setHeader('Content-Type', 'text/html');
    res.end('Hello World');
});

server.listen(3000, () => {
    console.log('服务器运行在 http://localhost:3000');
});

// ========== 请求对象 req ==========
req.method      // 'GET' / 'POST' / 'PUT' / 'DELETE'
req.url         // '/path?query=string'
req.headers     // { 'content-type': 'application/json' }
req.socket      // 底层 socket

// ========== 响应对象 res ==========
res.statusCode = 200;
res.statusMessage = 'OK';
res.setHeader('Content-Type', 'application/json');
res.writeHead(200, 'OK', { 'Content-Type': 'text/plain' });
res.write('partial data');
res.end('final data');

// ========== 静态文件服务 ==========
const fs = require('fs');
const path = require('path');

const filePath = path.join(__dirname, 'public', req.url);
const stream = fs.createReadStream(filePath);
stream.pipe(res);
// 错误处理
stream.on('error', () => {
    res.statusCode = 404;
    res.end('Not Found');
});
```


## 演示：HTTP 服务器

点击按钮查看


## 实战：完整的 HTTP 服务器

```javascript
// ========== 完整 HTTP 服务器示例 ==========
const http = require('http');
const fs = require('fs');
const path = require('path');
const url = require('url');

const MIME_TYPES = {
    '.html': 'text/html; charset=utf-8',
    '.css': 'text/css; charset=utf-8',
    '.js': 'application/javascript; charset=utf-8',
    '.json': 'application/json',
    '.png': 'image/png',
    '.jpg': 'image/jpeg',
    '.gif': 'image/gif',
    '.svg': 'image/svg+xml',
    '.ico': 'image/x-icon',
};

const server = http.createServer(async (req, res) => {
    const parsedUrl = url.parse(req.url, true);
    const pathname = parsedUrl.pathname;

    // API 路由
    if (pathname.startsWith('/api')) {
        res.setHeader('Content-Type', 'application/json');

        if (pathname === '/api/health' && req.method === 'GET') {
            res.end(JSON.stringify({ status: 'ok', uptime: process.uptime() }));
            return;
        }

        if (pathname === '/api/time' && req.method === 'GET') {
            res.end(JSON.stringify({ time: new Date().toISOString() }));
            return;
        }

        // 404 API
        res.statusCode = 404;
        res.end(JSON.stringify({ error: 'API not found' }));
        return;
    }

    // 静态文件服务
    let filePath = path.join(__dirname, 'public', pathname);
    if (pathname === '/') filePath = path.join(__dirname, 'public', 'index.html');

    try {
        const stat = await fs.promises.stat(filePath);
        if (stat.isDirectory()) {
            filePath = path.join(filePath, 'index.html');
        }

        const ext = path.extname(filePath);
        const contentType = MIME_TYPES[ext] || 'application/octet-stream';

        res.setHeader('Content-Type', contentType);
        res.setHeader('Cache-Control', 'public, max-age=3600');

        const stream = fs.createReadStream(filePath);
        stream.pipe(res);
        stream.on('error', () => {
            res.statusCode = 500;
            res.end('Internal Server Error');
        });
    } catch {
        res.statusCode = 404;
        res.setHeader('Content-Type', 'text/html; charset=utf-8');
        res.end('<h1>404 Not Found</h1>');
    }
});

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`服务器运行在 http://localhost:${PORT}`);
    console.log(`进程 PID: ${process.pid}`);
});

// 优雅关闭
process.on('SIGTERM', () => {
    server.close(() => process.exit(0));
});
```

## 请求体解析

```javascript
// ========== 解析不同类型的请求体 ==========
async function parseBody(req) {
    return new Promise((resolve, reject) => {
        const chunks = [];
        req.on('data', (chunk) => chunks.push(chunk));
        req.on('end', () => {
            const rawBody = Buffer.concat(chunks).toString();
            const contentType = req.headers['content-type'] || '';

            try {
                if (contentType.includes('application/json')) {
                    resolve(JSON.parse(rawBody));
                } else if (contentType.includes('application/x-www-form-urlencoded')) {
                    const params = new URLSearchParams(rawBody);
                    resolve(Object.fromEntries(params));
                } else if (contentType.includes('text/plain')) {
                    resolve(rawBody);
                } else {
                    resolve(rawBody);
                }
            } catch (err) {
                reject(new Error('请求体解析失败'));
            }
        });
        req.on('error', reject);
    });
}

// 使用
const server = http.createServer(async (req, res) => {
    if (req.method === 'POST' && req.url === '/api/data') {
        try {
            const body = await parseBody(req);
            console.log('收到数据:', body);
            res.setHeader('Content-Type', 'application/json');
            res.end(JSON.stringify({ received: body }));
        } catch (err) {
            res.statusCode = 400;
            res.end(JSON.stringify({ error: err.message }));
        }
    }
});
```

## HTTPS 服务器

```javascript
// ========== HTTPS 服务器 ==========
const https = require('https');
const fs = require('fs');

const options = {
    key: fs.readFileSync('./certs/server.key'),
    cert: fs.readFileSync('./certs/server.crt'),
};

const httpsServer = https.createServer(options, (req, res) => {
    res.end('Hello HTTPS!');
});

httpsServer.listen(443, () => {
    console.log('HTTPS 服务器运行在 https://localhost:443');
});
```

## 性能优化

- **使用 Stream 传输文件**：避免将整个文件读入内存
- **启用 keep-alive**：减少 TCP 连接建立开销
- **设置超时**：`server.timeout = 30000`（30秒）
- **压缩响应**：使用 `zlib.createGzip()` 压缩响应体
- **连接池**：高并发时使用 cluster 模块

<!-- Converted from: 12_HTTP服务器.html -->
