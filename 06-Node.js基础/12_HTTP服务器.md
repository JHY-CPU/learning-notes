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


<!-- Converted from: 12_HTTP服务器.html -->
