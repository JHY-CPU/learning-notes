# Express compression 压缩


## 🗜️ Express compression 压缩


compression 中间件、gzip/brotli/deflate 压缩算法、动态压缩 vs 静态压缩、压缩级别权衡、跳过已压缩内容、Nginx 压缩层、性能优化。


## compression 基础


```
// ========== compression 中间件 ==========
// 对 HTTP 响应体进行压缩
// 减少传输大小, 加快页面加载

// 安装:
// npm install compression

const compression = require('compression');

// ========== 基础使用 ==========
app.use(compression());

// 所有通过此中间件的响应都会被压缩
// 客户端需支持: Accept-Encoding: gzip, deflate, br

// ========== 配置选项 ==========
app.use(compression({
    // 压缩阈值 (小于此不压缩)
    threshold: 1024,  // 1KB, 默认

    // 压缩级别 (1-9, 越高越小但越慢)
    level: 6,  // 默认, 推荐平衡

    // 只压缩某些类型
    filter: (req, res) => {
        // 不要压缩图片 (已压缩过)
        if (req.url.match(/\.(jpg|jpeg|png|gif|webp|zip|gz)$/)) {
            return false;
        }
        // compression 默认会检查 Content-Type
        return compression.filter(req, res);
    },

    // 缓存动态压缩的结果
    // (不常用, 一般 Nginx 处理)
}));

// ========== 压缩效果 ==========
// JSON API 响应:
// 无压缩: 100KB
// gzip:    15KB  (85% 减少)
// brotli:  12KB  (88% 减少)
//
// HTML:
// 无压缩: 50KB
// gzip:   8KB   (84% 减少)
//
// CSS/JS:
// 无压缩: 200KB
// gzip:   40KB  (80% 减少)
```


## 压缩算法


```
// ========== 压缩算法对比 ==========

// ┌──────────┬──────────┬──────────┬──────────────┐
// │ 算法      │ 压缩比   │ 速度     │ 浏览器支持    │
// ├──────────┼──────────┼──────────┼──────────────┤
// │ gzip     │ 中等      │ 快       │ 全部          │
// │ brotli   │ 高        │ 慢(压缩) │ 现代浏览器    │
// │ deflate  │ 中等      │ 中等     │ 部分          │
// └──────────┴──────────┴──────────┴──────────────┘

// ========== gzip (通用) ==========
// npm install compression 默认使用 zlib
// 兼容性最好, 所有浏览器和 CDN 支持

// ========== Brotli (Google) ==========
// 比 gzip 压缩率高 20-30%
// Node.js zlib 内置 brotli 支持

const zlib = require('zlib');

// 使用 brotli 压缩:
app.use((req, res, next) => {
    const acceptEncoding = req.headers['accept-encoding'] || '';

    if (acceptEncoding.includes('br')) {
        // Brotli 压缩
        res.setHeader('Content-Encoding', 'br');
        const brotli = zlib.createBrotliCompress({
            params: {
                [zlib.constants.BROTLI_PARAM_QUALITY]: 4,  // 0-11
            },
        });
        // 拦截 res.send 等进行压缩...

        // 简单方案: 使用 shrink-ray 或 express-brotli
    }
    next();
});

// ========== express-brotli (替代) ==========
// npm install express-brotli
// const brotli = require('express-brotli');
// app.use(brotli({ quality: 4, threshold: 1024 }));

// ========== 压缩级别权衡 ==========
// gzip level:
// 1 — 最快, 压缩率低 (开发用)
// 6 — 平衡 (默认, 推荐)
// 9 — 最慢, 压缩率最高 (静态资源)

// ========== 跳过已压缩内容 ==========
// 图片/视频一般已压缩, 不再压缩
app.use(compression({
    filter: (req, res) => {
        const type = res.getHeader('Content-Type') || '';
        if (type.startsWith('image/')) return false;
        if (type.startsWith('video/')) return false;
        if (type === 'application/octet-stream') return false;
        return true;
    },
}));
```


## Nginx 压缩层


```
// ========== Nginx 压缩 ==========
// 生产环境通常由 Nginx 处理压缩
// Node.js 做 gzip 会消耗 CPU

// ========== Nginx gzip 配置 ==========
// /etc/nginx/nginx.conf:
//
// http {
//     # 启用 gzip
//     gzip on;
//     gzip_vary on;
//     gzip_proxied any;
//     gzip_comp_level 6;
//     gzip_min_length 1024;
//     gzip_types
//         text/plain
//         text/css
//         text/javascript
//         application/json
//         application/javascript
//         application/xml
//         image/svg+xml;
//
//     # Brotli (需安装 ngx_brotli)
//     brotli on;
//     brotli_comp_level 4;
//     brotli_types text/plain text/css application/json application/javascript;
// }

// ========== Node.js 压缩策略 ==========
// 开发环境: Express compression
// 生产环境: Nginx 压缩 (Node 不压缩)

// 生产环境条件启用:
if (config.isDev()) {
    app.use(compression());
}
// 生产环境由 Nginx 处理, Node 不需要

// ========== 动静分离 ==========
// 静态文件: Nginx 直接提供 + 压缩 + 缓存
// 动态 API: Nginx 压缩后转发

// Nginx:
// location /api/ {
//     proxy_pass http://node_app;
//     gzip on;  # 压缩 API 响应
// }
//
// location /static/ {
//     root /var/www;
//     gzip_static on;  # 使用预生成的 .gz 文件
//     expires 365d;
// }

// ========== gzip_static 预压缩 ==========
// 构建时生成 .gz 文件, Nginx 直接使用
// 避免实时压缩消耗 CPU

// 构建脚本:
// const zlib = require('zlib');
// const fs = require('fs');
//
// async function precompress(file) {
//     const content = fs.readFileSync(file);
//     const gzipped = zlib.gzipSync(content, { level: 9 });
//     fs.writeFileSync(file + '.gz', gzipped);
// }
//
// 遍历 dist/ 目录对所有 .js .css 预压缩
```


## 性能考量


```
// ========== 压缩性能权衡 ==========

// ========== CPU vs 带宽 ==========
// 压缩: 节省带宽, 消耗 CPU
// 不压缩: 消耗带宽, 节省 CPU

// 建议:
// 小型响应 (< 1KB): 不压缩 (阈值)
// JSON API: gzip level 6
// 大文件: 预压缩 (gzip_static)
// 图片: 不压缩 (已经压缩)
// 流式响应: 压缩 (但注意 chunk 延迟)

// ========== 条件压缩 ==========
// API 路由选择压缩
const apiCompression = compression({
    threshold: 512,    // API 响应更小
    level: 6,
    filter: (req, res) => {
        return req.url.startsWith('/api/');
    },
});

app.use('/api', apiCompression);

// 静态文件用更高压缩级别
const staticCompression = compression({
    threshold: 2048,
    level: 9,          // 静态资源最高压缩
    filter: (req, res) => {
        return req.url.startsWith('/static/');
    },
});

app.use('/static', staticCompression);

// ========== 压缩检测 ==========
// 查看压缩效果:
// curl -H "Accept-Encoding: gzip" -o /dev/null -w "%{size_download}" http://localhost:3000/api/users
// curl -o /dev/null -w "%{size_download}" http://localhost:3000/api/users
// 对比压缩前后大小

// ========== 禁用压缩场景 ==========
// 1. WebSocket (已经用了更高效的协议)
// 2. Server-Sent Events (实时性要求高)
// 3. 小文件 (< 1KB 反而变大)
// 4. 已加密内容 (随机性, 无法压缩)
// 5. 硬件负载均衡器处理 (AWS ALB)
app.use('/socket.io', (req, res, next) => {
    // 跳过 Socket.io 的 compression
    next();
});

// ========== Node.js 压缩负载测试 ==========
// 模拟: 100并发, 10KB JSON 响应
// 无压缩: CPU 5%,  带宽 100Mbps
// gzip 6: CPU 30%, 带宽 15Mbps
// brotli: CPU 60%, 带宽 10Mbps
//
// 结论: 高并发场景 Nginx 处理压缩
```


> **Note:** 💡 compression 要点: 减小传输体积 80%+; gzip 兼容性最好, brotli 压缩率最高; threshold 跳过小响应; filter 跳过图片/视频; Nginx 生产处理压缩更高效; gzip_static 预压缩免实时 CPU; 高并发考虑 Nginx 压缩; 分路由配置不同级别; 小文件不压缩反而增大。


## 练习


<!-- Converted from: 38_Express compression 压缩.html -->
