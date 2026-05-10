# Socket.io 生产部署与扩展


## 🚀 Socket.io 生产部署与扩展


多进程扩展 (cluster)、Redis 适配器 (Adapter)、粘性会话 (Sticky Session)、跨进程广播、连接状态管理、监控与指标、Nginx 反向代理配置、降级与回退。


## 多进程架构


```
// ========== Socket.io 扩展挑战 ==========
// Socket.io 默认单进程, 生产需要多进程
// 核心问题: 进程间无法直接通信
// -> 需要 Redis Adapter 实现跨进程广播

// ========== 方案 1: Cluster + Sticky Session ==========
const http = require('http');
const { Server } = require('socket.io');
const cluster = require('cluster');
const os = require('os');

if (cluster.isMaster) {
    const numCPUs = os.cpus().length;

    for (let i = 0; i < numCPUs; i++) {
        cluster.fork();
    }

    cluster.on('exit', (worker) => {
        console.log(`Worker ${worker.process.pid} died`);
        cluster.fork();  // 自动重启
    });
} else {
    const app = require('express')();
    const server = http.createServer(app);
    const io = new Server(server);

    io.on('connection', (socket) => {
        // 此 socket 只属于当前 worker
        socket.on('message', (data) => {
            // 只能广播到当前 worker 的连接
            io.emit('message', data);
        });
    });

    server.listen(3000);
}

// ========== 方案 2: Cluster + Redis Adapter (推荐) ==========
// 所有 worker 通过 Redis 同步消息

// ========== Nginx 配置 (Sticky Session) ==========
// upstream socket_nodes {
//     ip_hash;  # 基于 IP 的粘性会话
//     server 127.0.0.1:3001;
//     server 127.0.0.1:3002;
//     server 127.0.0.1:3003;
// }
//
// server {
//     listen 80;
//     location / {
//         proxy_set_header Upgrade $http_upgrade;
//         proxy_set_header Connection "upgrade";
//         proxy_set_header Host $host;
//         proxy_set_header X-Real-IP $remote_addr;
//         proxy_pass http://socket_nodes;
//     }
// }
```


## Redis Adapter


```
// ========== Redis Adapter ==========
// 安装: npm install @socket.io/redis-adapter ioredis

const { Server } = require('socket.io');
const { createAdapter } = require('@socket.io/redis-adapter');
const { Cluster } = require('ioredis');

// ========== 创建 Redis 连接 ==========
// 需要两个 Redis 连接 (pub/sub)
const pubClient = new Cluster([
    { host: 'redis-cluster-0', port: 6379 },
    { host: 'redis-cluster-1', port: 6379 },
]);

const subClient = pubClient.duplicate();

// ========== 应用到 Socket.io ==========
const io = new Server(server, {
    adapter: createAdapter(pubClient, subClient),
});

// 现在跨进程广播自动生效!
io.on('connection', (socket) => {
    // io.emit / io.to(room).emit 会广播到所有 worker
    socket.on('message', (data) => {
        io.to(data.room).emit('message:new', data);
    });
});

// ========== 单机 Redis ==========
const Redis = require('ioredis');
const pub = new Redis();
const sub = new Redis();

io.adapter(createAdapter(pub, sub));

// ========== Redis Adapter 原理 ==========
// 1. Worker A 调用 io.emit()
// 2. Redis Adapter 将消息发布到 Redis
// 3. Redis 广播给所有订阅的 worker
// 4. Worker B, C, ... 接收到消息并转发给本地 socket
//
// ┌──────────┐     ┌──────────┐     ┌──────────┐
// │ Worker A │────▶│   Redis  │────▶│ Worker B │
// └──────────┘     │  (Pub)   │     └──────────┘
//                  │  (Sub)   │────▶│ Worker C │
//                  └──────────┘     └──────────┘
```


## Sticky Session 粘性会话


```
// ========== Sticky Session 必要性 ==========
// WebSocket 连接建立后保持在同一进程
// 否则 Socket.io 无法找到 socket 实例

// ========== Nginx ip_hash ==========
// upstream io_nodes {
//     ip_hash;                     # 按 IP 哈希
//     server 127.0.0.1:3001 max_fails=3 fail_timeout=30s;
//     server 127.0.0.1:3002 max_fails=3 fail_timeout=30s;
//     server 127.0.0.1:3003 max_fails=3 fail_timeout=30s;
//     keepalive 64;
// }
//
// server {
//     listen 80;
//     server_name socket.example.com;
//
//     # Socket.io 需要 WebSocket 升级
//     location /socket.io/ {
//         proxy_pass http://io_nodes;
//         proxy_http_version 1.1;
//         proxy_set_header Upgrade $http_upgrade;
//         proxy_set_header Connection "upgrade";
//         proxy_set_header Host $host;
//         proxy_set_header X-Real-IP $remote_addr;
//         proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
//         proxy_set_header X-Forwarded-Proto $scheme;
//
//         # 长连接超时
//         proxy_read_timeout 86400s;
//         proxy_send_timeout 86400s;
//
//         # 缓冲关闭 (实时通信不需要)
//         proxy_buffering off;
//         proxy_buffer_size 4k;
//     }
// }

// ========== 使用 sticky 包 ==========
// 也可以使用 Node.js 层面的 sticky:
// npm install sticky-socket

const sticky = require('sticky-socket');
const http = require('http');

const server = http.createServer(app);
const io = new Server(server);

// 自动分配连接到 worker
sticky(server, {
    workers: require('os').cpus().length,
    env: { workerId: 1 },
});

// ========== Kubernetes 场景 ==========
// K8s 中每个 Pod 是独立进程
// 需要 Service 的 sessionAffinity:
// sessionAffinity: ClientIP
// sessionAffinityConfig:
//   clientIP:
//     timeoutSeconds: 10800  // 3小时
```


## 连接状态管理


```
// ========== 分布式在线状态 ==========
// Redis 存储所有在线用户 (替代 Map)

const Redis = require('ioredis');
const redis = new Redis();

// ========== 连接时注册 ==========
io.on('connection', async (socket) => {
    const { user } = socket;

    // Redis 哈希: online:{userId}
    await redis.hset(
        `online:${user.id}`,
        'socketId', socket.id,
        'name', user.name,
        'joinedAt', new Date().toISOString(),
        'worker', process.pid.toString()
    );

    // 过期时间 (防止僵尸记录)
    await redis.expire(`online:${user.id}`, 300);  // 5分钟

    // 设置心跳续期
    const heartbeat = setInterval(async () => {
        await redis.expire(`online:${user.id}`, 300);
    }, 120000);  // 2分钟

    // ========== 断开时清理 ==========
    socket.on('disconnect', async () => {
        clearInterval(heartbeat);
        await redis.del(`online:${user.id}`);
    });
});

// ========== 跨进程查询 ==========
async function getOnlineUser(userId) {
    const data = await redis.hgetall(`online:${userId}`);
    return Object.keys(data).length > 0 ? data : null;
}

async function getOnlineCount() {
    // SCAN 匹配所有 online: 键
    let count = 0;
    let cursor = '0';
    do {
        const [nextCursor, keys] = await redis.scan(
            cursor, 'MATCH', 'online:*', 'COUNT', 100
        );
        count += keys.length;
        cursor = nextCursor;
    } while (cursor !== '0');
    return count;
}

// ========== 跨进程发送消息 ==========
async function sendToUser(userId, event, data) {
    const user = await getOnlineUser(userId);
    if (user) {
        io.to(`user:${userId}`).emit(event, data);
    }
}

// ========== 用户多设备 ==========
// 同一用户多个连接 (手机 + 电脑)
io.on('connection', async (socket) => {
    const { user } = socket;

    // 加入用户房间
    socket.join(`user:${user.id}`);

    // 记录设备
    await redis.sadd(`devices:${user.id}`, socket.id);

    socket.on('disconnect', async () => {
        await redis.srem(`devices:${user.id}`, socket.id);
    });
});
```


## 监控与指标


```
// ========== Socket.io 监控 ==========
// Prometheus 指标暴露

const prometheus = require('prom-client');

// ========== 自定义指标 ==========
const connectionsGauge = new prometheus.Gauge({
    name: 'socket_io_connections_total',
    help: 'Total active socket connections',
    labelNames: ['namespace', 'worker'],
});

const eventsCounter = new prometheus.Counter({
    name: 'socket_io_events_total',
    help: 'Total socket events received',
    labelNames: ['event', 'namespace'],
});

const roomsGauge = new prometheus.Gauge({
    name: 'socket_io_rooms_count',
    help: 'Number of active rooms',
});

// ========== 指标收集 ==========
io.on('connection', (socket) => {
    connectionsGauge.inc({ namespace: socket.nsp.name, worker: process.pid });

    // 事件计数
    socket.onAny((event) => {
        eventsCounter.inc({ event, namespace: socket.nsp.name });
    });

    socket.on('disconnect', () => {
        connectionsGauge.dec({ namespace: socket.nsp.name, worker: process.pid });
    });
});

// 定时更新房间数
setInterval(() => {
    const rooms = io.sockets.adapter.rooms;
    roomsGauge.set(rooms.size);
}, 15000);

// ========== Express 监控端点 ==========
app.get('/metrics', async (req, res) => {
    res.set('Content-Type', prometheus.register.contentType);
    res.end(await prometheus.register.metrics());
});

// ========== Socket.io Admin UI ==========
// 官方管理面板
// npm install @socket.io/admin-ui

const { instrument } = require('@socket.io/admin-ui');

instrument(io, {
    auth: {
        type: 'basic',
        credentials: { username: 'admin', password: process.env.ADMIN_PWD },
    },
    mode: 'development',  // 生产环境设为 'production'
});

// 访问: https://admin.socket.io
// 输入服务器地址即可查看实时状态

// ========== 连接状态调试 ==========
app.get('/admin/socket/status', authenticate, authorize('admin'), (req, res) => {
    const adapter = io.sockets.adapter;

    res.json({
        connections: io.engine.clientsCount,
        namespaces: {
            total: Object.keys(io.nsps).length,
            chat: io.of('/chat')?.sockets?.size || 0,
        },
        rooms: {
            total: adapter.rooms.size,
            rooms: Array.from(adapter.rooms.keys())
                .filter(r => !r.startsWith('/')),  // 排除 namespace
        },
        worker: {
            pid: process.pid,
            uptime: process.uptime(),
            memory: process.memoryUsage(),
        },
    });
});
```


## 生产配置


```
// ========== 完整生产配置 ==========

const { Server } = require('socket.io');
const { createAdapter } = require('@socket.io/redis-adapter');
const { createClient } = require('redis');
const { instrument } = require('@socket.io/admin-ui');

// ========== Redis 客户端 ==========
const pubClient = createClient({
    url: process.env.REDIS_URL || 'redis://localhost:6379',
    socket: {
        reconnectStrategy: (retries) => Math.min(retries * 100, 3000),
    },
});
const subClient = pubClient.duplicate();

Promise.all([pubClient.connect(), subClient.connect()]);

// ========== Socket.io 服务端 ==========
const io = new Server(server, {
    // CORS
    cors: {
        origin: process.env.CORS_ORIGIN?.split(',') || ['http://localhost:5173'],
        credentials: true,
    },

    // Redis Adapter (跨进程)
    adapter: createAdapter(pubClient, subClient),

    // 传输配置
    transports: ['websocket', 'polling'],  // 优先 WebSocket

    // 允许升级
    allowEIO3: true,

    // 连接配置
    connectTimeout: 10000,
    maxHttpBufferSize: 1e6,  // 1MB

    // Ping/Pong
    pingInterval: 25000,
    pingTimeout: 20000,
});

// ========== 认证中间件 ==========
io.use(async (socket, next) => {
    try {
        const token = socket.handshake.auth.token;
        if (!token) throw new Error('Auth required');

        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        socket.user = decoded;
        next();
    } catch (err) {
        next(new Error('Authentication failed'));
    }
});

// ========== 连接限制 ==========
const CONNECTIONS_PER_USER = 5;
const connectionCounts = new Map();

io.use((socket, next) => {
    const userId = socket.handshake.auth.userId;
    const count = (connectionCounts.get(userId) || 0) + 1;

    if (count > CONNECTIONS_PER_USER) {
        return next(new Error('Too many connections'));
    }

    connectionCounts.set(userId, count);

    socket.on('disconnect', () => {
        const c = connectionCounts.get(userId);
        if (c <= 1) connectionCounts.delete(userId);
        else connectionCounts.set(userId, c - 1);
    });

    next();
});

// ========== 优雅关闭 ==========
process.on('SIGTERM', async () => {
    console.log('Shutting down Socket.io...');

    // 停止接受新连接
    io.close(() => {
        console.log('All connections closed');
    });

    // 断开所有客户端
    const sockets = await io.fetchSockets();
    sockets.forEach(socket => {
        socket.emit('server:shutdown', {
            message: 'Server maintenance, please reconnect',
            reconnectAfter: 5000,
        });
        socket.disconnect(true);
    });

    // 关闭 Redis
    await pubClient.quit();
    await subClient.quit();

    process.exit(0);
});

// ========== 动态命名空间 ==========
io.of(/^\/\w+$/).on('connection', (socket) => {
    console.log(`Connected to namespace: ${socket.nsp.name}`);
});

// ========== 客户端连接示例 ==========
// const socket = io('https://socket.example.com', {
//     transports: ['websocket'],
//     auth: { token: 'jwt...' },
//     reconnection: true,
//     reconnectionAttempts: Infinity,
//     reconnectionDelay: 1000,
//     reconnectionDelayMax: 30000,
//     randomizationFactor: 0.3,
// });
```


> **Note:** 💡 生产要点: 多进程用 Redis Adapter 实现跨进程广播; Sticky Session (ip_hash) 保证 WebSocket 在同一进程; Nginx 配置 WebSocket 升级和长超时; 连接状态用 Redis 持久化替代内存 Map; Prometheus 监控连接数和事件; Admin UI 管理面板; 连接数限制防滥用; 优雅关闭 + SIGTERM; 传输优先 WebSocket 降级 polling。


## 练习


<!-- Converted from: 28_Socket.io 生产部署与扩展.html -->
