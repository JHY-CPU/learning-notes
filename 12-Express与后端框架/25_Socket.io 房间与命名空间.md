# Socket.io 房间与命名空间


## 🏠 Socket.io 房间与命名空间


命名空间隔离 (namespace)、动态房间管理、房间成员追踪、跨房间消息、通配符事件、动态命名空间、适配器 (adapter) 与多进程扩展。


## 命名空间 (Namespace)


```
// ========== 命名空间 ==========
// 在同一连接上隔离不同功能
// 每个 namespace 有独立的 room/event

// ========== 服务端定义 ==========
const { Server } = require('socket.io');
const io = new Server(server);

// 默认命名空间: /
const main = io;

// 自定义命名空间:
const chat = io.of('/chat');
const news = io.of('/news');
const admin = io.of('/admin');

// ========== 各命名空间独立 ==========
// /chat 命名空间:
chat.on('connection', (socket) => {
    socket.on('message', (data) => {
        socket.emit('reply', 'Chat message received');
    });
});

// /news 命名空间:
news.on('connection', (socket) => {
    socket.emit('headline', { title: 'Breaking News!' });
});

// /admin 命名空间:
admin.on('connection', (socket) => {
    // 只发送给 admin 命名空间
    io.of('/admin').emit('stats', { users: 100 });
});

// ========== 客户端连接 ==========
// 连接到默认:
const socket = io();                    // → /

// 连接到特定命名空间:
const chatSocket = io('/chat');         // → /chat
const newsSocket = io('/news');         // → /news

// 每个连接独立:
chatSocket.emit('message', 'Hi');
newsSocket.on('headline', (data) => {});

// ========== 使用场景 ==========
// /chat    — 实时聊天
// /notif   — 通知推送
// /admin   — 管理监控
// /game    — 游戏状态同步
```


## 动态命名空间


```
// ========== 动态命名空间 ==========
// 根据条件创建命名空间

// ========== 动态创建 ==========
io.of(/^\/dynamic-\d+$/).on('connection', (socket) => {
    const namespace = socket.nsp.name;  // /dynamic-123
    socket.emit('welcome', { namespace });
});

// ========== 房间与命名空间交互 ==========
io.on('connection', (socket) => {
    // 用户加入聊天室
    socket.join('chat:general');

    // 加入项目房间
    socket.join(`project:${projectId}`);

    // 加入用户自己的房间 (私信)
    socket.join(`user:${userId}`);
});

// ========== 房间设计模式 ==========
// ┌────────────────────────────────────────┐
// │ 房间命名规范:                           │
// │   chat:general     — 公共聊天           │
// │   chat:room:123    — 聊天室            │
// │   user:456         — 用户私信          │
// │   project:789      — 项目协作          │
// │   notify:all       — 全局通知          │
// │   game:match:abc   — 游戏对战          │
// └────────────────────────────────────────┘

// ========== 跨房间消息 ==========
// 消息同时发送到多个房间:
function sendToUser(userId, event, data) {
    io.to(`user:${userId}`).emit(event, data);
}

function sendToProject(projectId, event, data) {
    io.to(`project:${projectId}`).emit(event, data);
}

// 组合:
app.post('/api/projects/:id/tasks', authenticate, asyncHandler(async (req, res) => {
    const task = await Task.create({ ...req.body, project: req.params.id });

    // 通知项目成员
    io.to(`project:${req.params.id}`).emit('task:created', task);

    // 通知创建者
    io.to(`user:${req.user.sub}`).emit('notification', {
        type: 'task_created',
        message: 'Task created successfully',
    });

    res.created(task);
}));
```


## 中间件与事件拦截


```
// ========== 命名空间中间件 ==========
// 在每个命名空间连接前执行

// ========== 全局中间件 ==========
io.use((socket, next) => {
    // 检查用户认证
    const token = socket.handshake.auth.token;
    if (!token) {
        return next(new Error('Authentication required'));
    }

    try {
        const user = jwt.verify(token, process.env.JWT_SECRET);
        socket.user = user;       // 附加用户信息到 socket
        next();
    } catch (err) {
        next(new Error('Invalid token'));
    }
});

// ========== 命名空间级中间件 ==========
admin.use((socket, next) => {
    if (socket.user?.role !== 'admin') {
        return next(new Error('Admin access required'));
    }
    next();
});

// ========== 事件拦截 ==========
// 监听所有事件 (通配符):
socket.use(([event, ...args], next) => {
    console.log(`Event: ${event}`, args);

    // 阻止特定事件
    if (event === 'secret' && socket.user?.role !== 'admin') {
        return;  // 不调用 next() 就阻止了
    }

    // 限流检查
    const now = Date.now();
    if (now - lastEvent < 100) {
        return;  // 节流
    }
    lastEvent = now;

    next();
});

// ========== 动态加入/离开 ==========
io.on('connection', (socket) => {
    // 连接后根据用户角色加入房间
    if (socket.user?.role === 'admin') {
        socket.join('admins');
    }

    if (socket.user?.premium) {
        socket.join('premium');
    }

    // 用户状态广播
    socket.broadcast.emit('user:online', {
        userId: socket.user.sub,
    });

    socket.on('disconnect', () => {
        socket.broadcast.emit('user:offline', {
            userId: socket.user.sub,
        });
    });
});
```


## 房间状态监控


```
// ========== 房间监控 ==========
// 查看和管理房间状态

// ========== 管理 API ==========
// Express 路由 + Socket.io 状态:

app.get('/admin/socket/rooms', authenticate, authorize('admin'), (req, res) => {
    const rooms = {};
    const allRooms = io.sockets.adapter.rooms;

    for (const [room, sockets] of allRooms) {
        if (!sockets.has(room)) { // 排除 socket 自带的房间
            rooms[room] = sockets.size;
        }
    }

    res.json({
        totalRooms: Object.keys(rooms).length,
        rooms,
    });
});

app.get('/admin/socket/clients', authenticate, authorize('admin'), (req, res) => {
    const clients = [];
    const sockets = io.sockets.sockets;

    for (const [id, socket] of sockets) {
        clients.push({
            id: socket.id,
            userId: socket.user?.sub,
            rooms: [...socket.rooms].filter(r => r !== id),
            connectedAt: socket.connected,
        });
    }

    res.json({
        totalClients: clients.length,
        clients,
    });
});

// ========== 强制断开 ==========
app.post('/admin/socket/disconnect/:userId',
    authenticate, authorize('admin'), (req, res) => {
    const sockets = io.sockets.sockets;
    for (const [id, socket] of sockets) {
        if (socket.user?.sub === req.params.userId) {
            socket.disconnect(true);
        }
    }
    res.success(null, 'User disconnected');
});

// ========== 统计 ==========
// 实时在线数:
const onlineCount = io.engine.clientsCount;

// 命名空间连接数:
const chatCount = io.of('/chat').sockets.size;
```


> **Note:** 💡 命名空间与房间要点: 命名空间隔离不同功能模块 (chat/news/admin); 房间逻辑分组 (chat:general/user:123/project:456); 中间件在连接和事件前执行; 通配符 event 拦截可做日志/限流; 用户连接时自动加入个人房间 (user:userId); 断开自动清理; 管理 API 可监控在线状态。


## 练习


<!-- Converted from: 25_Socket.io 房间与命名空间.html -->
