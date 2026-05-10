# Socket.io 认证中间件


## 🛡️ Socket.io 认证中间件


Socket.io 认证流程 (JWT Token)、连接中间件 (io.use)、socket.handshake.auth/headers/query、认证后自动加入房间、角色权限控制、断开与踢出、认证与 Express 共享 session。


## JWT 认证中间件


```
// ========== Socket.io 认证 ==========
// 连接时验证 JWT, 拒绝未认证连接

// ========== 认证中间件 ==========
const jwt = require('jsonwebtoken');

io.use((socket, next) => {
    // 取值顺序: auth → query → headers
    const token = socket.handshake.auth.token
        || socket.handshake.query.token
        || socket.handshake.headers['x-auth-token'];

    if (!token) {
        return next(new Error('Authentication required'));
    }

    try {
        const decoded = jwt.verify(token, process.env.JWT_SECRET);
        // 附加用户信息到 socket 实例
        socket.user = {
            id: decoded.sub,
            role: decoded.role,
            name: decoded.name,
        };
        next();
    } catch (err) {
        return next(new Error('Invalid token'));
    }
});

// ========== 客户端连接 ==========
// 通过 auth 参数传 token:
const socket = io('http://localhost:3000', {
    auth: {
        token: 'jwt-token-here'
    }
});

// 或通过 query:
const socket = io('http://localhost:3000?token=jwt-token-here');

// ========== 连接后 ==========
io.on('connection', (socket) => {
    console.log(`User ${socket.user.id} (${socket.user.role}) connected`);
    // socket.user 可用
});

// ========== 认证失败处理 ==========
// 客户端:
socket.on('connect_error', (err) => {
    if (err.message === 'Authentication required' ||
        err.message === 'Invalid token') {
        // 跳转登录页
        window.location.href = '/login';
    }
});
```


## 认证后自动加入房间


```
// ========== 用户房间策略 ==========
// 连接后根据用户信息自动加入相应房间

io.use(authMiddleware);  // 先认证

io.on('connection', (socket) => {
    const { user } = socket;

    // ========== 1. 个人房间 (私信) ==========
    socket.join(`user:${user.id}`);

    // ========== 2. 角色房间 ==========
    if (user.role === 'admin') {
        socket.join('role:admin');
    }
    socket.join(`role:${user.role}`);

    // ========== 3. 项目房间 (从数据库) ==========
    UserProject.find({ userId: user.id }).then(projects => {
        projects.forEach(p => {
            socket.join(`project:${p.projectId}`);
        });
    });

    // ========== 4. 在线状态 ==========
    // 通知其他人
    socket.broadcast.emit('user:online', {
        userId: user.id,
        name: user.name,
    });

    // 存储在线用户
    onlineUsers.set(user.id, {
        socketId: socket.id,
        name: user.name,
        joinedAt: new Date(),
    });

    // 发送当前在线列表 (给新连接)
    socket.emit('online:users', Array.from(onlineUsers.entries()));

    // ========== 5. 断开清理 ==========
    socket.on('disconnect', () => {
        onlineUsers.delete(user.id);
        socket.broadcast.emit('user:offline', {
            userId: user.id,
        });
    });
});

// ========== 在线用户管理 ==========
const onlineUsers = new Map();  // 生产用 Redis

// 查询用户是否在线:
function isUserOnline(userId) {
    return onlineUsers.has(userId);
}

// 获取在线用户数:
function getOnlineCount() {
    return onlineUsers.size;
}

// 向特定用户发消息:
function sendToUser(userId, event, data) {
    io.to(`user:${userId}`).emit(event, data);
}
```


## 角色权限控制


```
// ========== 事件级权限 ==========
// 在事件处理中检查权限

io.on('connection', (socket) => {
    const { user } = socket;

    // ========== 管理事件 ==========
    socket.on('admin:ban-user', async (data, callback) => {
        // 角色检查
        if (user.role !== 'admin') {
            return callback({ error: 'Forbidden' });
        }

        await User.findByIdAndUpdate(data.userId, { isActive: false });
        callback({ success: true });
    });

    // ========== 踢出用户 ==========
    if (user.role === 'admin') {
        socket.on('admin:kick', (data) => {
            const targetSocket = findSocketByUserId(data.userId);
            if (targetSocket) {
                targetSocket.emit('kicked', { reason: data.reason });
                targetSocket.disconnect(true);
            }
        });
    }

    // ========== 房间内角色 ==========
    socket.on('room:mute', ({ roomId, userId, duration }) => {
        if (user.role !== 'admin' && user.role !== 'moderator') {
            return socket.emit('error', { message: 'No permission' });
        }
        // 禁言用户...
    });
});

// ========== 共享 Express Session ==========
const sessionMiddleware = session({ ... });

// Express 使用:
app.use(sessionMiddleware);

// Socket.io 共享:
io.use((socket, next) => {
    sessionMiddleware(socket.request, {}, next);
});

// 然后在 socket 中访问:
io.on('connection', (socket) => {
    const session = socket.request.session;
    // session.userId
});

// ========== 认证检查辅助 ==========
function requireAuth(socket, callback) {
    if (!socket.user) {
        callback({ error: 'Not authenticated' });
        return false;
    }
    return true;
}

function requireRole(role) {
    return (socket, callback) => {
        if (socket.user?.role !== role) {
            callback({ error: 'Insufficient permissions' });
            return false;
        }
        return true;
    };
}

socket.on('admin:action', (data, callback) => {
    if (!requireRole('admin')(socket, callback)) return;
    // ... 执行操作
});
```


## 断开与重连


```
// ========== 断开连接 ==========
// 服务端主动断开:
socket.disconnect(true);  // 强制断开, 不重连

// 客户端断开:
socket.disconnect();
// 服务器不会自动重连

socket.close();  // 别名

// ========== 重连事件 ==========
// Socket.io 默认自动重连

socket.on('reconnect_attempt', (attempt) => {
    console.log(`Reconnect attempt #${attempt}`);
});

socket.on('reconnect', (attempt) => {
    console.log(`Reconnected after ${attempt} attempts`);
});

socket.on('reconnect_error', (err) => {
    console.error('Reconnect error:', err);
});

socket.on('reconnect_failed', () => {
    console.log('All reconnect attempts failed');
});

// ========== 重连后恢复 ==========
let currentRoom = null;

socket.on('connect', () => {
    // 重新加入之前的房间
    if (currentRoom) {
        socket.emit('join-room', currentRoom);
    }
});

socket.on('disconnect', (reason) => {
    if (reason === 'io server disconnect') {
        // 服务器主动断开, 不自动重连
    } else {
        // 网络问题, 自动重连
    }
});

// ========== 心跳检测 ==========
// Socket.io 内置 ping/pong
// 可配置:
const io = new Server(server, {
    pingInterval: 25000,     // 25 秒发一次 ping
    pingTimeout: 20000,      // 20 秒无 pong 断开
});

// 自定义心跳:
setInterval(() => {
    io.emit('heartbeat', { time: Date.now() });
}, 30000);
```


> **Note:** 💡 Socket.io 认证要点: io.use() 中间件在连接前执行; socket.handshake.auth 传 token; 认证后 socket.user 可用; 连接后自动加入个人/角色房间; 事件处理中检查具体权限; 共享 Express session 中间件; 断开事件区分服务器/网络原因; 重连后恢复到之前状态。


## 练习


<!-- Converted from: 26_Socket.io 认证中间件.html -->
