# Socket.io 基础


## 🔌 Socket.io 基础


WebSocket 协议 vs HTTP、Socket.io 安装与建立连接、emit/on 事件收发、命名空间 (namespace)、房间 (room)、广播、连接生命周期、断开重连。


## WebSocket 与 Socket.io


```
// ========== WebSocket ==========
// 全双工通信协议 (HTTP 升级)
// 服务器可主动推送消息到客户端

// HTTP: 请求-响应 (客户端发起)
// WebSocket: 双向实时 (任一方发起)

// ========== Socket.io ==========
// WebSocket 的封装库
// 提供: 自动重连、房间、命名空间、事件、回退

// ========== 安装 ==========
npm install socket.io socket.io-client

// ========== 建立连接 (服务端) ==========
const http = require('http');
const express = require('express');
const { Server } = require('socket.io');

const app = express();
const server = http.createServer(app);
const io = new Server(server, {
    cors: {
        origin: 'http://localhost:5173',
        methods: ['GET', 'POST'],
        credentials: true,
    },
});

// ========== 连接事件 ==========
io.on('connection', (socket) => {
    console.log('Client connected:', socket.id);

    // 监听客户端消息
    socket.on('chat message', (msg) => {
        console.log('Received:', msg);

        // 广播给所有客户端
        io.emit('chat message', msg);
    });

    // 断开连接
    socket.on('disconnect', (reason) => {
        console.log('Client disconnected:', socket.id, reason);
    });
});

// 启动:
server.listen(3000, () => {
    console.log('Server + Socket.io running on port 3000');
});
```


## 客户端连接


```
// ========== 客户端 (浏览器) ==========
// 使用 socket.io-client 或 CDN


// ========== Node.js 客户端 ==========
const { io } = require('socket.io-client');
const socket = io('http://localhost:3000');
socket.on('connect', () => console.log('Connected'));
```


## 事件与消息


```
// ========== 事件系统 ==========
// Socket.io 基于事件通信

// ========== 服务端 ==========
io.on('connection', (socket) => {
    // 发送消息到客户端
    socket.emit('hello', { msg: 'Welcome!' });

    // 发送带确认
    socket.emit('question', 'what is your name?', (answer) => {
        console.log('Client answered:', answer);
    });

    // 监听消息
    socket.on('message', (data) => {
        console.log('Message:', data);
    });

    // 监听带确认
    socket.on('join', (room, callback) => {
        socket.join(room);
        callback({ status: 'ok' });
    });

    // 一次性监听
    socket.once('init', (data) => {
        console.log('Init data:', data);
    });

    // 移除监听
    // socket.off('message');
});

// ========== 广播 ==========
// 向所有客户端发送 (包括发送者)
io.emit('global notification', 'Server maintenance soon');

// 向所有客户端发送 (不包括发送者)
socket.broadcast.emit('user joined', { userId: socket.id });

// 向房间内所有人 (包括发送者)
io.to('room1').emit('room message', 'Hello room!');

// 向房间内所有人 (不包括发送者)
socket.to('room1').emit('room message', 'Hello room!');

// 向多个房间
io.to('room1').to('room2').emit('multi room', 'data');

// ========== 命名空间 ==========
// 默认命名空间: /
// 自定义命名空间: /chat, /admin

const chatNamespace = io.of('/chat');
chatNamespace.on('connection', (socket) => {
    socket.on('message', (msg) => {
        chatNamespace.emit('message', msg);
    });
});

const adminNamespace = io.of('/admin');
adminNamespace.use((socket, next) => {
    // 认证中间件
    const token = socket.handshake.auth.token;
    if (token !== 'admin-token') {
        return next(new Error('Not authorized'));
    }
    next();
});
```


## 房间管理


```
// ========== 房间 (Room) ==========
// 逻辑分组, 同一房间内可广播

io.on('connection', (socket) => {
    console.log(`User ${socket.id} connected`);

    // ========== 加入房间 ==========
    socket.on('join-room', (roomId, callback) => {
        socket.join(roomId);
        console.log(`${socket.id} joined room ${roomId}`);

        // 通知房间其他人
        socket.to(roomId).emit('user-joined', {
            userId: socket.id,
            roomId,
        });

        // 给客户端确认
        if (callback) callback({ success: true });
    });

    // ========== 离开房间 ==========
    socket.on('leave-room', (roomId) => {
        socket.leave(roomId);
        socket.to(roomId).emit('user-left', {
            userId: socket.id,
        });
    });

    // ========== 房间内消息 ==========
    socket.on('room-message', ({ roomId, message }) => {
        io.to(roomId).emit('new-message', {
            userId: socket.id,
            message,
            timestamp: Date.now(),
        });
    });

    // ========== 断开时自动离开 ==========
    socket.on('disconnect', () => {
        // 可以获取 socket.rooms 知道他在哪些房间
        socket.rooms.forEach(room => {
            if (room !== socket.id) {
                socket.to(room).emit('user-disconnected', {
                    userId: socket.id,
                });
            }
        });
    });
});

// ========== 房间管理 API ==========
// 获取房间内成员数:
io.sockets.adapter.rooms.get('room1')?.size
// 获取 socket 在哪些房间:
socket.rooms  // Set { socket.id, 'room1', 'room2' }
// 获取所有房间:
io.sockets.adapter.rooms  // Map { roomId => Set { socketId } }
```


> **Note:** 💡 Socket.io 要点: 基于 WebSocket 全双工通信; emit/on 事件驱动; 房间 (room) 逻辑分组; 命名空间 (namespace) 隔离不同功能; broadcast 排除发送者; 自动重连; 回调确认; cors 配置跨域; 断开时清理房间。


## 练习


<!-- Converted from: 24_Socket.io 基础.html -->
