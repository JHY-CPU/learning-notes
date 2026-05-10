# 项目实战 Socket.io 实时聊天


## 📦 项目实战 3: Socket.io 实时聊天


Socket.io 建立连接、房间机制 (join/leave)、Redis 消息持久化、在线状态追踪、多进程扩展。


## 项目结构


```
// chat-app/
// ├── src/
// │   ├── server.js          # HTTP + Socket.io 服务器
// │   ├── config.js          # 配置
// │   ├── socket/            # Socket.io 事件处理
// │   │   ├── index.js           # 初始化
// │   │   ├── auth.js            # 认证中间件
// │   │   ├── chatHandler.js     # 聊天事件
// │   │   └── presenceHandler.js # 在线状态
// │   ├── db/                # 数据库
// │   │   └── redis.js           # Redis 客户端
// │   ├── models/            # 数据模型
// │   │   └── message.js
// │   └── utils/             # 工具
// │       └── format.js
// ├── client/                # 前端示例
// │   └── index.html
// ├── docker-compose.yml
// └── package.json

// ========== 服务器入口 ==========
// server.js
const http = require('http');
const express = require('express');
const { Server } = require('socket.io');
const { createAdapter } = require('@socket.io/redis-adapter');
const { createClient } = require('redis');

const app = express();
const server = http.createServer(app);

const io = new Server(server, {
  cors: {
    origin: process.env.CORS_ORIGIN || '*',
    methods: ['GET', 'POST'],
  },
  // 生产配置
  pingInterval: 25000,   // 心跳间隔
  pingTimeout: 20000,    // 超时断开
  maxHttpBufferSize: 1e6, // 最大消息 1MB
});

// Redis 适配器 (多进程/多服务器)
const pubClient = createClient({ url: process.env.REDIS_URL });
const subClient = pubClient.duplicate();

io.adapter(createAdapter(pubClient, subClient));

// 认证中间件
require('./socket/auth')(io);
// 事件处理
require('./socket/chatHandler')(io);
require('./socket/presenceHandler')(io);

server.listen(3000, () => {
  console.log('Chat server running on port 3000');
});
```


## Socket.io 事件处理


```
// ========== 认证中间件 ==========
// socket/auth.js
const jwt = require('jsonwebtoken');
const config = require('../config');

module.exports = (io) => {
  io.use((socket, next) => {
    const token = socket.handshake.auth.token;

    if (!token) {
      return next(new Error('认证失败'));
    }

    try {
      const decoded = jwt.verify(token, config.jwt.secret);
      socket.userId = decoded.userId;
      socket.userName = decoded.name;
      next();
    } catch (err) {
      next(new Error('令牌无效'));
    }
  });
};

// ========== 聊天事件处理 ==========
// socket/chatHandler.js
const Message = require('../models/message');

module.exports = (io) => {
  io.on('connection', (socket) => {
    console.log(`用户 ${socket.userName} 已连接`);

    // 加入房间
    socket.on('join:room', async (roomId) => {
      socket.join(`room:${roomId}`);

      // 加载历史消息
      const history = await Message.getRecent(roomId, 50);
      socket.emit('room:messages', history);

      // 通知房间用户上线
      socket.to(`room:${roomId}`).emit('user:online', {
        userId: socket.userId,
        name: socket.userName,
      });
    });

    // 离开房间
    socket.on('leave:room', (roomId) => {
      socket.leave(`room:${roomId}`);
      socket.to(`room:${roomId}`).emit('user:offline', {
        userId: socket.userId,
      });
    });

    // 发送消息
    socket.on('message:send', async ({ roomId, content, type = 'text' }) => {
      // 保存到数据库
      const message = await Message.create({
        roomId,
        userId: socket.userId,
        userName: socket.userName,
        content,
        type,
      });

      // 广播给房间所有人 (包含发送者)
      io.to(`room:${roomId}`).emit('message:new', message);

      // 发送回执
      socket.emit('message:ack', { messageId: message.id });
    });

    // 输入中...
    socket.on('typing:start', (roomId) => {
      socket.to(`room:${roomId}`).emit('typing:update', {
        userId: socket.userId,
        name: socket.userName,
        isTyping: true,
      });
    });

    socket.on('typing:stop', (roomId) => {
      socket.to(`room:${roomId}`).emit('typing:update', {
        userId: socket.userId,
        isTyping: false,
      });
    });

    // 断开连接
    socket.on('disconnect', () => {
      // 更新用户在线状态
      const rooms = [...socket.rooms]
        .filter(r => r.startsWith('room:'));
      rooms.forEach(roomId => {
        io.to(roomId).emit('user:offline', {
          userId: socket.userId,
        });
      });
    });
  });
};
```


## 在线状态与消息模型


```
// ========== 在线状态处理 ==========
// socket/presenceHandler.js
const redis = require('../db/redis');

const PRESENCE_KEY = 'presence';
const USER_ROOMS_KEY = 'user:rooms';

module.exports = (io) => {
  io.on('connection', async (socket) => {
    const userId = `user:${socket.userId}`;

    // 添加到在线集合
    await redis.sadd(PRESENCE_KEY, userId);
    await redis.hset(PRESENCE_KEY + ':info', userId, JSON.stringify({
      name: socket.userName,
      connectedAt: new Date(),
      socketId: socket.id,
    }));

    // 获取在线用户数
    const onlineCount = await redis.scard(PRESENCE_KEY);
    io.emit('presence:count', { count: onlineCount });

    // 断开清理
    socket.on('disconnect', async () => {
      await redis.srem(PRESENCE_KEY, userId);
      await redis.hdel(PRESENCE_KEY + ':info', userId);
      io.emit('presence:count', { count: await redis.scard(PRESENCE_KEY) });
    });
  });
};

// ========== 消息模型 ==========
// models/message.js
const redis = require('../db/redis');

const MESSAGE_KEY = 'messages';

class Message {
  // 创建消息
  static async create({ roomId, userId, userName, content, type }) {
    const message = {
      id: `msg:${Date.now()}:${Math.random().toString(36).slice(2, 8)}`,
      roomId,
      userId,
      userName,
      content,
      type,
      createdAt: new Date(),
    };

    // 存储到 Redis Stream
    await redis.xadd(
      `${MESSAGE_KEY}:${roomId}`,
      'MAXLEN', '~', 1000, // 最多保留 1000 条
      '*',
      'data', JSON.stringify(message)
    );

    return message;
  }

  // 获取最近消息
  static async getRecent(roomId, count = 50) {
    const results = await redis.xrevrange(
      `${MESSAGE_KEY}:${roomId}`,
      '+',
      '-',
      'COUNT', count
    );

    return results
      .map(([id, fields]) => ({
        id,
        ...JSON.parse(fields[1]), // fields = ['data', '{...}']
      }))
      .reverse();
  }
}

module.exports = Message;

// ========== Docker Compose ==========
// docker-compose.yml:
// version: '3.8'
// services:
//   chat:
//     build: .
//     ports:
//       - "3000:3000"
//     environment:
//       REDIS_URL: redis://redis:6379
//     depends_on:
//       - redis
//
//   redis:
//     image: redis:7-alpine
//     volumes:
//       - redisdata:/data
//
// volumes:
//   redisdata:
```


> **Note:** 💡 Socket.io 聊天要点: JWT 认证中间件 (socket.handshake.auth); 房间机制 join/leave; Redis Stream 消息持久化 (XADD/XREVRANGE); Redis Pub/Sub + Socket.io Redis Adapter 多进程扩展; 在线状态 (Set + Hash); typing 事件节流; 离线清理房间; 消息回执 ack。


## 练习


<!-- Converted from: 2_项目实战 Socket.io 实时聊天.html -->
