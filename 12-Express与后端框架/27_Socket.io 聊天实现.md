# Socket.io 聊天实现


## 💬 Socket.io 聊天实现


实时聊天系统设计、消息收发与持久化 (MongoDB)、聊天室管理、已读/未读、输入状态指示、消息历史加载、文件分享、表情/提及 (@user)。


## 聊天服务端架构


```
// ========== 聊天系统架构 ==========
// Server: Express + Socket.io + MongoDB
// Client: Socket.io-client

// ========== 消息模型 ==========
// models/Message.js:
const messageSchema = new mongoose.Schema({
    room: { type: String, required: true, index: true },
    sender: {
        userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
        name: String,
        avatar: String,
    },
    type: {
        type: String,
        enum: ['text', 'image', 'file', 'system'],
        default: 'text',
    },
    content: { type: String, required: true },
    mentions: [{ type: mongoose.Schema.Types.ObjectId, ref: 'User' }],
    attachments: [{
        url: String,
        name: String,
        size: Number,
        type: String,
    }],
    readBy: [{
        userId: { type: mongoose.Schema.Types.ObjectId, ref: 'User' },
        readAt: Date,
    }],
}, { timestamps: true });

// TTL 索引: 消息保留 90 天
messageSchema.index({ createdAt: 1 }, { expireAfterSeconds: 90 * 24 * 3600 });

// ========== Socket.io 聊天逻辑 ==========
// socket/chat.js:
function setupChat(io) {
    const chat = io.of('/chat');

    chat.use(authMiddleware);

    chat.on('connection', (socket) => {
        const { user } = socket;

        // 加入聊天室
        socket.on('room:join', async ({ roomId }, callback) => {
            socket.join(`room:${roomId}`);
            // 加载最近消息
            const recentMessages = await Message.find({ room: roomId })
                .sort({ createdAt: -1 })
                .limit(50)
                .lean();

            callback({ messages: recentMessages.reverse() });

            // 通知房间
            socket.to(`room:${roomId}`).emit('user:joined', {
                userId: user.id,
                name: user.name,
            });
        });

        // 发送消息
        socket.on('message:send', async (data, callback) => {
            const message = await Message.create({
                room: data.roomId,
                sender: { userId: user.id, name: user.name },
                content: data.content,
                type: data.type || 'text',
                mentions: extractMentions(data.content),
            });

            // 广播到房间
            io.of('/chat').to(`room:${data.roomId}`).emit('message:new', message);

            // 通知 @ 用户
            message.mentions.forEach(mentionId => {
                io.of('/chat').to(`user:${mentionId}`).emit('notification', {
                    type: 'mention',
                    message: `${user.name} mentioned you`,
                });
            });

            callback({ success: true, messageId: message._id });
        });

        // 输入状态
        socket.on('typing:start', ({ roomId }) => {
            socket.to(`room:${roomId}`).emit('typing', {
                userId: user.id,
                name: user.name,
            });
        });

        socket.on('typing:stop', ({ roomId }) => {
            socket.to(`room:${roomId}`).emit('typing:stop', {
                userId: user.id,
            });
        });
    });
}
```


## 已读与未读数


```
// ========== 已读回执 ==========

// ========== 标记已读 ==========
socket.on('message:read', async ({ roomId, messageIds }) => {
    await Message.updateMany(
        { _id: { $in: messageIds }, 'readBy.userId': { $ne: user.id } },
        { $push: { readBy: { userId: user.id, readAt: new Date() } } }
    );

    // 通知发送者已读
    socket.to(`room:${roomId}`).emit('message:read-receipt', {
        userId: user.id,
        messageIds,
    });
});

// ========== 未读数统计 ==========
// Express API:
app.get('/api/chat/unread', authenticate, asyncHandler(async (req, res) => {
    const rooms = await UserRoom.find({ userId: req.user.sub });

    const unreadCounts = await Promise.all(rooms.map(async (room) => {
        const count = await Message.countDocuments({
            room: room.roomId,
            'sender.userId': { $ne: req.user.sub },
            'readBy.userId': { $ne: req.user.sub },
        });
        return { roomId: room.roomId, count };
    }));

    res.success(unreadCounts.filter(r => r.count > 0));
}));

// ========== 已读/未读状态 ==========
// 每条消息显示 "Alice, Bob 已读"
// 消息详情中:
msg.readBy = [
    { userId: "u1", readAt: "2024-01-01T12:00:00Z" },
    { userId: "u2", readAt: "2024-01-01T12:01:00Z" },
];

// 获取未读用户 (房间总人数 - 已读数)
function getUnreadUsers(message, roomMemberCount) {
    return roomMemberCount - message.readBy.length;
}

// ========== 最近消息列表 ==========
app.get('/api/chat/rooms', authenticate, asyncHandler(async (req, res) => {
    const userRooms = await UserRoom.find({ userId: req.user.sub })
        .populate('room')
        .sort({ lastActivity: -1 })
        .lean();

    // 取每个房间最新消息
    const roomsWithLastMsg = await Promise.all(userRooms.map(async (ur) => {
        const lastMsg = await Message.findOne({ room: ur.room._id })
            .sort({ createdAt: -1 })
            .lean();

        const unread = await Message.countDocuments({
            room: ur.room._id,
            'sender.userId': { $ne: req.user.sub },
            'readBy.userId': { $ne: req.user.sub },
        });

        return { ...ur, lastMessage: lastMsg, unreadCount: unread };
    }));

    res.success(roomsWithLastMsg);
}));
```


## 客户端聊天实现


```
// ========== 客户端聊天 ==========
// chat.js (前端):

class ChatClient {
    constructor(token) {
        this.socket = io('/chat', {
            auth: { token },
            transports: ['websocket'],
        });
        this.currentRoom = null;
        this.setupListeners();
    }

    setupListeners() {
        this.socket.on('connect', () => {
            console.log('Chat connected');
            if (this.currentRoom) {
                this.joinRoom(this.currentRoom);
            }
        });

        this.socket.on('message:new', (message) => {
            this.renderMessage(message);
            this.scrollToBottom();
        });

        this.socket.on('typing', ({ userId, name }) => {
            this.showTypingIndicator(name);
        });

        this.socket.on('typing:stop', ({ userId }) => {
            this.hideTypingIndicator(userId);
        });

        this.socket.on('user:joined', ({ name }) => {
            this.showSystemMessage(`${name} joined`);
        });

        this.socket.on('connect_error', (err) => {
            console.error('Chat error:', err.message);
        });
    }

    joinRoom(roomId) {
        this.currentRoom = roomId;
        this.socket.emit('room:join', { roomId }, (response) => {
            this.renderMessages(response.messages);
        });
    }

    sendMessage(content) {
        if (!this.currentRoom || !content.trim()) return;

        this.socket.emit('message:send', {
            roomId: this.currentRoom,
            content: content.trim(),
        }, (response) => {
            if (!response.success) {
                this.showError('Failed to send message');
            }
        });
    }

    startTyping() {
        if (!this.currentRoom) return;
        this.socket.emit('typing:start', { roomId: this.currentRoom });

        // 停止输入 2 秒后自动取消
        clearTimeout(this.typingTimeout);
        this.typingTimeout = setTimeout(() => {
            this.socket.emit('typing:stop', { roomId: this.currentRoom });
        }, 2000);
    }

    // ... renderMessage, showTypingIndicator, etc.
}

// ========== 使用 ==========
const chat = new ChatClient('jwt-token');
chat.joinRoom('room-123');

// 发送消息按钮:
document.getElementById('send-btn').addEventListener('click', () => {
    const input = document.getElementById('msg-input');
    chat.sendMessage(input.value);
    input.value = '';
});

// 输入监听:
document.getElementById('msg-input').addEventListener('input', () => {
    chat.startTyping();
});
```


## 消息历史与加载更多


```
// ========== 消息历史加载 ==========
// 服务端:
app.get('/api/chat/rooms/:roomId/messages', authenticate, asyncHandler(async (req, res) => {
    const { roomId } = req.params;
    const { before, limit = 50 } = req.query;

    const query = { room: roomId };
    if (before) {
        query.createdAt = { $lt: new Date(before) };
    }

    const messages = await Message.find(query)
        .sort({ createdAt: -1 })
        .limit(parseInt(limit) + 1)
        .lean();

    const hasMore = messages.length > parseInt(limit);
    if (hasMore) messages.pop();

    res.success({
        messages: messages.reverse(),
        hasMore,
        nextCursor: messages.length > 0 ? messages[0].createdAt : null,
    });
}));

// 客户端加载更多:
chat.loadMore = async function() {
    if (this.loading || !this.hasMore) return;
    this.loading = true;

    const res = await fetch(
        `/api/chat/rooms/${this.currentRoom}/messages?before=${this.oldestTimestamp}&limit=50`,
        { headers: { Authorization: `Bearer ${this.token}` } }
    ).then(r => r.json());

    if (res.success) {
        this.prependMessages(res.data.messages);
        this.hasMore = res.data.hasMore;
        this.oldestTimestamp = res.data.nextCursor;
    }

    this.loading = false;
};

// 滚动到顶部加载:
chatContainer.addEventListener('scroll', () => {
    if (chatContainer.scrollTop === 0) {
        chat.loadMore();
    }
});

// ========== 系统消息 ==========
// 加入/离开通知:
socket.on('user:joined', ({ name }) => {
    addSystemMessage(`${name} joined the room`);
});

socket.on('user:left', ({ name }) => {
    addSystemMessage(`${name} left the room`);
});

// 输入提示 (防抖):
let typingTimeout;
socket.on('typing', ({ name }) => {
    showTypingIndicator(name);
    clearTimeout(typingTimeout);
    typingTimeout = setTimeout(hideTypingIndicator, 3000);
});
```


> **Note:** 💡 聊天实现要点: MongoDB 持久化消息; Socket.io 实时收发; 房间管理; 已读回执 (readBy 数组); 未读数统计; 输入状态指示 (防抖); 消息历史分页加载; @提及通知; 系统消息 (加入/离开); 断开重连恢复房间; TTL 索引自动清理旧消息。


## 练习


<!-- Converted from: 27_Socket.io 聊天实现.html -->
