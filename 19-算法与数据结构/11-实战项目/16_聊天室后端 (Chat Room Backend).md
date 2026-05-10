# 聊天室后端 (Chat Room Backend)

## 项目需求与功能分析

实时聊天是 WebSocket 的经典应用场景。本项目使用 Python 实现一个支持多人聊天室的后端服务器，深入理解 WebSocket 通信协议和实时消息推送机制。

### 核心功能

- WebSocket 实时双向通信
- 多聊天室管理（创建、加入、离开）
- 在线用户列表
- 消息广播和私信
- 用户昵称管理
- 消息历史记录

### 技术方案

- 使用 Python `asyncio` + `websockets` 库
- JSON 消息协议
- 异步事件驱动架构

## 核心算法原理

### WebSocket 协议

WebSocket 是基于 TCP 的全双工通信协议：
- 通过 HTTP 升级握手建立连接
- 建立后保持长连接
- 支持文本和二进制消息
- 服务端可主动推送消息

### 消息广播

当用户在聊天室发送消息时，服务器将消息转发给同一聊天室的所有在线用户。

## 完整代码实现

```python
import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Dict, Set, Optional, List
from collections import defaultdict


@dataclass
class Message:
    type: str  # 'chat', 'system', 'private'
    sender: str
    content: str
    room: str = ""
    timestamp: float = field(default_factory=time.time)
    target: str = ""  # 私信目标

    def to_json(self):
        return json.dumps({
            'type': self.type,
            'sender': self.sender,
            'content': self.content,
            'room': self.room,
            'timestamp': self.timestamp,
            'target': self.target,
        })

    @classmethod
    def from_json(cls, data):
        d = json.loads(data) if isinstance(data, str) else data
        return cls(**d)


class ChatRoom:
    """聊天室"""

    def __init__(self, name: str):
        self.name = name
        self.members: Set[str] = set()
        self.history: List[Message] = []
        self.max_history = 100

    def add_member(self, nickname: str):
        self.members.add(nickname)

    def remove_member(self, nickname: str):
        self.members.discard(nickname)

    def add_message(self, msg: Message):
        self.history.append(msg)
        if len(self.history) > self.max_history:
            self.history = self.history[-self.max_history:]

    def is_empty(self):
        return len(self.members) == 0


class ChatServer:
    """聊天服务器（模拟，不依赖 WebSocket 库）"""

    def __init__(self):
        self.rooms: Dict[str, ChatRoom] = {}
        self.users: Dict[str, str] = {}  # conn_id -> nickname
        self.user_rooms: Dict[str, str] = {}  # nickname -> room
        self.connections: Dict[str, object] = {}  # conn_id -> writer
        self.default_room = "大厅"
        self.rooms[self.default_room] = ChatRoom(self.default_room)

    def create_room(self, room_name: str) -> bool:
        if room_name in self.rooms:
            return False
        self.rooms[room_name] = ChatRoom(room_name)
        return True

    async def connect(self, conn_id: str, writer):
        """用户连接"""
        self.connections[conn_id] = writer
        self.users[conn_id] = f"匿名{conn_id[:4]}"
        await self.join_room(conn_id, self.default_room)

    async def disconnect(self, conn_id: str):
        """用户断开"""
        nickname = self.users.get(conn_id, "")
        room = self.user_rooms.get(nickname, "")
        if room and room in self.rooms:
            self.rooms[room].remove_member(nickname)
            await self.broadcast(room, Message(
                'system', '系统', f"{nickname} 离开了聊天室", room
            ), exclude=nickname)
        self.users.pop(conn_id, None)
        self.user_rooms.pop(nickname, None)
        self.connections.pop(conn_id, None)

    async def set_nickname(self, conn_id: str, nickname: str) -> bool:
        """设置昵称"""
        if nickname in self.user_rooms:
            return False
        old_name = self.users.get(conn_id, "")
        room = self.user_rooms.get(old_name, "")
        self.users[conn_id] = nickname
        if old_name in self.user_rooms:
            del self.user_rooms[old_name]
            self.user_rooms[nickname] = room
            if room in self.rooms:
                self.rooms[room].remove_member(old_name)
                self.rooms[room].add_member(nickname)
        return True

    async def join_room(self, conn_id: str, room_name: str):
        """加入聊天室"""
        nickname = self.users.get(conn_id, "")
        # 离开旧房间
        old_room = self.user_rooms.get(nickname, "")
        if old_room and old_room in self.rooms:
            self.rooms[old_room].remove_member(nickname)
            await self.broadcast(old_room, Message(
                'system', '系统', f"{nickname} 离开了聊天室", old_room
            ), exclude=nickname)

        # 加入新房间
        if room_name not in self.rooms:
            self.create_room(room_name)
        self.rooms[room_name].add_member(nickname)
        self.user_rooms[nickname] = room_name

        # 发送历史消息
        await self._send(conn_id, Message(
            'system', '系统',
            f"欢迎来到 {room_name}! 在线: {', '.join(self.rooms[room_name].members)}",
            room_name
        ))
        # 广播加入消息
        await self.broadcast(room_name, Message(
            'system', '系统', f"{nickname} 加入了聊天室", room_name
        ), exclude=nickname)

    async def handle_message(self, conn_id: str, raw_data: str):
        """处理客户端消息"""
        try:
            data = json.loads(raw_data)
        except json.JSONDecodeError:
            return

        msg_type = data.get('type', '')
        nickname = self.users.get(conn_id, "")

        if msg_type == 'nick':
            new_name = data.get('content', '')
            success = await self.set_nickname(conn_id, new_name)
            await self._send(conn_id, Message(
                'system', '系统',
                f"昵称已设置为 {new_name}" if success else "昵称已被占用"
            ))

        elif msg_type == 'join':
            room = data.get('content', '')
            await self.join_room(conn_id, room)

        elif msg_type == 'create':
            room = data.get('content', '')
            if self.create_room(room):
                await self._send(conn_id, Message(
                    'system', '系统', f"聊天室 '{room}' 已创建"
                ))
            else:
                await self._send(conn_id, Message(
                    'system', '系统', f"聊天室 '{room}' 已存在"
                ))

        elif msg_type == 'chat':
            room = self.user_rooms.get(nickname, self.default_room)
            msg = Message('chat', nickname, data.get('content', ''), room)
            self.rooms[room].add_message(msg)
            await self.broadcast(room, msg)

        elif msg_type == 'private':
            target = data.get('target', '')
            msg = Message('private', nickname, data.get('content', ''), target=target)
            await self.send_to_user(target, msg)
            await self._send(conn_id, msg)  # 回显

        elif msg_type == 'list':
            room = self.user_rooms.get(nickname, self.default_room)
            members = list(self.rooms[room].members) if room in self.rooms else []
            await self._send(conn_id, Message(
                'system', '系统',
                f"在线用户 ({len(members)}): {', '.join(members)}", room
            ))

        elif msg_type == 'rooms':
            room_list = [f"{r.name} ({len(r.members)}人)" for r in self.rooms.values()]
            await self._send(conn_id, Message(
                'system', '系统', f"聊天室: {', '.join(room_list)}"
            ))

    async def broadcast(self, room_name: str, msg: Message, exclude: str = ""):
        """广播消息到聊天室"""
        if room_name not in self.rooms:
            return
        room = self.rooms[room_name]
        for conn_id, nick in self.users.items():
            if nick in room.members and nick != exclude:
                await self._send(conn_id, msg)

    async def send_to_user(self, nickname: str, msg: Message):
        """发送私信"""
        for conn_id, nick in self.users.items():
            if nick == nickname:
                await self._send(conn_id, msg)
                return

    async def _send(self, conn_id: str, msg: Message):
        """发送消息到指定连接"""
        writer = self.connections.get(conn_id)
        if writer:
            data = msg.to_json() + '\n'
            writer.write(data.encode())
            await writer.drain()


class ChatClient:
    """聊天客户端（用于测试）"""

    def __init__(self, server: ChatServer, nickname: str):
        self.server = server
        self.nickname = nickname
        self.received: List[Message] = []
        self.conn_id = f"conn_{id(self)}"

    async def connect(self):
        self.server.users[self.conn_id] = self.nickname
        self.server.user_rooms[self.nickname] = self.server.default_room
        self.server.rooms[self.server.default_room].add_member(self.nickname)

    async def send(self, content: str, msg_type: str = 'chat'):
        data = json.dumps({'type': msg_type, 'content': content})
        await self.server.handle_message(self.conn_id, data)

    def get_messages(self):
        return self.received


def demo():
    """演示聊天室功能（使用 asyncio）"""
    async def run():
        server = ChatServer()

        # 模拟两个客户端
        client1 = ChatClient(server, "Alice")
        client2 = ChatClient(server, "Bob")
        await client1.connect()
        await client2.connect()

        await client1.send("大家好！")
        await client2.send("你好 Alice！")
        await client1.send("/rooms", 'rooms')
        await client1.send("/list", 'list')

    asyncio.run(run())


if __name__ == '__main__':
    demo()
```

## 测试用例

```python
import unittest

class TestChatServer(unittest.TestCase):
    def setUp(self):
        self.server = ChatServer()

    def test_create_room(self):
        self.assertTrue(self.server.create_room("技术"))
        self.assertFalse(self.server.create_room("技术"))
        self.assertIn("技术", self.server.rooms)

    def test_nickname(self):
        self.server.users["c1"] = "A"
        self.server.user_rooms["A"] = "大厅"
        self.assertEqual(self.server.users["c1"], "A")

    def test_message_creation(self):
        msg = Message('chat', 'Alice', 'hello', '大厅')
        data = json.loads(msg.to_json())
        self.assertEqual(data['sender'], 'Alice')
        self.assertEqual(data['content'], 'hello')

    def test_room_members(self):
        room = ChatRoom("test")
        room.add_member("A"); room.add_member("B")
        self.assertEqual(len(room.members), 2)
        room.remove_member("A")
        self.assertEqual(len(room.members), 1)

if __name__ == '__main__':
    unittest.main()
```

## 扩展方向

1. **真实 WebSocket**：集成 `websockets` 或 `Socket.IO` 库
2. **消息持久化**：使用 SQLite / Redis 存储聊天记录
3. **文件传输**：支持图片和文件分享
4. **用户认证**：登录注册和 Token 验证
5. **消息已读**：消息送达和已读回执
6. **群组管理**：管理员、踢人、禁言
7. **消息加密**：端到端加密消息
