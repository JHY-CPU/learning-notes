# gRPC 流式传输

## 流式传输概述

gRPC 支持四种通信模式，其中三种涉及流式传输。流式传输允许在单个 RPC 调用中发送或接收多个消息，适用于实时数据推送、大文件传输和双向通信等场景。

## 服务端流式传输

服务端流式 RPC 中，客户端发送一个请求，服务端返回一个消息流。

### .proto 定义

```protobuf
syntax = "proto3";

package notification;

message SubscribeRequest {
  string user_id = 1;
  repeated string topics = 2;
}

message Notification {
  string id = 1;
  string topic = 2;
  string title = 3;
  string body = 4;
  int64 timestamp = 5;
}

service NotificationService {
  // 服务端流：客户端订阅后，服务端持续推送通知
  rpc Subscribe(SubscribeRequest) returns (stream Notification);
}
```

### Python 服务端实现

```python
# server/notification_server.py
import grpc
import time
import uuid
from concurrent import futures
import notification_pb2
import notification_pb2_grpc
import threading
import queue


class NotificationServiceServicer(
    notification_pb2_grpc.NotificationServiceServicer
):
    def __init__(self):
        self.subscribers = {}  # user_id -> queue
        self.lock = threading.Lock()

    def Subscribe(self, request, context):
        """服务端流式 RPC"""
        user_id = request.user_id
        topics = list(request.topics)
        print(f"用户 {user_id} 订阅了: {topics}")

        # 为该用户创建消息队列
        msg_queue = queue.Queue()
        with self.lock:
            self.subscribers[user_id] = msg_queue

        try:
            # 持续从队列中获取消息并发送
            while context.is_active():
                try:
                    notification = msg_queue.get(timeout=1)
                    yield notification  # 关键：使用 yield 返回流式数据
                except queue.Empty:
                    continue
        finally:
            # 清理订阅
            with self.lock:
                self.subscribers.pop(user_id, None)
            print(f"用户 {user_id} 已取消订阅")

    def broadcast(self, topic, title, body):
        """广播通知给订阅者"""
        notification = notification_pb2.Notification(
            id=str(uuid.uuid4()),
            topic=topic,
            title=title,
            body=body,
            timestamp=int(time.time())
        )
        with self.lock:
            for uid, q in self.subscribers.items():
                q.put(notification)
```

### Python 客户端实现

```python
# client/notification_client.py
import grpc
import notification_pb2
import notification_pb2_grpc


def subscribe_notifications(user_id, topics):
    """客户端接收服务端流"""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = notification_pb2_grpc.NotificationServiceStub(channel)

        request = notification_pb2.SubscribeRequest(
            user_id=user_id,
            topics=topics
        )

        # 调用流式 RPC，返回迭代器
        notifications = stub.Subscribe(request, timeout=300)

        print(f"开始接收通知...")
        try:
            for notification in notifications:
                print(f"[{notification.topic}] {notification.title}")
                print(f"  {notification.body}")
                print(f"  时间: {notification.timestamp}")
                print()
        except grpc.RpcError as e:
            print(f"流结束: {e.code()} - {e.details()}")


if __name__ == "__main__":
    subscribe_notifications("user-001", ["news", "alerts"])
```

### Go 服务端实现

```go
// server/notification_server.go
package main

import (
    "fmt"
    "log"
    "sync"
    "time"

    pb "example.com/project/gen"
    "google.golang.org/grpc/codes"
    "google.golang.org/grpc/status"
)

type NotificationServer struct {
    pb.UnimplementedNotificationServiceServer
    mu          sync.RWMutex
    subscribers map[string]chan *pb.Notification
}

func (s *NotificationServer) Subscribe(
    req *pb.SubscribeRequest,
    stream pb.NotificationService_SubscribeServer,
) error {
    userID := req.UserId
    log.Printf("用户 %s 订阅了: %v", userID, req.Topics)

    // 创建通知通道
    ch := make(chan *pb.Notification, 100)
    s.mu.Lock()
    s.subscribers[userID] = ch
    s.mu.Unlock()

    defer func() {
        s.mu.Lock()
        delete(s.subscribers, userID)
        s.mu.Unlock()
        close(ch)
    }()

    // 从通道中读取通知并发送
    for {
        select {
        case <-stream.Context().Done():
            return status.Error(codes.Canceled, "客户端断开连接")
        case notification, ok := <-ch:
            if !ok {
                return nil
            }
            if err := stream.Send(notification); err != nil {
                return err
            }
        }
    }
}
```

## 客户端流式传输

客户端流式 RPC 中，客户端发送一个消息流，服务端接收后返回单个响应。

### .proto 定义

```protobuf
syntax = "proto3";

package upload;

message FileChunk {
  string filename = 1;
  bytes data = 2;
  int32 sequence = 3;
  bool is_last = 4;
}

message UploadResponse {
  bool success = 1;
  string message = 2;
  int64 total_bytes = 3;
  string file_hash = 4;
}

service UploadService {
  // 客户端流：客户端上传文件分块
  rpc Upload(stream FileChunk) returns (UploadResponse);
}
```

### Python 服务端实现

```python
# server/upload_server.py
import grpc
import hashlib
from concurrent import futures
import upload_pb2
import upload_pb2_grpc


class UploadServiceServicer(upload_pb2_grpc.UploadServiceServicer):
    def Upload(self, request_iterator, context):
        """客户端流式 RPC"""
        total_bytes = 0
        filename = ""
        hasher = hashlib.sha256()

        for chunk in request_iterator:
            filename = chunk.filename
            total_bytes += len(chunk.data)
            hasher.update(chunk.data)

            print(f"接收分块 {chunk.sequence}: {len(chunk.data)} 字节")

            # 模拟处理延迟
            import time
            time.sleep(0.01)

        file_hash = hasher.hexdigest()
        print(f"文件 {filename} 上传完成，共 {total_bytes} 字节")

        return upload_pb2.UploadResponse(
            success=True,
            message=f"文件上传成功",
            total_bytes=total_bytes,
            file_hash=file_hash
        )
```

### Python 客户端实现

```python
# client/upload_client.py
import grpc
import upload_pb2
import upload_pb2_grpc

CHUNK_SIZE = 1024 * 64  # 64KB 分块


def generate_chunks(filepath):
    """生成文件分块的生成器"""
    filename = filepath.split("/")[-1]
    sequence = 0

    with open(filepath, "rb") as f:
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                # 发送最后一个空分块标记
                yield upload_pb2.FileChunk(
                    filename=filename,
                    data=b"",
                    sequence=sequence,
                    is_last=True
                )
                break

            is_last = len(data) < CHUNK_SIZE
            yield upload_pb2.FileChunk(
                filename=filename,
                data=data,
                sequence=sequence,
                is_last=is_last
            )
            sequence += 1


def upload_file(filepath):
    """客户端流式上传"""
    with grpc.insecure_channel("localhost:50051") as channel:
        stub = upload_pb2_grpc.UploadServiceStub(channel)

        response = stub.Upload(generate_chunks(filepath))
        print(f"上传结果: {response.message}")
        print(f"总字节数: {response.total_bytes}")
        print(f"文件哈希: {response.file_hash}")


if __name__ == "__main__":
    upload_file("/path/to/large-file.zip")
```

### Go 客户端实现

```go
// client/upload_client.go
package main

import (
    "context"
    "io"
    "log"
    "os"

    pb "example.com/project/gen"
    "google.golang.org/grpc"
)

const chunkSize = 64 * 1024 // 64KB

func uploadFile(client pb.UploadServiceClient, filepath string) error {
    ctx := context.Background()

    // 创建客户端流
    stream, err := client.Upload(ctx)
    if err != nil {
        return err
    }

    file, err := os.Open(filepath)
    if err != nil {
        return err
    }
    defer file.Close()

    buf := make([]byte, chunkSize)
    sequence := int32(0)

    for {
        n, err := file.Read(buf)
        if err == io.EOF {
            // 发送结束标记
            stream.Send(&pb.FileChunk{
                Filename: filepath,
                Data:     []byte{},
                Sequence: sequence,
                IsLast:   true,
            })
            break
        }
        if err != nil {
            return err
        }

        stream.Send(&pb.FileChunk{
            Filename: filepath,
            Data:     buf[:n],
            Sequence: sequence,
            IsLast:   false,
        })
        sequence++
    }

    // 接收响应
    resp, err := stream.CloseAndRecv()
    if err != nil {
        return err
    }

    log.Printf("上传成功: %s, %d 字节", resp.Message, resp.TotalBytes)
    return nil
}
```

## 双向流式传输

双向流式 RPC 中，客户端和服务端都可以独立地发送消息流。

### .proto 定义

```protobuf
syntax = "proto3";

package chat;

message ChatMessage {
  string user_id = 1;
  string room_id = 2;
  string content = 3;
  int64 timestamp = 4;
  enum Type {
    TEXT = 0;
    IMAGE = 1;
    SYSTEM = 2;
  }
  Type type = 5;
}

service ChatService {
  // 双向流：实时聊天
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

### Python 服务端实现

```python
# server/chat_server.py
import grpc
import time
import threading
from concurrent import futures
import chat_pb2
import chat_pb2_grpc
from collections import defaultdict


class ChatServiceServicer(chat_pb2_grpc.ChatServiceServicer):
    def __init__(self):
        # room_id -> list of (user_id, queue)
        self.rooms = defaultdict(list)
        self.lock = threading.Lock()

    def Chat(self, request_iterator, context):
        """双向流式 RPC"""
        import queue
        user_id = None
        room_id = None
        msg_queue = queue.Queue()

        # 接收线程：从客户端流中接收消息
        def receive():
            nonlocal user_id, room_id
            for msg in request_iterator:
                user_id = msg.user_id
                room_id = msg.room_id

                # 首次收到消息时加入房间
                if room_id and user_id:
                    with self.lock:
                        if (user_id, msg_queue) not in self.rooms[room_id]:
                            self.rooms[room_id].append((user_id, msg_queue))

                # 广播给同房间的其他用户
                with self.lock:
                    for uid, q in self.rooms.get(room_id, []):
                        if uid != user_id:
                            q.put(msg)

        recv_thread = threading.Thread(target=receive, daemon=True)
        recv_thread.start()

        # 发送消息给当前客户端
        try:
            while context.is_active():
                try:
                    msg = msg_queue.get(timeout=1)
                    yield msg
                except queue.Empty:
                    continue
        finally:
            # 清理：从房间中移除
            if room_id and user_id:
                with self.lock:
                    self.rooms[room_id] = [
                        (uid, q) for uid, q in self.rooms[room_id]
                        if uid != user_id
                    ]
```

### Python 客户端实现

```python
# client/chat_client.py
import grpc
import time
import threading
import chat_pb2
import chat_pb2_grpc


class ChatClient:
    def __init__(self, user_id, room_id):
        self.user_id = user_id
        self.room_id = room_id
        self.channel = grpc.insecure_channel("localhost:50051")
        self.stub = chat_pb2_grpc.ChatServiceStub(self.channel)

    def message_generator(self):
        """生成要发送的消息流"""
        # 发送加入消息
        yield chat_pb2.ChatMessage(
            user_id=self.user_id,
            room_id=self.room_id,
            content=f"{self.user_id} 加入了聊天室",
            timestamp=int(time.time()),
            type=chat_pb2.ChatMessage.SYSTEM
        )

        # 从控制台读取消息
        while True:
            try:
                content = input()
                if content.lower() == "/quit":
                    break
                yield chat_pb2.ChatMessage(
                    user_id=self.user_id,
                    room_id=self.room_id,
                    content=content,
                    timestamp=int(time.time()),
                    type=chat_pb2.ChatMessage.TEXT
                )
            except EOFError:
                break

    def run(self):
        """启动聊天客户端"""
        responses = self.stub.Chat(self.message_generator())

        print(f"已加入聊天室 {self.room_id}，输入消息开始聊天（/quit 退出）")

        # 接收消息的线程
        def receive_messages():
            for msg in responses:
                type_str = {0: "消息", 1: "图片", 2: "系统"}[msg.type]
                print(f"[{type_str}] {msg.user_id}: {msg.content}")

        recv_thread = threading.Thread(target=receive_messages, daemon=True)
        recv_thread.start()

        # 等待退出
        recv_thread.join()
        self.channel.close()


if __name__ == "__main__":
    client = ChatClient("user-001", "room-1")
    client.run()
```

### Go 双向流实现

```go
// Go 服务端双向流实现
func (s *ChatServer) Chat(
    stream pb.ChatService_ChatServer,
) error {
    msgChan := make(chan *pb.ChatMessage, 100)
    userID := ""
    roomID := ""

    // 接收协程
    go func() {
        for {
            msg, err := stream.Recv()
            if err == io.EOF {
                close(msgChan)
                return
            }
            if err != nil {
                close(msgChan)
                return
            }
            userID = msg.UserId
            roomID = msg.RoomId
            s.broadcast(roomID, userID, msg)
        }
    }()

    // 发送循环
    for {
        select {
        case <-stream.Context().Done():
            s.leaveRoom(roomID, userID)
            return nil
        case msg, ok := <-s.getMessageChan(stream, roomID, userID):
            if !ok {
                return nil
            }
            if err := stream.Send(msg); err != nil {
                return err
            }
        }
    }
}
```

## 流式传输的最佳实践

### 1. 背压处理

```python
# 服务端控制发送速度
def StreamData(self, request, context):
    for item in large_dataset:
        if not context.is_active():
            break  # 客户端已断开
        yield item
        time.sleep(0.001)  # 控制发送速率
```

### 2. 错误处理

```python
# 流中的错误处理
def stream_with_error_handling(stub, request):
    try:
        for response in stub.Subscribe(request):
            process(response)
    except grpc.RpcError as e:
        if e.code() == grpc.StatusCode.CANCELLED:
            print("流被取消")
        elif e.code() == grpc.StatusCode.UNAVAILABLE:
            print("服务不可用")
        else:
            print(f"错误: {e.code()} - {e.details()}")
```

### 3. 取消流

```go
// Go 中取消流
ctx, cancel := context.WithCancel(context.Background())
stream, err := client.Subscribe(ctx, req)

// 在需要时取消
cancel()
```

## 小结

gRPC 的流式传输提供了强大的实时通信能力。服务端流适用于数据推送，客户端流适用于批量上传，双向流适用于实时交互。合理使用流式 RPC 可以构建高效的实时系统，但需要注意背压控制、错误处理和资源清理。
