# gRPC 服务定义

## 在 .proto 文件中定义服务

gRPC 服务通过在 `.proto` 文件中使用 `service` 关键字定义。每个服务包含一组 RPC 方法，每个方法定义了请求和响应的消息类型。

```protobuf
syntax = "proto3";

package user;

// 基本的服务定义
service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
  rpc ListUsers(ListUsersRequest) returns (ListUsersResponse);
}
```

## 四种调用模式详解

### 1. 一元 RPC（Unary RPC）

最基础的调用模式，客户端发送单个请求，服务端返回单个响应。

```protobuf
syntax = "proto3";

package calculator;

// 请求消息
message AddRequest {
  int32 a = 1;
  int32 b = 2;
}

// 响应消息
message AddResponse {
  int32 result = 1;
}

// 一元 RPC：简单的加法运算
service Calculator {
  rpc Add(AddRequest) returns (AddResponse);
  rpc Multiply(MultiplyRequest) returns (MultiplyResponse);
}

message MultiplyRequest {
  int32 a = 1;
  int32 b = 2;
}

message MultiplyResponse {
  int32 result = 1;
}
```

```python
# Python 服务端实现
import grpc
from concurrent import futures
import calculator_pb2
import calculator_pb2_grpc

class CalculatorServicer(calculator_pb2_grpc.CalculatorServicer):
    def Add(self, request, context):
        return calculator_pb2.AddResponse(result=request.a + request.b)

    def Multiply(self, request, context):
        return calculator_pb2.MultiplyResponse(result=request.a * request.b)

# 启动服务
server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
calculator_pb2_grpc.add_CalculatorServicer_to_server(
    CalculatorServicer(), server
)
server.add_insecure_port("[::]:50051")
server.start()
server.wait_for_termination()
```

### 2. 服务端流式 RPC（Server Streaming）

客户端发送单个请求，服务端返回一个消息流。适用于服务端需要持续推送数据的场景。

```protobuf
syntax = "proto3";

package stock;

// 股票价格消息
message StockPrice {
  string symbol = 1;
  double price = 2;
  int64 timestamp = 3;
}

// 订阅请求
message SubscribeRequest {
  repeated string symbols = 1;
}

// 服务端流式 RPC：实时股票价格推送
service StockService {
  // 返回 stream 关键字表示服务端流
  rpc SubscribePrices(SubscribeRequest) returns (stream StockPrice);
}
```

```python
# Python 服务端实现
import grpc
import time
import random
from concurrent import futures
import stock_pb2
import stock_pb2_grpc

class StockServiceServicer(stock_pb2_grpc.StockServiceServicer):
    def SubscribePrices(self, request, context):
        # 持续推送股票价格
        while True:
            for symbol in request.symbols:
                price = stock_pb2.StockPrice(
                    symbol=symbol,
                    price=random.uniform(100, 200),
                    timestamp=int(time.time())
                )
                yield price  # 流式返回
            time.sleep(1)
```

```go
// Go 服务端实现
func (s *server) SubscribePrices(
    req *pb.SubscribeRequest,
    stream pb.StockService_SubscribePricesServer,
) error {
    for {
        for _, symbol := range req.Symbols {
            price := &pb.StockPrice{
                Symbol:    symbol,
                Price:     rand.Float64()*100 + 100,
                Timestamp: time.Now().Unix(),
            }
            if err := stream.Send(price); err != nil {
                return err
            }
        }
        time.Sleep(time.Second)
    }
}
```

### 3. 客户端流式 RPC（Client Streaming）

客户端发送一个消息流，服务端在接收完所有消息后返回单个响应。适用于客户端需要上传大量数据的场景。

```protobuf
syntax = "proto3";

package upload;

// 文件分块
message FileChunk {
  string filename = 1;
  bytes data = 2;
  int32 sequence_number = 3;
  bool is_last = 4;
}

// 上传响应
message UploadResponse {
  bool success = 1;
  string message = 2;
  int64 total_bytes = 3;
}

// 客户端流式 RPC：文件上传
service UploadService {
  // 请求是 stream，响应是普通消息
  rpc Upload(stream FileChunk) returns (UploadResponse);
}
```

```python
# Python 服务端实现
import grpc
from concurrent import futures
import upload_pb2
import upload_pb2_grpc

class UploadServiceServicer(upload_pb2_grpc.UploadServiceServicer):
    def Upload(self, request_iterator, context):
        total_bytes = 0
        filename = ""

        for chunk in request_iterator:
            filename = chunk.filename
            total_bytes += len(chunk.data)
            print(f"接收到分块 {chunk.sequence_number}: {len(chunk.data)} 字节")

        return upload_pb2.UploadResponse(
            success=True,
            message=f"文件 {filename} 上传成功",
            total_bytes=total_bytes
        )
```

```go
// Go 服务端实现
func (s *server) Upload(
    stream pb.UploadService_UploadServer,
) error {
    var totalBytes int64
    var filename string

    for {
        chunk, err := stream.Recv()
        if err == io.EOF {
            // 流结束，返回响应
            return stream.SendAndClose(&pb.UploadResponse{
                Success:     true,
                Message:     fmt.Sprintf("文件 %s 上传成功", filename),
                TotalBytes:  totalBytes,
            })
        }
        if err != nil {
            return err
        }
        filename = chunk.Filename
        totalBytes += int64(len(chunk.Data))
    }
}
```

### 4. 双向流式 RPC（Bidirectional Streaming）

客户端和服务端都可以发送消息流，两个流独立操作。适用于需要实时双向通信的场景。

```protobuf
syntax = "proto3";

package chat;

// 聊天消息
message ChatMessage {
  string user_id = 1;
  string content = 2;
  int64 timestamp = 3;
}

// 双向流式 RPC：实时聊天
service ChatService {
  // 请求和响应都是 stream
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

```python
# Python 服务端实现
import grpc
from concurrent import futures
import chat_pb2
import chat_pb2_grpc
import threading
import time

class ChatServiceServicer(chat_pb2_grpc.ChatServiceServicer):
    def __init__(self):
        self.clients = {}  # user_id -> queue

    def Chat(self, request_iterator, context):
        import queue
        msg_queue = queue.Queue()
        user_id = None

        # 启动接收线程
        def receive_messages():
            nonlocal user_id
            for msg in request_iterator:
                user_id = msg.user_id
                if user_id not in self.clients:
                    self.clients[user_id] = msg_queue
                # 广播消息给其他客户端
                for uid, q in self.clients.items():
                    if uid != user_id:
                        q.put(msg)

        recv_thread = threading.Thread(target=receive_messages)
        recv_thread.start()

        # 发送消息给当前客户端
        while context.is_active():
            try:
                msg = msg_queue.get(timeout=1)
                yield msg
            except queue.Empty:
                continue
```

```go
// Go 服务端实现
func (s *server) Chat(
    stream pb.ChatService_ChatServer,
) error {
    msgChan := make(chan *pb.ChatMessage, 100)

    // 启动接收协程
    go func() {
        for {
            msg, err := stream.Recv()
            if err == io.EOF {
                close(msgChan)
                return
            }
            if err != nil {
                return
            }
            // 广播消息
            s.broadcast(msg)
        }
    }()

    // 发送消息
    for msg := range s.getMessageChan(stream) {
        if err := stream.Send(msg); err != nil {
            return err
        }
    }
    return nil
}
```

## 高级服务定义模式

### 请求和响应中使用 Oneof

```protobuf
syntax = "proto3";

package notification;

message NotificationRequest {
  oneof target {
    string user_id = 1;
    string group_id = 2;
  }
  NotificationPayload payload = 3;
}

message NotificationPayload {
  string title = 1;
  string body = 2;
  map<string, string> data = 3;
}

message NotificationResponse {
  bool success = 1;
  int32 delivered_count = 2;
}

service NotificationService {
  rpc Send(NotificationRequest) returns (NotificationResponse);
}
```

### 使用 Empty 请求或响应

```protobuf
import "google/protobuf/empty.proto";

service HealthService {
  // 无请求参数
  rpc GetStatus(google.protobuf.Empty) returns (HealthStatus);

  // 无返回值
  rpc Ping(google.protobuf.Empty) returns (google.protobuf.Empty);
}
```

### 流式 RPC 中的元数据

```protobuf
// 在实际使用中，流式 RPC 可以通过元数据传递额外信息
// 例如认证令牌、追踪 ID 等
// 这些信息通过 gRPC 的 metadata 机制传递，不需要在 .proto 中定义
```

## 小结

gRPC 服务定义在 `.proto` 文件中，通过 `service` 关键字声明服务，每个方法通过 `rpc` 关键字定义。四种调用模式（一元、服务端流、客户端流、双向流）覆盖了从简单请求到实时双向通信的各种场景。在后续章节中，我们将学习如何实现这些服务定义。
