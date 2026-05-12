# gRPC 概述与架构

## 什么是 gRPC

gRPC 是 Google 开源的高性能远程过程调用（RPC）框架，基于 HTTP/2 协议传输，默认使用 Protocol Buffers 作为序列化格式。gRPC 支持多种编程语言，可以在任何环境中运行。

gRPC 的名称源自 Google Remote Procedure Call，其中 "g" 代表 Google（也有说法代表 good）。

```protobuf
// 一个简单的 gRPC 服务定义
syntax = "proto3";

package greeter;

message HelloRequest {
  string name = 1;
}

message HelloResponse {
  string message = 1;
}

service Greeter {
  rpc SayHello(HelloRequest) returns (HelloResponse);
}
```

## HTTP/2 传输协议

gRPC 基于 HTTP/2 协议，这带来了多项性能优势：

### HTTP/2 的核心特性

| 特性 | 说明 | gRPC 中的作用 |
|------|------|---------------|
| 多路复用 | 单个连接上并发多个请求 | 减少连接开销 |
| 头部压缩 | HPACK 算法压缩头部 | 减少传输数据量 |
| 服务端推送 | 服务端主动推送数据 | 支持服务端流式传输 |
| 二进制帧 | 数据以二进制帧传输 | 与 Protobuf 配合 |
| 流优先级 | 请求可以设置优先级 | 重要请求优先处理 |

```python
# HTTP/1.1 vs HTTP/2 对比
# HTTP/1.1: 每个请求需要一个 TCP 连接（或复用但有队头阻塞）
# HTTP/2:   单个 TCP 连接上并发多个请求/响应

# 在 gRPC 中，客户端与服务端建立一个 HTTP/2 连接后
# 可以在该连接上同时发起多个 RPC 调用
# 每个 RPC 调用对应 HTTP/2 中的一个 stream（流）
```

## gRPC 的四种服务类型

### 1. 一元 RPC（Unary RPC）

最简单的模式，客户端发送一个请求，服务端返回一个响应。类似于传统的函数调用。

```protobuf
// 一元 RPC
service UserService {
  rpc GetUser(GetUserRequest) returns (GetUserResponse);
  rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
  rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
}
```

### 2. 服务端流式 RPC

客户端发送一个请求，服务端返回一个消息流。适用于服务端需要发送大量数据的场景。

```protobuf
// 服务端流式 RPC
service StockService {
  rpc WatchStock(WatchStockRequest) returns (stream StockPrice);
}
```

### 3. 客户端流式 RPC

客户端发送一个消息流，服务端返回一个响应。适用于客户端需要上传大量数据的场景。

```protobuf
// 客户端流式 RPC
service UploadService {
  rpc UploadFile(stream FileChunk) returns (UploadResponse);
}
```

### 4. 双向流式 RPC

客户端和服务端都可以发送消息流，两个流独立操作。适用于需要实时双向通信的场景。

```protobuf
// 双向流式 RPC
service ChatService {
  rpc Chat(stream ChatMessage) returns (stream ChatMessage);
}
```

## gRPC 与 REST 的对比

| 特性 | gRPC | REST |
|------|------|------|
| 协议 | HTTP/2 | 通常 HTTP/1.1 |
| 序列化 | Protobuf（二进制） | JSON（文本） |
| 性能 | 高（二进制、多路复用） | 中（文本、单连接） |
| 类型安全 | 强类型（.proto 定义） | 弱类型（通常无 Schema） |
| 流式传输 | 原生支持四种模式 | 需要 WebSocket 等 |
| 浏览器支持 | 需要 gRPC-Web 代理 | 原生支持 |
| API 设计 | RPC 风格（动词导向） | RESTful 风格（资源导向） |
| 代码生成 | 自动生成客户端/服务端 | 需要手动实现或使用工具 |
| 可读性 | 二进制不可读 | JSON 人类可读 |
| 生态成熟度 | 相对较新 | 非常成熟 |

```python
# REST 风格的 API 设计
# GET    /api/users/123        # 获取用户
# POST   /api/users            # 创建用户
# PUT    /api/users/123        # 更新用户
# DELETE /api/users/123        # 删除用户

# gRPC 风格的 API 设计
# rpc GetUser(GetUserRequest) returns (GetUserResponse);
# rpc CreateUser(CreateUserRequest) returns (CreateUserResponse);
# rpc UpdateUser(UpdateUserRequest) returns (UpdateUserResponse);
# rpc DeleteUser(DeleteUserRequest) returns (DeleteUserResponse);
```

## gRPC 的核心概念

### 通道（Channel）

通道是客户端连接到服务端的抽象，管理底层的 HTTP/2 连接。

```go
// Go 语言中的通道创建
conn, err := grpc.Dial(
    "localhost:50051",
    grpc.WithTransportCredentials(insecure.NewCredentials()),
)
if err != nil {
    log.Fatal(err)
}
defer conn.Close()

// 使用通道创建客户端存根
client := pb.NewUserServiceClient(conn)
```

```python
# Python 中的通道创建
import grpc

channel = grpc.insecure_channel("localhost:50051")
client = user_pb2_grpc.UserServiceStub(channel)
```

### 存根（Stub）

存根是客户端调用远程方法的本地代理对象，隐藏了底层的网络通信细节。

### 服务端实现

服务端需要实现 `.proto` 中定义的接口，处理客户端的请求。

## gRPC 的适用场景

**非常适合的场景：**

- 微服务之间的内部通信
- 需要高性能的后端系统
- 实时通信（如聊天、游戏）
- 流式数据处理
- 多语言环境的统一 API

**不太适合的场景：**

- 需要浏览器直接访问的公共 API
- 需要人类直接调试的接口
- 简单的 CRUD 应用
- 团队不熟悉 Protobuf

## gRPC 生态系统

| 组件 | 说明 |
|------|------|
| gRPC-Go | Go 语言实现 |
| gRPC-Python | Python 语言实现 |
| gRPC-Java | Java 语言实现 |
| gRPC-Web | 浏览器端支持 |
| grpc-gateway | 将 gRPC 转换为 RESTful API |
| buf | 现代 Protobuf 工具链 |
| grpcurl | 命令行 gRPC 客户端 |
| BloomRPC | gRPC GUI 客户端 |

## 小结

gRPC 是基于 HTTP/2 和 Protobuf 的高性能 RPC 框架，支持一元、服务端流、客户端流和双向流四种通信模式。相比 REST，gRPC 在性能、类型安全和流式传输方面具有明显优势，特别适合微服务内部通信。在后续章节中，我们将深入学习 gRPC 的服务定义和具体实现。
