# gRPC 详解

## 一、gRPC 特点

```
gRPC 特点:
├── 高性能 - HTTP/2 + Protobuf
├── 强类型 - .proto 文件定义
├── 流式通信 - 支持流式传输
├── 多语言 - 跨语言支持
└── 双向流 - 客户端/服务端流
```

## 二、Proto 定义

```protobuf
syntax = "proto3";

service UserService {
    rpc GetUser(GetUserRequest) returns (User);
    rpc ListUsers(ListUsersRequest) returns (stream User);
}

message GetUserRequest {
    int64 id = 1;
}

message User {
    int64 id = 1;
    string name = 2;
    string email = 3;
}
```

## 三、工作原理

gRPC 基于 HTTP/2 协议和 Protobuf 序列化。Protobuf 将 .proto 文件编译为各语言的客户端和服务端代码，提供强类型接口。HTTP/2 支持多路复用（一个 TCP 连接上并发多个请求）、头部压缩、服务端推送。gRPC 支持四种通信模式：一元 RPC（请求-响应）、服务端流式（客户端一次请求，服务端返回流）、客户端流式（客户端发送流，服务端一次响应）、双向流式（双方同时发送流）。

## 四、优缺点

**优点：**
- 性能远高于 REST（Protobuf 序列化体积小 5-10 倍，速度快 3-5 倍）
- 强类型接口，编译期发现问题
- 流式通信适合大数据传输和实时场景

**缺点：**
- 调试困难，无法用浏览器直接测试
- Proto 文件版本管理需要团队规范
- 不适合对外暴露 API（前端无法直接调用）

## 五、最佳实践**

1. Proto 文件统一管理在独立 Git 仓库，CI 自动生成各语言代码
2. 内部服务间调用优先使用 gRPC，对外暴露用 REST 转接
3. 流式通信用于日志采集、实时数据推送等场景
4. gRPC 拦截器实现认证、日志、链路追踪等横切逻辑

## 六、常见陷阱

1. **Proto 文件不兼容变更**（删除字段、修改类型），导致已部署服务通信失败
2. **gRPC 负载均衡需要特殊配置**（因为 HTTP/2 长连接），默认不轮询
3. **流式通信未处理背压**，消费方处理不过来导致内存溢出
4. **gRPC 超时和重试未配置**，与 REST 调用一样的问题
