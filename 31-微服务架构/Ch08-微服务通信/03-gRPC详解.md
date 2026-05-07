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

## 三、注意事项

1. **gRPC 性能比 REST 高很多**
2. **适合内部服务间调用**
3. **Proto 文件要版本管理**
4. **流式通信适合大数据传输**
5. **调试比 REST 困难**
