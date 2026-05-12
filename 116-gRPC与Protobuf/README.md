# gRPC 与 Protocol Buffers 学习笔记

本模块系统整理了 gRPC 框架与 Protocol Buffers 序列化协议的核心知识，从基础语法到微服务实战，覆盖完整的学习路径。

## 目录结构

| 序号 | 文件 | 主题 |
|------|------|------|
| 01 | `01_Protocol_Buffers概述.md` | Protobuf 简介、二进制序列化、对比 JSON/XML |
| 02 | `02_Protobuf语法与数据类型.md` | proto3 语法、标量类型、枚举、嵌套消息、oneof、map |
| 03 | `03_Protobuf代码生成.md` | protoc 编译器、多语言代码生成、描述符 |
| 04 | `04_gRPC概述与架构.md` | gRPC 架构、HTTP/2、四种服务类型、对比 REST |
| 05 | `05_gRPC服务定义.md` | .proto 服务定义、四种调用模式 |
| 06 | `06_gRPC_Server实现.md` | Python/Go 服务端实现、拦截器 |
| 07 | `07_gRPC_Client实现.md` | Python/Go 客户端、通道配置、元数据 |
| 08 | `08_gRPC流式传输.md` | 服务端/客户端/双向流式传输 |
| 09 | `09_gRPC拦截器与中间件.md` | 服务端/客户端拦截器、认证、日志、指标 |
| 10 | `10_gRPC错误处理与重试.md` | 状态码、错误详情、重试策略、健康检查 |
| 11 | `11_gRPC安全.md` | TLS/SSL、双向 TLS、令牌认证、RBAC |
| 12 | `12_gRPC在微服务中的应用.md` | 服务发现、负载均衡、截止时间传播、链路追踪 |

## 学习路径建议

1. **基础阶段**：先学习 01-03，掌握 Protocol Buffers 的核心概念和使用方法
2. **框架阶段**：学习 04-05，理解 gRPC 的架构设计和服务定义方式
3. **实践阶段**：学习 06-08，动手实现 gRPC 服务端和客户端
4. **进阶阶段**：学习 09-12，掌握拦截器、安全机制和微服务集成

## 环境准备

```bash
# 安装 Protocol Buffers 编译器
# macOS
brew install protobuf
# Ubuntu
sudo apt install protobuf-compiler

# Python 依赖
pip install grpcio grpcio-tools protobuf

# Go 依赖
go install google.golang.org/protobuf/cmd/protoc-gen-go@latest
go install google.golang.org/grpc/cmd/protoc-gen-go-grpc@latest

# 验证安装
protoc --version
```

## 前置知识

- 基本的网络编程概念（HTTP、TCP）
- 至少掌握一门编程语言（Python 或 Go 推荐）
- 了解客户端-服务器架构
- 基本的命令行操作
