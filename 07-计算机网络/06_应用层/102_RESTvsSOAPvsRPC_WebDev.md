# RESTvsSOAPvsRPC

## ⚖️ REST vs SOAP vs RPC

三种 API 架构风格对比、SOAP 协议、gRPC、选型指南。

## 三巨头对比
```
// ========== REST vs SOAP vs RPC ==========
//
// 特性          REST              SOAP               RPC (gRPC)
// ───────────────────────────────────────────────────────────────
// 协议         HTTP              HTTP/SMTP/etc       HTTP/2
// 数据格式     JSON/XML          XML                  Protocol Buffers
// 面向         资源              操作 (函数)          操作 (函数)
// 状态         无状态            可有状态             无状态
// 性能         中等              较慢 (XML 大)        快 (二进制)
// 缓存         支持              不支持               不支持
// 浏览器友好   原生              需工具               需 gRPC-web
// 工具生态     Postman/curl      复杂 (WSDL/SOAP)     代码生成
// 学习曲线     低                高                   中
// 类型安全     弱 (JSON 无类型)   强 (XSD Schema)      强 (protobuf)
// 流式         不支持             不支持               支持
//
// ========== 核心差异 ==========
// REST: 操作资源 (CRUD)
//   GET /users/123
//
// SOAP: 调用操作 (远程过程 + 复杂标准)
//   POST /UserService
//   123
//
// gRPC: 调用远程方法 (接口定义 + 高性能)
//   rpc GetUser (GetUserRequest) returns (User);
```
## SOAP 协议详解
```
// ========== SOAP 基础 ==========
// Simple Object Access Protocol
// 基于 XML 的消息协议
// 与传输层无关 (HTTP, SMTP, JMS 均可)
// 企业级标准,复杂但功能完善
//
// ========== SOAP 消息结构 ==========
//
//
//
//     abc123
//
//
//
//       123
//
//
//
//     // 错误信息 (可选)
//
//

// ========== SOAP 相关标准 ==========
// WSDL (Web Services Description Language):
//   描述服务接口的 XML 格式
//   定义: 操作, 消息格式, 绑定, 端点
//
// XSD (XML Schema Definition):
//   定义 XML 数据结构
//   强类型验证
//
// UDDI (Universal Description, Discovery, and Integration):
//   服务注册和发现 (已淘汰)
//
// WS-* 标准栈:
//   WS-Security, WS-AtomicTransaction, WS-ReliableMessaging
//   WS-Addressing, WS-Policy 等
```
## gRPC 详解
```
// ========== gRPC 特点 ==========
// 由 Google 开发 (2016)
// 基于 HTTP/2, 默认使用 Protocol Buffers
// 强类型,代码自动生成
//
// ========== Protocol Buffers ==========
// // user.proto
// syntax = "proto3";
// package user;
//
// message GetUserRequest {
//   int32 id = 1;
// }
//
// message User {
//   int32 id = 1;
//   string name = 2;
//   string email = 3;
// }
//
// service UserService {
//   rpc GetUser (GetUserRequest) returns (User);
//   rpc ListUsers (Empty) returns (stream User);  // 服务器流
// }
//
// ========== gRPC 四种通信模式 ==========
// 1. 一元 RPC (Unary):
//    客户端发一个请求,服务器回一个响应
//    类似传统 HTTP
//
// 2. 服务器流 (Server Streaming):
//    客户端发一个请求,服务器返回流式响应
//    适用: 日志推送, 事件流
//
// 3. 客户端流 (Client Streaming):
//    客户端流式发送,服务器返回一个响应
//    适用: 大文件上传, 批量数据
//
// 4. 双向流 (Bidirectional Streaming):
//    双方独立发送消息
//    适用: 实时聊天, 游戏状态同步

// ========== SOAP vs REST vs gRPC 选型 ==========
// 选 SOAP:
//   金融/银行/保险 (遗留系统)
//   需要 WS-Security/WS-AtomicTransaction
//   B2B 集成,合同式接口
//
// 选 REST:
//   Web API / 移动端 API
//   公开 API (第三方开发者)
//   简单 CRUD 场景
//   需要缓存/CDN
//
// 选 gRPC:
//   微服务间通信
//   高性能/低延迟场景
//   流媒体/实时通信
//   强类型接口需求
```
> **Note**: 📊 选型口诀: 面向资源 → REST, 面向内部高性能 → gRPC, 面向遗留企业 → SOAP。gRPC 在微服务架构中越来越流行,但浏览器端仍需 REST 或 gRPC-Web。

## 实际应用中的 API 选型
```
// ========== 典型场景选型 ==========
//
// 公开 Web API (第三方):
//   REST (JSON) → 最简单,生态最好
//   GitHub API, Stripe API, Twilio API
//
// 内部微服务通信:
//   gRPC → 高性能,类型安全
//   Kubernetes 服务间调用
//
// 移动端 API:
//   REST (JSON) → 通用
//   GraphQL → 减少数据传输量
//
// 实时流式:
//   WebSocket → 双向实时
//   gRPC stream → 高性能流
//
// 遗留系统集成:
//   SOAP → 金融/保险/政府
//   或 REST 适配器包装

// ========== RPC 变体 ==========
// JSON-RPC: 轻量级远程调用
//   {"jsonrpc": "2.0", "method": "subtract",
//    "params": [42, 23], "id": 1}
//
// XML-RPC: SOAP 前身
//
//     subtract
//     42...
//
// Thrift: Facebook 的 RPC 框架
// Avro: Hadoop 生态的 RPC/序列化框架
// RSocket: 响应式 RPC (支持多种传输层)
```
