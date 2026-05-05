# 客户端服务器模型

## 🖥️ 客户端-服务器模型

C/S 架构、请求-响应模式、无状态协议、长连接 vs 短连接。

## C/S 模型概述

客户端-服务器模型是互联网最基本的通信架构，客户端发起请求，服务器处理并返回响应。
```
// ========== 客户端-服务器模型 ==========
//
//  ┌─────────┐   请求(Request)    ┌─────────┐
//  │ 客户端   │ ─────────────────→ │ 服务器   │
//  │ (Browser)│                   │ (Server) │
//  │          │ ←───────────────── │          │
//  └─────────┘   响应(Response)   └─────────┘
//
// ========== 客户端类型 ==========
// Web 浏览器: Chrome/Firefox/Safari
// 移动 App:   iOS/Android 应用
// 桌面应用:   Electron/原生应用
// IoT 设备:   智能家居/传感器
// API 客户端: curl/Postman/代码

// ========== 服务器类型 ==========
// Web 服务器:    Nginx/Apache/Caddy
// 应用服务器:    Node.js/Spring Boot/Django
// 数据库服务器:  MySQL/PostgreSQL/MongoDB
// 文件服务器:    S3/MinIO/NFS
// 缓存服务器:    Redis/Memcached
```
## 请求-响应模式
```
// ========== 典型 HTTP 请求-响应流程 ==========
//
// 请求 (客户端 → 服务器):
//   POST /api/login HTTP/1.1
//   Host: example.com
//   Content-Type: application/json
//   Authorization: Bearer
//   Body: {"username":"admin","password":"***"}
//
// 响应 (服务器 → 客户端):
//   HTTP/1.1 200 OK
//   Content-Type: application/json
//   Set-Cookie: session=abc123
//   Body: {"token":"xxx","expires":3600}
//
// ========== 请求头信息 ==========
// Host:          目标主机名
// User-Agent:    客户端标识
// Accept:        接受的响应类型
// Authorization: 认证信息
// Cookie:        Cookie 数据
// Content-Type:  请求体格式
// Cache-Control: 缓存策略
```
## 无状态协议

HTTP 是无状态协议——每个请求都是独立的，服务器默认不记住之前的请求。
```
// ========== 无状态的含义 ==========
// 优点:
//   - 服务器无需维护客户端状态
//   - 易于水平扩展 (任何服务器都可以处理请求)
//   - 故障恢复简单 (切换服务器不影响)
//
// 缺点:
//   - 每次请求需要重复认证
//   - 无法原生保持用户会话
//
// ========== 维持状态的方式 ==========
// Cookie:     服务器在响应中设置,浏览器自动发送
// Session:    Cookie+服务器存储,基于 Cookie ID
// JWT:        客户端持有加密 token,服务器验证
// 隐藏字段:   表单中的 hidden input
// URL 参数:   查询字符串中带状态标识

// ========== Cookie 工作流程 ==========
// 1. 客户端 POST /login → 服务器验证身份
// 2. 服务器响应 Set-Cookie: session_id=abc123
// 3. 浏览器存储 Cookie
// 4. 后续请求自动携带 Cookie
// 5. 服务器解析 Cookie 识别用户
```
> **Note**: 🔑 无状态让互联网得以大规模扩展——想象一下如果每个服务器都要记住所有用户的购物车，亿级访问量会是什么场景。

## 长连接 vs 短连接
```
// ========== 短连接 ==========
// 每个请求新建 TCP 连接,用完关闭
// HTTP/1.0 默认行为
// 缺点: 频繁三次握手和四次挥手
// 适用于: 低频请求
//
// 示例:
//   GET /page1 → [TCP 连接] → [响应] → [关闭]
//   GET /page2 → [TCP 连接] → [响应] → [关闭]
//
// ========== 长连接 (Keep-Alive) ==========
// 复用 TCP 连接,传输多个请求
// HTTP/1.1 默认行为
// 优点: 减少握手开销,降低延迟
// 适用于: 高频请求 (API/SSR/流媒体)
//
// 示例:
//   GET /page1 → GET /page2 → GET /page3 → [关闭]
//   └───────────── 同一 TCP 连接 ────────────┘
//
// ========== 连接池 ==========
// 数据库连接池、HTTP 连接池
// 维护一组"预热"的连接
// 避免频繁创建/销毁连接的开销
```
