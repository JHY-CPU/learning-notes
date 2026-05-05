# HTTP请求与响应结构

## 📝 HTTP 请求与响应结构

HTTP 请求报文格式、响应报文格式、起始行、头部、空行、消息体。

## HTTP 请求结构

一个 HTTP 请求由三部分组成：请求行、请求头、请求体。
```
// ========== HTTP 请求报文 ==========
//
//  ┌──────────────────────────────────────────┐
//  │ 请求行 (Request Line)                     │
//  │ GET /api/users?page=1 HTTP/1.1           │
//  ├──────────────────────────────────────────┤
//  │ 请求头 (Headers)                          │
//  │ Host: api.example.com                     │
//  │ User-Agent: Mozilla/5.0                   │
//  │ Accept: application/json                  │
//  │ Authorization: Bearer eyJ...              │
//  │ Content-Type: application/json            │
//  ├──────────────────────────────────────────┤
//  │ (空行)                                    │
//  ├──────────────────────────────────────────┤
//  │ 请求体 (Body) — 仅 POST/PUT/PATCH 需要    │
//  │ { "name": "Alice", "email": "a@b.com" }  │
//  └──────────────────────────────────────────┘

// ========== 请求行详解 ==========
// 格式: 方法 + 空格 + URL + 空格 + 版本
//
// GET    /api/users?page=1    HTTP/1.1
// POST   /api/users           HTTP/2
// PUT    /api/users/123       HTTP/1.1
// DELETE /api/users/123       HTTP/1.1

// ========== 常见请求头 ==========
// Host:          目标服务器 (HTTP/1.1 必需)
// User-Agent:    客户端信息
// Accept:        接受的响应类型
// Accept-Language: 接受的语言
// Accept-Encoding: 接受的压缩算法
// Authorization: 认证凭据
// Cookie:        Cookie 数据
// Content-Type:  请求体 MIME 类型
// Content-Length: 请求体字节长度
// Referer:       来源页面 URL
// Origin:        请求来源 (CORS)
// Cache-Control: 缓存策略
```
## HTTP 响应结构

HTTP 响应由三部分组成：状态行、响应头、响应体。
```
// ========== HTTP 响应报文 ==========
//
//  ┌──────────────────────────────────────────┐
//  │ 状态行 (Status Line)                      │
//  │ HTTP/1.1 200 OK                          │
//  ├──────────────────────────────────────────┤
//  │ 响应头 (Headers)                          │
//  │ Date: Mon, 01 Jan 2024 12:00:00 GMT      │
//  │ Content-Type: application/json           │
//  │ Content-Length: 123                      │
//  │ Cache-Control: max-age=3600              │
//  │ Set-Cookie: session=abc123; HttpOnly     │
//  ├──────────────────────────────────────────┤
//  │ (空行)                                    │
//  ├──────────────────────────────────────────┤
//  │ 响应体 (Body)                             │
//  │ { "id": 1, "name": "Alice", ... }       │
//  └──────────────────────────────────────────┘

// ========== 状态行详解 ==========
// 格式: 版本 + 空格 + 状态码 + 空格 + 原因短语
//
// HTTP/1.1 200 OK
// HTTP/2   404 Not Found
// HTTP/1.1 500 Internal Server Error
// HTTP/1.1 302 Found (重定向)

// ========== 常见响应头 ==========
// Content-Type:    响应体类型
// Content-Length:  响应体长度
// Content-Encoding:压缩算法 (gzip/brotli)
// Cache-Control:   缓存指令
// ETag:            资源版本标识
// Set-Cookie:      设置 Cookie
// Location:        重定向 URL
// Access-Control-Allow-Origin: CORS
// WWW-Authenticate: 认证挑战
// Retry-After:     重试等待时间
```
## MIME 类型

Content-Type 使用 MIME 类型标识资源格式。
```
// ========== 常见 MIME 类型 ==========
//
// 文本:
//   text/html              HTML 文档
//   text/plain             纯文本
//   text/css               CSS
//   text/javascript        JavaScript (传统)
//
// 应用:
//   application/json       JSON
//   application/xml        XML
//   application/pdf        PDF 文档
//   application/octet-stream 二进制文件下载
//   application/x-www-form-urlencoded  表单提交
//
// 多部分:
//   multipart/form-data    文件上传
//   multipart/byteranges   范围请求
//
// 图片/音视频:
//   image/jpeg, image/png, image/webp
//   audio/mpeg, video/mp4

// ========== 内容协商 ==========
// 客户端告诉服务器它"偏好"什么格式:
//   Accept: application/json, text/html;q=0.9
//   Accept-Language: zh-CN, en;q=0.7
//   Accept-Encoding: gzip, br
//
// q=权重, 0-1, 默认 1
// 服务器根据 Accept 头选择最佳响应格式
```
## 模拟演示
