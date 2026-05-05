# 请求头与响应头

## 📋 HTTP 请求头与响应头

通用头、请求头、响应头、实体头分类详解，自定义头规范。

## HTTP 头部分类

HTTP 头部分为四大类：通用头、请求头、响应头、实体头。
```
// ========== 通用头 (General Headers) ==========
// 适用于请求和响应,但与传输数据无关
//
// Date:           报文创建时间
// Cache-Control:  缓存指令
// Connection:     连接管理选项
// Via:            代理服务器信息
// Trailer:        分块传输末尾的头部
// Transfer-Encoding: 传输编码 (如 chunked)

// ========== 请求头 (Request Headers) ==========
// 客户端 → 服务器, 携带请求额外信息
//
// Host:             目标服务器 (必需)
// User-Agent:       客户端标识
// Referer:          请求来源 URL
// Origin:           请求来源 (CORS)
// Authorization:    认证凭据
// Accept:           接受的响应类型
// Accept-Language:  接受的语言
// Accept-Encoding:  接受的压缩算法
// Cookie:           Cookie 数据
// If-Modified-Since: 条件请求
// If-None-Match:    ETag 条件请求
// Range:            范围请求

// ========== 响应头 (Response Headers) ==========
// 服务器 → 客户端, 携带响应额外信息
//
// Age:              缓存响应的时间
// Location:         重定向 URL
// Set-Cookie:       设置 Cookie
// WWW-Authenticate: 认证挑战
// Retry-After:      重试等待时间
// Server:           服务器名称

// ========== 实体头 (Entity Headers) ==========
// 描述消息体内容的头部
//
// Content-Type:     媒体类型
// Content-Length:   体长度 (bytes)
// Content-Encoding: 压缩算法
// Content-Language: 内容的语言
// Content-Location: 内容的实际位置
// Content-Range:    范围响应的偏移
// ETag:             实体标签 (缓存验证)
// Expires:          过期时间
// Last-Modified:    最后修改时间
```
## 重要头部详解
```
// ========== Host 头 ==========
// HTTP/1.1 唯一必需头
// 使一台服务器托管多个域名 (虚拟主机)
// Host: www.example.com:8080
//
// ========== User-Agent 头 ==========
// 标识发出请求的客户端软件
// Mozilla/5.0 (Windows NT 10.0; Win64; x64)
//   AppleWebKit/537.36 (KHTML, like Gecko)
//   Chrome/120.0.0.0 Safari/537.36
//
// 用途: 统计分析, 内容适配, 反爬虫
//
// ========== Content-Type 头 ==========
// 指定请求/响应的媒体类型
// Content-Type: application/json; charset=utf-8
//
// 常见的值:
//   text/html               HTML
//   text/plain              纯文本
//   application/json         JSON
//   application/xml          XML
//   multipart/form-data      文件上传
//   application/octet-stream 二进制流

// ========== Content-Length ==========
// 消息体大小 (bytes)
// 用于确定请求/响应的结束位置
// Content-Length: 348
//
// 与 Transfer-Encoding: chunked 互斥
// chuncked 没有 Content-Length, 逐块传输

// ========== Referer 头 ==========
// 当前请求的来源页面 URL
// 用途: 统计流量来源, 防盗链
// 注意: 隐私敏感, Referrer-Policy 可控制
```
## 自定义头部与标准
```
// ========== 自定义头命名规范 ==========
// 非标准头使用 X- 前缀 (已弃用,但广泛使用)
//   X-Request-ID:      请求追踪 ID
//   X-Trace-ID:        链路追踪 ID
//   X-RateLimit-Limit: 限流限额
//   X-CSRF-Token:      CSRF 令牌
//
// 新的 RFC 6648 建议:
//   不再推荐 X- 前缀
//   如: RateLimit-Limit 替代 X-RateLimit-Limit
//
// 供应商专属头:
//   X-Amz-Date:          AWS 签名日期
//   X-Cloud-Trace-Context: Google Cloud 追踪
//   X-Forwarded-For:     真实客户端 IP
//   X-Real-IP:           Nginx 真实 IP

// ========== 安全相关响应头 ==========
// Strict-Transport-Security: 强制 HTTPS
//   max-age=31536000; includeSubDomains
//
// X-Content-Type-Options: 禁用 MIME 嗅探
//   nosniff
//
// X-Frame-Options: 防止点击劫持
//   DENY | SAMEORIGIN
//
// Content-Security-Policy: 内容安全策略
//   default-src 'self'; script-src 'self' cdn.example.com
//
// X-XSS-Protection: XSS 过滤 (已废弃)
//   1; mode=block
//
// Referrer-Policy: 控制 Referer 发送策略
//   no-referrer | same-origin | strict-origin-when-cross-origin
```
## 请求头模拟
