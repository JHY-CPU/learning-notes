# PATCH与状态码

## 🔧 PATCH 与 HTTP 状态码

PATCH 方法详解、JSON Patch/JSON Merge Patch、HTTP 状态码分类与含义。

## PATCH 方法详解

PATCH 用于对资源进行部分修改，不同于 PUT 的完全替换。
```
// ========== PUT vs PATCH ==========
//
// 原始资源:
//   { "name": "Alice", "email": "alice@a.com", "age": 25 }
//
// PUT 全量替换:
//   PUT /users/123
//   { "name": "Alice", "email": "new@a.com" }
//   结果: { "name": "Alice", "email": "new@a.com" }
//   age 字段消失了! (被置空)
//
// PATCH 部分更新:
//   PATCH /users/123
//   { "email": "new@a.com" }
//   结果: { "name": "Alice", "email": "new@a.com", "age": 25 }
//   只更新 email, 其他字段保持不变

// ========== JSON Merge Patch (RFC 7386) ==========
// 简单体合并规则:
//   提供字段 → 更新
//   null     → 删除
//   未提供   → 不变
//
// PATCH /users/123
// { "email": "new@a.com", "age": null }
// 结果: 更新 email, 删除 age, 其他不变

// ========== JSON Patch (RFC 6902) ==========
// 操作序列格式:
// PATCH /users/123
// Content-Type: application/json-patch+json
// [
//   { "op": "replace", "path": "/email", "value": "new@a.com" },
//   { "op": "remove", "path": "/age" },
//   { "op": "add", "path": "/tags/-", "value": "premium" }
// ]
//
// 支持的操作: add, remove, replace, move, copy, test
```
## HTTP 状态码分类
```
// ========== 状态码分类 ==========
//
// 1xx (信息性): 请求已接收,继续处理
// 2xx (成功):   请求已成功处理
// 3xx (重定向): 需要进一步操作完成请求
// 4xx (客户端错误): 请求包含错误或无法处理
// 5xx (服务器错误): 服务器处理请求时出错
//
// ========== 1xx 信息性 ==========
// 100 Continue:      继续发送请求体
// 101 Switching Protocols: 升级协议 (WebSocket)
// 102 Processing:    正在处理 (WebDAV)
//
// ========== 2xx 成功 ==========
// 200 OK:            成功 (默认)
// 201 Created:       创建成功 (POST)
// 202 Accepted:      已接受但未完成 (异步任务)
// 204 No Content:    成功但无响应体 (DELETE)
// 206 Partial Content: 部分内容 (范围请求/断点续传)
```
## 3xx 与 4xx 状态码
```
// ========== 3xx 重定向 ==========
// 301 Moved Permanently:   永久重定向 (SEO 迁移)
// 302 Found:               临时重定向 (登录后跳转)
// 303 See Other:           查看其他资源 (POST 后重定向)
// 304 Not Modified:        缓存未过期 (条件 GET)
// 307 Temporary Redirect:  临时重定向 (保持请求方法)
// 308 Permanent Redirect:  永久重定向 (保持请求方法)
//
// 301 vs 302:
//   301: 浏览器缓存重定向,更新书签
//   302: 临时跳转,下次还走原 URL
//
// ========== 4xx 客户端错误 ==========
// 400 Bad Request:         请求格式错误
// 401 Unauthorized:        未认证 (未登录)
// 403 Forbidden:           已认证但无权限
// 404 Not Found:           资源不存在
// 405 Method Not Allowed:  请求方法不支持
// 406 Not Acceptable:      无法生成接受的响应格式
// 408 Request Timeout:     请求超时
// 409 Conflict:            资源冲突 (并发更新)
// 410 Gone:                资源已永久删除
// 413 Payload Too Large:   请求体太大
// 415 Unsupported Media Type: 不支持的媒体类型
// 422 Unprocessable Entity: 语义错误 (表单验证失败)
// 429 Too Many Requests:   请求限流
// 451 Unavailable For Legal Reasons: 法律原因不可用
```
## 5xx 状态码与最佳实践
```
// ========== 5xx 服务器错误 ==========
// 500 Internal Server Error:      服务器内部错误 (未处理异常)
// 501 Not Implemented:            请求的方法不支持
// 502 Bad Gateway:                网关/代理上游无响应
// 503 Service Unavailable:        服务暂时不可用 (维护/过载)
// 504 Gateway Timeout:            网关/代理上游超时
// 505 HTTP Version Not Supported: HTTP 版本不支持
//
// ========== 状态码选择最佳实践 ==========
// ✅ 正确选择:
//   POST /articles:         201 Created
//   DELETE /articles/123:   204 No Content
//   表单验证失败:           422 Unprocessable Entity
//   用户未登录:             401 Unauthorized
//   权限不足:               403 Forbidden
//   请求频率过高:           429 Too Many Requests
//   上游超时:               504 Gateway Timeout
//
// ❌ 常见错误:
//   用 200 表示失败:        ❌ 应使用 4xx
//   用 500 表示验证失败:    ❌ 应使用 422
//   用 400 表示未认证:      ❌ 应使用 401
//   未登录返回 403:         ❌ 应使用 401

// ========== 响应体最佳实践 ==========
// 错误响应格式:
// {
//   "error": {
//     "code": "VALIDATION_ERROR",
//     "message": "邮箱格式不正确",
//     "details": [
//       { "field": "email", "message": "请输入有效的邮箱地址" }
//     ]
//   }
// }
```
> **Note**: 🎯 状态码是 API 设计中最容易被忽视的细节——用对状态码能显著提升 API 的可用性和可理解性。比如用 422 表示验证失败,远比返回 200 + 错误码更清晰。
