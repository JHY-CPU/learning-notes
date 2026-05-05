# 状态码分类

## 🔢 HTTP 状态码分类

状态码详细分类速查、各状态码含义、使用场景、最佳实践对照表。

## 状态码速查表
```
// ========== 完整状态码速查 ==========
//
// 1xx (信息):
//   100 Continue          继续发送请求体
//   101 Switching Protocols 切换协议 (WebSocket)
//   102 Processing        正在处理 (WebDAV)
//
// 2xx (成功):
//   200 OK                成功
//   201 Created           创建成功
//   202 Accepted          已接受 (异步)
//   203 Non-Authoritative  非权威信息
//   204 No Content        成功无响应体
//   205 Reset Content     重置内容
//   206 Partial Content   部分内容 (范围请求)
//
// 3xx (重定向):
//   300 Multiple Choices  多个可选资源
//   301 Moved Permanently 永久重定向
//   302 Found             临时重定向
//   303 See Other         查看其他位置
//   304 Not Modified      缓存未过期
//   307 Temporary Redirect 临时重定向 (方法不变)
//   308 Permanent Redirect 永久重定向 (方法不变)
//
// 4xx (客户端错误):
//   400 Bad Request       请求格式错误
//   401 Unauthorized      未认证
//   402 Payment Required  需付款 (预留)
//   403 Forbidden         禁止访问
//   404 Not Found         未找到
//   405 Method Not Allowed 方法不允许
//   406 Not Acceptable    无法接受
//   408 Request Timeout   请求超时
//   409 Conflict          冲突
//   410 Gone              已消失
//   412 Precondition Failed 前置条件失败
//   413 Payload Too Large  负载太大
//   415 Unsupported Media Type 不支持的媒体
//   422 Unprocessable Entity 无法处理的实体
//   429 Too Many Requests 请求太多
//
// 5xx (服务器错误):
//   500 Internal Server Error 内部错误
//   501 Not Implemented   未实现
//   502 Bad Gateway       网关错误
//   503 Service Unavailable 服务不可用
//   504 Gateway Timeout   网关超时
//   505 Version Not Supported 版本不支持
```
## 场景化状态码选择
```
// ========== CRUD 场景 ==========
//
// 创建资源:
//   POST /articles → 201 Created
//     Location: /articles/123
//
// 读取资源:
//   GET /articles/123 → 200 OK
//   GET /articles/999 → 404 Not Found
//
// 更新资源:
//   PUT /articles/123 → 200 OK
//   PUT 条件失败     → 409 Conflict (版本冲突)
//
// 删除资源:
//   DELETE /articles/123 → 204 No Content
//   DELETE /articles/999 → 404 Not Found

// ========== 认证与授权场景 ==========
// 未登录:
//   GET /api/profile → 401 Unauthorized
//   WWW-Authenticate: Bearer realm="api"
//
// 登录成功:
//   POST /api/login → 200 OK
//
// 已登录但无权限:
//   GET /api/admin → 403 Forbidden
//
// 登录失败(密码错):
//   POST /api/login → 401 Unauthorized

// ========== API 错误场景 ==========
// 请求体格式错:
//   POST /articles → 400 Bad Request
//   (JSON 解析失败)
//
// 字段验证失败:
//   POST /register → 422 Unprocessable Entity
//   { errors: { email: "格式不正确" } }
//
// 请求频率过高:
//   GET /api/search → 429 Too Many Requests
//   Retry-After: 120
```
## 常见状态码误用
```
// ========== 状态码误用案例 ==========
//
// ❌ 错误: 用 200 表示自定义错误
//   200 OK + {"code": -1, "msg": "token过期"}
//   ✅ 正确: 401 Unauthorized
//
// ❌ 错误: 未认证时返回 403
//   GET /api/profile → 403 Forbidden
//   ✅ 正确: 401 Unauthorized
//
// ❌ 错误: 用 500 表示验证失败
//   POST /register → 500 + "邮箱已存在"
//   ✅ 正确: 409 Conflict 或 422
//
// ❌ 错误: 用 404 掩盖权限问题
//   GET /api/admin → 404 (存在但用户无权限)
//   ✅ 正确: 403 Forbidden
//
// ❌ 错误: 同步接口用 202
//   POST /login → 202 Accepted
//   ✅ 正确: 200 OK (除非异步)
//
// ❌ 错误: 用 400 反所有错误
//   验证也 400, 权限也 400, 全用 400
//   ✅ 正确: 用精确状态码

// ========== 选择决策树 ==========
// 请求成功了?
//   ├── 是: 2xx
//   │   ├── 创建了资源? → 201
//   │   ├── 删除了资源? → 204
//   │   └── 其他 → 200
//   └── 否:
//       ├── 客户端问题? → 4xx
//       │   ├── 未认证? → 401
//       │   ├── 无权限? → 403
//       │   ├── 不存在? → 404
//       │   ├── 验证失败? → 422
//       │   └── 限流? → 429
//       └── 服务器问题? → 5xx
//           ├── 未知错误? → 500
//           ├── 上游问题? → 502/504
//           └── 过载/维护? → 503
```
> **Note**: 📋 一个好的 HTTP API 应该让调用者仅通过状态码就了解大致结果——而不需要解析响应体。滥用 200 + 自定义错误码是常见的反模式。

## 模拟演示
