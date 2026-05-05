# HTTP方法

## 🛠️ HTTP 方法

GET/POST/PUT/DELETE/PATCH/HEAD/OPTIONS 方法详解、安全与幂等、CRUD 映射。

## HTTP 方法大全
```
// ========== 核心 HTTP 方法 ==========
//
// GET:    获取资源 (只读)
// POST:   创建资源 (非幂等)
// PUT:    替换资源 (幂等)
// PATCH:  部分更新资源 (非幂等)
// DELETE: 删除资源 (幂等)
// HEAD:   获取响应头 (同 GET 但不返回体)
// OPTIONS: 查询服务器支持的请求方法
//
// ========== 方法详解 ==========
//
// GET /api/users          → 获取用户列表
// GET /api/users/123      → 获取单个用户
// POST /api/users          → 创建新用户
// PUT /api/users/123       → 完全替换用户
// PATCH /api/users/123     → 部分更新用户
// DELETE /api/users/123    → 删除用户

// ========== 完整方法列表 ==========
// GET      读取资源
// POST     提交新资源
// PUT      全量更新资源
// PATCH    部分更新资源
// DELETE   删除资源
// HEAD     获取响应头 (如检查资源是否存在)
// OPTIONS  查询支持的跨域方法
// TRACE    回显请求 (诊断用)
// CONNECT  建立隧道 (HTTPS 代理)
```
## 安全方法与幂等方法

安全 (Safe): 不改变服务器状态；幂等 (Idempotent): 多次执行结果相同。
```
// ========== 安全与幂等 ==========
//
// 方法      安全    幂等   请求体
// ─────────────────────────────────────
// GET       ✅     ✅     ❌
// HEAD      ✅     ✅     ❌
// OPTIONS   ✅     ✅     ❌
// DELETE    ❌     ✅     可选
// PUT       ❌     ✅     ✅
// PATCH     ❌     ❌     ✅
// POST      ❌     ❌     ✅

// ========== 幂等的重要性 ==========
// 网络不可靠,请求可能重试
// 幂等方法确保重试安全:
//
// PUT /api/users/123 (重试 3 次=1 次效果)
//   第 1 次: 更新成功
//   第 2 次: 同样数据再更新 (无变化)
//   第 3 次: 再次更新 (无变化)
//   → 最终结果一致
//
// POST /api/users (重试 = 多次创建)
//   第 1 次: 创建用户 A
//   第 2 次: 创建用户 B (重复!)
//   → 需幂等键 (Idempotency-Key) 防止重复

// ========== 实际应用 ==========
// 浏览器刷新页面:
//   安全方法 (GET) → 安全刷新
//   非安全方法 (POST) → 浏览器提示"确认重新提交表单"
//
// CDN 缓存:
//   只缓存安全方法 (GET/HEAD)
//
// 爬虫:
//   遵循安全方法约束
```
> **Note**: ⚠️ GET 请求必须有"幂等"特性——同一个 GET 请求无论执行多少次,服务器状态都应相同。如果你用 GET 来删除数据,爬虫路过时你的数据就没了。

## CRUD 与 REST 映射
```
// ========== CRUD 到 HTTP 映射 ==========
//
// Create  → POST   (或 PUT,若客户端决定 ID)
// Read    → GET
// Update  → PUT (全量) 或 PATCH (部分)
// Delete  → DELETE
//
// ========== RESTful API 设计 ==========
//
// 操作          POST              GET               PUT               DELETE
// ──────────────────────────────────────────────────────────────────────────
// 所有资源      /users            /users
// 单个资源                          /users/:id        /users/:id        /users/:id
// 子资源                            /users/:id/posts
// 操作                              /users/:id/profile
//
// GET    /api/articles              # 列表
// GET    /api/articles/123          # 详情
// POST   /api/articles              # 创建
// PUT    /api/articles/123          # 全量更新
// PATCH  /api/articles/123          # 部分更新
// DELETE /api/articles/123          # 删除

// ========== 注意事项 ==========
// PUT vs PATCH:
//   PUT: 替换整个资源
//     PUT /users/1 → body: {name, email, age}
//     缺失的字段会被置空
//
//   PATCH: 只更新指定字段
//     PATCH /users/1 → body: {name: "new"}
//     只更新 name,其他字段不变
```
## 表单场景的方法支持
```
// ========== HTML 表单的方法限制 ==========
// HTML 表单只支持 GET 和 POST:
//
//
//
// 要使用 PUT/DELETE/PATCH:
// 1. JavaScript AJAX/Fetch
//   fetch('/api/users/123', { method: 'DELETE' })
//
// 2. 方法伪造 (Method Spoofing):
//
//
//

// ========== 方法覆盖头 ==========
// 某些代理/防火墙只允许 GET 和 POST
// 使用自定义头解决:
//   X-HTTP-Method-Override: PUT
//   X-HTTP-Method-Override: DELETE
//
// 服务器检查该头来覆盖实际方法

// ========== OPTIONS 预检请求 ==========
// 跨域复杂请求前,浏览器先发 OPTIONS:
//   OPTIONS /api/users
//   Origin: https://frontend.com
//   Access-Control-Request-Method: POST
//
// 服务器响应:
//   Access-Control-Allow-Methods: GET,POST,PUT,DELETE
//   Access-Control-Allow-Origin: https://frontend.com
//   Access-Control-Max-Age: 86400
```
