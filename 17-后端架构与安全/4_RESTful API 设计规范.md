# RESTful API 设计规范


## 📐 RESTful API 设计规范


REST 成熟度模型、资源命名、URL 设计、版本化、分页/过滤/排序、错误格式、HATEOAS、幂等性。


## REST 设计原则


```
// ========== REST 成熟度模型 ==========
// Level 0: 单个端点, 所有操作通过参数区分 (沼泽)
//   POST /api
//   ?action=getUser&id=1

// Level 1: 资源识别 (资源)
//   GET /users/1
//   POST /createUser

// Level 2: HTTP 动词 (方法)
//   GET    /users/1     → 获取用户
//   PUT    /users/1     → 更新用户
//   DELETE /users/1     → 删除用户

// Level 3: HATEOAS (超媒体)
//   GET /users/1 返回含操作的链接
//   { "id": 1, "name": "John",
//     "links": {
//       "orders": "/users/1/orders",
//       "self": "/users/1"
//     }
//   }

// ========== 资源命名 ==========
// ✅ 好的命名:
//   GET    /users              // 用户列表
//   GET    /users/123          // 单个用户
//   POST   /users              // 创建用户
//   PUT    /users/123          // 替换用户
//   PATCH  /users/123          // 部分更新
//   DELETE /users/123          // 删除用户

//   GET    /users/123/orders   // 子资源
//   GET    /orders/456         // 订单

// ❌ 坏的命名:
//   GET    /getUsers           // 动词
//   POST   /createUser         // 动词
//   GET    /userList           // 未统一复数
//   GET    /users/getOrders    // 路径混乱

// ========== 命名规范 ==========
// 1. 使用复数名词: /users, /orders
// 2. 小写 + 连字符: /order-items
// 3. 层级用斜杠: /users/123/orders
// 4. 查询用 ?: /users?status=active
// 5. 不用动词: GET /users 不是 GET /getUsers
// 6. 一致格式: 所有端点同风格
```


## 版本化与分页


```
// ========== API 版本化 ==========
// 方式 1: URL 前缀 (最常用)
//   /api/v1/users
//   /api/v2/users

// 方式 2: 请求头
//   Accept: application/vnd.myapp.v1+json

// 方式 3: 查询参数
//   /api/users?version=1

// 推荐: URL 前缀, 简单明确, 易于路由
// v1 兼容, v2 新功能, 逐步淘汰 v1

// ========== 分页 ==========
// 推荐格式:
// GET /api/users?page=1&per_page=20&sort=created_atℴ=desc

// 响应:
// {
//   "data": [...],
//   "meta": {
//     "page": 1,
//     "per_page": 20,
//     "total": 100,
//     "total_pages": 5
//   }
// }

// 游标分页 (适合实时数据):
// GET /api/events?cursor=abc123&limit=20
// {
//   "data": [...],
//   "meta": {
//     "next_cursor": "def456",
//     "has_more": true
//   }
// }

// ========== 过滤与排序 ==========
// 过滤:
// GET /api/users?status=active
// GET /api/users?created_at[gte]=2024-01-01&created_at[lte]=2024-12-31
// GET /api/users?search=john

// 字段选择:
// GET /api/users?fields=id,name,email

// 排序:
// GET /api/users?sort=created_atℴ=desc
// GET /api/users?sort=-created_at  (负号=降序)

// ========== HTTP 状态码 ==========
// 200 OK          — 成功 (GET, PATCH)
// 201 Created     — 创建成功 (POST)
// 204 No Content  — 删除成功 (DELETE)
// 301 Moved       — 永久重定向
// 400 Bad Request — 请求参数错误
// 401 Unauthorized — 未认证
// 403 Forbidden   — 无权限
// 404 Not Found   — 资源不存在
// 409 Conflict    — 冲突 (重复创建)
// 422 Unprocessable — 验证失败
// 429 Too Many Requests — 限流
// 500 Internal    — 服务器错误
// 503 Unavailable — 服务不可用
```


## 错误格式与幂等性


```
// ========== 统一错误格式 ==========
// {
//   "error": {
//     "code": "VALIDATION_ERROR",
//     "message": "请求参数验证失败",
//     "details": [
//       {
//         "field": "email",
//         "message": "邮箱格式不正确",
//         "code": "INVALID_FORMAT"
//       }
//     ],
//     "request_id": "req-abc-123",
//     "timestamp": "2024-01-01T00:00:00Z"
//   }
// }

// ========== Go 统一错误响应 ==========
type APIError struct {
    Code      string       `json:"code"`
    Message   string       `json:"message"`
    Details   []ErrorDetail `json:"details,omitempty"`
    RequestID string       `json:"request_id,omitempty"`
    Timestamp time.Time    `json:"timestamp"`
}

type ErrorDetail struct {
    Field   string `json:"field,omitempty"`
    Message string `json:"message"`
    Code    string `json:"code,omitempty"`
}

func ErrorResponse(c *gin.Context, status int, code, msg string) {
    c.AbortWithStatusJSON(status, APIError{
        Code:      code,
        Message:   msg,
        RequestID: c.GetString("request_id"),
        Timestamp: time.Now(),
    })
}

func ValidationError(c *gin.Context, details []ErrorDetail) {
    c.AbortWithStatusJSON(422, APIError{
        Code:      "VALIDATION_ERROR",
        Message:   "请求参数验证失败",
        Details:   details,
        RequestID: c.GetString("request_id"),
        Timestamp: time.Now(),
    })
}

// ========== 幂等性 ==========
// 安全方法: GET, HEAD, OPTIONS
// 幂等方法: PUT, DELETE
// 不幂等: POST

// 幂等键 (Idempotency-Key):
// 客户端生成唯一键, 重试时使用相同键

// 实现:
// func IdempotencyMiddleware() gin.HandlerFunc {
//     return func(c *gin.Context) {
//         if c.Request.Method != "POST" {
//             c.Next()
//             return
//         }
//
//         key := c.GetHeader("Idempotency-Key")
//         if key == "" {
//             c.Next()
//             return
//         }
//
//         // 检查是否已处理
//         result, _ := redis.Get(ctx, "idem:"+key).Result()
//         if result != "" {
//             // 返回之前的结果
//             c.Data(200, "application/json", []byte(result))
//             c.Abort()
//             return
//         }
//
//         c.Set("idempotency_key", key)
//         c.Next()
//     }
// }
```


## HATEOAS 与 OpenAPI


```
// ========== HATEOAS 示例 ==========
// 返回资源相关操作链接
// {
//   "id": 123,
//   "name": "John Doe",
//   "email": "john@example.com",
//   "_links": {
//     "self": { "href": "/api/v1/users/123", "method": "GET" },
//     "orders": { "href": "/api/v1/users/123/orders", "method": "GET" },
//     "update": { "href": "/api/v1/users/123", "method": "PUT" },
//     "delete": { "href": "/api/v1/users/123", "method": "DELETE" }
//   }
// }

// ========== OpenAPI 3.0 规范 ==========
// openapi: 3.0.0
// info:
//   title: 用户管理 API
//   version: 1.0.0
//   description: 用户管理系统 RESTful API
//
// servers:
//   - url: https://api.example.com/v1
//
// paths:
//   /users:
//     get:
//       summary: 获取用户列表
//       parameters:
//         - name: page
//           in: query
//           schema: { type: integer, default: 1 }
//         - name: per_page
//           in: query
//           schema: { type: integer, default: 20 }
//       responses:
//         '200':
//           description: 成功
//           content:
//             application/json:
//               schema:
//                 $ref: '#/components/schemas/UserList'
//
// components:
//   schemas:
//     User:
//       type: object
//       properties:
//         id: { type: integer }
//         name: { type: string }
//         email: { type: string, format: email }
//         created_at: { type: string, format: date-time }
//
//     UserList:
//       type: object
//       properties:
//         data: { type: array, items: { $ref: '#/components/schemas/User' } }
//         meta:
//           type: object
//           properties:
//             page: { type: integer }
//             total: { type: integer }
```


> **Note:** 💡 REST 要点: 复数名词 /users, HTTP 动词 (GET/POST/PUT/DELETE); 统一版本 /v1/; 分页 (page/per_page + meta); 过滤/排序/字段选择; 状态码精确: 201 Created, 204 No Content, 422 Validation, 429 Rate Limit; 统一错误 {code,message,details,request_id}; 幂等键 Idempotency-Key; HATEOAS 链接; OpenAPI 3.0 文档规范。


## 练习


<!-- Converted from: 4_RESTful API 设计规范.html -->
