# Express RESTful API 设计


## 🏗️ Express RESTful API 设计


REST 架构原则、资源命名规范、URL 设计最佳实践、HTTP 方法与 CRUD 映射、RESTful 路由设计 (嵌套/过滤/排序/分页)、API 版本控制、HATEOAS 概念、API 设计规范 checklist。


## REST 架构原则


```
// ========== REST ==========
// Representational State Transfer
// Roy Fielding 2000 年博士论文

// ========== 六大约束 ==========
// 1. 客户端-服务端分离 (关注点分离)
// 2. 无状态 (每个请求包含所有信息)
// 3. 可缓存 (响应明确缓存策略)
// 4. 统一接口 (资源/方法/表述)
// 5. 分层系统 (代理/网关/负载均衡)
// 6. 按需代码 (可选, 很少用)

// ========== 统一接口 ==========
// 1. 资源标识 (URL)
// 2. 资源表述 (JSON/XML)
// 3. 自描述消息 (Content-Type)
// 4. HATEOAS (超媒体作为应用状态引擎)

// ========== 资源命名 ==========
// ✅ 名词复数:     /users, /orders, /products
// ✅ 层级关系:     /users/:id/orders
// ✅ 查询参数过滤:  /users?role=admin&status=active
// ❌ 动词:         /getUsers, /createUser
// ❌ 文件扩展:     /users.json, /users.xml
// ❌ 混用单复数:   /user 和 /users 混用
```


## URL 设计规范


```
// ========== 资源 URL 模式 ==========
// ┌──────────┬──────────────────────────┬──────────────────┐
// │ 方法      │ URL                      │ 用途             │
// ├──────────┼──────────────────────────┼──────────────────┤
// │ GET      │ /users                   │ 列表             │
// │ GET      │ /users/:id               │ 详情             │
// │ POST     │ /users                   │ 创建             │
// │ PUT      │ /users/:id               │ 完整更新         │
// │ PATCH    │ /users/:id               │ 部分更新         │
// │ DELETE   │ /users/:id               │ 删除             │
// ├──────────┼──────────────────────────┼──────────────────┤
// │ GET      │ /users/:id/orders        │ 子资源列表       │
// │ POST     │ /users/:id/orders        │ 创建子资源       │
// │ GET      │ /users/:id/orders/:oid   │ 子资源详情       │
// └──────────┴──────────────────────────┴──────────────────┘

// ========== 查询参数规范 ==========
// 分页:
GET /users?page=1&limit=20
GET /users?offset=0&limit=20
GET /users?cursor=abc123&limit=20    // 游标分页

// 过滤:
GET /users?role=admin
GET /users?status=active&createdAfter=2024-01-01

// 排序:
GET /users?sort=name              // 升序
GET /users?sort=-name             // 降序 (负号)
GET /users?sort=name,createdAt    // 多字段

// 字段选择:
GET /users?fields=id,name,email   // 只返回指定字段

// 搜索:
GET /users?search=john
GET /users?q=keyword              // 通用搜索

// ========== API 版本控制 ==========
// 方案 1: URL 前缀 (最常用)
GET /api/v1/users
GET /api/v2/users

// 方案 2: 请求头
GET /users
Accept: application/vnd.myapp.v1+json

// 方案 3: 查询参数
GET /users?version=1

// Express 实现:
const v1Router = require('./routes/v1');
const v2Router = require('./routes/v2');
app.use('/api/v1', v1Router);
app.use('/api/v2', v2Router);
```


## 响应格式


```
// ========== 统一响应格式 ==========
// 所有 API 响应保持一致的结构

// ========== 成功响应 ==========
// 列表:
GET /users?page=1&limit=20
{
    "success": true,
    "data": [
        { "id": 1, "name": "Alice", "email": "alice@example.com" }
    ],
    "pagination": {
        "page": 1,
        "limit": 20,
        "total": 100,
        "totalPages": 5,
        "hasNext": true,
        "hasPrev": false
    }
}

// 详情:
GET /users/1
{
    "success": true,
    "data": { "id": 1, "name": "Alice", "email": "alice@example.com" }
}

// 创建:
POST /users
201 Created
{
    "success": true,
    "data": { "id": 2, "name": "Bob" },
    "message": "User created successfully"
}

// ========== 错误响应 ==========
// 验证错误 422:
{
    "success": false,
    "code": "VALIDATION_ERROR",
    "message": "Validation failed",
    "errors": [
        { "field": "email", "message": "Invalid email format" }
    ]
}

// 通用错误 500:
{
    "success": false,
    "code": "INTERNAL_ERROR",
    "message": "Internal server error",
    "requestId": "req-abc-123"
}

// ========== 响应中间件 ==========
app.use((req, res, next) => {
    res.success = (data, message = 'Success', status = 200) => {
        return res.status(status).json({
            success: true,
            data,
            message,
        });
    };

    res.paginated = (data, pagination) => {
        return res.status(200).json({
            success: true,
            data,
            pagination,
        });
    };

    res.error = (message, code = 'ERROR', status = 400) => {
        return res.status(status).json({
            success: false,
            code,
            message,
        });
    };

    next();
});
```


## API 设计 Checklist


```
// ========== API 设计检查清单 ==========

// ✅ 1. 资源命名
//   - 复数名词: /users, /orders, /products
//   - 小写 + 连字符: /order-items, /blog-posts
//   - 层级: /users/:id/orders
//   - 避免动词

// ✅ 2. HTTP 方法
//   - GET: 查询 (幂等, 安全)
//   - POST: 创建 (不幂等)
//   - PUT: 完整更新 (幂等)
//   - PATCH: 部分更新
//   - DELETE: 删除 (幂等)

// ✅ 3. 状态码
//   200 OK: GET/PUT 成功
//   201 Created: POST 成功
//   204 No Content: DELETE 成功
//   400 Bad Request: 请求错误
//   401 Unauthorized: 未认证
//   403 Forbidden: 无权限
//   404 Not Found: 资源不存在
//   409 Conflict: 冲突 (重复)
//   422 Unprocessable: 验证失败
//   429 Too Many: 限流
//   500 Server Error: 服务器错误

// ✅ 4. 一致性
//   - 统一响应格式 (success/data/message)
//   - 一致的分页格式
//   - 一致的错误格式
//   - 一致的日期格式 (ISO 8601)

// ✅ 5. 安全
//   - HTTPS 必须
//   - JWT/Token 认证
//   - 限流
//   - 输入验证
//   - CORS 配置

// ✅ 6. 文档
//   - OpenAPI/Swagger
//   - API 变更日志
//   - 示例请求/响应
```


> **Note:** 💡 RESTful API 设计要点: 资源用复数名词, 方法表达 CRUD; 一致响应格式 (success/data/pagination/error); 状态码精准选择; 版本控制推荐 URL 前缀 /api/v1; 嵌套路由表达资源关系; 过滤/排序/分页用查询参数; 安全 HTTPS + JWT + 限流; 文档用 OpenAPI/Swagger。


## 练习


<!-- Converted from: 12_Express RESTful API 设计.html -->
