# RESTful API 设计规范


## RESTful API 设计规范


RESTAPI设计HTTP方法


REST（Representational State Transfer）是一种架构风格，通过统一的接口和无状态的请求来设计 Web API。核心原则包括资源导向、统一接口和状态无关。


## 资源命名规范


```
URL 设计原则：
  1. 使用名词而非动词
  2. 使用复数形式
  3. 使用小写字母和连字符
  4. 层级关系用嵌套表示
  5. 避免文件扩展名

正确示例：
  GET    /api/v1/users              // 获取用户列表
  GET    /api/v1/users/123          // 获取用户123
  POST   /api/v1/users              // 创建用户
  PUT    /api/v1/users/123          // 完整更新用户
  PATCH  /api/v1/users/123          // 部分更新用户
  DELETE /api/v1/users/123          // 删除用户

错误示例：
  GET    /api/getUsers              // 动词命名
  GET    /api/user_list             // 下划线命名
  POST   /api/v1/createUser         // 动词命名
  GET    /api/v1/users.json         // 文件扩展名

资源嵌套：
  GET    /api/v1/users/123/orders        // 用户123的订单
  GET    /api/v1/users/123/orders/456    // 用户123的订单456
  GET    /api/v1/categories/5/products   // 分类5下的产品

嵌套层级限制：
  // 建议最多2-3层嵌套
  /api/v1/users/123/orders/456/items     // 过深
  // 改为扁平化：
  /api/v1/order-items?order_id=456       // 更好

操作型接口（非CRUD）：
  // 用 action 子资源表示
  POST   /api/v1/users/123/activate      // 激活用户
  POST   /api/v1/orders/456/cancel       // 取消订单
  POST   /api/v1/files/789/upload        // 上传文件
  // 或者用 HTTP 方法 + 自定义动作
  POST   /api/v1/users/123/actions       // body: {"action": "activate"}

批量操作：
  POST   /api/v1/users/batch             // body: {"action": "delete", "ids": [1,2,3]}
  DELETE /api/v1/users?ids=1,2,3         // 批量删除
```


## HTTP 方法语义


```
┌──────────┬───────────┬──────────┬───────────┬──────────┐
│ 方法     │ 幂等性    │ 安全性   │ 有请求体  │ 用途     │
├──────────┼───────────┼──────────┼───────────┼──────────┤
│ GET      │ 是        │ 是       │ 否        │ 查询资源 │
│ POST     │ 否        │ 否       │ 是        │ 创建资源 │
│ PUT      │ 是        │ 否       │ 是        │ 完整更新 │
│ PATCH    │ 否/可选   │ 否       │ 是        │ 部分更新 │
│ DELETE   │ 是        │ 否       │ 可选      │ 删除资源 │
│ HEAD     │ 是        │ 是       │ 否        │ 获取头部 │
│ OPTIONS  │ 是        │ 是       │ 可选      │ 查询方法 │
└──────────┴───────────┴──────────┴───────────┴──────────┘

幂等性（Idempotent）：
  多次执行与一次执行结果相同
  GET    /users/123        → 每次返回相同结果
  PUT    /users/123        → 多次更新结果相同
  DELETE /users/123        → 多次删除结果相同（第一次200，之后404）
  POST   /users            → 多次创建产生多条记录（不幂等）

PUT vs PATCH：
  PUT：完整替换资源
  PATCH：部分更新资源

  // PUT 请求（完整）
  PUT /api/v1/users/123
  {"name": "张三", "email": "zhangsan@example.com", "age": 25}

  // PATCH 请求（部分）
  PATCH /api/v1/users/123
  {"email": "new@example.com"}
  // 只更新 email 字段

POST 用于多种场景：
  创建资源：POST /users → 201 Created
  执行操作：POST /orders/123/cancel → 200 OK
  复杂查询：POST /search body: {filter} → 200 OK
  批量操作：POST /users/batch body: {ids} → 200 OK
```


## HTTP 状态码


```
2xx 成功：
  200 OK                - 请求成功（GET/PUT/DELETE）
  201 Created           - 资源创建成功（POST）
  202 Accepted          - 请求已接受，异步处理中
  204 No Content        - 成功但无响应体（DELETE）

3xx 重定向：
  301 Moved Permanently - 永久重定向
  302 Found             - 临时重定向
  304 Not Modified      - 缓存命中（条件请求）
  307 Temporary Redirect - 临时重定向（保持方法）
  308 Permanent Redirect - 永久重定向（保持方法）

4xx 客户端错误：
  400 Bad Request       - 请求格式错误
  401 Unauthorized      - 未认证（需登录）
  403 Forbidden         - 已认证但无权限
  404 Not Found         - 资源不存在
  405 Method Not Allowed - HTTP方法不允许
  409 Conflict           - 资源冲突（如重复创建）
  422 Unprocessable Entity - 请求格式正确但语义错误
  429 Too Many Requests  - 请求频率超限

5xx 服务端错误：
  500 Internal Server Error - 服务器内部错误
  502 Bad Gateway           - 网关/代理错误
  503 Service Unavailable   - 服务暂时不可用
  504 Gateway Timeout       - 网关超时

常见误用：
  // 错误：用 200 表示业务失败
  HTTP/1.1 200 OK
  {"code": 500, "msg": "服务器错误"}

  // 正确：HTTP状态码表示请求结果
  HTTP/1.1 500 Internal Server Error
  {"error": "服务器错误", "detail": "..."}

统一响应格式：
  // 成功响应
  {
    "code": 0,
    "message": "success",
    "data": { ... },
    "timestamp": 1698765432
  }

  // 错误响应
  {
    "code": 40001,
    "message": "用户名已存在",
    "errors": [
      {"field": "username", "message": "该用户名已被注册"}
    ],
    "timestamp": 1698765432
  }
```


## HATEOAS（超媒体驱动）


```
HATEOAS = Hypermedia As The Engine Of Application State
  API 响应中包含相关操作的链接
  客户端通过链接导航，不需要硬编码 URL

响应示例：
  GET /api/v1/users/123

  {
    "id": 123,
    "name": "张三",
    "email": "zhangsan@example.com",
    "status": "active",
    "_links": {
      "self": { "href": "/api/v1/users/123" },
      "orders": { "href": "/api/v1/users/123/orders" },
      "deactivate": { "href": "/api/v1/users/123/deactivate", "method": "POST" },
      "delete": { "href": "/api/v1/users/123", "method": "DELETE" }
    }
  }

HAL（Hypertext Application Language）格式：
  {
    "_embedded": {
      "users": [
        {"id": 1, "name": "张三", "_links": {"self": {"href": "/users/1"}}},
        {"id": 2, "name": "李四", "_links": {"self": {"href": "/users/2"}}}
      ]
    },
    "_links": {
      "self": { "href": "/users?page=1&size=10" },
      "next": { "href": "/users?page=2&size=10" },
      "last": { "href": "/users?page=50&size=10" }
    },
    "page": { "size": 10, "totalElements": 500, "totalPages": 50 }
  }

HATEOAS 的优缺点：
  优点：
  - 客户端与服务端解耦
  - API 演进不影响客户端
  - 自描述 API

  缺点：
  - 增加响应体积
  - 实现复杂度高
  - 大多数 REST API 不完全实现
```


## API 版本控制


```
版本控制策略：

1. URL 路径版本（最常用）：
   GET /api/v1/users
   GET /api/v2/users
   优点：直观、易缓存
   缺点：URL 变化，可能需要重新索引

2. 请求参数版本：
   GET /api/users?version=1
   GET /api/users?version=2
   优点：URL 不变
   缺点：参数可选时容易遗漏

3. Header 版本：
   GET /api/users
   Accept-Version: v1
   // 或
   Accept: application/vnd.myapi.v1+json
   优点：URL 干净
   缺点：不够直观，缓存复杂

4. Host 版本：
   GET https://v1.api.example.com/users
   GET https://v2.api.example.com/users
   优点：完全隔离
   缺点：需要管理多个域名

版本兼容策略：
  // 向后兼容的变更（不需要新版本）
  - 新增可选字段
  - 新增端点
  - 新增可选查询参数

  // 破坏性变更（需要新版本）
  - 删除字段
  - 修改字段类型
  - 修改字段语义
  - 修改验证规则

  // 推荐做法
  - 尽量保持向后兼容
  - 新字段设为可选
  - 旧字段标记废弃（deprecation）
  - 设置废弃时间线
```


## 分页、过滤与排序


```
分页：
  // 偏移量分页（Offset Pagination）
  GET /api/v1/users?page=2&size=20
  // page 从 1 开始
  // 适合：数据量小，需要跳页

  // 游标分页（Cursor Pagination）
  GET /api/v1/users?cursor=eyJpZCI6MTAwfQ&size=20
  // cursor = 上一页最后一条记录的编码
  // 适合：数据量大，实时数据流
  // 优点：不受数据插入/删除影响

  // 时间戳分页
  GET /api/v1/feed?after=2025-10-21T10:00:00Z&limit=20

分页响应格式：
  {
    "data": [...],
    "pagination": {
      "page": 2,
      "size": 20,
      "total": 500,
      "total_pages": 25,
      "has_next": true,
      "has_prev": true
    },
    "_links": {
      "first": "/api/v1/users?page=1&size=20",
      "prev": "/api/v1/users?page=1&size=20",
      "next": "/api/v1/users?page=3&size=20",
      "last": "/api/v1/users?page=25&size=20"
    }
  }

过滤：
  // 字段过滤
  GET /api/v1/users?status=active&role=admin

  // 范围过滤
  GET /api/v1/products?price_min=100&price_max=500
  GET /api/v1/orders?created_after=2025-01-01&created_before=2025-12-31

  // 搜索
  GET /api/v1/users?q=张三
  GET /api/v1/products?search=手机

  // 字段选择（Sparse Fieldsets）
  GET /api/v1/users?fields=id,name,email
  // 只返回指定字段

排序：
  GET /api/v1/users?sort=created_at        // 升序
  GET /api/v1/users?sort=-created_at       // 降序（负号）
  GET /api/v1/users?sort=role,-created_at  // 多字段排序

  // 或使用 order 参数
  GET /api/v1/users?sort_by=created_atℴ=desc

组合示例：
  GET /api/v1/products
    ?category=electronics
    &price_min=100
    &price_max=1000
    &sort=-rating
    &page=1
    &size=20
    &fields=id,name,price,rating

  // 返回：电子产品分类，价格100-1000，按评分降序，第1页，每页20条
```


> **Note:** RESTful API 设计的核心是资源导向和统一接口。URL 使用名词复数表示资源，HTTP 方法表示操作，状态码表示结果。版本控制推荐 URL 路径方式，分页建议大数据量使用游标分页。HATEOAS 理论上最优雅但实际项目中可简化实现。


<!-- Converted from: 01_RESTful API设计规范.html -->
