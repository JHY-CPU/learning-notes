# RESTful API设计

## 一、概念说明

使用 Rust 框架构建符合 REST 规范的 API。

```rust
use axum::{routing::{get, post, put, delete}, Router, Json};

let app = Router::new()
    .route("/api/users", get(list_users).post(create_user))
    .route("/api/users/:id", get(get_user).put(update_user).delete(delete_user));
```

## 二、具体用法

### 2.1 CRUD 操作

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize)]
struct User {
    id: u64,
    name: String,
    email: String,
}

async fn list_users() -> Json<Vec<User>> { Json(vec![]) }
async fn create_user(Json(user): Json<User>) -> Json<User> { Json(user) }
async fn get_user(Path(id): Path<u64>) -> Json<User> { todo!() }
async fn update_user(Path(id): Path<u64>, Json(user): Json<User>) -> Json<User> { Json(user) }
async fn delete_user(Path(id): Path<u64>) -> &'static str { "已删除" }
```

### 2.2 错误处理

```rust
use axum::http::StatusCode;
use axum::response::IntoResponse;

#[derive(Serialize)]
struct ApiError {
    code: u16,
    message: String,
}

impl IntoResponse for ApiError {
    fn into_response(self) -> axum::response::Response {
        (StatusCode::from_u16(self.code).unwrap(), Json(self)).into_response()
    }
}
```

### 2.3 分页查询

```rust
use serde::{Deserialize, Serialize};
use axum::extract::Query;

#[derive(Deserialize)]
struct Pagination {
    page: Option<u32>,
    per_page: Option<u32>,
}

#[derive(Serialize)]
struct PaginatedResponse<T> {
    data: Vec<T>,
    page: u32,
    per_page: u32,
    total: u64,
}

async fn list_users_paginated(
    Query(pagination): Query<Pagination>,
) -> Json<PaginatedResponse<User>> {
    let page = pagination.page.unwrap_or(1);
    let per_page = pagination.per_page.unwrap_or(20).min(100);

    let offset = (page - 1) * per_page;
    // 查询数据库...

    Json(PaginatedResponse {
        data: vec![],
        page,
        per_page,
        total: 1000,
    })
}
```

### 2.4 API 版本控制

```rust
// 路径版本
let app = Router::new()
    .nest("/api/v1", v1_routes())
    .nest("/api/v2", v2_routes());

// 或使用 Header 版本
async fn handler(headers: HeaderMap) -> Response {
    let version = headers.get("X-API-Version")
        .and_then(|v| v.to_str().ok())
        .unwrap_or("v1");

    match version {
        "v1" => handle_v1().await,
        "v2" => handle_v2().await,
        _ => StatusCode::BAD_REQUEST.into_response(),
    }
}
```

### 2.5 输入验证

```rust
use validator::Validate;

#[derive(Deserialize, Validate)]
struct CreateUserRequest {
    #[validate(length(min = 1, max = 100))]
    name: String,

    #[validate(email)]
    email: String,

    #[validate(range(min = 18, max = 120))]
    age: u8,
}

async fn create_user(Json(req): Json<CreateUserRequest>) -> Result<Response, AppError> {
    req.validate().map_err(|e| AppError::ValidationError(e.to_string()))?;

    // 处理有效的请求...
    Ok(Json(json!({"status": "created"})).into_response())
}
```

## 四、HTTP 状态码速查

| 状态码 | 含义 | 使用场景 |
|--------|------|---------|
| 200 OK | 成功 | GET、PUT、PATCH |
| 201 Created | 已创建 | POST 创建资源 |
| 204 No Content | 无内容 | DELETE 成功 |
| 400 Bad Request | 请求错误 | 输入验证失败 |
| 401 Unauthorized | 未认证 | 缺少认证信息 |
| 403 Forbidden | 禁止访问 | 无权限 |
| 404 Not Found | 未找到 | 资源不存在 |
| 409 Conflict | 冲突 | 资源已存在 |
| 429 Too Many Requests | 请求过多 | 限流 |
| 500 Internal Server Error | 服务器错误 | 未预期错误 |

## 五、注意事项与常见陷阱

1. **状态码选择**：正确使用 HTTP 状态码，遵循 REST 规范
2. **分页处理**：大数据集使用分页，避免一次返回过多数据
3. **版本控制**：API 版本管理策略，保持向后兼容
4. **输入验证**：验证所有用户输入，防止恶意请求
5. **文档生成**：使用 OpenAPI 自动生成文档，保持文档与代码同步
6. **幂等性**：PUT 和 DELETE 应该是幂等的
7. **CORS 配置**：配置正确的 CORS 策略
