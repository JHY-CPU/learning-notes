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

## 三、注意事项与常见陷阱

1. **状态码选择**：正确使用 HTTP 状态码
2. **分页处理**：大数据集使用分页
3. **版本控制**：API 版本管理策略
4. **输入验证**：验证所有用户输入
5. **文档生成**：使用 OpenAPI 自动生成文档
