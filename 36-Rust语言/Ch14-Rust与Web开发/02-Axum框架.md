# Axum框架

## 一、概念说明

Axum 是基于 Tower 和 Hyper 的现代 Web 框架，设计简洁，类型安全，易于扩展。

```rust
use axum::{routing::get, Router};

#[tokio::main]
async fn main() {
    let app = Router::new()
        .route("/", get(|| async { "你好，Axum！" }));

    axum::Server::bind(&"127.0.0.1:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

## 二、具体用法

### 2.1 路由与提取器

```rust
use axum::{routing::{get, post}, Json, extract::Path};

async fn get_user(Path(id): Path<u64>) -> String {
    format!("用户ID: {}", id)
}

async fn create_user(Json(payload): Json<CreateUser>) -> Json<User> {
    Json(User { id: 1, name: payload.name })
}
```

### 2.2 中间件

```rust
use tower::ServiceBuilder;
use tower_http::trace::TraceLayer;

let app = Router::new()
    .route("/", get(|| async { "Hello" }))
    .layer(
        ServiceBuilder::new()
            .layer(TraceLayer::new_for_http())
    );
```

## 三、注意事项与常见陷阱

1. **类型安全**：提取器类型需正确
2. **错误处理**：实现 IntoResponse trait
3. **状态共享**：使用 Extension 或 State
4. **异步兼容**：确保所有依赖支持 async
5. **性能调优**：合理配置连接池和超时
