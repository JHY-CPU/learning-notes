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

### 2.3 状态管理

```rust
use axum::{extract::State, routing::get, Router};
use std::sync::Arc;

#[derive(Clone)]
struct AppState {
    db_pool: sqlx::PgPool,
    config: AppConfig,
}

#[derive(Clone)]
struct AppConfig {
    app_name: String,
}

async fn index(State(state): State<Arc<AppState>>) -> String {
    format!("欢迎来到 {}", state.config.app_name)
}

#[tokio::main]
async fn main() {
    let state = Arc::new(AppState {
        db_pool: sqlx::PgPool::connect("postgres://...").await.unwrap(),
        config: AppConfig { app_name: "我的应用".into() },
    });

    let app = Router::new()
        .route("/", get(index))
        .with_state(state);

    axum::Server::bind(&"127.0.0.1:3000".parse().unwrap())
        .serve(app.into_make_service())
        .await
        .unwrap();
}
```

### 2.4 错误处理

```rust
use axum::http::StatusCode;
use axum::response::{IntoResponse, Response};
use axum::Json;
use serde_json::json;

struct AppError {
    status: StatusCode,
    message: String,
}

impl IntoResponse for AppError {
    fn into_response(self) -> Response {
        (self.status, Json(json!({"error": self.message}))).into_response()
    }
}

impl From<anyhow::Error> for AppError {
    fn from(err: anyhow::Error) -> Self {
        AppError {
            status: StatusCode::INTERNAL_SERVER_ERROR,
            message: err.to_string(),
        }
    }
}

async fn handler() -> Result<String, AppError> {
    // 使用 ? 自动转换错误
    let data = fetch_data().await?;
    Ok(data)
}
```

### 2.5 子路由与嵌套路由

```rust
use axum::{routing::{get, post}, Router};

fn api_routes() -> Router {
    Router::new()
        .route("/users", get(list_users).post(create_user))
        .route("/users/:id", get(get_user).put(update_user).delete(delete_user))
        .route("/posts", get(list_posts).post(create_post))
}

let app = Router::new()
    .route("/", get(index))
    .nest("/api/v1", api_routes())
    .nest("/admin", admin_routes());
```

### 2.6 WebSocket 支持

```rust
use axum::{extract::ws::{WebSocket, WebSocketUpgrade}, response::IntoResponse};

async fn ws_handler(ws: WebSocketUpgrade) -> impl IntoResponse {
    ws.on_upgrade(handle_socket)
}

async fn handle_socket(mut socket: WebSocket) {
    while let Some(Ok(msg)) = socket.recv().await {
        match msg {
            axum::extract::ws::Message::Text(text) => {
                socket.send(text.into()).await.unwrap();
            }
            axum::extract::ws::Message::Close(_) => break,
            _ => {}
        }
    }
}
```

## 四、Axum vs Actix-web 对比

| 特性 | Axum | Actix-web |
|------|------|-----------|
| 设计理念 | 类型安全、可组合 | 性能优先、功能全面 |
| 底层 | Tower + Hyper | 自研 Actor |
| 学习曲线 | 中等 | 中等 |
| 性能 | 优秀 | 优秀 |
| 生态系统 | 快速发展 | 成熟 |
| 中间件 | Tower 中间件 | 自定义中间件 |

## 五、注意事项与常见陷阱

1. **类型安全**：提取器类型需正确，类型错误在编译时捕获
2. **错误处理**：实现 `IntoResponse` trait 统一错误响应
3. **状态共享**：使用 `State` 提取器，避免全局变量
4. **异步兼容**：确保所有依赖支持 async，避免阻塞操作
5. **性能调优**：合理配置连接池和超时，启用 HTTP/2
6. **版本兼容**：Axum API 可能在版本间变化，注意迁移指南
7. **调试**：使用 tracing 中间件记录请求和响应
