# Actix-web基础

## 一、概念说明

Actix-web 是 Rust 最流行的 Web 框架，基于 Actor 模型，提供高性能的 HTTP 服务器。

```rust
use actix_web::{web, App, HttpServer, HttpResponse};

async fn index() -> HttpResponse {
    HttpResponse::Ok().body("你好，世界！")
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    HttpServer::new(|| {
        App::new().route("/", web::get().to(index))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

## 二、具体用法

### 2.1 路由定义

```rust
use actix_web::{web, HttpResponse};

fn config(cfg: &mut web::ServiceConfig) {
    cfg.route("/", web::get().to(index))
       .route("/users/{id}", web::get().to(get_user))
       .route("/users", web::post().to(create_user));
}

async fn get_user(path: web::Path<u32>) -> HttpResponse {
    HttpResponse::Ok().json(format!("用户ID: {}", path))
}
```

### 2.2 请求提取器

```rust
use actix_web::{web, HttpResponse};
use serde::Deserialize;

#[derive(Deserialize)]
struct Info {
    username: String,
}

async fn login(info: web::Json<Info>) -> HttpResponse {
    HttpResponse::Ok().body(format!("欢迎，{}", info.username))
}

async fn query(query: web::Query<Info>) -> HttpResponse {
    HttpResponse::Ok().body(format!("查询：{}", query.username))
}
```

### 2.3 状态管理

```rust
use actix_web::{web, App, HttpServer};
use std::sync::Mutex;

struct AppState {
    app_name: String,
    visitor_count: Mutex<u64>,
}

async fn index(data: web::Data<AppState>) -> String {
    let mut count = data.visitor_count.lock().unwrap();
    *count += 1;
    format!("欢迎来到 {}! 访客数: {}", data.app_name, count)
}

#[actix_web::main]
async fn main() -> std::io::Result<()> {
    let data = web::Data::new(AppState {
        app_name: "我的应用".to_string(),
        visitor_count: Mutex::new(0),
    });

    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .route("/", web::get().to(index))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
```

### 2.4 文件上传处理

```rust
use actix_multipart::Multipart;
use actix_web::HttpResponse;
use futures::StreamExt;

async fn upload(mut payload: Multipart) -> HttpResponse {
    while let Some(item) = payload.next().await {
        let mut field = item.unwrap();
        let content_disposition = field.content_disposition();
        let filename = content_disposition.get_filename().unwrap();

        let mut data = Vec::new();
        while let Some(chunk) = field.next().await {
            data.extend_from_slice(&chunk.unwrap());
        }

        println!("文件: {}, 大小: {} 字节", filename, data.len());
    }
    HttpResponse::Ok().body("上传完成")
}
```

### 2.5 错误处理统一

```rust
use actix_web::{HttpResponse, ResponseError};
use std::fmt;

#[derive(Debug)]
enum AppError {
    NotFound(String),
    InternalError(String),
}

impl fmt::Display for AppError {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        match self {
            AppError::NotFound(msg) => write!(f, "未找到: {}", msg),
            AppError::InternalError(msg) => write!(f, "内部错误: {}", msg),
        }
    }
}

impl ResponseError for AppError {
    fn error_response(&self) -> HttpResponse {
        match self {
            AppError::NotFound(msg) => {
                HttpResponse::NotFound().json(serde_json::json!({"error": msg}))
            }
            AppError::InternalError(msg) => {
                HttpResponse::InternalServerError().json(serde_json::json!({"error": msg}))
            }
        }
    }
}
```

## 四、Actix-web 性能优化

```rust
// 连接池配置
use sqlx::postgres::PgPoolOptions;

// 启用 HTTP/2
use actix_web::HttpServer;
use actix_tls::accept::rustls::TlsConfig;

// 压缩响应
use actix_web::middleware::Compress;

// 静态文件缓存
use actix_files::Files;

fn configure_app(cfg: &mut web::ServiceConfig) {
    cfg.service(
        Files::new("/static", "./static")
            .use_last_modified(true)
            .use_etag(true),
    );
}
```

## 五、注意事项与常见陷阱

1. **异步函数**：处理函数必须是 async，不能在 handler 中使用阻塞操作
2. **错误处理**：使用 `Result<T, E>` 返回错误，实现 `ResponseError` trait
3. **中间件**：合理使用中间件处理认证、日志、CORS 等
4. **状态管理**：使用 `web::Data` 共享状态，避免全局可变状态
5. **性能**：利用连接池和缓存，启用压缩和 HTTP/2
6. **安全性**：启用 CORS、CSRF 保护，验证所有输入
7. **优雅关闭**：处理 SIGTERM 信号，等待请求完成
