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

## 三、注意事项与常见陷阱

1. **异步函数**：处理函数必须是 async
2. **错误处理**：使用 Result 返回错误
3. **中间件**：合理使用中间件处理认证、日志等
4. **状态管理**：使用 web::Data 共享状态
5. **性能**：利用连接池和缓存
