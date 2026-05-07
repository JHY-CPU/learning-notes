# Serverless与Edge

## 一、概念说明

WASM 在 Serverless 和 Edge Computing 场景中提供比容器更快的冷启动和更小的资源占用。

```javascript
// Cloudflare Workers 使用 WASM
addEventListener('fetch', event => {
    event.respondWith(handleRequest(event.request));
});

async function handleRequest(request) {
    const { instance } = await WebAssembly.instantiate(WASM_MODULE);
    const result = instance.exports.process(request.body);
    return new Response(result);
}
```

## 二、具体用法

### 2.1 Cloudflare Workers

```rust
use worker::*;

#[event(fetch)]
pub async fn main(req: Request, env: Env, _ctx: Context) -> Result<Response> {
    let router = Router::new();

    router
        .get_async("/api/data", handle_get)
        .post_async("/api/data", handle_post)
        .run(req, env)
        .await
}

async fn handle_get(_req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let data = ctx.kv("DATA_STORE")?.get("key").text().await?;
    Response::ok(data.unwrap_or_default())
}

async fn handle_post(mut req: Request, ctx: RouteContext<()>) -> Result<Response> {
    let body = req.text().await?;
    ctx.kv("DATA_STORE")?.put("key", body)?.execute().await?;
    Response::ok("OK")
}
```

### 2.2 Fastly Compute@Edge

```rust
use fastly::http::{header, Method, StatusCode};
use fastly::{mime, Request, Response};

#[fastly::main]
fn main(req: Request) -> Response {
    match req.get_path() {
        "/" => Response::from_status(StatusCode::OK)
            .with_content_type(mime::TEXT_HTML_UTF_8)
            .with_body("<h1>Hello from WASM!</h1>"),

        "/api" => {
            let body = req.into_body_str();
            Response::from_status(StatusCode::OK)
                .with_content_type(mime::APPLICATION_JSON)
                .with_body(format!("{{\"received\": \"{}\"}}", body))
        }

        _ => Response::from_status(StatusCode::NOT_FOUND),
    }
}
```

### 2.3 WASI 运行时

```rust
// 通用 WASI 应用
use std::io::{self, Read};

fn main() {
    let mut input = String::new();
    io::stdin().read_to_string(&mut input).unwrap();

    let result = process_data(&input);
    println!("{}", result);
}

fn process_data(input: &str) -> String {
    // 处理逻辑
    input.to_uppercase()
}
```

```bash
# 编译为 WASI
cargo build --target wasm32-wasi --release

# 使用 wasmtime 运行
wasmtime target/wasm32-wasi/release/my_app.wasm
```

### 2.4 微服务模式

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn route_request(method: &str, path: &str, body: &str) -> JsValue {
    let response = match (method, path) {
        ("GET", "/health") => Response {
            status: 200,
            body: "OK".to_string(),
        },
        ("POST", "/process") => Response {
            status: 200,
            body: format!("处理: {}", body),
        },
        _ => Response {
            status: 404,
            body: "Not Found".to_string(),
        },
    };
    serde_wasm_bindgen::to_value(&response).unwrap()
}

#[derive(serde::Serialize)]
struct Response {
    status: u16,
    body: String,
}
```

### 2.5 边缘计算最佳实践

```toml
# fastly.toml
[package]
name = "my-edge-app"
version = "0.1.0"

[local_server]
[local_server.backends]
[local_server.backends.origin]
url = "https://origin.example.com"
```

```rust
// 缓存策略
use fastly::{CacheKey, Request, Response};

fn cached_response(req: &Request) -> Option<Response> {
    let key = CacheKey::from(req.get_url_str());
    fastly::Cache::lookup(key).and_then(|found| found.to_stream().ok())
}
```

## 三、注意事项与常见陷阱

1. **冷启动**：WASM 冷启动比容器快 10-100 倍
2. **运行时限制**：Edge 运行时有 CPU 时间、内存限制
3. **WASI 兼容性**：不同运行时的 WASI 支持程度不同
4. **网络访问**：Edge 运行时可能限制出站网络
5. **调试工具**：Edge 调试比本地开发更困难
