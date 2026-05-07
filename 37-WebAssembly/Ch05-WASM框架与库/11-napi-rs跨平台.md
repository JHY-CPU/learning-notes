# napi-rs跨平台

## 一、概念说明

napi-rs 允许用 Rust 编写 Node.js 原生扩展，同时支持 WASM 降级，实现跨平台运行。

```rust
use napi_derive::napi;

#[napi]
pub fn fibonacci(n: u32) -> u64 {
    match n {
        0 => 0,
        1 => 1,
        _ => {
            let mut a = 0u64;
            let mut b = 1u64;
            for _ in 2..=n {
                let tmp = a + b;
                a = b;
                b = tmp;
            }
            b
        }
    }
}
```

## 二、具体用法

### 2.1 项目配置

```toml
# Cargo.toml
[package]
name = "my-node-addon"
version = "0.1.0"
edition = "2021"

[lib]
crate-type = ["cdylib"]

[dependencies]
napi = { version = "2", features = ["napi8"] }
napi-derive = "2"

[build-dependencies]
napi-build = "2"
```

### 2.2 函数与结构体导出

```rust
use napi_derive::napi;

#[napi]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[napi]
pub struct Config {
    pub host: String,
    pub port: u16,
}

#[napi]
impl Config {
    #[napi(constructor)]
    pub fn new(host: String, port: u16) -> Self {
        Config { host, port }
    }

    #[napi]
    pub fn to_url(&self, ssl: Option<bool>) -> String {
        let scheme = if ssl.unwrap_or(false) { "https" } else { "http" };
        format!("{}://{}:{}", scheme, self.host, self.port)
    }
}
```

### 2.3 异步函数

```rust
use napi::bindgen_prelude::*;

#[napi]
pub async fn fetch_data(url: String) -> Result<String> {
    let resp = reqwest::get(&url).await
        .map_err(|e| Error::from_reason(e.to_string()))?;
    let text = resp.text().await
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(text)
}
```

### 2.4 WASM 降级方案

```javascript
// package.json 中配置条件导出
{
    "name": "my-lib",
    "exports": {
        ".": {
            "node": {
                "import": "./my-addon.mjs",
                "require": "./my-addon.cjs"
            },
            "browser": "./wasm/my_lib.js",
            "default": "./wasm/my_lib.js"
        }
    }
}
```

### 2.5 类型增强

```rust
use napi::bindgen_prelude::*;

#[napi(object)]
pub struct UserInput {
    pub name: String,
    pub age: Option<u32>,
    pub tags: Vec<String>,
}

#[napi]
pub fn process_user(input: UserInput) -> Result<String> {
    Ok(format!("{} ({}岁)", input.name, input.age.unwrap_or(0)))
}

// 自动为 TypeScript 生成类型声明
// export interface UserInput { name: string; age?: number; tags: string[] }
// export declare function processUser(input: UserInput): string;
```

## 三、注意事项与常见陷阱

1. **构建工具**：需要 @napi-rs/cli 或 napi-build 进行交叉编译
2. **平台差异**：不同平台需要编译不同的 .node 文件
3. **WASM 限制**：降级到 WASM 时无法使用 Node.js 原生 API
4. **内存管理**：napi 对象的生命周期由 Node.js GC 管理
5. **错误传播**：使用 napi::Error 将 Rust 错误转为 JS 异常
