# serde-wasm-bindgen

## 一、概念说明

serde-wasm-bindgen 提供了 Rust 结构体与 JsValue 之间的高性能序列化/反序列化，比 js-sys 手动操作更高效。

```rust
use serde::{Serialize, Deserialize};
use wasm_bindgen::prelude::*;

#[derive(Serialize, Deserialize)]
pub struct User {
    pub name: String,
    pub age: u32,
    pub active: bool,
}

#[wasm_bindgen]
pub fn get_user() -> JsValue {
    let user = User {
        name: "Alice".to_string(),
        age: 30,
        active: true,
    };
    serde_wasm_bindgen::to_value(&user).unwrap()
}

#[wasm_bindgen]
pub fn set_user(val: JsValue) {
    let user: User = serde_wasm_bindgen::from_value(val).unwrap();
    web_sys::console::log_1(&format!("用户: {}", user.name).into());
}
```

## 二、具体用法

### 2.1 复杂结构体序列化

```rust
use std::collections::HashMap;

#[derive(Serialize)]
pub struct Config {
    pub database: DatabaseConfig,
    pub features: Vec<String>,
    pub settings: HashMap<String, String>,
}

#[derive(Serialize)]
pub struct DatabaseConfig {
    pub host: String,
    pub port: u16,
    pub pool_size: u32,
}

#[wasm_bindgen]
pub fn get_config() -> JsValue {
    let config = Config {
        database: DatabaseConfig {
            host: "localhost".to_string(),
            port: 5432,
            pool_size: 10,
        },
        features: vec!["auth".into(), "logging".into()],
        settings: HashMap::from([
            ("theme".into(), "dark".into()),
            ("lang".into(), "zh".into()),
        ]),
    };
    serde_wasm_bindgen::to_value(&config).unwrap()
}
```

### 2.2 从 JS 接收数据

```rust
#[derive(Deserialize)]
pub struct SearchParams {
    pub query: String,
    pub page: Option<u32>,
    pub limit: Option<u32>,
    pub filters: Option<Vec<String>>,
}

#[wasm_bindgen]
pub fn search(params: JsValue) -> Result<JsValue, JsValue> {
    let params: SearchParams = serde_wasm_bindgen::from_value(params)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let page = params.page.unwrap_or(1);
    let limit = params.limit.unwrap_or(20);

    // 执行搜索逻辑
    let results: Vec<String> = vec!["结果1".into(), "结果2".into()];

    serde_wasm_bindgen::to_value(&results)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

### 2.3 嵌套结构与枚举

```rust
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
pub enum Message {
    Text { content: String },
    Image { url: String, width: u32, height: u32 },
    Command { action: String, args: Vec<String> },
}

#[wasm_bindgen]
pub fn process_message(msg: JsValue) -> Result<JsValue, JsValue> {
    let msg: Message = serde_wasm_bindgen::from_value(msg)
        .map_err(|e| JsValue::from_str(&e.to_string()))?;

    let response = match msg {
        Message::Text { content } => {
            format!("收到文本: {}", content)
        }
        Message::Image { url, width, height } => {
            format!("收到图片: {} ({}x{})", url, width, height)
        }
        Message::Command { action, args } => {
            format!("执行命令: {} {:?}", action, args)
        }
    };

    serde_wasm_bindgen::to_value(&response)
        .map_err(|e| JsValue::from_str(&e.to_string()))
}
```

### 2.4 性能对比

```rust
// js-sys 方式 - 较慢
fn js_sys_way(data: &[u32]) -> JsValue {
    let arr = js_sys::Array::new();
    for &val in data {
        arr.push(&JsValue::from(val));
    }
    arr.into()
}

// serde-wasm-bindgen 方式 - 更快
fn serde_way(data: &[u32]) -> JsValue {
    serde_wasm_bindgen::to_value(data).unwrap()
}
```

### 2.5 自定义序列化

```rust
#[derive(Serialize)]
pub struct Response<T: Serialize> {
    pub success: bool,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub data: Option<T>,
    #[serde(skip_serializing_if = "Option::is_none")]
    pub error: Option<String>,
}

#[wasm_bindgen]
pub fn create_response(success: bool, data: JsValue) -> JsValue {
    if success {
        serde_wasm_bindgen::to_value(&Response::<()> {
            success: true,
            data: Some(()),
            error: None,
        }).unwrap()
    } else {
        serde_wasm_bindgen::to_value(&Response::<()> {
            success: false,
            data: None,
            error: Some("操作失败".into()),
        }).unwrap()
    }
}
```

## 三、注意事项与常见陷阱

1. **性能优势**：比 js-sys 手动构建对象快 2-5 倍
2. **类型安全**：反序列化到具体类型，编译时检查字段
3. **Option 映射**：`Option<T>` 映射为 `T | undefined`
4. **枚举处理**：通过 `#[serde(tag)]` 控制 JS 表示形式
5. **大对象**：超大结构体序列化仍可能有性能瓶颈
