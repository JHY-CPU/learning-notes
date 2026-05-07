# Promise与异步

## 一、概念说明

WASM 支持 JavaScript Promise，实现异步操作。

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;

#[wasm_bindgen]
pub async fn fetch_data(url: &str) -> Result<String, JsValue> {
    let window = web_sys::window().unwrap();
    let resp = JsFuture::from(window.fetch_with_str(url)).await?;
    let resp: web_sys::Response = resp.dyn_into()?;
    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}
```

## 二、具体用法

### 2.1 Promise 创建

```rust
use js_sys::Promise;

#[wasm_bindgen]
pub fn create_promise() -> Promise {
    Promise::new(&mut |resolve, _reject| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, 1000)
            .unwrap();
    })
}
```

### 2.2 Promise 链

```rust
#[wasm_bindgen]
pub async fn chain_promises() -> Result<String, JsValue> {
    let first = fetch_data("api/step1").await?;
    let second = fetch_data(&format!("api/step2?data={}", first)).await?;
    Ok(second)
}
```

### 2.3 错误处理

```rust
#[wasm_bindgen]
pub async fn safe_fetch(url: &str) -> Result<String, JsValue> {
    match fetch_data(url).await {
        Ok(data) => Ok(data),
        Err(e) => {
            console::log_1(&format!("错误: {:?}", e).into());
            Err(e)
        }
    }
}
```

### 2.4 并发操作

```rust
use futures::future::join_all;

#[wasm_bindgen]
pub async fn fetch_multiple(urls: &js_sys::Array) -> js_sys::Array {
    let mut futures = vec![];
    for i in 0..urls.length() {
        let url = urls.get(i).as_string().unwrap();
        futures.push(fetch_data(&url));
    }

    let results = join_all(futures).await;
    let arr = js_sys::Array::new();
    for result in results {
        if let Ok(data) = result {
            arr.push(&data.into());
        }
    }
    arr
}
```

## 三、注意事项与常见陷阱

1. **异步运行时**：使用 wasm-bindgen-futures
2. **错误传播**：正确处理异步错误
3. **取消操作**：支持取消异步操作
4. **性能**：异步操作有开销
5. **调试**：异步调试更困难
