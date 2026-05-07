# WebAPI封装

## 一、概念说明

web-sys crate 为大多数 Web API 提供了 Rust 绑定，包括 Fetch、WebSocket、WebGL、Web Audio 等。

```rust
use web_sys::window;

fn get_location() -> String {
    let location = window().unwrap().location();
    format!("{}{}", location.origin().unwrap(), location.pathname().unwrap())
}
```

## 二、具体用法

### 2.1 Fetch API

```rust
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, RequestMode, Response};

pub async fn api_request(url: &str, method: &str) -> Result<String, JsValue> {
    let opts = RequestInit::new();
    opts.set_method(method);
    opts.set_mode(RequestMode::Cors);

    let request = Request::new_with_str_and_init(url, &opts)?;
    request.headers().set("Content-Type", "application/json")?;

    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}
```

### 2.2 WebSocket

```rust
use web_sys::WebSocket;

pub fn create_ws(url: &str) -> Result<WebSocket, JsValue> {
    let ws = WebSocket::new(url)?;

    let onmessage = Closure::wrap(Box::new(move |e: web_sys::MessageEvent| {
        let msg = e.data().as_string().unwrap();
        web_sys::console::log_1(&format!("收到: {}", msg).into());
    }) as Box<dyn FnMut(_)>);

    ws.set_onmessage(Some(onmessage.as_ref().unchecked_ref()));
    onmessage.forget();

    Ok(ws)
}
```

### 2.3 LocalStorage

```rust
pub fn save_to_storage(key: &str, value: &str) -> Result<(), JsValue> {
    let storage = web_sys::window()
        .unwrap()
        .local_storage()?
        .unwrap();
    storage.set_item(key, value)
}

pub fn load_from_storage(key: &str) -> Option<String> {
    web_sys::window()
        .unwrap()
        .local_storage()
        .unwrap()
        .unwrap()
        .get_item(key)
        .unwrap()
}
```

### 2.4 Clipboard API

```rust
use wasm_bindgen_futures::JsFuture;

pub async fn copy_to_clipboard(text: &str) -> Result<(), JsValue> {
    let clipboard = web_sys::window()
        .unwrap()
        .navigator()
        .clipboard()
        .unwrap();

    JsFuture::from(clipboard.write_text(text)).await?;
    Ok(())
}

pub async fn read_clipboard() -> Result<String, JsValue> {
    let clipboard = web_sys::window()
        .unwrap()
        .navigator()
        .clipboard()
        .unwrap();

    let text = JsFuture::from(clipboard.read_text()).await?;
    Ok(text.as_string().unwrap())
}
```

## 三、注意事项与常见陷阱

1. **API 可用性**：部分 Web API 仅在 HTTPS 下可用（如 Clipboard）
2. **异步处理**：许多 Web API 是异步的，需配合 wasm-bindgen-futures
3. **浏览器兼容性**：不同浏览器 API 支持程度不同
4. **错误处理**：Web API 可能抛出异常，需用 Result 处理
5. **Feature Flags**：web-sys 需启用对应 feature 才能使用对应 API
