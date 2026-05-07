# web-sys使用

## 一、概念说明

web-sys 提供了 Rust 对 Web API 的绑定，可以在 WASM 中直接调用浏览器 API。

```rust
use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen]
pub fn log_message(msg: &str) {
    console::log_1(&msg.into());
}
```

## 二、具体用法

### 2.1 DOM 操作

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement};

#[wasm_bindgen]
pub fn create_element(tag: &str) -> Result<Element, JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let element = document.create_element(tag)?;
    Ok(element)
}

#[wasm_bindgen]
pub fn set_text_content(element: &Element, text: &str) {
    element.set_text_content(Some(text));
}
```

### 2.2 事件处理

```rust
use wasm_bindgen::prelude::*;
use web_sys::{EventTarget, HtmlButtonElement};

#[wasm_bindgen]
pub fn add_click_handler(button: &HtmlButtonElement) -> Result<(), JsValue> {
    let closure = Closure::wrap(Box::new(move || {
        console::log_1(&"按钮被点击！".into());
    }) as Box<dyn Fn()>);

    button.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;
    closure.forget(); // 防止闭包被回收
    Ok(())
}
```

### 2.3 Fetch API

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

#[wasm_bindgen]
pub async fn fetch_data(url: &str) -> Result<String, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("GET");

    let request = Request::new_with_str_and_init(url, &opts)?;
    let window = web_sys::window().unwrap();
    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}
```

### 2.4 Canvas 操作

```rust
use wasm_bindgen::prelude::*;
use web_sys::{CanvasRenderingContext2d, HtmlCanvasElement};

#[wasm_bindgen]
pub fn draw_circle(canvas: &HtmlCanvasElement, x: f64, y: f64, radius: f64) {
    let context = canvas
        .get_context("2d")
        .unwrap()
        .unwrap()
        .dyn_into::<CanvasRenderingContext2d>()
        .unwrap();

    context.begin_path();
    context.arc(x, y, radius, 0.0, 2.0 * std::f64::consts::PI).unwrap();
    context.fill();
}
```

## 三、注意事项与常见陷阱

1. **可选值**：许多 web-sys 方法返回 Option
2. **错误处理**：使用 Result 处理错误
3. **生命周期**：注意 JavaScript 对象的生命周期
4. **性能**：频繁 DOM 操作有性能开销
5. **类型转换**：使用 dyn_into 进行类型转换
