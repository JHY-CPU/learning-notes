# wasm-bindgen入门

## 一、概念说明

wasm-bindgen 是 Rust 和 JavaScript 之间的桥梁，生成类型安全的绑定代码。

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("你好，{}！", name)
}
```

## 二、具体用法

### 2.1 基本绑定

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn add(a: i32, b: i32) -> i32 {
    a + b
}

#[wasm_bindgen]
pub fn fibonacci(n: u32) -> u32 {
    match n {
        0 => 0,
        1 => 1,
        _ => fibonacci(n - 1) + fibonacci(n - 2),
    }
}
```

### 2.2 结构体导出

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub struct Calculator {
    value: f64,
}

#[wasm_bindgen]
impl Calculator {
    #[wasm_bindgen(constructor)]
    pub fn new(initial: f64) -> Calculator {
        Calculator { value: initial }
    }

    pub fn add(&mut self, x: f64) {
        self.value += x;
    }

    pub fn get_value(&self) -> f64 {
        self.value
    }
}
```

### 2.3 JavaScript 集成

```javascript
import init, { greet, Calculator } from './pkg/my_lib.js';

async function main() {
  await init();

  console.log(greet('世界'));

  const calc = new Calculator(0);
  calc.add(5);
  console.log(calc.get_value());
}

main();
```

### 2.4 构建配置

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
```

```bash
# 构建
wasm-pack build --target web
```

## 三、注意事项与常见陷阱

1. **类型转换**：注意 Rust 和 JavaScript 类型转换
2. **所有权**：wasm-bindgen 管理所有权
3. **内存管理**：注意内存泄漏
4. **异步支持**：使用 wasm-bindgen-futures
5. **错误处理**：Result 类型转换为 JavaScript 异常

## 五、类型映射速查

| Rust 类型 | JavaScript 类型 | 说明 |
|-----------|---------------|------|
| `i32` | `number` | 32 位有符号整数 |
| `u32` | `number` | 32 位无符号整数 |
| `f32` / `f64` | `number` | 浮点数 |
| `bool` | `boolean` | 布尔值 |
| `String` / `&str` | `string` | 字符串 |
| `Vec<T>` | `Array` | 数组 |
| `Box<[T]>` | `Array` | 固定数组 |
| `JsValue` | `any` | 任意 JS 值 |
| `Result<T, JsValue>` | 抛出异常 | 错误处理 |
| `()` | `undefined` | 空值 |
| `Option<T>` | `T \| undefined` | 可选值 |
| `Closure` | `Function` | 闭包 |

## 六、使用 web-sys 操作 DOM

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement, Window};

#[wasm_bindgen]
pub fn setup_page() -> Result<(), JsValue> {
    let window: Window = web_sys::window().unwrap();
    let document: Document = window.document().unwrap();

    // 创建一个 div 元素
    let div: Element = document.create_element("div")?;
    div.set_id("app");
    div.set_inner_html("<h1>Hello from Rust WASM!</h1>");

    // 添加到 body
    let body: HtmlElement = document.body().unwrap();
    body.append_child(&div)?;

    // 设置样式
    let style = div.dyn_ref::<HtmlElement>().unwrap();
    style.style().set_property("color", "blue")?;
    style.style().set_property("font-size", "24px")?;

    Ok(())
}

#[wasm_bindgen]
pub fn add_click_listener(element_id: &str) -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();

    let element = document
        .get_element_by_id(element_id)
        .ok_or_else(|| JsValue::from_str("Element not found"))?;

    let closure = Closure::wrap(Box::new(move || {
        web_sys::console::log_1(&"Button clicked!".into());
    }) as Box<dyn Fn()>);

    element.add_event_listener_with_callback("click", closure.as_ref().unchecked_ref())?;

    // 防止闭包被释放
    closure.forget();

    Ok(())
}
```

Cargo.toml 依赖配置：

```toml
[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = [
  "Document",
  "Element",
  "HtmlElement",
  "Window",
  "console",
  "EventTarget",
] }
js-sys = "0.3"
```

## 七、使用 js-sys 操作 JavaScript 对象

```rust
use wasm_bindgen::prelude::*;
use js_sys::{Array, Date, JSON, Map, Object, Reflect};

#[wasm_bindgen]
pub fn create_js_object() -> JsValue {
    let obj = Object::new();

    // 设置属性
    Reflect::set(&obj, &"name".into(), &"Rust WASM".into()).unwrap();
    Reflect::set(&obj, &"version".into(), &"1.0.0".into()).unwrap();
    Reflect::set(&obj, &"timestamp".into(), &Date::new_0().get_time().into()).unwrap();

    obj.into()
}

#[wasm_bindgen]
pub fn process_js_array(arr: JsValue) -> Result<JsValue, JsValue> {
    let array: Array = arr.dyn_into()?;

    let result = Array::new();
    for i in 0..array.length() {
        let val = array.get(i);
        // 将每个元素翻倍（如果是数字）
        if let Some(num) = val.as_f64() {
            result.push(&(num * 2.0).into());
        } else {
            result.push(&val);
        }
    }

    Ok(result.into())
}

#[wasm_bindgen]
pub fn stringify_json(obj: JsValue) -> Result<String, JsValue> {
    let json_string = JSON::stringify(&obj)?;
    Ok(json_string.into())
}
```

## 八、错误处理最佳实践

```rust
use wasm_bindgen::prelude::*;

// 定义错误类型
#[wasm_bindgen]
pub struct WasmError {
    message: String,
    code: u32,
}

#[wasm_bindgen]
impl WasmError {
    #[wasm_bindgen(constructor)]
    pub fn new(message: &str, code: u32) -> WasmError {
        WasmError {
            message: message.to_string(),
            code,
        }
    }

    #[wasm_bindgen(getter)]
    pub fn message(&self) -> String {
        self.message.clone()
    }

    #[wasm_bindgen(getter)]
    pub fn code(&self) -> u32 {
        self.code
    }
}

// 使用 Result 处理错误
#[wasm_bindgen]
pub fn safe_divide(a: f64, b: f64) -> Result<f64, JsValue> {
    if b == 0.0 {
        return Err(JsValue::from_str("Division by zero"));
    }
    Ok(a / b)
}

// 将 Rust 错误转换为 JsValue
impl From<std::io::Error> for WasmError {
    fn from(err: std::io::Error) -> Self {
        WasmError {
            message: err.to_string(),
            code: 500,
        }
    }
}
```

```javascript
// JavaScript 端错误处理
import init, { safe_divide } from './pkg/my_lib.js';

await init();
try {
  const result = safe_divide(10, 0);
} catch (error) {
  console.error('WASM error:', error); // "Division by zero"
}
```

## 九、异步操作

```rust
use wasm_bindgen::prelude::*;
use wasm_bindgen_futures::JsFuture;
use web_sys::{Request, RequestInit, Response};

#[wasm_bindgen]
pub async fn fetch_url(url: &str) -> Result<String, JsValue> {
    let mut opts = RequestInit::new();
    opts.method("GET");

    let request = Request::new_with_str_and_init(url, &opts)?;
    let window = web_sys::window().unwrap();

    let resp_value = JsFuture::from(window.fetch_with_request(&request)).await?;
    let resp: Response = resp_value.dyn_into()?;

    if !resp.ok() {
        return Err(JsValue::from_str(&format!(
            "HTTP error: {}",
            resp.status()
        )));
    }

    let text = JsFuture::from(resp.text()?).await?;
    Ok(text.as_string().unwrap())
}

// 定时器
#[wasm_bindgen]
pub async fn delay(ms: i32) {
    let promise = js_sys::Promise::new(&mut |resolve, _| {
        web_sys::window()
            .unwrap()
            .set_timeout_with_callback_and_timeout_and_arguments_0(&resolve, ms)
            .unwrap();
    });
    JsFuture::from(promise).await.unwrap();
}
```

```toml
# Cargo.toml 异步依赖
[dependencies]
wasm-bindgen-futures = "0.4"
js-sys = "0.3"
```
