# WebAssembly与Rust

## 一、概念说明

Rust 可以编译为 WebAssembly，在浏览器中运行高性能代码。

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("你好，{}！", name)
}
```

## 二、具体用法

### 2.1 wasm-bindgen 基础

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
extern "C" {
    fn alert(s: &str);
}

#[wasm_bindgen]
pub fn greet(name: &str) {
    alert(&format!("你好，{}！", name));
}
```

### 2.2 与 JavaScript 交互

```rust
use wasm_bindgen::prelude::*;
use web_sys::console;

#[wasm_bindgen(start)]
pub fn main() {
    console::log_1(&"WASM 已加载".into());
}

#[wasm_bindgen]
pub fn process_data(data: &[u8]) -> Vec<u8> {
    data.iter().map(|b| b.wrapping_mul(2)).collect()
}
```

### 2.3 构建配置

```toml
# Cargo.toml
[lib]
crate-type = ["cdylib"]

[dependencies]
wasm-bindgen = "0.2"
web-sys = "0.3"
```

```bash
wasm-pack build --target web
```

### 2.4 DOM 操作

```rust
use wasm_bindgen::prelude::*;
use web_sys::{Document, Element, HtmlElement};

#[wasm_bindgen]
pub fn create_element() -> Result<(), JsValue> {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();

    let div: Element = document.create_element("div")?;
    div.set_text_content(Some("Hello from Rust WASM!"));
    div.set_id("rust-element");

    let body = document.body().unwrap();
    body.append_child(&div)?;

    Ok(())
}
```

### 2.5 包大小优化

```toml
# Cargo.toml 优化配置
[profile.release]
opt-level = "z"  # 优化大小
lto = true
codegen-units = 1
strip = true
```

```bash
# 使用 wasm-opt 进一步优化
wasm-opt -Oz -o output.wasm input.wasm

# 使用 wee_alloc 减小分配器大小
# Cargo.toml: wee_alloc = "0.4"
```

## 四、WASM 与 JS 互操作

| 功能 | Rust 侧 | JS 侧 |
|------|---------|-------|
| 导出函数 | `#[wasm_bindgen]` | 直接调用 |
| 导入函数 | `extern "C"` | 定义在 JS 中 |
| 复杂类型 | `JsValue` | 任意 JS 对象 |
| 异步 | `wasm_bindgen_futures` | Promise |
| DOM | `web_sys` | 原生 DOM API |

## 五、注意事项与常见陷阱

1. **包大小**：优化 WASM 包大小，使用 wasm-opt 和 LTO
2. **调试工具**：使用浏览器开发者工具调试 WASM 代码
3. **内存管理**：注意 WASM 线性内存管理，避免内存泄漏
4. **异步支持**：WASM 支持异步操作，使用 wasm_bindgen_futures
5. **生态系统**：使用 wasm-bindgen 和 web-sys 与浏览器交互
6. **浏览器兼容**：检查目标浏览器的 WASM 支持
7. **性能分析**：使用性能分析工具找出瓶颈
