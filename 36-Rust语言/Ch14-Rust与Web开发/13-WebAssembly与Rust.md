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

## 三、注意事项与常见陷阱

1. **包大小**：优化 WASM 包大小
2. **调试工具**：使用 browser 开发者工具调试
3. **内存管理**：注意 WASM 内存管理
4. **异步支持**：WASM 支持异步操作
5. **生态系统**：使用 wasm-bindgen 和 web-sys 与浏览器交互
