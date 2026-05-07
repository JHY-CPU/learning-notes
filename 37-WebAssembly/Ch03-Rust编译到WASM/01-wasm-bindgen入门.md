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
