# wasm-bindgen深入

## 一、概念说明

wasm-bindgen 是 Rust 与 JavaScript 互操作的核心工具，提供了自动类型转换、类导出、模块绑定等功能。

```rust
use wasm_bindgen::prelude::*;

#[wasm_bindgen]
pub fn greet(name: &str) -> String {
    format!("Hello, {}!", name)
}

// 生成对应的 JS 绑定代码
// export function greet(name: string): string;
```

## 二、具体用法

### 2.1 结构体导出

```rust
#[wasm_bindgen]
pub struct GameEngine {
    width: u32,
    height: u32,
    state: Vec<u8>,
}

#[wasm_bindgen]
impl GameEngine {
    #[wasm_bindgen(constructor)]
    pub fn new(width: u32, height: u32) -> GameEngine {
        GameEngine {
            width,
            height,
            state: vec![0; (width * height) as usize],
        }
    }

    #[wasm_bindgen(getter)]
    pub fn width(&self) -> u32 { self.width }

    pub fn tick(&mut self) {
        // 游戏逻辑
    }

    pub fn render(&self) -> Vec<u8> {
        self.state.clone()
    }
}
```

### 2.2 枚举导出

```rust
#[wasm_bindgen]
#[derive(Clone, Copy)]
pub enum Direction {
    Up = 0,
    Down = 1,
    Left = 2,
    Right = 3,
}

#[wasm_bindgen]
pub fn move_player(dir: Direction) -> (i32, i32) {
    match dir {
        Direction::Up => (0, -1),
        Direction::Down => (0, 1),
        Direction::Left => (-1, 0),
        Direction::Right => (1, 0),
    }
}
```

### 2.3 JS 名称映射

```rust
#[wasm_bindgen]
extern "C" {
    // 导入 JS 函数并重命名
    #[wasm_bindgen(js_name = setTimeout)]
    fn set_timeout(callback: &Closure<dyn FnMut()>, ms: i32) -> i32;

    // 导入 JS 类
    type HTMLCanvasElement;
    #[wasm_bindgen(method, getter)]
    fn width(this: &HTMLCanvasElement) -> u32;

    // 可选参数
    #[wasm_bindgen(js_namespace = console)]
    fn log(s: &str);
}
```

### 2.4 catch 属性

```rust
#[wasm_bindgen]
extern "C" {
    // 捕获 JS 异常返回 Result
    #[wasm_bindgen(catch)]
    fn eval(code: &str) -> Result<JsValue, JsValue>;

    // 忽略异常
    #[wasm_bindgen(catch, js_name = document.getElementById)]
    fn get_element_by_id(id: &str) -> Result<Option<web_sys::Element>, JsValue>;
}
```

### 2.5 类型细化

```rust
#[wasm_bindgen(typescript_type = "Array<number>")]
pub struct NumberArray(Vec<f64>);

#[wasm_bindgen]
impl NumberArray {
    #[wasm_bindgen(constructor)]
    pub fn new() -> NumberArray { NumberArray(vec![]) }

    pub fn push(&mut self, val: f64) { self.0.push(val); }

    pub fn sum(&self) -> f64 { self.0.iter().sum() }
}
```

## 三、注意事项与常见陷阱

1. **pub 限制**：只有 `pub` 的函数和结构体会被导出
2. **引用限制**：不能导出包含引用的结构体（除 `&str`）
3. **async 支持**：async fn 自动转换为返回 Promise 的函数
4. **命名转换**：Rust snake_case 自动转为 JS camelCase
5. **生成文件**：wasm-bindgen 会生成 `.js` 和 `.d.ts` 文件
