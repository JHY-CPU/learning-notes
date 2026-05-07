# wasm-pack工具

## 一、概念说明

wasm-pack 是构建、测试和发布 Rust WASM 包的命令行工具，可生成 npm 兼容的包。

```bash
# 安装
cargo install wasm-pack

# 构建 npm 包
wasm-pack build --target web

# 发布到 npm
wasm-pack publish
```

## 二、具体用法

### 2.1 构建目标

```bash
# browser 目标 - 标准 npm 包，require/import 使用
wasm-pack build --target bundler

# web 目标 - ES 模块，浏览器原生 import
wasm-pack build --target web

# nodejs 目标 - Node.js CommonJS 模块
wasm-pack build --target nodejs

# no-modules 目标 - 全局变量，适合简单页面
wasm-pack build --target no-modules
```

### 2.2 Cargo.toml 配置

```toml
[package]
name = "my-wasm-lib"
version = "0.1.0"
edition = "2021"
authors = ["Your Name <you@example.com>"]
description = "A WASM library"
license = "MIT"
repository = "https://github.com/user/repo"

[lib]
crate-type = ["cdylib", "rlib"]

[dependencies]
wasm-bindgen = "0.2"
web-sys = { version = "0.3", features = ["console"] }
js-sys = "0.3"

[dev-dependencies]
wasm-bindgen-test = "0.3"

[profile.release]
opt-level = "s"       # 优化体积
lto = true            # 链接时优化
codegen-units = 1     # 单编译单元
```

### 2.3 测试

```bash
# 在 headless 浏览器中运行测试
wasm-pack test --headless --chrome

# 在 Firefox 中测试
wasm-pack test --headless --firefox

# 使用 Node.js 测试
wasm-pack test --nodejs
```

```rust
// tests/web.rs
use wasm_bindgen_test::*;

wasm_bindgen_test_configure!(run_in_browser);

#[wasm_bindgen_test]
fn pass() {
    assert_eq!(1 + 1, 2);
}

#[wasm_bindgen_test]
fn test_dom() {
    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let div = document.create_element("div").unwrap();
    div.set_inner_text("test");
    assert_eq!(div.text_content().unwrap(), "test");
}
```

### 2.4 生成的包结构

```
pkg/
├── my_wasm_lib.js          # JS 绑定
├── my_wasm_lib.d.ts        # TypeScript 类型声明
├── my_wasm_lib_bg.wasm     # WASM 二进制
├── my_wasm_lib_bg.wasm.d.ts # WASM 类型
└── package.json            # npm 包元数据
```

### 2.5 前端集成

```javascript
// Vite/Webpack 项目
import init, { greet } from 'my-wasm-lib';

async function main() {
    await init(); // 初始化 WASM
    greet('World');
}
main();
```

```html
<!-- 直接在 HTML 中使用 -->
<script type="module">
    import init, { greet } from './pkg/my_wasm_lib.js';
    await init();
    greet('Browser');
</script>
```

## 三、注意事项与常见陷阱

1. **target 选择**：web target 最灵活，bundler target 兼容性最好
2. **npm 命名**：包名使用 kebab-case，文件名自动转为 snake_case
3. **版本管理**：Cargo.toml version 和 package.json version 同步
4. **优化**：release profile 配置对 WASM 体积影响很大
5. **CI 集成**：GitHub Actions 有专门的 wasm-pack action
