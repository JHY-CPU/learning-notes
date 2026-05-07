# trunk构建工具

## 一、概念说明

Trunk 是专为 Rust WASM 前端应用设计的构建工具，类似 webpack/vite，支持自动编译、打包、HMR 热更新。

```bash
cargo install trunk
trunk serve --open       # 开发服务器 + 热更新
trunk build --release    # 生产构建
```

## 二、具体用法

### 2.1 index.html 配置

```html
<!DOCTYPE html>
<html>
<head>
    <!-- Trunk 会自动处理以下标签 -->
    <link data-trunk rel="rust" data-wasm-opt="s" />
    <link data-trunk rel="css" href="style.css" />
    <link data-trunk rel="copy-dir" href="assets/" />
</head>
<body>
    <div id="app"></div>
</body>
</html>
```

### 2.2 项目结构

```
my-app/
├── index.html          # 入口 HTML
├── Cargo.toml          # Rust 包配置
├── src/
│   └── main.rs         # WASM 入口
├── style.css           # 样式文件
├── assets/             # 静态资源
└── Trunk.toml          # Trunk 配置（可选）
```

### 2.3 Trunk.toml 配置

```toml
[build]
target = "index.html"
dist = "dist"
release = false
public_url = "/"

[watch]
watch = ["src/", "assets/"]
ignore = []

[serve]
address = "127.0.0.1"
port = 8080
open = false
```

### 2.4 资源处理

```html
<!-- 编译 Rust -->
<link data-trunk rel="rust" data-wasm-opt="s" data-bin="my-app" />

<!-- 复制单文件 -->
<link data-trunk rel="copy-file" href="favicon.ico" />

<!-- 复制目录 -->
<link data-trunk rel="copy-dir" href="public/" />

<!-- SCSS 编译 -->
<link data-trunk rel="scss" href="style.scss" />

<!-- 内联 CSS -->
<link data-trunk rel="inline-css" href="critical.css" />
```

### 2.5 main.rs 示例

```rust
use wasm_bindgen::prelude::*;

fn main() {
    wasm_logger::init(wasm_logger::Config::default());
    console_error_panic_hook::set_once();

    let window = web_sys::window().unwrap();
    let document = window.document().unwrap();
    let app = document.get_element_by_id("app").unwrap();
    app.set_inner_html("<h1>Hello Trunk!</h1>");

    log::info!("应用已启动");
}
```

## 三、注意事项与常见陷阱

1. **HTML 标签**：必须使用 `data-trunk` 属性标记资源
2. **路径解析**：资源路径相对于 index.html 所在目录
3. **HMR 限制**：Rust 代码变更需要重新编译，非真正的 HMR
4. **优化选项**：`data-wasm-opt="s"` 可减小 WASM 体积
5. **兼容性**：需 Rust 1.60+，建议使用最新稳定版
