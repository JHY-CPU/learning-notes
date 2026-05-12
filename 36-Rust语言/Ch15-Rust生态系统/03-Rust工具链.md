# Rust工具链

## 一、概念说明

Rust 工具链包括 rustc（编译器）、rustup（版本管理）、cargo（包管理）等核心工具。

```bash
# 安装和管理
rustup install stable
rustup install nightly
rustup default stable
rustup update
```

## 二、具体用法

### 2.1 rustup 版本管理

```bash
# 查看已安装版本
rustup show
rustup toolchain list

# 切换版本
rustup default nightly

# 项目特定版本
echo "nightly" > rust-toolchain

# 添加组件
rustup component add clippy
rustup component add rustfmt
rustup component add rust-analyzer
```

### 2.2 rustc 编译器

```bash
# 基本编译
rustc main.rs -o main

# 优化级别
rustc -O main.rs

# 目标平台
rustc --target wasm32-unknown-unknown main.rs

# 生成汇编
rustc --emit=asm main.rs
```

### 2.3 cargo 子命令

```bash
# 安装子命令
cargo install cargo-watch
cargo install cargo-edit
cargo install cargo-expand

# 常用子命令
cargo watch -x run  # 自动重新运行
cargo add serde     # 添加依赖
cargo rm serde      # 移除依赖
cargo expand        # 展开宏
```

### 2.4 rust-analyzer IDE 配置

```json
// VS Code settings.json
{
    "rust-analyzer.checkOnSave.command": "clippy",
    "rust-analyzer.inlayHints.typeHints.enable": true,
    "rust-analyzer.inlayHints.chainingHints.enable": true,
    "rust-analyzer.lens.enable": true,
    "rust-analyzer.completion.autoimport.enable": true,
    "[rust]": {
        "editor.formatOnSave": true,
        "editor.defaultFormatter": "rust-lang.rust-analyzer"
    }
}
```

### 2.5 项目特定工具链

```toml
# rust-toolchain.toml
[toolchain]
channel = "stable"
components = ["rustfmt", "clippy", "rust-analyzer"]
targets = ["wasm32-unknown-unknown", "aarch64-unknown-linux-gnu"]

# 或简单版本
# rust-toolchain
stable
```

### 2.6 环境变量配置

```bash
# 加速编译
export CARGO_BUILD_JOBS=8
export RUSTC_WRAPPER=sccache  # 使用 sccache 缓存编译

# 配置代理
export HTTP_PROXY=http://proxy:8080
export HTTPS_PROXY=http://proxy:8080

# 配置镜像源（中国区）
export CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse
```

## 四、工具链组件速查

| 组件 | 用途 | 安装 |
|------|------|------|
| rustc | 编译器 | 自动安装 |
| cargo | 包管理器 | 自动安装 |
| rustfmt | 代码格式化 | `rustup component add rustfmt` |
| clippy | Lint 工具 | `rustup component add clippy` |
| rust-analyzer | LSP 服务器 | `rustup component add rust-analyzer` |
| rust-docs | 文档 | `rustup component add rust-docs` |
| miri | 未定义行为检测 | `rustup component add miri` |
| rust-src | 标准库源码 | `rustup component add rust-src` |

## 五、注意事项与常见陷阱

1. **版本兼容**：确保工具链版本兼容，使用 rust-toolchain.toml 固定版本
2. **夜间版本**：某些功能需要 nightly，仅在需要时使用
3. **组件管理**：定期更新组件，保持工具链最新
4. **缓存清理**：rustup self uninstall 清理，或手动删除 ~/.rustup
5. **代理配置**：配置代理以加速下载，特别在中国网络环境
6. **sccache**：使用 sccache 加速重复编译，特别适合 CI
7. **交叉编译工具**：交叉编译需要安装目标平台的工具链
