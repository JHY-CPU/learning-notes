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

## 三、注意事项与常见陷阱

1. **版本兼容**：确保工具链版本兼容
2. **夜间版本**：某些功能需要 nightly
3. **组件管理**：定期更新组件
4. **缓存清理**：rustup self uninstall 清理
5. **代理配置**：配置代理以加速下载
