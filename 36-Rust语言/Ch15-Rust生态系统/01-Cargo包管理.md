# Cargo包管理

## 一、概念说明

Cargo 是 Rust 的包管理器和构建工具，管理依赖、构建、测试和发布。

```toml
# Cargo.toml
[package]
name = "my_project"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

## 二、具体用法

### 2.1 常用命令

```bash
# 创建新项目
cargo new my_project --lib
cargo init --bin

# 构建
cargo build
cargo build --release

# 运行
cargo run
cargo run -- --help

# 测试
cargo test
cargo test -- --nocapture

# 文档
cargo doc --open

# 发布
cargo publish
```

### 2.2 依赖管理

```toml
[dependencies]
# 版本指定
serde = "1.0"
serde = "^1.0"  # 兼容版本
serde = "~1.0.123"  # 补丁版本
serde = "=1.0.123"  # 精确版本

# Git 依赖
my_lib = { git = "https://github.com/user/repo", branch = "main" }

# 本地依赖
my_lib = { path = "../my_lib" }

# 可选依赖
[dependencies]
optional_dep = { version = "1.0", optional = true }

[features]
default = ["optional_dep"]
```

### 2.3 工作空间

```toml
# Cargo.toml (root)
[workspace]
members = [
    "crate1",
    "crate2",
]
resolver = "2"
```

## 三、注意事项与常见陷阱

1. **版本锁定**：使用 Cargo.lock 确保可重复构建
2. **依赖更新**：定期运行 cargo update
3. **安全审计**：使用 cargo audit 检查安全漏洞
4. **构建缓存**：清理缓存用 cargo clean
5. **工作空间**：大型项目使用工作空间管理
