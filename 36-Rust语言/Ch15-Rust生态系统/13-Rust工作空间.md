# Rust工作空间

## 一、概念说明

工作空间（Workspace）允许在单个仓库中管理多个相关的 crate。

```toml
# Cargo.toml (root)
[workspace]
members = [
    "core",
    "server",
    "client",
    "shared",
]
resolver = "2"
```

## 二、具体用法

### 2.1 工作空间配置

```toml
[workspace]
members = ["crates/*"]
exclude = ["crates/old"]

[workspace.dependencies]
serde = "1.0"
tokio = "1"

# 子 crate 使用
[dependencies]
serde = { workspace = true }
```

### 2.2 构建和测试

```bash
# 构建所有 crate
cargo build --workspace

# 测试所有 crate
cargo test --workspace

# 构建特定 crate
cargo build -p my_crate

# 查看依赖图
cargo tree --workspace
```

### 2.3 共享配置

```toml
# workspace Cargo.toml
[workspace.package]
version = "1.0.0"
edition = "2021"
license = "MIT"

# 子 crate 继承
[package]
name = "my_crate"
version.workspace = true
edition.workspace = true
```

### 2.4 内部依赖

```toml
# server/Cargo.toml
[dependencies]
shared = { path = "../shared" }
core = { path = "../core" }
```

## 三、注意事项与常见陷阱

1. **版本同步**：使用 workspace 版本保持同步
2. **依赖共享**：避免重复依赖声明
3. **构建顺序**：Cargo 自动处理构建顺序
4. **CI 配置**：CI 中构建整个工作空间
5. **发布顺序**：按依赖顺序发布 crate
