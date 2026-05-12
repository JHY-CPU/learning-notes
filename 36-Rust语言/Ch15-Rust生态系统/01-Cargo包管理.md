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

### 2.4 Feature Flags

```toml
# Cargo.toml
[features]
default = ["std"]
std = []
async = ["tokio"]
full = ["std", "async", "serde"]

[dependencies]
tokio = { version = "1", optional = true, features = ["full"] }
serde = { version = "1.0", optional = true, features = ["derive"] }
```

```rust
// 条件编译
#[cfg(feature = "async")]
async fn async_function() { /* ... */ }

#[cfg(not(feature = "std"))]
fn no_std_function() { /* ... */ }
```

### 2.5 构建脚本 (build.rs)

```rust
// build.rs
fn main() {
    // 重新编译当文件变化时
    println!("cargo:rerun-if-changed=proto/");
    println!("cargo:rerun-if-changed=build.rs");

    // 条件编译标志
    if cfg!(target_os = "linux") {
        println!("cargo:rustc-cfg=linux_backend");
    }

    // 链接系统库
    println!("cargo:rustc-link-lib=ssl");
    println!("cargo:rustc-link-search=native=/usr/lib");
}
```

### 2.6 依赖优化技巧

```toml
# 选择轻量级依赖
# 使用 default-features = false 减少依赖树
serde = { version = "1.0", default-features = false, features = ["derive"] }

# 使用 workspace 共享依赖版本
[workspace.dependencies]
tokio = { version = "1", features = ["full"] }
serde = { version = "1.0", features = ["derive"] }

# 子 crate 中
[dependencies]
tokio = { workspace = true }
```

## 四、Cargo 命令速查

| 命令 | 用途 |
|------|------|
| `cargo new` | 创建新项目 |
| `cargo init` | 初始化已有目录 |
| `cargo build` | 构建项目 |
| `cargo run` | 构建并运行 |
| `cargo test` | 运行测试 |
| `cargo bench` | 运行基准测试 |
| `cargo doc` | 生成文档 |
| `cargo publish` | 发布到 crates.io |
| `cargo install` | 安装二进制 crate |
| `cargo update` | 更新依赖 |
| `cargo clean` | 清理构建产物 |
| `cargo tree` | 显示依赖树 |
| `cargo audit` | 检查安全漏洞 |
| `cargo fmt` | 格式化代码 |
| `cargo clippy` | 运行 lint 检查 |

## 五、注意事项与常见陷阱

1. **版本锁定**：使用 Cargo.lock 确保可重复构建，提交到版本控制
2. **依赖更新**：定期运行 cargo update，检查 breaking changes
3. **安全审计**：使用 cargo audit 检查安全漏洞，设置 CI 自动检查
4. **构建缓存**：清理缓存用 cargo clean，增量编译优化构建时间
5. **工作空间**：大型项目使用工作空间管理，共享依赖版本
6. **feature flags**：合理使用 feature flags 减少编译时间和二进制大小
7. **依赖树检查**：使用 `cargo tree` 检查依赖关系，避免循环依赖
