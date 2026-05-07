# Rust 语言概述

## 一、概念说明

Rust 是一门系统级编程语言，以安全性、并发性和高性能为核心设计目标。它通过所有权系统在编译时消除内存安全问题，无需垃圾回收。

```rust
// Rust 核心特性
// 1. 零成本抽象 - 高级抽象不引入运行时开销
// 2. 所有权系统 - 编译时内存安全保证
// 3. 无数据竞争 - 编译时并发安全保证
// 4. 跨平台 - 支持多种操作系统和架构

fn main() {
    println!("你好，Rust！");
}
```

## 二、安装与配置

```bash
# 安装 Rust (使用 rustup)
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# Windows: 下载 rustup-init.exe
# https://rustup.rs

# 验证安装
rustc --version
cargo --version

# 更新 Rust
rustup update

# 卸载
rustup self uninstall
```

## 三、Hello World

```rust
// main.rs
fn main() {
    // 打印到控制台
    println!("Hello, world!");

    // 带参数的打印
    let name = "Rust";
    println!("你好，{}！", name);

    // 格式化打印
    let x = 42;
    println!("十进制: {}", x);
    println!("十六进制: {:x}", x);
    println!("二进制: {:b}", x);
    println!("八进制: {:o}", x);
}
```

## 四、Cargo 项目管理

```bash
# 创建新项目
cargo new hello_world
cd hello_world

# 项目结构
/*
hello_world/
├── Cargo.toml      # 项目配置
├── src/
│   └── main.rs     # 入口文件
├── .gitignore
└── .git/
*/

# 编译并运行
cargo run

# 只编译
cargo build

# 编译发布版本
cargo build --release

# 检查代码（不生成二进制）
cargo check

# 运行测试
cargo test

# 生成文档
cargo doc --open
```

```toml
# Cargo.toml
[package]
name = "hello_world"
version = "0.1.0"
edition = "2021"

[dependencies]
serde = { version = "1.0", features = ["derive"] }
tokio = { version = "1", features = ["full"] }
```

## 五、开发工具

```bash
# 推荐 IDE: VS Code + rust-analyzer 扩展

# 常用 cargo 命令
cargo fmt          # 格式化代码
cargo clippy       # 代码检查（lint）
cargo audit        # 安全审计
cargo watch -x run # 文件变化自动运行
```

## 六、注意事项

1. **编译器严格**：Rust 编译器非常严格，未通过编译的代码不会运行
2. **所有权概念**：学习 Rust 最大的障碍是理解所有权系统
3. **借用检查器**：编译时检查引用的有效性
4. **社区友好**：Rust 社区非常友好，遇到问题可以求助
5. **学习曲线**：Rust 学习曲线较陡，但掌握后效率很高
