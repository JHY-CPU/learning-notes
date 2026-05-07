# Clippy与lint

## 一、概念说明

Clippy 是 Rust 官方 lint 工具，检查代码中的常见问题和改进建议。

```bash
cargo clippy
cargo clippy -- -W clippy::all
cargo clippy --fix
```

## 二、具体用法

### 2.1 配置 Clippy

```toml
# .clippy.toml
too-many-arguments-threshold = 7

# Cargo.toml
[lints.clippy]
all = "warn"
pedantic = "warn"
module_name_repetitions = "allow"
```

### 2.2 常用 lint 规则

```rust
// 常见警告
// clippy::needless_return: 不需要的 return
// clippy::single_match: 使用 if let 替代
// clippy::clone_on_copy: Copy 类型无需 clone
// clippy::redundant_closure: 简化闭包

// 示例
let x = Some(5);
// Clippy 建议
if let Some(v) = x {
    println!("{}", v);
}
```

### 2.3 自定义 lint

```rust
// 使用 #[allow] 抑制特定警告
#[allow(clippy::too_many_arguments)]
fn complex_function(a: i32, b: i32, c: i32, d: i32, e: i32, f: i32, g: i32) {}

// 在模块级别抑制
#![allow(clippy::module_name_repetitions)]
```

## 三、注意事项与常见陷阱

1. **CI 集成**：在 CI 中运行 Clippy
2. **渐进采用**：逐步启用更严格的 lint
3. **误报处理**：合理使用 allow
4. **自动修复**：使用 --fix 自动修复
5. **团队规范**：团队统一 lint 配置
