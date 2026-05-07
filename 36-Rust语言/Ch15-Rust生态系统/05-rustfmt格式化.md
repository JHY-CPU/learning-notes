# rustfmt格式化

## 一、概念说明

rustfmt 是 Rust 官方代码格式化工具，统一代码风格。

```bash
cargo fmt
cargo fmt -- --check
```

## 二、具体用法

### 2.1 配置 rustfmt

```toml
# rustfmt.toml
max_width = 100
tab_spaces = 4
edition = "2021"
use_field_init_shorthand = true
imports_granularity = "Crate"
```

### 2.2 常用选项

```toml
# 基本配置
max_width = 100
hard_tabs = false
tab_spaces = 4

# 导入排序
imports_layout = "Vertical"
group_imports = "StdExternalCrate"

# 函数格式
fn_params_layout = "Tall"
brace_style = "SameLineWhere"
```

### 2.3 忽略格式化

```rust
// #[rustfmt::skip]
#[rustfmt::skip]
let matrix = vec![
    1, 0, 0,
    0, 1, 0,
    0, 0, 1,
];
```

## 三、注意事项与常见陷阱

1. **CI 检查**：在 CI 中检查格式
2. **IDE 集成**：配置保存时自动格式化
3. **团队规范**：团队统一格式配置
4. **Git hooks**：使用 pre-commit hook
5. **历史代码**：逐步格式化历史代码
