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

## 三、CI集成

```yaml
# GitHub Actions中检查格式
- name: Check formatting
  run: cargo fmt -- --check
# 返回非零退出码表示有文件未格式化
```

## 四、注意事项与常见陷阱

1. **CI检查**：在CI中使用`cargo fmt -- --check`确保代码风格一致，不通过则阻断合并
2. **IDE集成**：VS Code安装rust-analyzer扩展，配置`editor.formatOnSave: true`实现保存自动格式化
3. **团队规范**：将`rustfmt.toml`提交到仓库根目录，确保所有成员使用相同配置
4. **Git hooks**：使用pre-commit hook在提交前自动格式化，避免忘记手动执行

## 五、常见陷阱

1. **版本差异**：不同rustfmt版本格式化结果可能不同，团队应统一使用同一版本（通过`rust-toolchain.toml`固定）
2. **历史代码爆炸**：对大型历史项目一次性格式化会产生巨大diff，应分模块逐步格式化
3. **属性跳过滥用**：过度使用`#[rustfmt::skip]`会导致代码风格不一致，仅在矩阵等特殊布局时使用
4. **宏内代码**：rustfmt不格式化宏内部代码，宏定义内的格式需要手动维护

## 六、高级配置详解

### 6.1 完整配置参考

```toml
# rustfmt.toml - 企业级配置示例
# 基础设置
max_width = 100
hard_tabs = false
tab_spaces = 4
edition = "2021"

# 换行与缩进
newline_style = "Unix"
use_small_heuristics = "Default"
indent_style = "Block"

# 导入组织
imports_granularity = "Crate"       # 合并同crate导入
group_imports = "StdExternalCrate"  # std/external/crate分组
imports_layout = "Vertical"         # 垂直排列导入
reorder_imports = true              # 自动排序导入
reorder_modules = true              # 自动排序模块声明

# 函数格式
fn_params_layout = "Tall"
brace_style = "SameLineWhere"
control_brace_style = "AlwaysSameLine"
empty_item_single_line = true
fn_single_line = false

# 结构体与枚举
struct_lit_single_line = true
enum_discrim_align_threshold = 30
struct_field_align_threshold = 30

# 代码风格
use_field_init_shorthand = true     # S { x } 而非 S { x: x }
use_try_shorthand = true            # x? 而非 Try::into(x)
format_code_in_doc_comments = true  # 格式化文档中的代码
wrap_comments = true                # 自动换行注释
normalize_comments = false
normalize_doc_attributes = true

# 宏处理
format_macro_matchers = true
format_macro_bodies = true
hex_literal_case = "Upper"
```

### 6.2 条件编译属性

```rust
// 单行跳过
#[rustfmt::skip]
const LOOKUP_TABLE: [u8; 256] = [
    0x00, 0x01, 0x02, 0x03, /* ... */
];

// 块级跳过（rustfmt 2.0+）
#[rustfmt::skip]
mod manual_layout {
    fn example() {
        // 此模块内代码不会被格式化
        let x=1;let y=2;
    }
}

// 仅跳过特定项目
#[rustfmt::skip::attributes(serde)]
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiPayload {
    user_name: String,
    email_address: String,
}
```

### 6.3 自定义格式化规则

```toml
# 针对特定场景的配置
# 宽屏显示器友好
max_width = 120
use_small_heuristics = "Max"

# 嵌入式项目紧凑风格
max_width = 80
tab_spaces = 2
imports_granularity = "Item"
```

## 七、团队工作流集成

### 7.1 Pre-commit Hook

```bash
#!/bin/sh
# .git/hooks/pre-commit
# 检查暂存文件的格式
files=$(git diff --cached --name-only --diff-filter=ACM | grep '\.rs$')
if [ -n "$files" ]; then
    cargo fmt -- --check $files
    if [ $? -ne 0 ]; then
        echo "代码格式不正确，请运行 'cargo fmt' 后重新提交"
        exit 1
    fi
fi
```

### 7.2 GitHub Actions 完整配置

```yaml
# .github/workflows/format.yml
name: Format Check
on: [push, pull_request]

jobs:
  fmt:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: rustfmt
      - run: cargo fmt --all -- --check

  # 同时检查 clippy
  clippy:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: dtolnay/rust-toolchain@stable
        with:
          components: clippy
      - run: cargo clippy -- -D warnings
```

### 7.3 统一工具链版本

```toml
# rust-toolchain.toml - 确保团队使用相同版本
[toolchain]
channel = "1.77.0"
components = ["rustfmt", "clippy"]
```

### 7.4 VS Code 配置

```json
// .vscode/settings.json
{
    "[rust]": {
        "editor.defaultFormatter": "rust-lang.rust-analyzer",
        "editor.formatOnSave": true,
        "editor.formatOnSaveTimeout": 5000
    },
    "rust-analyzer.rustfmt.extraArgs": [
        "--config-path",
        "${workspaceFolder}/rustfmt.toml"
    ]
}
```

## 八、rustfmt 与 clippy 配合

```bash
# 一键检查：格式 + lint
cargo fmt -- --check && cargo clippy -- -D warnings

# 一键修复
cargo fmt && cargo clippy --fix --allow-dirty

# 仅格式化特定包（工作空间）
cargo fmt -p my-package -- --check
```

## 九、格式化前后对比

```rust
// 格式化前：风格混乱
fn   process( x:i32,y :i32)->i32{
    if x>y { x } else {y}
}

use std::collections::HashMap;
use std::io::Result;
use std::fs::File;

// 格式化后：统一风格
fn process(x: i32, y: i32) -> i32 {
    if x > y { x } else { y }
}

use std::{collections::HashMap, fs::File, io::Result};
// 注意：imports_granularity = "Crate" 时会自动合并导入
```

## 十、rustfmt 稳定版与 nightly 特性

```bash
# 使用 nightly 的不稳定特性
cargo +nightly fmt -- --edition 2021 --config unstable_features=true

# 查看所有可用配置项
rustfmt --help=config

# 生成默认配置文件
rustfmt --print-config default rustfmt.toml
```

```toml
# 仅 nightly 可用的配置
# unstable_features = true
# hex_literal_case = "Upper"
# format_code_in_doc_comments = true
```

## 十一、性能与注意事项

1. **格式化速度**：rustfmt 处理大型项目通常在数秒内完成，增量格式化更快
2. **不可逆操作**：格式化后的代码无法自动还原，建议先提交再格式化
3. **宏限制**：`macro_rules!` 内部代码不被格式化，过程宏的输出会被格式化
4. **属性干扰**：某些 `#[cfg]` 条件编译属性可能影响格式化结果
5. **版本锁定**：强烈建议通过 `rust-toolchain.toml` 锁定 rustfmt 版本，避免不同版本产生不同格式
6. **首次格式化**：对已有项目首次运行 `cargo fmt` 会产生大量变更，建议单独提交此变更
