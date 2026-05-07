# Cow写时复制

## 一、概念说明

`Cow<'a, T>`（Clone on Write）是一种智能指针，用于延迟克隆。它在可能时借用数据，只在需要修改时才进行克隆。

```rust
use std::borrow::Cow;

fn process(input: &str) -> Cow<'_, str> {
    if input.contains("bad") {
        Cow::Owned(input.replace("bad", "good"))
    } else {
        Cow::Borrowed(input)
    }
}
```

## 二、具体用法

### 2.1 基本用法

```rust
use std::borrow::Cow;

let borrowed: Cow<str> = Cow::Borrowed("hello");
let owned: Cow<str> = Cow::Owned(String::from("world"));

// 自动选择
fn process<'a>(input: Cow<'a, str>) -> Cow<'a, str> {
    if input.len() > 5 {
        Cow::Owned(input.to_uppercase().to_string())
    } else {
        input // 保持借用
    }
}
```

### 2.2 性能优化

```rust
use std::borrow::Cow;

fn normalize_whitespace(input: &str) -> Cow<'_, str> {
    if input.contains("  ") {
        // 需要修改时克隆
        let mut result = String::with_capacity(input.len());
        let mut last_was_space = false;
        for c in input.chars() {
            if c.is_whitespace() {
                if !last_was_space {
                    result.push(' ');
                }
                last_was_space = true;
            } else {
                result.push(c);
                last_was_space = false;
            }
        }
        Cow::Owned(result)
    } else {
        // 不需要修改时借用
        Cow::Borrowed(input)
    }
}
```

### 2.3 与标准库集成

```rust
use std::borrow::Cow;
use std::ffi::CStr;
use std::path::PathBuf;

fn to_string_lossy(input: &[u8]) -> Cow<'_, str> {
    String::from_utf8_lossy(input)
}

// PathBuf 的 to_string_lossy
fn path_to_string(path: &PathBuf) -> Cow<'_, str> {
    path.to_string_lossy()
}
```

## 三、注意事项与常见陷阱

1. **生命周期**：Cow 的生命周期与借用的数据绑定
2. **自动解引用**：Cow 实现了 Deref，可直接调用内部类型方法
3. **Clone 行为**：to_mut() 会触发克隆
4. **适用场景**：大多读取、偶尔修改的场景
5. **内存开销**：Cow 本身比直接引用更大
