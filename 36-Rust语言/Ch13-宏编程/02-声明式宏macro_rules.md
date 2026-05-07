# 声明式宏macro_rules

## 一、概念说明

`macro_rules!` 是 Rust 的声明式宏系统，通过模式匹配和代码替换实现代码复用。它类似于 Scheme 的语法宏。

```rust
macro_rules! vec {
    ($($x:expr),*) => {
        {
            let mut v = Vec::new();
            $(v.push($x);)*
            v
        }
    };
}
```

## 二、具体用法

### 2.1 重复模式

```rust
macro_rules! create_vec {
    // 零个或多个
    ($($x:expr),*) => {{
        let mut v = Vec::new();
        $(v.push($x);)*
        v
    }};

    // 带尾逗号
    ($($x:expr),+ ,) => {
        create_vec!($($x),*)
    };
}

let v = create_vec![1, 2, 3];
let v = create_vec![1, 2, 3,]; // 支持尾逗号
```

### 2.2 多分支匹配

```rust
macro_rules! calculate {
    (add $x:expr, $y:expr) => { $x + $y };
    (sub $x:expr, $y:expr) => { $x - $y };
    (mul $x:expr, $y:expr) => { $x * $y };
    (div $x:expr, $y:expr) => { $x / $y };
}

let result = calculate!(add 1, 2);
let result = calculate!(mul 3, 4);
```

### 2.3 递归宏

```rust
macro_rules! count {
    () => { 0 };
    ($x:tt $($rest:tt)*) => { 1 + count!($($rest)*) };
}

assert_eq!(count!(a b c d), 4);
```

### 2.4 生成代码

```rust
macro_rules! impl_display {
    ($($t:ty),*) => {
        $(
            impl std::fmt::Display for $t {
                fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                    write!(f, "{:?}", self)
                }
            }
        )*
    };
}
```

## 三、注意事项与常见陷阱

1. **卫生性**：宏内定义的变量不会污染外部作用域
2. **匹配优先级**：先匹配的规则优先
3. **递归限制**：递归宏有展开深度限制
4. **调试技巧**：使用 `cargo expand` 查看宏展开
5. **测试验证**：宏的测试应覆盖各种输入模式
