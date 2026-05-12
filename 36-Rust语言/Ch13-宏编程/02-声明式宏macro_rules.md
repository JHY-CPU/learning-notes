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

### 2.5 可变参数与尾逗号

```rust
macro_rules! flexible_vec {
    // 空
    () => { Vec::new() };

    // 单个元素
    ($elem:expr) => {{
        let mut v = Vec::new();
        v.push($elem);
        v
    }};

    // 多个元素，支持尾逗号
    ($($elem:expr),+ $(,)?) => {{
        let mut v = Vec::new();
        $(v.push($elem);)+
        v
    }};
}

fn flexible_vec_demo() {
    let v1 = flexible_vec![];
    let v2 = flexible_vec![1];
    let v3 = flexible_vec![1, 2, 3];
    let v4 = flexible_vec![1, 2, 3,]; // 尾逗号
}
```

### 2.6 使用 @__ 内部模式

```rust
macro_rules! impl_display_for_enum {
    // 入口：开始递归
    (@inner $enum_name:ident, $($variant:ident),*) => {
        impl std::fmt::Display for $enum_name {
            fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
                match self {
                    $($enum_name::$variant => write!(f, stringify!($variant)),)*
                }
            }
        }
    };

    // 公共接口
    ($enum_name:ident { $($variant:ident),* $(,)? }) => {
        impl_display_for_enum!(@inner $enum_name, $($variant),*);
    };
}
```

### 2.7 条件展开与 cfg

```rust
macro_rules! cfg_if {
    ($cfg:meta => $then:block else $else:block) => {
        #[cfg($cfg)]
        $then
        #[cfg(not($cfg))]
        $else
    };
}

// 使用
cfg_if! {
    target_os = "windows" => {
        fn platform_specific() { println!("Windows"); }
    } else {
        fn platform_specific() { println!("其他平台"); }
    }
}
```

### 2.8 调试宏展开

```rust
// 使用 stringify! 查看宏接收的输入
macro_rules! debug_macro {
    ($($tt:tt)*) => {
        eprintln!("宏接收: {}", stringify!($($tt)*));
    };
}

// 使用 dbg! 在宏内部调试
macro_rules! trace_expand {
    ($($tt:tt)*) => {{
        eprintln!("展开前: {}", stringify!($($tt)*));
        let result = { $($tt)* };
        eprintln!("展开后: {:?}", result);
        result
    }};
}

// 使用 cargo expand 查看完整展开
// $ cargo expand --lib
```

### 2.9 常见模式速查

```rust
// 模式1: 生成多个 impl 块
macro_rules! impl_traits {
    ($t:ty, $($trait:ident),+) => {
        $(
            impl $trait for $t {
                // ...
            }
        )+
    };
}

// 模式2: 类型映射
macro_rules! type_map {
    ($($from:ty => $to:ty),+ $(,)?) => {
        $(
            impl From<$from> for $to {
                fn from(val: $from) -> $to {
                    val as $to
                }
            }
        )+
    };
}

// 模式3: 静态注册
macro_rules! register_handler {
    ($name:ident, $handler:expr) => {
        paste::paste! {
            #[linkme::distributed_slice]
            static [<$name _HANDLER>]: fn() = $handler;
        }
    };
}
```

## 四、宏片段说明符完整列表

| 说明符 | 匹配内容 | 示例 |
|--------|---------|------|
| `expr` | 表达式 | `1 + 2`, `foo()` |
| `ty` | 类型 | `i32`, `Vec<String>` |
| `ident` | 标识符 | `foo`, `MyStruct` |
| `path` | 路径 | `std::io::Error` |
| `stmt` | 语句 | `let x = 1;` |
| `block` | 代码块 | `{ let x = 1; x }` |
| `item` | 项 | `fn foo() {}` |
| `meta` | 属性 | `derive(Debug)` |
| `tt` | 标记树 | 任意 token |
| `literal` | 字面量 | `42`, `"hello"` |
| `lifetime` | 生命周期 | `'a` |
| `vis` | 可见性 | `pub`, `pub(crate)` |
| `pat` | 模式 | `Some(x)`, `(a, b)` |
| `pat_param` | 模式（不含 `|`） | 同上 |

## 五、注意事项与常见陷阱

1. **卫生性**：宏内定义的变量不会污染外部作用域，但 `$name:ident` 引入的标识符在调用者作用域
2. **匹配优先级**：先匹配的规则优先，将更具体的规则放在前面
3. **递归限制**：递归宏有展开深度限制（默认128层），超过会编译失败
4. **调试技巧**：使用 `cargo expand` 或 `stringify!` 查看宏展开结果
5. **测试验证**：宏的测试应覆盖各种输入模式，包括边界情况
6. **类型推断**：宏展开后才进行类型检查，错误信息可能指向宏定义而非调用处
7. **宏的组合**：嵌套宏调用可能导致复杂性增加，保持宏的简单性
