# DSL构建与宏

## 一、概念说明

宏可用于在 Rust 中构建领域特定语言（DSL），提供更直观的语法。

```rust
// 自定义 DSL 示例
let config = config! {
    host: "localhost",
    port: 8080,
    timeout: 30,
};
```

## 二、具体用法

### 2.1 HTML DSL

```rust
macro_rules! html {
    ($tag:ident { $($inner:tt)* }) => {
        format!("<{}>{}</{}>",
            stringify!($tag),
            html!($($inner)*),
            stringify!($tag))
    };
    ($text:expr) => { $text.to_string() };
}

let page = html! {
    html {
        head { "标题" }
        body { "内容" }
    }
};
```

### 2.2 SQL DSL

```rust
macro_rules! select {
    ($($field:ident),+ from $table:ident) => {{
        let fields = vec![$(stringify!($field)),+];
        format!("SELECT {} FROM {}",
            fields.join(", "),
            stringify!($table))
    }};
}

let query = select!(name, age from users);
// "SELECT name, age FROM users"
```

### 2.3 状态机 DSL

```rust
macro_rules! state_machine {
    ($name:ident { $($state:ident),+ }) => {
        enum $name {
            $($state),+
        }

        impl $name {
            fn transitions(&self) -> Vec<&str> {
                match self {
                    $(
                        $name::$state => vec![stringify!($state)],
                    )+
                }
            }
        }
    };
}

state_machine!(TrafficLight { Red, Yellow, Green });
```

## 三、注意事项与常见陷阱

1. **语法限制**：Rust 宏的语法有一定限制
2. **错误消息**：DSL 的错误消息可能难以理解
3. **IDE 支持**：自定义 DSL 的 IDE 支持可能有限
4. **测试覆盖**：DSL 需要充分的测试
5. **文档**：提供清晰的 DSL 使用文档
