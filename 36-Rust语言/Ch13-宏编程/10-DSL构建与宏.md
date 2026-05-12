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

### 2.4 正则表达式 DSL

```rust
macro_rules! regex_dsl {
    // 匹配字符
    (char $c:literal) => {
        format!("[{}]", $c)
    };
    // 零次或多次
    (zero_or_more $pattern:expr) => {
        format!("({})*", $pattern)
    };
    // 一次或多次
    (one_or_more $pattern:expr) => {
        format!("({})+", $pattern)
    };
    // 选择
    (choice $($pattern:expr),+) => {
        format!("({})", [$($pattern),+].join("|"))
    };
    // 组合
    (seq $($pattern:expr),+) => {
        format!("{}", [$($pattern),+].join(""))
    };
}

fn regex_dsl_demo() {
    let pattern = regex_dsl!(seq
        regex_dsl!(char 'a'),
        regex_dsl!(one_or_more regex_dsl!(char 'b')),
        regex_dsl!(choice regex_dsl!(char 'c'), regex_dsl!(char 'd'))
    );
    // 生成: (a)((b)+)((c|d))
}
```

### 2.5 验证 DSL

```rust
macro_rules! validate {
    ($value:expr, min $min:expr) => {
        if $value < $min {
            Err(format!("值 {} 小于最小值 {}", $value, $min))
        } else {
            Ok($value)
        }
    };
    ($value:expr, max $max:expr) => {
        if $value > $max {
            Err(format!("值 {} 大于最大值 {}", $value, $max))
        } else {
            Ok($value)
        }
    };
    ($value:expr, range $min:expr to $max:expr) => {
        validate!($value, min $min)
            .and_then(|v| validate!(v, max $max))
    };
    ($value:expr, in [$($allowed:expr),+]) => {
        if [$($allowed),+].contains(&$value) {
            Ok($value)
        } else {
            Err(format!("值 {} 不在允许列表中", $value))
        }
    };
}

fn validation_dsl_demo() {
    let result = validate!(42, range 0 to 100);
    let result = validate!("admin", in ["admin", "user", "guest"]);
}
```

### 2.6 测试 DSL

```rust
macro_rules! test_case {
    ($name:ident: $input:expr => $expected:expr) => {
        #[test]
        fn $name() {
            let result = $input;
            assert_eq!(result, $expected);
        }
    };
    ($name:ident: $input:expr => $expected:expr, $msg:expr) => {
        #[test]
        fn $name() {
            let result = $input;
            assert_eq!(result, $expected, $msg);
        }
    };
}

// 使用
test_case!(test_add: 1 + 1 => 2);
test_case!(test_mul: 3 * 4 => 12, "乘法应该正确");
```

## 四、DSL 设计原则

```
1. 直观性：语法应该接近领域语言
2. 一致性：类似的操作使用类似的语法
3. 错误友好：提供清晰的错误消息
4. 可组合性：小的 DSL 元素可以组合成复杂表达式
5. 类型安全：利用 Rust 的类型系统防止错误
```

## 五、注意事项与常见陷阱

1. **语法限制**：Rust 宏的语法有一定限制，不能完全自由定义语法
2. **错误消息**：DSL 的错误消息可能难以理解，需要自定义错误处理
3. **IDE 支持**：自定义 DSL 的 IDE 支持可能有限，语法高亮和补全不完整
4. **测试覆盖**：DSL 需要充分的测试，覆盖各种输入组合
5. **文档**：提供清晰的 DSL 使用文档和示例
6. **性能**：DSL 宏展开可能生成大量代码，注意编译时间
7. **演进**：DSL 的变更可能破坏现有代码，需要版本管理
