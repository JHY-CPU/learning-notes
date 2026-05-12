# 宏与serde

## 一、概念说明

Serde 使用派生宏自动实现序列化/反序列化。理解其宏实现有助于自定义序列化行为。

```rust
use serde::{Serialize, Deserialize};

#[derive(Serialize, Deserialize, Debug)]
struct Person {
    name: String,
    age: u32,
}
```

## 二、具体用法

### 2.1 自定义序列化

```rust
use serde::{Serialize, Serializer};

#[derive(Serialize)]
struct Custom {
    #[serde(serialize_with = "serialize_timestamp")]
    timestamp: u64,
}

fn serialize_timestamp<S: Serializer>(
    value: &u64,
    serializer: S,
) -> Result<S::Ok, S::Error> {
    let datetime = chrono::NaiveDateTime::from_timestamp(*value as i64, 0);
    serializer.serialize_str(&datetime.to_string())
}
```

### 2.2 容器属性

```rust
#[derive(Serialize, Deserialize)]
#[serde(rename_all = "camelCase")]
struct ApiResponse {
    status_code: u32,
    error_message: Option<String>,
}

#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum Message {
    Text { content: String },
    Image { url: String },
}
```

### 2.3 反序列化验证

```rust
use serde::Deserialize;

#[derive(Deserialize)]
struct Config {
    #[serde(deserialize_with = "validate_port")]
    port: u16,
}

fn validate_port<'de, D: serde::Deserializer<'de>>(
    deserializer: D,
) -> Result<u16, D::Error> {
    let port = u16::deserialize(deserializer)?;
    if port < 1024 {
        return Err(serde::de::Error::custom("端口必须 >= 1024"));
    }
    Ok(port)
}
```

### 2.4 枚举序列化策略

```rust
use serde::{Serialize, Deserialize};

// 外部标记（默认）
#[derive(Serialize, Deserialize)]
enum External {
    A { x: i32 },
    B(i32),
    C,
}
// {"A": {"x": 1}} / {"B": 2} / "C"

// 内部标记
#[derive(Serialize, Deserialize)]
#[serde(tag = "type")]
enum Internal {
    A { x: i32 },
    B { y: i32 },
}
// {"type": "A", "x": 1}

// 相邻标记
#[derive(Serialize, Deserialize)]
#[serde(tag = "type", content = "data")]
enum Adjacent {
    A(i32),
    B(String),
}
// {"type": "A", "data": 1} / {"type": "B", "data": "hello"}

// 无标记
#[derive(Serialize, Deserialize)]
#[serde(untagged)]
enum Untagged {
    Int(i32),
    Text(String),
}
// 1 / "hello"（没有包装）
```

### 2.5 序列化新类型模式

```rust
use serde::{Serialize, Serializer, Deserialize, Deserializer};

// 包装类型，自定义序列化
struct TrimmedString(String);

impl Serialize for TrimmedString {
    fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
        serializer.serialize_str(self.0.trim())
    }
}

impl<'de> Deserialize<'de> for TrimmedString {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
        let s = String::deserialize(deserializer)?;
        Ok(TrimmedString(s.trim().to_string()))
    }
}
```

### 2.6 条件序列化

```rust
use serde::Serialize;

#[derive(Serialize)]
struct Response {
    data: Vec<String>,

    // 仅在 debug 模式下序列化
    #[cfg_attr(debug_assertions, serde(rename = "debug_info"))]
    #[cfg_attr(not(debug_assertions), serde(skip))]
    debug_info: Option<String>,

    // 使用自定义函数决定是否序列化
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}
```

## 四、serde 属性速查

| 属性 | 用途 |
|------|------|
| `rename = "name"` | 重命名字段 |
| `rename_all = "camelCase"` | 重命名所有字段 |
| `default` | 使用 Default trait |
| `skip` | 跳过序列化/反序列化 |
| `flatten` | 扁平化嵌套结构 |
| `tag = "type"` | 枚举标记方式 |
| `with = "module"` | 使用自定义模块 |
| `borrow = "'a"` | 借用而非拥有 |

## 五、注意事项与常见陷阱

1. **属性语法**：熟悉 serde 的属性语法，参考 serde.rs 文档
2. **性能**：派生宏生成的代码通常足够高效，但复杂序列化可能需要自定义
3. **错误消息**：提供清晰的反序列化错误，使用 `#[serde(deny_unknown_fields)]` 检测未知字段
4. **版本兼容**：使用 `#[serde(default)]` 处理新增字段，保持向后兼容
5. **测试**：测试各种序列化格式（JSON、YAML、TOML 等）
6. **泛型序列化**：泛型类型需要正确的 trait bounds
7. **大整数**：JSON 不支持大整数，考虑使用字符串表示
