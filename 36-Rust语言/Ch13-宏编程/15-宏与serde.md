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

## 三、注意事项与常见陷阱

1. **属性语法**：熟悉 serde 的属性语法
2. **性能**：派生宏生成的代码通常足够高效
3. **错误消息**：提供清晰的反序列化错误
4. **版本兼容**：使用 #[serde(default)] 处理新增字段
5. **测试**：测试各种序列化格式
