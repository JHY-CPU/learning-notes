# 字符串String详解

## 一、概念说明

Rust 中有两种主要的字符串类型：`String` 和 `&str`。`String` 是堆分配的、可增长的、可变的UTF-8编码字符串；`&str` 是字符串切片，是对一段UTF-8数据的引用。

```rust
// 字面量类型是 &str
let s1 = "hello";

// 创建 String
let s2 = String::from("hello");
let s3 = "hello".to_string();
let s4 = String::new();
```

## 二、具体用法

### 2.1 更新字符串

```rust
let mut s = String::from("foo");

// push_str 追加字符串切片
s.push_str("bar"); // "foobar"

// push 追加单个字符
s.push('!'); // "foobar!"

// 使用 + 运算符拼接
let s1 = String::from("Hello, ");
let s2 = String::from("world!");
let s3 = s1 + &s2; // s1 被移动，不再有效

// 使用 format! 宏
let s1 = String::from("tic");
let s2 = String::from("tac");
let s3 = String::from("toe");
let s = format!("{}-{}-{}", s1, s2, s3); // "tic-tac-toe"
```

### 2.2 索引与切片

```rust
let s = String::from("你好世界");

// Rust 不支持直接索引字符串
// let c = s[0]; // 编译错误！

// 使用 bytes() 获取字节
for b in "你好".bytes() {
    println!("{}", b);
}

// 使用 chars() 获取字符
for c in "你好世界".chars() {
    println!("{}", c);
}

// 字符串切片（需注意UTF-8边界）
let hello = &s[0..6]; // "你好" (每个中文3字节)
```

### 2.3 常用方法

```rust
let s = String::from("  Hello, Rust!  ");

// 去除空白
let trimmed = s.trim(); // "Hello, Rust!"

// 大小写转换
let upper = "hello".to_uppercase(); // "HELLO"
let lower = "HELLO".to_lowercase(); // "hello"

// 替换
let replaced = "hello world".replace("world", "Rust");

// 分割
let parts: Vec<&str> = "a,b,c".split(",").collect();
// ["a", "b", "c"]

// 包含检查
let contains = "hello world".contains("world"); // true
let starts = "hello".starts_with("he"); // true

// 重复
let repeated = "ha".repeat(3); // "hahaha"
```

### 2.4 字符串与其他类型转换

```rust
// 数字转字符串
let num_str = 42.to_string();
let float_str = std::f64::consts::PI.to_string();

// 字符串转数字
let num: i32 = "42".parse().unwrap();
let float: f64 = "3.14".parse().unwrap();

// 处理解析错误
match "hello".parse::<i32>() {
    Ok(n) => println!("数字: {}", n),
    Err(e) => println!("解析失败: {}", e),
}
```

## 三、注意事项与常见陷阱

1. **UTF-8编码**：Rust 字符串是 UTF-8 的，不能用索引直接访问字符
2. **所有权转移**：`+` 运算符会取得第一个字符串的所有权
3. **字节vs字符**：一个 Unicode 字符可能占多个字节
4. **性能考虑**：频繁拼接字符串时，使用 `format!` 或预分配容量
5. **OsString/CStr**：与操作系统交互时可能需要 `OsString` 或 `CString`
