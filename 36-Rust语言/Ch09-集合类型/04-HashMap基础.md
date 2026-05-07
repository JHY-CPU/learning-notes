# HashMap基础

## 一、概念说明

`HashMap<K, V>` 是 Rust 标准库提供的哈希映射类型，用于存储键值对。它基于开放寻址的哈希表实现，键必须是同一类型，值也必须是同一类型。

```rust
use std::collections::HashMap;

// 创建空 HashMap
let mut scores: HashMap<String, i32> = HashMap::new();

// 使用元组向量初始化
let teams = vec![String::from("蓝队"), String::from("黄队")];
let initial_scores = vec![10, 50];
let scores: HashMap<_, _> = teams.iter().zip(initial_scores.iter()).collect();
```

## 二、具体用法

### 2.1 插入与更新

```rust
use std::collections::HashMap;

let mut map = HashMap::new();

// insert 插入键值对
map.insert(String::from("蓝队"), 10);
map.insert(String::from("黄队"), 50);

// 仅在键不存在时插入
map.entry(String::from("红队")).or_insert(30);

// 根据旧值更新
let text = "hello world wonderful world";
let mut word_count = HashMap::new();
for word in text.split_whitespace() {
    let count = word_count.entry(word).or_insert(0);
    *count += 1;
}
// {"hello": 1, "world": 2, "wonderful": 1}
```

### 2.2 访问元素

```rust
let mut scores = HashMap::new();
scores.insert(String::from("蓝队"), 10);

// get 方法返回 Option<&V>
match scores.get("蓝队") {
    Some(score) => println!("分数: {}", score),
    None => println!("没有该队伍"),
}

// 遍历键值对
for (key, value) in &scores {
    println!("{}: {}", key, value);
}

// 获取所有键
for key in scores.keys() {
    println!("{}", key);
}

// 获取所有值
for value in scores.values() {
    println!("{}", value);
}
```

### 2.3 删除与检查

```rust
let mut map = HashMap::new();
map.insert("a", 1);
map.insert("b", 2);

// 删除键值对
map.remove("a");

// 检查键是否存在
let has_key = map.contains_key("b"); // true

// 获取长度
println!("长度: {}", map.len());
println!("是否为空: {}", map.is_empty());
```

### 2.4 自定义类型作为键

```rust
use std::collections::HashMap;
use std::hash::{Hash, Hasher};

#[derive(Hash, Eq, PartialEq, Debug)]
struct Student {
    id: u32,
    name: String,
}

let mut grades: HashMap<Student, f64> = HashMap::new();
let s = Student { id: 1, name: String::from("张三") };
grades.insert(s, 95.5);
```

## 三、注意事项与常见陷阱

1. **所有权转移**：插入的键值对如果是堆数据，所有权会转移给 HashMap
2. **引用类型键**：需要确保引用的生命周期不短于 HashMap
3. **哈希函数**：默认使用 SipHash，可通过 `with_hasher` 自定义
4. **无序性**：HashMap 不保证元素顺序，遍历顺序可能每次不同
5. **entry API**：推荐使用 `entry` API 处理"存在则更新，不存在则插入"的场景
