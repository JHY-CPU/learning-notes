# Entry API详解

## 一、概念说明

Entry API 是 Rust 标准库为 HashMap 和 BTreeMap 提供的高效"存在则获取，不存在则插入"模式。它避免了重复查找，代码更简洁高效。

```rust
use std::collections::HashMap;

let mut map: HashMap<String, Vec<i32>> = HashMap::new();

// 传统方式：需要查找两次
map.entry("scores".to_string()).or_insert_with(Vec::new).push(95);
```

## 二、具体用法

### 2.1 or_insert 与 or_insert_with

```rust
use std::collections::HashMap;

let mut word_count: HashMap<&str, i32> = HashMap::new();

// or_insert: 不存在则插入默认值
*word_count.entry("hello").or_insert(0) += 1;

// or_insert_with: 不存在时用闭包创建默认值
let mut groups: HashMap<String, Vec<String>> = HashMap::new();
groups.entry("team_a".to_string())
    .or_insert_with(Vec::new)
    .push("张三".to_string());
```

### 2.2 and_modify 与 or_default

```rust
use std::collections::HashMap;

let mut scores: HashMap<&str, i32> = HashMap::new();

// and_modify: 存在时修改，不存在时插入默认值
scores.entry("alice")
    .and_modify(|v| *v += 10)
    .or_insert(50);

// 链式调用
scores.entry("bob")
    .and_modify(|v| *v *= 2)
    .or_insert(60);

// or_default: 插入类型的默认值
let mut map: HashMap<&str, Vec<i32>> = HashMap::new();
map.entry("data").or_default().push(42);
```

### 2.3 Entry 枚举模式匹配

```rust
use std::collections::HashMap;

let mut map: HashMap<&str, i32> = HashMap::new();

match map.entry("key") {
    std::collections::hash_map::Entry::Occupied(mut entry) => {
        *entry.get_mut() += 1;
    }
    std::collections::hash_map::Entry::Vacant(entry) => {
        entry.insert(1);
    }
}

// 使用 if let 简化
if let std::collections::hash_map::Entry::Vacant(e) = map.entry("new_key") {
    e.insert(100);
}
```

### 2.4 高级用法：累计统计

```rust
use std::collections::HashMap;

fn word_frequency(text: &str) -> HashMap<&str, usize> {
    let mut freq = HashMap::new();
    for word in text.split_whitespace() {
        *freq.entry(word).or_insert(0) += 1;
    }
    freq
}

// 组合使用
let mut counters: HashMap<&str, Vec<i32>> = HashMap::new();
let data = vec![("a", 1), ("b", 2), ("a", 3), ("b", 4)];

for (key, val) in data {
    counters.entry(key).or_default().push(val);
}
// {"a": [1, 3], "b": [2, 4]}
```

## 三、注意事项与常见陷阱

1. **所有权**：entry 方法会取得键的所有权
2. **返回类型**：Entry 是枚举，分为 Occupied 和 Vacant
3. **性能优势**：相比先 `contains_key` 再 `insert`，Entry API 只做一次查找
4. **链式调用**：and_modify 和 or_insert 可任意组合
5. **可变借用**：Entry 获取的是 `&mut self`，注意借用规则
