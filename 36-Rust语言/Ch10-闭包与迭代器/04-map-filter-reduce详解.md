# map/filter/reduce详解

## 一、概念说明

map、filter、reduce 是函数式编程的三大核心操作。Rust 的迭代器提供了这些操作的惰性版本，可以高效地链式组合处理数据。

```rust
let result: i32 = (1..=10)
    .filter(|x| x % 2 == 0)   // 过滤
    .map(|x| x * x)           // 映射
    .fold(0, |acc, x| acc + x); // 归约
// 4 + 16 + 36 + 64 + 100 = 220
```

## 二、具体用法

### 2.1 map 变换

```rust
// 基本映射
let doubled: Vec<i32> = vec![1, 2, 3]
    .iter()
    .map(|x| x * 2)
    .collect();
// [2, 4, 6]

// 类型转换
let strings: Vec<String> = vec![1, 2, 3]
    .iter()
    .map(|x| format!("第{}个", x))
    .collect();

// 链式 map
let result: Vec<String> = vec!["hello", "world"]
    .iter()
    .map(|s| s.to_uppercase())
    .map(|s| s.len())
    .map(|len| format!("长度: {}", len))
    .collect();
```

### 2.2 filter 过滤

```rust
// 基本过滤
let evens: Vec<&i32> = vec![1, 2, 3, 4, 5]
    .iter()
    .filter(|x| *x % 2 == 0)
    .collect();
// [2, 4]

// 复合条件
let results: Vec<&str> = vec!["apple", "banana", "cherry", "date"]
    .iter()
    .filter(|s| s.len() > 4)
    .filter(|s| s.starts_with('a') || s.starts_with('c'))
    .map(|s| s.as_ref())
    .collect();
// ["apple", "cherry"]

// filter_map 合并过滤和映射
let numbers: Vec<i32> = vec!["1", "hi", "3", "bye", "5"]
    .iter()
    .filter_map(|s| s.parse::<i32>().ok())
    .collect();
// [1, 3, 5]
```

### 2.3 reduce 与 fold

```rust
// fold 累积（指定初始值）
let sum = vec![1, 2, 3, 4, 5]
    .iter()
    .fold(0, |acc, x| acc + x);
// 15

// fold 构建复杂结构
let map = vec![("a", 1), ("b", 2), ("a", 3)]
    .into_iter()
    .fold(std::collections::HashMap::new(), |mut acc, (k, v)| {
        *acc.entry(k).or_insert(0) += v;
        acc
    });
// {"a": 4, "b": 2}

// reduce（使用第一个元素作为初始值）
let max = vec![3, 1, 4, 1, 5, 9]
    .into_iter()
    .reduce(|a, b| if a > b { a } else { b });
// Some(9)

// scan 带状态累积
let cumulative: Vec<i32> = vec![1, 2, 3, 4, 5]
    .iter()
    .scan(0, |state, x| {
        *state += x;
        Some(*state)
    })
    .collect();
// [1, 3, 6, 10, 15]
```

## 三、注意事项与常见陷阱

1. **惰性求值**：map/filter 需 collect 或 for 循环才执行
2. **引用vs所有权**：iter() 产生引用，into_iter() 消耗所有权
3. **编译优化**：迭代器链编译为单一循环，性能与手写相当
4. **内存效率**：大数据量避免中间 collect，保持迭代器链
5. **替代方案**：简单场景 for 循环可读性可能更好
