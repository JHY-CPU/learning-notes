# Vec高级操作与性能

## 一、概念说明

向量的高级操作涉及排序、搜索、过滤、转换等常见数据处理场景。Rust 提供了丰富的迭代器方法和原地操作来高效处理向量数据。

```rust
let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6];
v.sort();           // 原地排序
v.dedup();          // 去除连续重复元素
```

## 二、具体用法

### 2.1 排序与比较

```rust
let mut v = vec![3, 1, 4, 1, 5, 9, 2, 6];

// 基本排序（要求元素实现 Ord）
v.sort();
// 结果: [1, 1, 2, 3, 4, 5, 6, 9]

// 使用比较函数排序
v.sort_by(|a, b| b.cmp(a)); // 降序

// 使用键排序
let mut v = vec!["hello", "world", "a", "by"];
v.sort_by_key(|s| s.len());
// 结果: ["a", "by", "hello", "world"]

// 浮点数排序（f64 不实现 Ord）
let mut floats = vec![1.0, 3.5, 2.1, 0.5];
floats.sort_by(|a, b| a.partial_cmp(b).unwrap());
```

### 2.2 过滤与映射

```rust
let v = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// filter 过滤
let evens: Vec<&i32> = v.iter().filter(|x| *x % 2 == 0).collect();
// [2, 4, 6, 8, 10]

// map 映射
let squares: Vec<i32> = v.iter().map(|x| x * x).collect();
// [1, 4, 9, 16, 25, 36, 49, 64, 81, 100]

// filter_map 合并过滤与映射
let v = vec!["1", "hello", "3", "world", "5"];
let numbers: Vec<i32> = v.iter()
    .filter_map(|s| s.parse::<i32>().ok())
    .collect();
// [1, 3, 5]
```

### 2.3 聚合操作

```rust
let v = vec![1, 2, 3, 4, 5];

let sum: i32 = v.iter().sum();        // 15
let product: i32 = v.iter().product(); // 120
let max = v.iter().max();             // Some(&5)
let min = v.iter().min();             // Some(&1)

// fold 累积
let sum = v.iter().fold(0, |acc, x| acc + x);
```

### 2.4 容量与内存管理

```rust
let mut v: Vec<i32> = Vec::with_capacity(10);

println!("长度: {}", v.len());       // 0
println!("容量: {}", v.capacity()); // 10

// 缩小容量以匹配实际长度
v.shrink_to_fit();

// 保留至少指定容量
v.reserve(100);
```

## 三、注意事项与常见陷阱

1. **排序稳定性**：`sort` 是稳定排序，`sort_unstable` 更快但不稳定
2. **drain 方法**：移除范围元素并返回迭代器，注意范围语法
3. **retain 方法**：保留满足条件的元素，是 filter 的原地版本
4. **性能考量**：优先使用迭代器链代替手动循环，编译器能更好优化
5. **VecDeque**：需要频繁在头部插入/删除时，使用 `VecDeque` 更高效
