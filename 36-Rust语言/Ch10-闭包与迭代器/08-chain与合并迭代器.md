# chain与合并迭代器

## 一、概念说明

`chain` 将两个迭代器顺序连接成一个。`interleave` 交替选取元素。这些操作用于合并多个数据源。

```rust
let a = vec![1, 2, 3];
let b = vec![4, 5, 6];
let chained: Vec<i32> = a.into_iter().chain(b).collect();
// [1, 2, 3, 4, 5, 6]
```

## 二、具体用法

### 2.1 chain 连接

```rust
// 连接多个迭代器
let result: Vec<i32> = vec![1, 2]
    .into_iter()
    .chain(vec![3, 4])
    .chain(vec![5, 6])
    .collect();
// [1, 2, 3, 4, 5, 6]

// 连接单个元素
let result: Vec<i32> = vec![1, 2, 3]
    .into_iter()
    .chain(std::iter::once(4))
    .chain(std::iter::repeat(0).take(3))
    .collect();
// [1, 2, 3, 4, 0, 0, 0]

// 连接切片
let a = [1, 2, 3];
let b = [4, 5, 6];
let combined: Vec<i32> = a.iter().chain(b.iter()).cloned().collect();
```

### 2.2 生成器迭代器

```rust
// repeat: 无限重复
let ones: Vec<i32> = std::iter::repeat(1).take(5).collect();
// [1, 1, 1, 1, 1]

// repeat_with: 使用闭包生成
let mut n = 0;
let powers: Vec<i32> = std::iter::repeat_with(|| {
    n += 1;
    n * n
}).take(5).collect();
// [1, 4, 9, 16, 25]

// empty 和 once
let empty: Vec<i32> = std::iter::empty().collect();
let single: Vec<i32> = std::iter::once(42).collect();
```

### 2.3 合并策略

```rust
// 合并排序的迭代器
fn merge_sorted<I>(a: I, b: I) -> Vec<i32>
where
    I: Iterator<Item = i32>,
{
    let mut result: Vec<i32> = a.chain(b).collect();
    result.sort();
    result.dedup();
    result
}

// 去重合并
let a = vec![1, 3, 5];
let b = vec![2, 3, 4, 5];
let merged: Vec<i32> = a.into_iter()
    .chain(b)
    .collect::<std::collections::HashSet<_>>()
    .into_iter()
    .collect();
```

## 三、注意事项与常见陷阱

1. **长度计算**：chain 的 size_hint 是两者之和
2. **类型一致**：被连接的迭代器元素类型必须相同
3. **所有权**：chain 获取两个迭代器的所有权
4. **惰性求值**：chain 不会立即执行，仍需消费操作
5. **替代方案**：简单连接也可用 extend 或 append
