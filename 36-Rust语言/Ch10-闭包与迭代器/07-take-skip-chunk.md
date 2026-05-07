# take/skip/chunk迭代器

## 一、概念说明

这类迭代器适配器用于从序列中选取或跳过元素：`take` 取前N个，`skip` 跳过前N个，`take_while`/`skip_while` 按条件选取/跳过，`chunks` 将序列分块。

```rust
let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// take 取前3个
let first_three: Vec<i32> = data.iter().take(3).cloned().collect();
// [1, 2, 3]

// skip 跳过前3个
let after_three: Vec<i32> = data.iter().skip(3).cloned().collect();
// [4, 5, 6, 7, 8, 9, 10]
```

## 二、具体用法

### 2.1 take 与 skip

```rust
let data = (1..=20).collect::<Vec<i32>>();

// 分页模拟
fn paginate<T: Clone>(data: &[T], page: usize, size: usize) -> Vec<T> {
    data.iter()
        .skip((page - 1) * size)
        .take(size)
        .cloned()
        .collect()
}

let page2 = paginate(&data, 2, 5);
// [6, 7, 8, 9, 10]

// 前N个和后N个
let first = data.iter().take(3).cloned().collect::<Vec<_>>();
let last = data.iter().rev().take(3).rev().cloned().collect::<Vec<_>>();
```

### 2.2 take_while 与 skip_while

```rust
// take_while: 条件为true时取元素
let result: Vec<i32> = vec![1, 2, 3, 4, 5, 1, 2]
    .into_iter()
    .take_while(|x| *x < 4)
    .collect();
// [1, 2, 3]（遇到4停止）

// skip_while: 条件为true时跳过
let result: Vec<i32> = vec![1, 2, 3, 4, 5, 1, 2]
    .into_iter()
    .skip_while(|x| *x < 4)
    .collect();
// [4, 5, 1, 2]（跳过直到遇到4）
```

### 2.3 chunks 分块

```rust
let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9];

// chunks: 不重叠分块（切片用法）
for chunk in data.chunks(3) {
    println!("{:?}", chunk);
}
// [1, 2, 3]
// [4, 5, 6]
// [7, 8, 9]

// windows: 滑动窗口
for window in data.windows(3) {
    println!("{:?}", window);
}
// [1, 2, 3]
// [2, 3, 4]
// ...

// chunks_exact: 精确分块（不足的丢弃）
for chunk in data.chunks_exact(4) {
    println!("{:?}", chunk);
}
// [1, 2, 3, 4]
// [5, 6, 7, 8]
```

## 三、注意事项与常见陷阱

1. **惰性求值**：take/skip 是迭代器方法，chunks 是切片方法
2. **边界处理**：take 超过实际长度不会 panic
3. **所有权**：skip 会消耗跳过的元素
4. **chunks 稳定**：chunks/windows/chunks_exact 需切片
5. **性能**：这些操作都是 O(1) 或 O(n)，不会产生中间集合
