# 并行迭代器rayon

## 一、概念说明

`rayon` 是 Rust 的数据并行库，提供了与标准库迭代器类似的 API，但能自动将计算分发到多个 CPU 核心。只需将 `iter()` 改为 `par_iter()` 即可实现并行化。

```rust
// Cargo.toml: rayon = "1"
use rayon::prelude::*;

// 并行计算平方和
let sum: i32 = (1..=1000)
    .into_par_iter()
    .map(|x| x * x)
    .sum();
```

## 二、具体用法

### 2.1 并行迭代器基础

```rust
use rayon::prelude::*;

let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];

// par_iter: 并行不可变迭代
let squares: Vec<i32> = data.par_iter()
    .map(|x| x * x)
    .collect();

// par_iter_mut: 并行可变迭代
let mut data = vec![1, 2, 3, 4, 5];
data.par_iter_mut().for_each(|x| *x *= 2);

// into_par_iter: 并行所有权转移迭代
let sum: i32 = data.into_par_iter()
    .map(|x| x * 2)
    .sum();
```

### 2.2 并行搜索与过滤

```rust
use rayon::prelude::*;

// 并行查找
let data: Vec<i32> = (1..1000000).collect();
let found = data.par_iter()
    .find_any(|&&x| x == 500000);

// 并行过滤
let evens: Vec<i32> = (1..100)
    .into_par_iter()
    .filter(|x| x % 2 == 0)
    .collect();

// 并行 any/all
let has_negative = data.par_iter().any(|x| *x < 0);
```

### 2.3 自定义线程池

```rust
use rayon::ThreadPoolBuilder;

let pool = ThreadPoolBuilder::new()
    .num_threads(4)
    .thread_name(|i| format!("worker-{}", i))
    .build()
    .unwrap();

pool.install(|| {
    let sum: i32 = (1..=100).into_par_iter().sum();
    println!("总和: {}", sum);
});
```

## 三、注意事项与常见陷阱

1. **开销平衡**：小数据量并行可能比串行更慢
2. **数据竞争**：par_iter_mut 需要数据可安全分割
3. **全局线程池**：rayon 默认使用全局线程池
4. **死锁风险**：在并行任务中获取锁可能死锁
5. **调试困难**：并行执行使得调试和复现更困难
