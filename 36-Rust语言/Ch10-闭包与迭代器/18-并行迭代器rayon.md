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

### 2.4 并行排序

```rust
use rayon::prelude::*;

fn parallel_sort_example() {
    let mut data: Vec<i32> = (0..1_000_000).rev().collect();

    // 并行排序
    data.par_sort();
    assert!(data.windows(2).all(|w| w[0] <= w[1]));

    // 自定义比较
    let mut people = vec![
        ("张三", 30),
        ("李四", 25),
        ("王五", 35),
    ];
    people.par_sort_by(|a, b| a.1.cmp(&b.1));
}
```

### 2.5 并行 map-reduce

```rust
use rayon::prelude::*;

fn map_reduce_example() {
    let data: Vec<i64> = (1..=1_000_000).collect();

    // map-reduce: 分割数据，分别计算，再合并结果
    let (sum, count, max) = data.par_iter()
        .map(|&x| (x, 1i64, x))  // (sum, count, max)
        .reduce(
            || (0, 0, i64::MIN),
            |(s1, c1, m1), (s2, c2, m2)| {
                (s1 + s2, c1 + c2, m1.max(m2))
            },
        );

    let average = sum as f64 / count as f64;
    println!("平均值: {:.2}, 最大值: {}", average, max);
}
```

### 2.6 并行文件处理

```rust
use rayon::prelude::*;
use std::fs;
use std::path::Path;

fn process_files_parallel(dir: &Path) -> Vec<(String, usize)> {
    let entries: Vec<_> = fs::read_dir(dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .collect();

    entries.par_iter()
        .filter_map(|entry| {
            let path = entry.path();
            if path.is_file() {
                let content = fs::read_to_string(&path).ok()?;
                let name = path.file_name()?.to_string_lossy().into_owned();
                Some((name, content.len()))
            } else {
                None
            }
        })
        .collect()
}
```

### 2.7 分割策略与微基准

```rust
use rayon::prelude::*;

fn demonstrate_split_behavior() {
    // rayon 自动将数据分割成小块分配给线程
    // 可以使用 split 来观察分割行为

    let data: Vec<i32> = (1..=20).collect();

    // 使用 chunks 并行处理
    let results: Vec<i32> = data.par_chunks(5)
        .map(|chunk| chunk.iter().sum())
        .collect();
    // results: [15, 40, 65, 90]

    // 自定义分割阈值
    let large_data: Vec<i32> = (0..10_000_000).collect();
    let result: i64 = large_data.par_iter()
        .with_min_len(1000)  // 每个任务至少处理 1000 个元素
        .map(|&x| x as i64)
        .sum();
}
```

### 2.8 并行迭代器与异步结合

```rust
use rayon::prelude::*;

async fn async_with_rayon() {
    // 异步环境中使用 rayon 处理 CPU 密集型任务
    let data: Vec<i64> = (0..10_000_000).collect();

    // 使用 spawn_blocking 将 rayon 任务移到阻塞线程池
    let result = tokio::task::spawn_blocking(move || {
        data.par_iter()
            .map(|&x| {
                let mut val = x;
                for _ in 0..100 {
                    val = val.wrapping_mul(17).wrapping_add(31);
                }
                val
            })
            .sum::<i64>()
    }).await.unwrap();

    println!("计算结果: {}", result);
}
```

## 四、性能调优技巧

### 4.1 选择正确的并行策略

| 数据规模 | 推荐策略 | 说明 |
|---------|---------|------|
| < 1000 | 串行 | 并行开销大于收益 |
| 1000 - 100000 | par_iter | 标准并行 |
| > 100000 | par_iter + with_min_len | 减少任务分割开销 |

### 4.2 自定义线程池最佳实践

```rust
use rayon::ThreadPoolBuilder;

fn custom_pool_example() {
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())  // 匹配 CPU 核心数
        .thread_name(|i| format!("rayon-worker-{}", i))
        .stack_size(8 * 1024 * 1024)  // 8MB 栈
        .panic_handler(|e| eprintln!("线程 panic: {:?}", e))
        .build()
        .unwrap();

    pool.install(|| {
        // 在自定义线程池中执行
        let sum: i64 = (0..1_000_000)
            .into_par_iter()
            .sum();
    });
}
```

## 五、注意事项与常见陷阱

1. **开销平衡**：小数据量并行可能比串行更慢，通常 1000+ 元素才考虑并行
2. **数据竞争**：par_iter_mut 需要数据可安全分割，避免共享可变状态
3. **全局线程池**：rayon 默认使用全局线程池，可通过 `ThreadPoolBuilder` 自定义
4. **死锁风险**：在并行任务中获取锁可能死锁，考虑使用无锁数据结构
5. **调试困难**：并行执行使得调试和复现更困难，使用确定性种子辅助调试
6. **Side Effect**：避免在 `map`/`filter` 中有副作用，应使用 `for_each`
7. **嵌套并行**：嵌套的 `par_iter` 可能导致线程池饥饿，使用 `scope` 管理
