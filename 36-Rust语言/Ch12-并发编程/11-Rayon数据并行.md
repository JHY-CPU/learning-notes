# Rayon数据并行

## 一、概念说明

Rayon 是 Rust 的数据并行库，提供并行迭代器和工作窃取调度器。只需将 iter 改为 par_iter 即可实现并行化。

```rust
use rayon::prelude::*;

let sum: i32 = (1..=1000)
    .into_par_iter()
    .map(|x| x * x)
    .sum();
```

## 二、具体用法

### 2.1 并行迭代器

```rust
use rayon::prelude::*;

// 并行 map/filter/collect
let data = vec![1, 2, 3, 4, 5, 6, 7, 8, 9, 10];
let squares: Vec<i32> = data.par_iter()
    .filter(|x| *x % 2 == 0)
    .map(|x| x * x)
    .collect();

// 并行排序
let mut data = vec![3, 1, 4, 1, 5, 9, 2, 6];
data.par_sort();

// 并行 search
let found = data.par_iter()
    .find_any(|x| **x == 5);
```

### 2.2 自定义并行操作

```rust
use rayon::prelude::*;

fn parallel_sum(data: &[i64]) -> i64 {
    if data.len() < 1000 {
        // 小数据串行计算
        data.iter().sum()
    } else {
        // 大数据并行计算
        let mid = data.len() / 2;
        let (left, right) = data.split_at(mid);
        let (sum_left, sum_right) = rayon::join(
            || parallel_sum(left),
            || parallel_sum(right),
        );
        sum_left + sum_right
    }
}
```

## 三、注意事项与常见陷阱

1. **开销平衡**：小数据量并行反而更慢
2. **数据竞争**：确保操作是无状态的
3. **side effect**：避免在并行迭代器中有副作用
4. **线程池大小**：默认使用 CPU 核心数
5. **调试困难**：并行代码调试和复现更困难
