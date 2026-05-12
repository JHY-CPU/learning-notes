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

### 2.3 并行归约操作

```rust
use rayon::prelude::*;

fn parallel_reduction() {
    // 自定义归约：计算最大值及其位置
    let data: Vec<i64> = (0..10_000_000).rev().collect();

    let result = data.par_iter()
        .enumerate()
        .reduce(
            || (0usize, &i64::MIN),
            |(idx1, val1), (idx2, val2)| {
                if val1 >= val2 { (idx1, val1) } else { (idx2, val2) }
            },
        );

    println!("最大值 {} 在位置 {}", result.1, result.0);
}
```

### 2.4 并行文件系统操作

```rust
use rayon::prelude::*;
use std::fs;
use std::path::Path;

fn parallel_dir_scan(root: &Path) -> Vec<(String, u64)> {
    // 递归收集所有文件
    fn collect_files(dir: &Path) -> Vec<std::path::PathBuf> {
        let mut files = Vec::new();
        if let Ok(entries) = fs::read_dir(dir) {
            for entry in entries.flatten() {
                let path = entry.path();
                if path.is_dir() {
                    files.extend(collect_files(&path));
                } else {
                    files.push(path);
                }
            }
        }
        files
    }

    let all_files = collect_files(root);

    // 并行处理文件
    all_files.par_iter()
        .filter_map(|path| {
            let metadata = fs::metadata(path).ok()?;
            let size = metadata.len();
            let name = path.to_string_lossy().into_owned();
            Some((name, size))
        })
        .collect()
}
```

### 2.5 并行字符串处理

```rust
use rayon::prelude::*;

fn parallel_text_analysis(texts: &[String]) -> (usize, usize) {
    // 并行统计字符数和行数
    let (total_chars, total_lines) = texts.par_iter()
        .map(|text| (text.len(), text.lines().count()))
        .reduce(
            || (0, 0),
            |(c1, l1), (c2, l2)| (c1 + c2, l1 + l2),
        );

    (total_chars, total_lines)
}
```

### 2.6 并行矩阵运算

```rust
use rayon::prelude::*;

fn parallel_matrix_multiply(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let m = b[0].len();
    let k = b.len();

    (0..n).into_par_iter()
        .map(|i| {
            (0..m).map(|j| {
                (0..k).map(|l| a[i][l] * b[l][j]).sum()
            }).collect()
        })
        .collect()
}
```

## 四、性能调优最佳实践

### 4.1 线程池大小

```rust
use rayon::ThreadPoolBuilder;

fn optimal_thread_pool() {
    // 对于 CPU 密集型：线程数 = CPU 核心数
    // 对于 I/O 混合型：可以更多线程
    let pool = ThreadPoolBuilder::new()
        .num_threads(num_cpus::get())
        .build()
        .unwrap();

    pool.install(|| {
        // 在优化的线程池中执行
    });
}
```

### 4.2 最小任务粒度

```rust
use rayon::prelude::*;

fn control_granularity() {
    let data: Vec<i64> = (0..10_000_000).collect();

    // 使用 with_min_len 控制每个任务的最小大小
    let sum: i64 = data.par_iter()
        .with_min_len(10_000)  // 每个任务至少处理 10000 个元素
        .map(|&x| x)
        .sum();
}
```

## 五、注意事项与常见陷阱

1. **开销平衡**：小数据量并行反而更慢，通常 1000+ 元素才考虑并行
2. **数据竞争**：确保操作是无状态的，避免共享可变状态
3. **Side Effect**：避免在并行迭代器中有副作用，使用 for_each 代替
4. **线程池大小**：默认使用 CPU 核心数，可通过 ThreadPoolBuilder 自定义
5. **调试困难**：并行代码调试和复现更困难，使用确定性种子辅助
6. **嵌套并行**：避免在并行迭代器内嵌套另一个并行迭代器
7. **内存使用**：并行处理可能增加内存使用，注意大数据集的内存限制
