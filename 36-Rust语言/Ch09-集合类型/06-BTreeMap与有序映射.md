# BTreeMap与有序映射

## 一、概念说明

`BTreeMap<K, V>` 是基于B树实现的有序键值映射。与 HashMap 不同，BTreeMap 中的键始终保持有序排列，适合需要范围查询和有序遍历的场景。

```rust
use std::collections::BTreeMap;

let mut map = BTreeMap::new();
map.insert(3, "三");
map.insert(1, "一");
map.insert(2, "二");
// 遍历顺序: 1, 2, 3
```

## 二、具体用法

### 2.1 范围查询

```rust
use std::collections::BTreeMap;

let mut map = BTreeMap::new();
for i in 1..=10 {
    map.insert(i, format!("值{}", i));
}

// 获取范围 (2..5): 包含2，不包含5
let range: Vec<_> = map.range(2..5).collect();
// [(2, "值2"), (3, "值3"), (4, "值4")]

// 包含范围 (2..=5)
let range_inclusive: Vec<_> = map.range(2..=5).collect();

// 获取第一个和最后一个
let first = map.iter().next();
let last = map.iter().last();
```

### 2.2 分离与合并

```rust
let mut map = BTreeMap::new();
map.insert(1, "a");
map.insert(2, "b");
map.insert(3, "c");
map.insert(4, "d");

// 在键2处分离
let mut upper = map.split_off(&2);
// map: {1: "a"}, upper: {2: "b", 3: "c", 4: "d"}

// 合并
map.append(&mut upper);
// map: {1: "a", 2: "b", 3: "c", 4: "d"}
```

### 2.3 查找前后键

```rust
let mut map = BTreeMap::new();
map.insert(1, "一");
map.insert(3, "三");
map.insert(5, "五");

// 获取最近的小于等于的键
let floor = map.range(..=3).last();
// Some((3, "三"))

// 获取最近的大于的键
let higher = map.range(3..).skip(1).next();
// Some((5, "五"))
```

### 2.4 自定义键类型

```rust
use std::collections::BTreeMap;
use std::cmp::Ordering;

#[derive(Eq, PartialEq, Ord, PartialOrd, Debug)]
struct Person {
    age: u32,
    name: String,
}

// 注意：derive 的 Ord 会按字段声明顺序比较
// age 相同时会比较 name

fn custom_key_example() {
    let mut map = BTreeMap::new();
    map.insert(Person { age: 25, name: "张三".into() }, "工程师");
    map.insert(Person { age: 30, name: "李四".into() }, "设计师");
    map.insert(Person { age: 25, name: "王五".into() }, "产品经理");

    // 按 age 升序，age 相同时按 name 升序
    for (person, role) in &map {
        println!("{}({}岁): {}", person.name, person.age, role);
    }
}
```

### 2.5 BTreeMap 与 HashMap 选择对比

```rust
use std::collections::{BTreeMap, HashMap};
use std::time::Instant;

fn compare_maps() {
    let n = 1_000_000;

    // HashMap: 无序，O(1) 查找
    let start = Instant::now();
    let mut hmap = HashMap::new();
    for i in 0..n { hmap.insert(i, i * 2); }
    println!("HashMap 插入: {:?}", start.elapsed());

    // BTreeMap: 有序，O(log n) 查找，范围查询高效
    let start = Instant::now();
    let mut bmap = BTreeMap::new();
    for i in 0..n { bmap.insert(i, i * 2); }
    println!("BTreeMap 插入: {:?}", start.elapsed());

    // 范围查询：BTreeMap 的优势场景
    let range: Vec<_> = bmap.range(100..200).collect();
    println!("范围查询结果数: {}", range.len());
}
```

### 2.6 Entry API 与条件插入

```rust
use std::collections::BTreeMap;

fn entry_api_examples() {
    let mut scores: BTreeMap<String, Vec<i32>> = BTreeMap::new();

    // or_insert_with: 仅在键不存在时计算默认值
    scores.entry("数学".into()).or_insert_with(Vec::new).push(95);
    scores.entry("数学".into()).or_insert_with(Vec::new).push(88);

    // and_modify: 仅在键存在时修改
    scores.entry("数学".into())
        .and_modify(|v| v.push(76));

    // or_default: 使用 Default trait
    scores.entry("英语".into()).or_default().push(90);

    for (subject, grades) in &scores {
        let avg: f64 = grades.iter().sum::<i32>() as f64 / grades.len() as f64;
        println!("{}: 平均分 {:.1}", subject, avg);
    }
}
```

## 四、实际应用场景

### 4.1 时间序列数据存储

```rust
use std::collections::BTreeMap;

struct TimeSeries {
    data: BTreeMap<u64, f64>, // timestamp -> value
}

impl TimeSeries {
    fn new() -> Self {
        TimeSeries { data: BTreeMap::new() }
    }

    fn insert(&mut self, timestamp: u64, value: f64) {
        self.data.insert(timestamp, value);
    }

    /// 查询时间范围内的所有数据点
    fn query_range(&self, start: u64, end: u64) -> Vec<(u64, f64)> {
        self.data.range(start..end)
            .map(|(&ts, &val)| (ts, val))
            .collect()
    }

    /// 计算移动平均
    fn moving_average(&self, window: usize) -> Vec<f64> {
        self.data.values()
            .collect::<Vec<_>>()
            .windows(window)
            .map(|w| w.iter().copied().sum::<f64>() / window as f64)
            .collect()
    }
}
```

### 4.2 有序排行榜

```rust
use std::collections::BTreeMap;

struct Leaderboard {
    // score -> list of players (BTreeMap 自动按分数排序)
    entries: BTreeMap<std::cmp::Reverse<i32>, Vec<String>>,
}

impl Leaderboard {
    fn new() -> Self {
        Leaderboard { entries: BTreeMap::new() }
    }

    fn add_player(&mut self, name: String, score: i32) {
        self.entries.entry(std::cmp::Reverse(score))
            .or_default()
            .push(name);
    }

    fn top_n(&self, n: usize) -> Vec<(&i32, &Vec<String>)> {
        self.entries.iter()
            .take(n)
            .map(|(k, v)| (&k.0, v))
            .collect()
    }
}
```

## 五、性能考虑

1. **B 树节点大小**：标准库使用 B=6（每个节点最多 11 个键），在现代 CPU 缓存行上表现良好
2. **大数据量测试**：在 100 万级数据上，BTreeMap 范围查询比 HashMap 全遍历快数百倍
3. **内存开销**：BTreeMap 的每个节点比 HashMap 的桶更紧凑，但 HashMap 有更低的平均查找成本
4. **迭代性能**：BTreeMap 顺序迭代非常快（内存局部性好），HashMap 迭代则较慢（遍历所有桶）

## 六、注意事项与常见陷阱

1. **性能差异**：小数据量时 BTreeMap 比 HashMap 更快（缓存友好）
2. **键约束**：键必须实现 `Ord` trait，自定义类型需要正确实现比较逻辑
3. **有序保证**：遍历始终按升序排列，反向遍历使用 `.rev()`
4. **范围边界**：注意 `range` 方法的包含/排除边界，`..` 排除，`..=` 包含
5. **内存布局**：BTreeMap 内存局部性好，适合大规模有序数据
6. **分裂合并**：`split_off` 和 `append` 是 O(n) 操作，大数据量时注意性能
7. **键比较顺序**：derive 的 Ord 按字段声明顺序比较，可能导致不符合预期的排序
