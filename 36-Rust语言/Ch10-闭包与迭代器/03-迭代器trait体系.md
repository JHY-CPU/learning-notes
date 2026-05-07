# 迭代器trait体系

## 一、概念说明

Rust 的迭代器系统围绕 `Iterator` trait 构建，提供了惰性求值的序列处理能力。迭代器是 Rust 零成本抽象的核心体现，编译器能将其优化为等效的手写循环。

```rust
pub trait Iterator {
    type Item;
    fn next(&mut self) -> Option<Self::Item>;

    // 提供了80+个默认方法
    fn map<B, F>(self, f: F) -> Map<Self, F>
    where Self: Sized, F: FnMut(Self::Item) -> B;

    fn filter<P>(self, predicate: P) -> Filter<Self, P>
    where Self: Sized, P: FnMut(&Self::Item) -> bool;

    // ...更多适配器方法
}
```

## 二、具体用法

### 2.1 创建迭代器

```rust
// 从集合创建
let v = vec![1, 2, 3];
let iter = v.iter();        // 产生 &T
let iter = v.iter_mut();    // 产生 &mut T
let iter = v.into_iter();   // 产生 T（消耗集合）

// 范围迭代器
let range = 1..10;
let range_inclusive = 1..=10;

// 自定义迭代器
struct Counter {
    count: u32,
    max: u32,
}

impl Iterator for Counter {
    type Item = u32;

    fn next(&mut self) -> Option<Self::Item> {
        if self.count < self.max {
            self.count += 1;
            Some(self.count)
        } else {
            None
        }
    }
}
```

### 2.2 消费迭代器

```rust
let v = vec![1, 2, 3, 4, 5];

// collect 收集为集合
let vec: Vec<i32> = v.iter().map(|x| x * 2).collect();

// sum/product 聚合
let sum: i32 = v.iter().sum();
let product: i32 = v.iter().product();

// any/all 条件检查
let has_even = v.iter().any(|x| x % 2 == 0);
let all_positive = v.iter().all(|x| *x > 0);

// count/last/position
let count = v.iter().count();
let last = v.iter().last();
let pos = v.iter().position(|x| *x == 3);

// reduce/fold 累积
let sum = v.iter().fold(0, |acc, x| acc + x);
```

### 2.3 迭代器适配器链

```rust
let result: Vec<String> = (1..=100)
    .filter(|x| x % 2 == 0)       // 偶数
    .filter(|x| x % 3 == 0)       // 且是3的倍数
    .map(|x| format!("数字: {}", x))
    .take(5)                       // 取前5个
    .collect();
```

## 三、注意事项与常见陷阱

1. **惰性求值**：适配器不执行计算，只在消费时才执行
2. **所有权消耗**：into_iter 消耗集合，iter 只借用
3. **零成本抽象**：迭代器链编译为高效的机器码
4. **短路求值**：any/all/find 等方法会提前终止
5. **大小提示**：实现 size_hint 优化 collect 性能
