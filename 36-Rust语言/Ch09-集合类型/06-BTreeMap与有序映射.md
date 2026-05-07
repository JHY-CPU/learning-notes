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

## 三、注意事项与常见陷阱

1. **性能差异**：小数据量时 BTreeMap 比 HashMap 更快（缓存友好）
2. **键约束**：键必须实现 `Ord` trait
3. **有序保证**：遍历始终按升序排列
4. **范围边界**：注意 `range` 方法的包含/排除边界
5. **内存布局**：BTreeMap 内存局部性好，适合大规模有序数据
