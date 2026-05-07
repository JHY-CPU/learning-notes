# HashSet与BTreeSet

## 一、概念说明

`HashSet<T>` 是基于哈希表的集合，元素唯一且无序。`BTreeSet<T>` 是基于B树的集合，元素唯一且有序。两者都要求元素实现 `Eq + Hash`（HashSet）或 `Ord`（BTreeSet）。

```rust
use std::collections::{HashSet, BTreeSet};

let mut hash_set = HashSet::new();
hash_set.insert("苹果");
hash_set.insert("香蕉");

let mut btree_set = BTreeSet::new();
btree_set.insert(3);
btree_set.insert(1);
btree_set.insert(2);
// 遍历顺序为 1, 2, 3
```

## 二、具体用法

### 2.1 基本操作

```rust
use std::collections::HashSet;

let mut a = HashSet::new();
a.insert(1);
a.insert(2);
a.insert(3);

let mut b = HashSet::new();
b.insert(2);
b.insert(3);
b.insert(4);

// 并集
let union: HashSet<_> = a.union(&b).collect();

// 交集
let intersection: HashSet<_> = a.intersection(&b).collect();

// 差集
let difference: HashSet<_> = a.difference(&b).collect();

// 对称差集
let sym_diff: HashSet<_> = a.symmetric_difference(&b).collect();

println!("并集: {:?}", union);           // {1, 2, 3, 4}
println!("交集: {:?}", intersection);     // {2, 3}
println!("差集: {:?}", difference);       // {1}
println!("对称差: {:?}", sym_diff);       // {1, 4}
```

### 2.2 子集与超集检查

```rust
let a: HashSet<_> = [1, 2].iter().cloned().collect();
let b: HashSet<_> = [1, 2, 3].iter().cloned().collect();

println!("a 是 b 的子集: {}", a.is_subset(&b));  // true
println!("b 是 a 的超集: {}", b.is_superset(&a)); // true
println!("a 和 b 是否不相交: {}", a.is_disjoint(&b)); // false
```

### 2.3 去重应用

```rust
let data = vec![1, 2, 2, 3, 3, 3, 4, 4, 4, 4];

// 使用 HashSet 去重
let unique: HashSet<_> = data.iter().cloned().collect();
let mut unique_vec: Vec<_> = unique.into_iter().collect();
unique_vec.sort();
// [1, 2, 3, 4]

// 使用 BTreeSet 去重并自动排序
let unique_sorted: BTreeSet<_> = data.iter().cloned().collect();
// {1, 2, 3, 4}
```

## 三、注意事项与常见陷阱

1. **哈希性能**：HashSet 使用 SipHash，大量小整数时 BTreeSet 可能更快
2. **有序遍历**：需要有序遍历时使用 BTreeSet
3. **借用规则**：集合中的元素不能同时被借用和修改
4. **容量预分配**：HashSet 也支持 `with_capacity` 预分配
5. **迭代器顺序**：HashSet 迭代顺序不稳定，不要依赖它
