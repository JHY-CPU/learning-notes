# LinkedList链表

## 一、概念说明

`LinkedList<T>` 是标准库提供的双向链表实现。与 Vec 和 VecDeque 相比，LinkedList 在分割和拼接操作上有优势，但随机访问性能较差，缓存局部性也不好。

```rust
use std::collections::LinkedList;

let mut list = LinkedList::new();
list.push_back(1);
list.push_back(2);
list.push_front(0);
// [0, 1, 2]
```

## 二、具体用法

### 2.1 基本操作

```rust
use std::collections::LinkedList;

let mut list = LinkedList::new();

// 添加元素
list.push_back('a');
list.push_back('b');
list.push_front('z');

// 弹出元素
let front = list.pop_front(); // Some('z')
let back = list.pop_back();   // Some('b')

// 检查属性
println!("长度: {}", list.len());
println!("是否为空: {}", list.is_empty());

// 清空
list.clear();
```

### 2.2 分割与拼接

```rust
use std::collections::LinkedList;

let mut list1: LinkedList<i32> = (1..=5).collect();
let mut list2 = list1.split_off(3);
// list1: [1, 2, 3]
// list2: [4, 5]

// 拼接
list1.append(&mut list2);
// list1: [1, 2, 3, 4, 5]
```

### 2.3 迭代与转换

```rust
use std::collections::LinkedList;

let mut list: LinkedList<i32> = (1..=5).collect();

// 迭代
for item in &list {
    println!("{}", item);
}

// 转换为 Vec
let vec: Vec<i32> = list.iter().cloned().collect();

// 从迭代器创建
let list: LinkedList<_> = vec![1, 2, 3].into_iter().collect();
```

## 三、注意事项与常见陷阱

1. **性能劣势**：几乎所有场景下 Vec 或 VecDeque 都比 LinkedList 更快
2. **缓存局部性**：链表节点分散在堆上，缓存命中率低
3. **适用场景**：仅在需要频繁分割/拼接列表时考虑使用
4. **所有权管理**：每个节点独立分配在堆上
5. **实现 trait**：LinkedList 实现了 IntoIterator，可用于迭代器链
