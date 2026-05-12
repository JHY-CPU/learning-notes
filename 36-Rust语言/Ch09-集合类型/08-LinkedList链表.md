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

### 2.4 Cursor 游标操作

```rust
use std::collections::LinkedList;

fn cursor_example() {
    let mut list: LinkedList<i32> = (1..=5).collect();

    // Cursor 提供在链表中间插入/删除的能力
    let mut cursor = list.cursor_front();
    if let Some(current) = cursor.current() {
        println!("当前元素: {}", current);
    }
    cursor.move_next();
    if let Some(current) = cursor.current() {
        println!("下一个元素: {}", current);
    }

    // 注意：cursor API 目前在 nightly 中
    // 在 stable 中需要手动实现遍历逻辑
}
```

### 2.5 手动实现链表节点

```rust
// 当需要更复杂的链表操作时，可以手动实现
use std::ptr::NonNull;
use std::marker::PhantomData;

struct Node<T> {
    next: Option<NonNull<Node<T>>>,
    prev: Option<NonNull<Node<T>>>,
    data: T,
}

// 简单的栈实现用于对比
fn stack_vs_list() {
    // Vec 作为栈：高效的 push/pop
    let mut stack = Vec::new();
    stack.push(1);
    stack.push(2);
    stack.pop(); // O(1)

    // LinkedList 作为队列：push_back + pop_front
    let mut queue = LinkedList::new();
    queue.push_back(1);
    queue.push_back(2);
    queue.pop_front(); // O(1) 但有堆分配开销
}
```

### 2.6 性能基准对比

```rust
use std::collections::LinkedList;
use std::time::Instant;

fn benchmark_comparison() {
    let n = 100_000;

    // Vec 插入性能
    let start = Instant::now();
    let mut vec = Vec::new();
    for i in 0..n { vec.push(i); }
    // 随机访问
    let _ = vec[n / 2];
    println!("Vec: {:?}", start.elapsed());

    // LinkedList 插入性能
    let start = Instant::now();
    let mut list = LinkedList::new();
    for i in 0..n { list.push_back(i); }
    // 需要遍历访问中间元素
    let middle = n / 2;
    let mut iter = list.iter();
    for _ in 0..middle { iter.next(); }
    println!("LinkedList: {:?}", start.elapsed());

    // 结论：LinkedList 在几乎所有操作上都比 Vec 慢
    // 唯一优势：split_off 和 append 在大数据量时更快
}
```

### 2.7 VecDeque 替代方案

```rust
use std::collections::VecDeque;

fn vecdeque_example() {
    // VecDeque 结合了 Vec 和 LinkedList 的优点
    let mut deque = VecDeque::new();

    // 两端操作都是 O(1)
    deque.push_front(1);
    deque.push_back(2);
    deque.push_back(3);
    deque.pop_front(); // Some(1)
    deque.pop_back();  // Some(3)

    // 旋转操作
    let mut deque: VecDeque<i32> = (1..=5).collect();
    deque.rotate_left(2);  // [3, 4, 5, 1, 2]
    deque.rotate_right(1); // [2, 3, 4, 5, 1]

    // 分割（比 LinkedList 更高效）
    let mut deque: VecDeque<i32> = (1..=10).collect();
    let other = deque.split_off(5);
    // deque: [1,2,3,4,5], other: [6,7,8,9,10]
}
```

## 四、实际应用场景

### 4.1 LRU 缓存

```rust
use std::collections::{LinkedList, HashMap};

struct LRUCache<K, V> {
    capacity: usize,
    order: LinkedList<K>,
    map: HashMap<K, V>,
}

impl<K: Clone + Eq + std::hash::Hash, V> LRUCache<K, V> {
    fn new(capacity: usize) -> Self {
        LRUCache {
            capacity,
            order: LinkedList::new(),
            map: HashMap::new(),
        }
    }

    fn get(&mut self, key: &K) -> Option<&V> {
        if self.map.contains_key(key) {
            // 将 key 移到链表头部
            self.order.retain(|k| k != key);
            self.order.push_front(key.clone());
            self.map.get(key)
        } else {
            None
        }
    }

    fn put(&mut self, key: K, value: V) {
        if self.map.contains_key(&key) {
            self.order.retain(|k| k != &key);
        } else if self.order.len() >= self.capacity {
            if let Some(old_key) = self.order.pop_back() {
                self.map.remove(&old_key);
            }
        }
        self.order.push_front(key.clone());
        self.map.insert(key, value);
    }
}
```

### 4.2 工作调度队列

```rust
use std::collections::LinkedList;

struct Task {
    id: u64,
    priority: u8,
    payload: String,
}

struct TaskScheduler {
    high_priority: LinkedList<Task>,
    normal_priority: LinkedList<Task>,
    low_priority: LinkedList<Task>,
}

impl TaskScheduler {
    fn new() -> Self {
        TaskScheduler {
            high_priority: LinkedList::new(),
            normal_priority: LinkedList::new(),
            low_priority: LinkedList::new(),
        }
    }

    fn enqueue(&mut self, task: Task) {
        match task.priority {
            p if p >= 8 => self.high_priority.push_back(task),
            p if p >= 4 => self.normal_priority.push_back(task),
            _ => self.low_priority.push_back(task),
        }
    }

    fn dequeue(&mut self) -> Option<Task> {
        self.high_priority.pop_front()
            .or_else(|| self.normal_priority.pop_front())
            .or_else(|| self.low_priority.pop_front())
    }
}
```

## 五、何时使用 LinkedList

| 场景 | 推荐类型 | 原因 |
|------|---------|------|
| 随机访问 | Vec | O(1) vs O(n) |
| 两端操作 | VecDeque | 缓存友好 |
| 分割/拼接 | LinkedList | O(1) 操作 |
| 排序 | Vec | TimSort 优化 |
| 栈 | Vec | 更快 push/pop |
| 队列 | VecDeque | 更快 enqueue/dequeue |

## 六、注意事项与常见陷阱

1. **性能劣势**：几乎所有场景下 Vec 或 VecDeque 都比 LinkedList 更快
2. **缓存局部性**：链表节点分散在堆上，缓存命中率低，现代 CPU 预取失效
3. **适用场景**：仅在需要频繁分割/拼接列表时考虑使用
4. **所有权管理**：每个节点独立分配在堆上，内存碎片化
5. **实现 trait**：LinkedList 实现了 IntoIterator，可用于迭代器链
6. **retain 方法**：`retain` 是 O(n) 操作，如果频繁调用考虑换用其他数据结构
7. **调试困难**：LinkedList 的 Debug 输出可能很长，大数据量时影响可读性
