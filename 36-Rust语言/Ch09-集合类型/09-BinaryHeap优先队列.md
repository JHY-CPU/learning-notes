# BinaryHeap优先队列

## 一、概念说明

`BinaryHeap<T>` 是标准库提供的二叉堆实现（最大堆），可用于实现优先队列。元素按降序弹出，最大值总是在堆顶。

```rust
use std::collections::BinaryHeap;

let mut heap = BinaryHeap::new();
heap.push(3);
heap.push(1);
heap.push(5);
heap.push(2);

println!("{}", heap.pop().unwrap()); // 5 (最大值)
println!("{}", heap.pop().unwrap()); // 3
```

## 二、具体用法

### 2.1 基本操作

```rust
use std::collections::BinaryHeap;

let mut heap = BinaryHeap::new();

// 插入元素
heap.push(10);
heap.push(20);
heap.push(5);

// 查看堆顶（不移除）
if let Some(&top) = heap.peek() {
    println!("堆顶: {}", top); // 20
}

// 弹出最大值
let max = heap.pop(); // Some(20)

// 长度与清空
println!("长度: {}", heap.len());
heap.clear();
```

### 2.2 最小堆实现

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

// 使用 Reverse 包装实现最小堆
let mut min_heap = BinaryHeap::new();
min_heap.push(Reverse(3));
min_heap.push(Reverse(1));
min_heap.push(Reverse(5));

println!("{}", min_heap.pop().unwrap().0); // 1 (最小值)
println!("{}", min_heap.pop().unwrap().0); // 3
```

### 2.3 Dijkstra最短路径应用

```rust
use std::collections::BinaryHeap;
use std::cmp::Reverse;

#[derive(Eq, PartialEq)]
struct State {
    cost: usize,
    node: usize,
}

impl Ord for State {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        other.cost.cmp(&self.cost) // 反转实现最小堆
    }
}

impl PartialOrd for State {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

fn shortest_path(graph: &Vec<Vec<(usize, usize)>>, start: usize) -> Vec<usize> {
    let n = graph.len();
    let mut dist = vec![usize::MAX; n];
    let mut heap = BinaryHeap::new();
    dist[start] = 0;
    heap.push(State { cost: 0, node: start });

    while let Some(State { cost, node }) = heap.pop() {
        if cost > dist[node] { continue; }
        for &(next, weight) in &graph[node] {
            let new_cost = cost + weight;
            if new_cost < dist[next] {
                dist[next] = new_cost;
                heap.push(State { cost: new_cost, node: next });
            }
        }
    }
    dist
}
```

## 三、注意事项与常见陷阱

1. **最大堆特性**：默认是最大堆，需要最小堆时用 `Reverse` 包装
2. **Ord 实现**：自定义类型需实现 `Ord` trait
3. **迭代无序**：迭代 BinaryHeap 不保证排序顺序
4. **耗尽迭代**：需要有序弹出应使用 `pop` 循环
5. **内存效率**：内部使用 Vec，支持 `with_capacity` 预分配
