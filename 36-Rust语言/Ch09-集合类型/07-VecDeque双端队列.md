# VecDeque双端队列

## 一、概念说明

`VecDeque<T>`（双端队列）是一种允许在两端高效插入和删除的动态数组。与 Vec 相比，VecDeque 在头部操作的时间复杂度为 O(1)，适合需要队列或栈的行为。

```rust
use std::collections::VecDeque;

let mut deque = VecDeque::new();
deque.push_back(1);
deque.push_back(2);
deque.push_front(0);
// [0, 1, 2]
```

## 二、具体用法

### 2.1 基本操作

```rust
use std::collections::VecDeque;

let mut deque = VecDeque::with_capacity(10);

// 两端插入
deque.push_back(1);
deque.push_back(2);
deque.push_front(0);

// 两端弹出
let front = deque.pop_front(); // Some(0)
let back = deque.pop_back();   // Some(2)

// 获取元素
if let Some(val) = deque.get(0) {
    println!("第一个元素: {}", val);
}

// 长度与容量
println!("长度: {}", deque.len());
println!("容量: {}", deque.capacity());
println!("是否为空: {}", deque.is_empty());
```

### 2.2 循环队列应用

```rust
use std::collections::VecDeque;

fn sliding_window_max(data: &[i32], k: usize) -> Vec<i32> {
    let mut result = Vec::new();
    let mut deque: VecDeque<usize> = VecDeque::new();

    for i in 0..data.len() {
        // 移除超出窗口的元素
        while let Some(&idx) = deque.front() {
            if idx + k <= i {
                deque.pop_front();
            } else {
                break;
            }
        }
        // 维护递减队列
        while let Some(&idx) = deque.back() {
            if data[idx] <= data[i] {
                deque.pop_back();
            } else {
                break;
            }
        }
        deque.push_back(i);
        if i >= k - 1 {
            result.push(data[*deque.front().unwrap()]);
        }
    }
    result
}

let data = vec![1, 3, -1, -3, 5, 3, 6, 7];
let max_vals = sliding_window_max(&data, 3);
// [3, 3, 5, 5, 6, 7]
```

### 2.3 旋转与交换

```rust
use std::collections::VecDeque;

let mut deque: VecDeque<i32> = (1..=5).collect();
// [1, 2, 3, 4, 5]

// 向左旋转
deque.rotate_left(2);
// [3, 4, 5, 1, 2]

// 向右旋转
deque.rotate_right(1);
// [2, 3, 4, 5, 1]

// 交换元素
deque.swap(0, 4);
// [1, 3, 4, 5, 2]
```

## 三、注意事项与常见陷阱

1. **环形缓冲区**：VecDeque 内部使用环形缓冲区实现，内存不连续
2. **索引访问**：get 操作是 O(1)，但可能比 Vec 稍慢
3. **容量策略**：push 时可能触发扩容，类似 Vec
4. **切片获取**：使用 `make_contiguous` 获取连续切片
5. **适用场景**：需要队列行为时用 VecDeque，需要随机访问时用 Vec
