# deque详解

## 一、概念说明

`std::deque`（双端队列，C++标准 §23.3.8）支持两端高效插入删除。底层由多个固定大小的内存块（chunk）组成，通过中控映射表管理，兼具vector的随机访问和list的两端操作优势。但分段连续的内存布局使其缓存性能不如vector。

### 1.1 内存布局

```
中控映射表（map）:
[ptr] → [chunk0: 元素0-7]
[ptr] → [chunk1: 元素8-15]
[ptr] → [chunk2: 元素16-23]

特点：
- 随机访问：通过中控表计算chunk和偏移，O(1)但有间接开销
- 两端插入：只需在首/尾chunk操作，满时分配新chunk
- 无连续data()：不能直接传给C接口
```

## 二、具体用法

### 2.1 基本操作

```cpp
#include <deque>
#include <iostream>

int main() {
    std::deque<int> d = {1, 2, 3, 4, 5};

    // 两端操作（O(1)）
    d.push_front(0);
    d.push_back(6);
    d.emplace_front(-1);   // C++11
    d.emplace_back(7);     // C++11

    std::cout << "front: " << d.front() << std::endl;  // -1
    std::cout << "back: " << d.back() << std::endl;    // 7

    // 随机访问（O(1)，但比vector慢）
    std::cout << "d[3]: " << d[3] << std::endl;

    // 弹出
    d.pop_front();
    d.pop_back();

    // 容量
    std::cout << "size: " << d.size() << std::endl;
    std::cout << "empty: " << d.empty() << std::endl;

    // 注意：deque没有capacity()和reserve()！
}
```

### 2.2 插入与删除

```cpp
void modify_demo() {
    std::deque<int> d = {1, 2, 3, 4, 5};

    // 中间插入（O(n)，需要移动元素）
    d.insert(d.begin() + 2, 99);           // 在索引2前插入99
    d.insert(d.end(), 3, 88);              // 尾部插入3个88
    d.emplace(d.begin() + 1, 77);          // 原地构造

    // 删除
    d.erase(d.begin());                     // 删除第一个
    d.erase(d.begin(), d.begin() + 3);     // 删除前3个

    // 清空
    d.clear();

    // 批量插入
    std::deque<int> more = {10, 20, 30};
    d.insert(d.end(), more.begin(), more.end());
}
```

### 2.3 与vector对比

```cpp
#include <vector>
#include <chrono>

void compare_vector_deque() {
    const int N = 100000;

    // 头部插入：deque优势
    auto start = std::chrono::high_resolution_clock::now();
    std::deque<int> d;
    for (int i = 0; i < N; ++i) d.push_front(i);  // O(1)
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "deque push_front: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    start = std::chrono::high_resolution_clock::now();
    std::vector<int> v;
    for (int i = 0; i < N; ++i) v.insert(v.begin(), i);  // O(n)
    end = std::chrono::high_resolution_clock::now();
    std::cout << "vector insert(begin): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    // 随机访问：vector优势（缓存友好）
    // 大多数场景vector仍更快
}
```

### 2.4 作为适配器底层容器

```cpp
#include <stack>
#include <queue>

// deque是stack和queue的默认底层容器
std::stack<int> stk;          // 实际是 stack<int, deque<int>>
std::queue<int> que;          // 实际是 queue<int, deque<int>>

// 选择deque的原因：
// 1. 两端操作O(1)
// 2. 比list缓存友好
// 3. 内存分配比vector灵活（不需要连续大块）
```

## 三、迭代器失效规则

```cpp
/*
| 操作                  | 迭代器失效           |
|-----------------------|---------------------|
| push_front/back       | 所有迭代器失效       |
| pop_front/back        | 所有迭代器失效       |
| insert（中间）         | 所有迭代器失效       |
| erase（中间）          | 所有迭代器失效       |
| operator[]            | 不失效              |
| at()                  | 不失效              |

对比vector：
- vector的push_back只在扩容时失效
- deque的push_front/back总是使所有迭代器失效
*/
```

## 四、注意事项与常见陷阱

1. **没有`reserve()`和`capacity()`**：无法预分配，内存管理由实现决定
2. **内存不是完全连续的**：`&d[0] + 1`不一定等于`&d[1]`，不能传给需要连续内存的C接口
3. **迭代器失效比vector更频繁**：两端操作总是使所有迭代器失效
4. **随机访问比vector慢**：需要通过中控表间接访问
5. **更多的内存开销**：管理多个内存块的中控表
6. **适用场景**：需要频繁两端操作且需要随机访问（如滑动窗口、BFS队列）
7. **大多数场景vector仍更快**：缓存友好性弥补了理论复杂度差距
