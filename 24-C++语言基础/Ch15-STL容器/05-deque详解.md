# deque详解

## 一、概念说明

`std::deque`（双端队列）是支持两端高效插入删除的序列容器。底层由多个固定大小的内存块组成（分段连续），兼具vector的随机访问和list的两端操作优势。

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
    std::cout << "front: " << d.front() << std::endl;  // 0
    std::cout << "back: " << d.back() << std::endl;    // 6

    // 随机访问（O(1)）
    std::cout << "d[3]: " << d[3] << std::endl;  // 3

    // 容量
    std::cout << "size: " << d.size() << std::endl;

    // 弹出
    d.pop_front();
    d.pop_back();
}
```

### 2.2 与vector对比

```cpp
void compare() {
    // deque没有capacity()和reserve()
    std::deque<int> d;
    // d.reserve(100);  // 编译错误

    // deque的内存不是完全连续的
    std::deque<int> d2 = {1, 2, 3};
    // &d2[0] + 1 不一定等于 &d2[1]（分段连续）

    // deque在头部插入不移动所有元素
    for (int i = 0; i < 1000; ++i) {
        d.push_front(i);  // O(1)，不需要移动后续元素
    }
}
```

### 2.3 适用场景

```cpp
#include <queue>

// 作为stack和queue的底层容器
std::deque<int> d;
std::stack<int, std::deque<int>> stk(d);  // 默认就是deque

// 需要频繁头部操作的场景
std::deque<std::string> history;
history.push_front("最新操作");
history.pop_front();
if (history.size() > 100) history.pop_back();

// 需要随机访问+两端操作
std::deque<int> sliding_window;
sliding_window.push_back(1);
sliding_window.push_back(2);
std::cout << sliding_window[0] << std::endl;  // 随机访问
```

## 三、注意事项与常见陷阱

- deque没有`reserve()`和`capacity()`方法
- deque的内存分段连续，不能直接传给需要连续内存的C接口
- deque的迭代器失效规则比vector复杂
- 随机插入删除仍为O(n)，只是两端操作为O(1)
- deque比vector有更多的内存开销（管理多个内存块）
- 大多数场景下vector仍比deque快（缓存友好性）
