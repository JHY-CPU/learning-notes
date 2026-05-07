# vector详解

## 一、概念说明

`std::vector`是最常用的序列容器，底层是动态数组。它支持随机访问，在尾部插入删除均摊O(1)，但扩容时需要重新分配内存并拷贝所有元素。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <vector>
#include <iostream>

int main() {
    // 初始化方式
    std::vector<int> v1;                      // 空
    std::vector<int> v2(5);                   // 5个0
    std::vector<int> v3(5, 42);               // 5个42
    std::vector<int> v4 = {1, 2, 3, 4, 5};   // 初始化列表
    std::vector<int> v5(v4);                  // 拷贝构造
    std::vector<int> v6(std::move(v4));       // 移动构造

    // 访问元素
    std::cout << v4[0] << std::endl;          // 1（无越界检查）
    std::cout << v4.at(0) << std::endl;       // 1（有越界检查）
    std::cout << v4.front() << std::endl;     // 1
    std::cout << v4.back() << std::endl;      // 5

    // 容量信息
    std::cout << "size: " << v4.size() << std::endl;
    std::cout << "capacity: " << v4.capacity() << std::endl;
    std::cout << "empty: " << v4.empty() << std::endl;
}
```

### 2.2 扩容策略

```cpp
void demonstrate_capacity() {
    std::vector<int> v;
    std::cout << "初始: size=" << v.size() << " capacity=" << v.capacity() << std::endl;

    // 预留空间（避免多次扩容）
    v.reserve(100);
    std::cout << "reserve(100): capacity=" << v.capacity() << std::endl;

    // push_back触发扩容
    for (int i = 0; i < 10; ++i) {
        v.push_back(i);
        std::cout << "push " << i << ": size=" << v.size()
                  << " capacity=" << v.capacity() << std::endl;
    }
    // 扩容通常是1.5倍或2倍增长
}
```

### 2.3 插入和删除

```cpp
#include <algorithm>

void demonstrate_ops() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 尾部操作（高效）
    v.push_back(6);           // O(1)均摊
    v.pop_back();             // O(1)

    // 任意位置操作（低效）
    v.insert(v.begin() + 2, 99);  // O(n)
    v.erase(v.begin());           // O(n)

    // 清空
    v.clear();                // size=0，capacity不变
    v.shrink_to_fit();        // 释放多余容量

    // 批量插入
    std::vector<int> more = {10, 20, 30};
    v.insert(v.end(), more.begin(), more.end());
}
```

### 2.4 数据指针访问

```cpp
void demonstrate_data() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 获取底层数据指针（C接口兼容）
    int* ptr = v.data();
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << ptr[i] << " ";  // 1 2 3 4 5
    }
    std::cout << std::endl;
}
```

## 三、注意事项与常见陷阱

- `reserve()`预分配空间可避免多次扩容
- `push_back`可能使所有迭代器失效（扩容时）
- `insert`和`erase`使被操作位置之后的迭代器失效
- `operator[]`不检查越界，`at()`会抛出`out_of_range`
- vector的元素必须可复制（C++11后可移动）
- 存储指针时注意内存管理，考虑`unique_ptr`
