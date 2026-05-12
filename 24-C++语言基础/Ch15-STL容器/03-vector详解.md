# vector详解

## 一、概念说明

`std::vector`是最常用的序列容器（C++标准 §23.3.11），底层是动态数组。它支持O(1)随机访问，在尾部插入删除均摊O(1)，但扩容时需要重新分配内存并拷贝/移动所有元素。vector的连续内存布局使其具有极佳的缓存友好性，是大多数场景的默认选择。

### 1.1 核心特性

| 特性 | 说明 |
|------|------|
| 内存布局 | 完全连续 |
| 随机访问 | O(1) |
| 尾部插入 | 均摊O(1) |
| 中间插入 | O(n) |
| 迭代器类别 | 随机访问迭代器 |
| 引用稳定性 | 扩容时全部失效 |

## 二、具体用法

### 2.1 初始化方式

```cpp
#include <vector>
#include <iostream>
#include <string>

int main() {
    // 各种初始化方式
    std::vector<int> v1;                        // 空
    std::vector<int> v2(5);                     // 5个0
    std::vector<int> v3(5, 42);                 // 5个42
    std::vector<int> v4 = {1, 2, 3, 4, 5};     // 初始化列表（C++11）
    std::vector<int> v5(v4);                    // 拷贝构造
    std::vector<int> v6(std::move(v4));         // 移动构造（C++11）
    std::vector<int> v7(v5.begin() + 1, v5.end() - 1);  // 迭代器范围

    // 从C数组构造（C++11）
    int arr[] = {1, 2, 3};
    std::vector<int> v8(std::begin(arr), std::end(arr));
}
```

### 2.2 元素访问

```cpp
void access_demo() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 下标访问（无越界检查，最快）
    std::cout << v[0] << std::endl;          // 1

    // at()访问（有越界检查，抛出out_of_range）
    try {
        std::cout << v.at(10) << std::endl;
    } catch (const std::out_of_range& e) {
        std::cerr << "越界: " << e.what() << std::endl;
    }

    // 首尾元素
    std::cout << v.front() << std::endl;     // 1
    std::cout << v.back() << std::endl;      // 5

    // data()获取底层指针（C++11，C接口兼容）
    int* ptr = v.data();
    std::cout << ptr[2] << std::endl;        // 3

    // 迭代器
    for (auto it = v.begin(); it != v.end(); ++it)
        std::cout << *it << " ";

    // 范围for（C++11）
    for (const auto& elem : v)
        std::cout << elem << " ";
}
```

### 2.3 容量管理

```cpp
void capacity_demo() {
    std::vector<int> v;

    // size vs capacity
    std::cout << "size: " << v.size() << std::endl;         // 0
    std::cout << "capacity: " << v.capacity() << std::endl; // 0

    // reserve预分配（避免多次扩容）
    v.reserve(100);
    std::cout << "capacity: " << v.capacity() << std::endl; // >= 100

    // push_back触发扩容
    for (int i = 0; i < 10; ++i) {
        v.push_back(i);
        // 扩容策略通常是1.5倍（MSVC/GCC）或2倍（早期实现）
    }

    // shrink_to_fit释放多余容量（C++11，非绑定请求）
    v.shrink_to_fit();

    // resize改变size
    v.resize(50);      // 扩充到50，新元素默认初始化
    v.resize(60, 42);  // 扩充到60，新元素为42
    v.resize(10);      // 截断到10

    // max_size：理论最大元素数
    std::cout << "max_size: " << v.max_size() << std::endl;
}
```

### 2.4 增删操作

```cpp
#include <algorithm>

void modify_demo() {
    std::vector<int> v = {1, 2, 3, 4, 5};

    // 尾部操作（高效）
    v.push_back(6);              // O(1)均摊
    v.emplace_back(7);           // C++11，原地构造，无额外拷贝
    v.pop_back();                // O(1)

    // 任意位置操作（低效，O(n)）
    v.insert(v.begin() + 2, 99);           // 在索引2前插入99
    v.insert(v.end(), 3, 88);              // 尾部插入3个88
    v.insert(v.end(), {10, 20, 30});       // C++11，插入初始化列表

    v.emplace(v.begin() + 1, 77);          // C++11，原地构造

    v.erase(v.begin());                     // 删除第一个元素
    v.erase(v.begin(), v.begin() + 3);     // 删除前3个元素

    v.clear();                              // 清空，size=0
}
```

## 三、性能优化

```cpp
// 1. 预分配避免多次扩容
std::vector<int> v;
v.reserve(10000);  // 一次性分配

// 2. emplace替代push（避免临时对象）
struct Point { int x, y; Point(int a, int b) : x(a), y(b) {} };
std::vector<Point> points;
points.emplace_back(1, 2);   // 直接构造，无拷贝
// points.push_back(Point(1, 2));  // 先构造临时对象，再移动

// 3. 移动语义
std::vector<std::string> v1 = {"hello", "world"};
std::vector<std::string> v2 = std::move(v1);  // O(1)，转移所有权

// 4. swap技巧释放内存
std::vector<int> huge(1000000);
huge.clear();
std::vector<int>().swap(huge);  // capacity归零
```

## 四、注意事项与常见陷阱

1. **`reserve()`预分配空间**：已知元素数量时使用，避免多次扩容
2. **`push_back`可能使所有迭代器失效**：扩容时重新分配内存
3. **`insert`和`erase`使被操作位置之后的迭代器失效**
4. **`operator[]`不检查越界**：调试模式可能有断言，生产环境用`at()`
5. **vector的元素必须可复制或可移动**（C++11后支持移动语义）
6. **存储指针时注意内存管理**：考虑使用`unique_ptr`避免内存泄漏
7. **`vector<bool>`是特化**：位压缩实现，`v[i]`返回代理对象而非`bool&`
8. **`shrink_to_fit`是非绑定请求**：编译器可能忽略
