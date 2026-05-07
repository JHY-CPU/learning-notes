# 动态数组vector概览

## 一、概念说明

`std::vector`是C++标准库最常用的容器，提供**动态大小**的连续数组。它自动管理内存，支持随机访问，尾部插入/删除为O(1)。

与原生数组相比，`vector`更安全（自动管理内存、支持边界检查）、更灵活（动态大小），但有少量额外开销。

## 二、具体用法

### 2.1 构造与初始化

```cpp
#include <vector>

std::vector<int> v1;                    // 空
std::vector<int> v2(5);                 // 5个0
std::vector<int> v3(5, 42);             // 5个42
std::vector<int> v4 = {1, 2, 3, 4, 5}; // 列表初始化

for (int x : v4) std::cout << x << " ";
// 输出: 1 2 3 4 5
```

### 2.2 增删改查

```cpp
std::vector<int> v = {1, 2, 3};

v.push_back(4);        // 尾部添加
v.emplace_back(5);     // 就地构造，更高效
v.insert(v.begin() + 1, 99);  // 在位置1插入99

for (int x : v) std::cout << x << " ";
// 输出: 1 99 2 3 4 5

v.pop_back();          // 删除最后一个
v.erase(v.begin() + 1);  // 删除位置1的元素

for (int x : v) std::cout << x << " ";
// 输出: 1 2 3 4
```

### 2.3 容量管理

```cpp
std::vector<int> v = {1, 2, 3};

std::cout << "size: " << v.size() << std::endl;
std::cout << "capacity: " << v.capacity() << std::endl;

v.reserve(100);   // 预留空间，避免多次扩容
std::cout << "capacity after reserve: " << v.capacity() << std::endl;

v.shrink_to_fit();  // 释放多余容量
std::cout << "capacity after shrink: " << v.capacity() << std::endl;

// 输出:
// size: 3
// capacity: 3 (或更大)
// capacity after reserve: 100
// capacity after shrink: 3
```

### 2.4 与原生数组对比

```cpp
// vector的优势
std::vector<int> v = {1, 2, 3};
v.push_back(4);              // 自动扩展
int sz = v.size();           // 知道大小
v = {5, 6, 7};               // 可赋值
auto v2 = v;                 // 可拷贝

// 原生数组
int arr[] = {1, 2, 3};
// arr[3] = 4;               // 越界！
// auto arr2 = arr;          // 只拷贝指针
```

## 三、注意事项与常见陷阱

- `push_back`可能导致**重新分配**（扩容），使所有迭代器/引用失效
- `emplace_back`就地构造，避免临时对象创建
- 频繁在中间插入/删除用`std::list`或`std::deque`
- `vector<bool>`是特化版本，每个bool可能只占1位
- 使用`at()`进行边界检查，`operator[]`不检查
