# STL容器概述

## 一、概念说明

STL容器是C++标准库中用于存储和管理数据集合的模板类（C++标准 §23.2）。它们分为五大类：**序列容器**、**关联容器**、**无序关联容器**、**容器适配器**和**C++17/20/23新增容器**。每种容器针对不同的使用场景进行了优化，选择合适的容器对程序性能至关重要。

### 1.1 容器统一接口

所有标准容器都提供以下基本接口：

| 接口 | 说明 |
|------|------|
| `value_type` | 元素类型 |
| `size()` | 元素数量 |
| `empty()` | 是否为空 |
| `begin()`/`end()` | 迭代器 |
| `operator==` | 相等比较 |

## 二、容器分类

### 2.1 序列容器（Sequence Containers）

```cpp
#include <vector>
#include <deque>
#include <list>
#include <forward_list>
#include <array>

// vector：动态数组，支持快速随机访问，尾部插入O(1)均摊
std::vector<int> vec = {1, 2, 3, 4, 5};

// deque：双端队列，两端高效插入删除，分段连续内存
std::deque<int> deq = {1, 2, 3, 4, 5};

// list：双向链表，任意位置O(1)插入删除（给定迭代器）
std::list<int> lst = {1, 2, 3, 4, 5};

// forward_list：单向链表，内存开销更小（C++11）
std::forward_list<int> flst = {1, 2, 3, 4, 5};

// array：固定大小数组，零开销抽象（C++11）
std::array<int, 5> arr = {1, 2, 3, 4, 5};
```

### 2.2 关联容器（Associative Containers）

```cpp
#include <map>
#include <set>

// 有序关联容器（基于红黑树，O(log n)查找）
std::map<std::string, int> scores = {{"Alice", 90}, {"Bob", 85}};
std::set<int> unique_nums = {1, 2, 3, 4, 5};

// 允许重复键
std::multimap<std::string, int> grades;
std::multiset<int> bag = {1, 1, 2, 2, 3};
```

### 2.3 无序关联容器（Unordered Containers）

```cpp
#include <unordered_map>
#include <unordered_set>

// 基于哈希表，平均O(1)查找（C++11）
std::unordered_map<std::string, int> hash_map = {{"key", 42}};
std::unordered_set<int> hash_set = {1, 2, 3, 4, 5};

// 允许重复
std::unordered_multimap<std::string, int> ummap;
std::unordered_multiset<int> umset;
```

### 2.4 容器适配器（Container Adapters）

```cpp
#include <stack>
#include <queue>

// 基于其他容器的接口适配
std::stack<int> stk;            // 默认基于deque，LIFO
std::queue<int> que;            // 默认基于deque，FIFO
std::priority_queue<int> pq;    // 默认基于vector，大顶堆
```

### 2.5 现代C++新增容器

```cpp
#include <optional>    // C++17
#include <variant>     // C++17
#include <any>         // C++17
#include <span>        // C++20
#include <flat_map>    // C++23

std::optional<int> opt = 42;           // 可选值
std::variant<int, double> v = 3.14;    // 类型安全联合
std::span<int> s(vec);                 // 非拥有视图
```

### 2.6 选择指南

```cpp
/*
| 需求                      | 推荐容器                    |
|--------------------------|---------------------------|
| 默认选择/随机访问          | vector                    |
| 频繁头部/尾部操作          | deque                     |
| 频繁中间插入删除           | list                      |
| 有序查找                  | map, set                  |
| 快速查找（无需有序）        | unordered_map, unordered_set |
| 固定大小                  | array                     |
| 栈操作                   | stack                     |
| 队列操作                  | queue                     |
| 优先级队列                | priority_queue            |
| 可选值                   | optional                  |
| 多类型值                 | variant                   |
| 非拥有视图                | span, string_view         |
*/
```

## 三、容器共性与差异

### 3.1 内存模型

```cpp
// 所有容器使用值语义（存储副本）
std::vector<std::string> v;
v.push_back("hello");  // 存储拷贝，不是指针

// 大对象考虑存储智能指针
std::vector<std::unique_ptr<BigObject>> objects;
objects.push_back(std::make_unique<BigObject>());
```

### 3.2 迭代器类别

```cpp
/*
| 容器              | 迭代器类别     |
|-------------------|---------------|
| vector, array     | 随机访问       |
| deque             | 随机访问       |
| list              | 双向          |
| forward_list      | 前向          |
| set, map          | 双向          |
| unordered_*       | 前向          |
*/
```

## 四、注意事项与常见陷阱

1. **vector是默认选择**：除非有明确理由（频繁头部操作、需要稳定引用等），否则使用vector
2. **值语义**：容器存储元素的副本，大对象考虑存储指针或智能指针
3. **迭代器失效**：不同容器的失效规则差异很大，插入删除时需特别注意
4. **移动语义**：C++11后容器支持移动，存储不可复制但可移动的对象
5. **allocator**：默认使用`std::allocator`，特殊需求可自定义（如内存池）
6. **异常安全**：大多数容器操作提供基本或强异常安全保证
7. **比较容器**：C++20引入`operator<=>`，简化自定义类型的比较
