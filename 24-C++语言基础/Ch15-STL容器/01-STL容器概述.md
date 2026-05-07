# STL容器概述

## 一、概念说明

STL容器是C++标准库中用于存储和管理数据集合的模板类。它们分为四大类：**序列容器**、**关联容器**、**无序关联容器**和**容器适配器**。每种容器针对不同的使用场景进行了优化。

## 二、容器分类

### 2.1 序列容器（Sequence Containers）

```cpp
#include <vector>
#include <deque>
#include <list>
#include <forward_list>
#include <array>

// vector：动态数组，支持快速随机访问
std::vector<int> vec = {1, 2, 3, 4, 5};

// deque：双端队列，两端高效插入删除
std::deque<int> deq = {1, 2, 3, 4, 5};

// list：双向链表，任意位置高效插入删除
std::list<int> lst = {1, 2, 3, 4, 5};

// forward_list：单向链表，内存开销更小
std::forward_list<int> flst = {1, 2, 3, 4, 5};

// array：固定大小数组，零开销
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

// 基于哈希表，平均O(1)查找
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
std::stack<int> stk;            // 默认基于deque
std::queue<int> que;            // 默认基于deque
std::priority_queue<int> pq;    // 默认基于vector（大顶堆）
```

### 2.5 选择指南

```cpp
/*
| 需求                      | 推荐容器                    |
|--------------------------|---------------------------|
| 随机访问                  | vector, array             |
| 频繁头部/尾部操作          | deque                     |
| 频繁中间插入删除           | list                      |
| 有序查找                  | map, set                  |
| 快速查找（无需有序）        | unordered_map, unordered_set |
| 栈操作                   | stack                     |
| 队列操作                  | queue                     |
| 优先级队列                | priority_queue            |
*/
```

## 三、注意事项与常见陷阱

- vector是默认选择，除非有明确理由使用其他容器
- 容器存储的是元素的副本（值语义），大对象考虑存储指针
- 所有容器都是模板类，需要指定元素类型
- 容器的迭代器失效规则因容器而异，需要特别注意
- C++11后容器支持移动语义，提高性能
