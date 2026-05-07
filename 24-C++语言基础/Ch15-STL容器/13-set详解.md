# set详解

## 一、概念说明

`std::set`是基于红黑树的有序唯一元素集合，支持O(log n)的查找、插入和删除。元素自动排序且不重复。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <set>
#include <iostream>

int main() {
    std::set<int> s = {5, 3, 1, 4, 2, 1};  // 重复的1被忽略

    s.insert(6);          // 插入
    s.erase(3);           // 删除
    s.erase(s.find(4));   // 通过迭代器删除

    // 遍历（自动排序）
    for (const auto& v : s) std::cout << v << " ";  // 1 2 5 6
    std::cout << std::endl;

    // 查找
    if (s.find(5) != s.end()) {
        std::cout << "找到5" << std::endl;
    }

    // 检查存在
    if (s.count(5) > 0) { /* 存在 */ }
    if (s.contains(5)) { /* C++20 */ }
}
```

### 2.2 集合运算

```cpp
#include <vector>
#include <algorithm>

void set_operations() {
    std::set<int> a = {1, 2, 3, 4};
    std::set<int> b = {3, 4, 5, 6};
    std::set<int> result;

    // 并集
    std::set_union(a.begin(), a.end(), b.begin(), b.end(),
                   std::inserter(result, result.begin()));
    // result: {1, 2, 3, 4, 5, 6}

    // 交集
    result.clear();
    std::set_intersection(a.begin(), a.end(), b.begin(), b.end(),
                          std::inserter(result, result.begin()));
    // result: {3, 4}

    // 差集
    result.clear();
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(result, result.begin()));
    // result: {1, 2}
}
```

### 2.3 范围查询

```cpp
void range_query() {
    std::set<int> s = {1, 3, 5, 7, 9};

    // lower_bound: >= value 的第一个元素
    auto lo = s.lower_bound(4);  // 指向5

    // upper_bound: > value 的第一个元素
    auto hi = s.upper_bound(6);  // 指向7

    // equal_range: [lower_bound, upper_bound)
    auto [begin, end] = s.equal_range(3);  // 只有3
}
```

## 三、注意事项与常见陷阱

- set的元素是const的，不能通过迭代器修改
- 自定义类型需要定义`operator<`或提供比较器
- set没有`operator[]`（没有值的概念）
- `insert`返回pair：是否插入成功+迭代器
- set的迭代器是双向的（bidirectional）
- 需要修改元素时，先erase再insert
