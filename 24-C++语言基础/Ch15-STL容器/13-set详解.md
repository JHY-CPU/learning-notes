# set详解

## 一、概念说明

`std::set`是基于红黑树的有序唯一元素集合（C++标准 §23.4.6.1），支持O(log n)的查找、插入和删除。元素自动排序且不重复。set可以看作只有键没有值的map（`set<T>`等价于`map<T, void*>`的简化版）。

### 1.1 核心特性

| 特性 | 说明 |
|------|------|
| 底层结构 | 红黑树 |
| 元素唯一性 | 是 |
| 排序 | 自动按键排序 |
| 查找复杂度 | O(log n) |
| 迭代器类别 | 双向（bidirectional） |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <set>
#include <iostream>

int main() {
    // 初始化
    std::set<int> s = {5, 3, 1, 4, 2, 1};  // 重复的1被忽略

    // 插入
    auto [it, inserted] = s.insert(6);
    std::cout << "插入6: " << (inserted ? "成功" : "失败") << std::endl;
    std::cout << "插入的元素: " << *it << std::endl;

    // 删除
    s.erase(3);             // 按值删除
    s.erase(s.find(4));     // 按迭代器删除

    // 遍历（自动排序）
    for (const auto& v : s) std::cout << v << " ";  // 1 2 5 6
    std::cout << std::endl;

    // 查找
    if (s.find(5) != s.end())
        std::cout << "找到5" << std::endl;

    // 检查存在
    if (s.count(5) > 0) { /* 存在 */ }
    if (s.contains(5)) { /* C++20 */ }

    // 大小
    std::cout << "size: " << s.size() << std::endl;
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

    // 差集 A-B
    result.clear();
    std::set_difference(a.begin(), a.end(), b.begin(), b.end(),
                        std::inserter(result, result.begin()));
    // result: {1, 2}

    // 对称差集
    result.clear();
    std::set_symmetric_difference(a.begin(), a.end(), b.begin(), b.end(),
                                   std::inserter(result, result.begin()));
    // result: {1, 2, 5, 6}

    // 子集判断
    bool is_subset = std::includes(a.begin(), a.end(),
                                    std::set<int>{1, 2}.begin(),
                                    std::set<int>{1, 2}.end());
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

    // 范围遍历 [3, 7]
    auto range_lo = s.lower_bound(3);  // 3
    auto range_hi = s.upper_bound(7);  // 9
    for (auto it = range_lo; it != range_hi; ++it)
        std::cout << *it << " ";  // 3 5 7

    // equal_range
    auto [begin, end] = s.equal_range(5);  // 只有5
}
```

### 2.4 自定义比较器

```cpp
// 降序
std::set<int, std::greater<int>> desc = {1, 2, 3};  // 3, 2, 1

// 自定义结构体
struct Point {
    int x, y;
    bool operator<(const Point& other) const {
        return std::tie(x, y) < std::tie(other.x, other.y);
    }
};
std::set<Point> points;
```

## 三、注意事项与常见陷阱

1. **set的元素是const的**：不能通过迭代器修改（会破坏排序），需erase+insert
2. **自定义类型需要定义`operator<`或提供比较器**
3. **set没有`operator[]`**：没有值的概念
4. **`insert`返回pair**：`pair<iterator, bool>`，第二个元素表示是否插入成功
5. **set的迭代器是双向的**：不支持随机访问，不能用`std::sort`
6. **需要修改元素时**：先erase再insert
7. **set vs unordered_set**：有序用set，快速查找用unordered_set
