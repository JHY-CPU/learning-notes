# multiset详解

## 一、概念说明

`std::multiset`是允许重复元素的有序集合。与`set`不同，它可以在同一位置存储多个相同的值。底层同样基于红黑树。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <set>
#include <iostream>

int main() {
    std::multiset<int> ms = {5, 3, 1, 4, 2, 1, 1};

    // 允许重复插入
    ms.insert(1);
    ms.insert(1);

    std::cout << "count(1): " << ms.count(1) << std::endl;  // 4

    // 遍历（排序+重复）
    for (const auto& v : ms) std::cout << v << " ";
    // 1 1 1 1 2 3 4 5

    // 删除（erase(key)删除所有匹配值）
    ms.erase(1);  // 删除所有1
    // 或删除单个
    auto it = ms.find(2);
    if (it != ms.end()) ms.erase(it);
}
```

### 2.2 查找与范围

```cpp
void demo() {
    std::multiset<int> ms = {1, 1, 2, 2, 3, 3};

    // equal_range
    auto [lo, hi] = ms.equal_range(2);
    std::cout << "2的范围: ";
    for (auto it = lo; it != hi; ++it) std::cout << *it << " ";

    // lower_bound / upper_bound
    auto lb = ms.lower_bound(2);  // 第一个2
    auto ub = ms.upper_bound(2);  // 第一个3
}
```

## 三、注意事项与常见陷阱

- `erase(key)`删除所有匹配值，用迭代器删除只删一个
- multiset元素是const的，不能修改（需erase+insert）
- 查找是O(log n)，计数是O(log n + count)
- 如果不需要重复，优先使用set
- multiset的迭代器在插入时不失效
