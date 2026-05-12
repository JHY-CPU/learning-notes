# multiset详解

## 一、概念说明

`std::multiset`是允许重复元素的有序集合（C++标准 §23.4.6.1）。与`set`不同，它可以在同一位置存储多个相同的值。底层同样基于红黑树，所有操作的时间复杂度与set相同。

### 1.1 与set的差异

| 特性 | set | multiset |
|------|-----|---------|
| 元素唯一性 | 唯一 | 可重复 |
| `insert`返回 | `pair<it, bool>` | 迭代器 |
| `erase(key)` | 删除1个 | 删除所有匹配 |
| 计数 | 0或1 | 0到N |

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

    std::cout << "count(1): " << ms.count(1) << std::endl;  // 5

    // 遍历（排序+重复）
    for (const auto& v : ms) std::cout << v << " ";
    // 1 1 1 1 1 2 3 4 5

    // 查找（返回第一个匹配）
    auto it = ms.find(1);
    if (it != ms.end())
        std::cout << "找到: " << *it << std::endl;

    // C++20 contains
    if (ms.contains(3)) { /* 存在 */ }
}
```

### 2.2 删除操作

```cpp
void erase_demo() {
    std::multiset<int> ms = {1, 1, 1, 2, 2, 3};

    // erase(key)删除所有匹配值
    size_t removed = ms.erase(1);
    std::cout << "删除了 " << removed << " 个1" << std::endl;  // 3

    // 用迭代器删除只删一个
    ms.insert(2);
    auto it = ms.find(2);
    if (it != ms.end()) ms.erase(it);  // 只删除一个2

    // 删除范围
    ms = {1, 1, 2, 2, 3, 3};
    auto [lo, hi] = ms.equal_range(2);
    ms.erase(lo, hi);  // 删除所有2
}
```

### 2.3 查找与范围

```cpp
void range_demo() {
    std::multiset<int> ms = {1, 1, 2, 2, 2, 3, 3};

    // equal_range
    auto [lo, hi] = ms.equal_range(2);
    std::cout << "2的范围: ";
    for (auto it = lo; it != hi; ++it) std::cout << *it << " ";
    // 2 2 2

    // lower_bound / upper_bound
    auto lb = ms.lower_bound(2);  // 第一个2
    auto ub = ms.upper_bound(2);  // 第一个3

    // 计算距离
    auto dist = std::distance(lb, ub);
    std::cout << "2的数量: " << dist << std::endl;  // 3
}
```

### 2.4 实用示例：排行榜

```cpp
#include <string>

void leaderboard() {
    std::multiset<std::pair<int, std::string>, std::greater<>> board;
    board.insert({100, "Alice"});
    board.insert({95, "Bob"});
    board.insert({95, "Charlie"});  // 同分
    board.insert({90, "Dave"});

    // 自动按分数降序排列
    std::cout << "排行榜:" << std::endl;
    int rank = 1;
    for (const auto& [score, name] : board) {
        std::cout << rank++ << ". " << name << " (" << score << ")" << std::endl;
    }
}
```

## 三、注意事项与常见陷阱

1. **`erase(key)`删除所有匹配值**：用迭代器删除只删一个，误用可能导致数据丢失
2. **multiset元素是const的**：不能修改（需erase+insert）
3. **查找是O(log n)**：计数是O(log n + count)
4. **如果不需要重复，优先使用set**：更高效
5. **multiset的迭代器在插入时不失效**：被删元素除外
6. **`insert`返回迭代器**（不是pair），因为总能插入成功
