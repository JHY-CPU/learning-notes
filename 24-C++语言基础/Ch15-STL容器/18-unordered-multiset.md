# unordered_multiset

## 一、概念说明

`std::unordered_multiset`是允许重复元素的无序集合（C++标准 §23.5.6.1，C++11），基于哈希表实现。相比`multiset`（红黑树），提供平均O(1)的查找但不保证元素顺序。

### 1.1 与相关容器的对比

| 特性 | unordered_multiset | multiset | unordered_set |
|------|-------------------|---------|--------------|
| 元素唯一性 | 可重复 | 可重复 | 唯一 |
| 有序 | 否 | 是 | 否 |
| 查找 | O(1) | O(log n) | O(1) |
| 范围查询 | 不支持 | 支持 | 不支持 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <unordered_set>
#include <iostream>

int main() {
    std::unordered_multiset<int> ums = {5, 3, 1, 4, 2, 1, 1};

    // 插入重复元素
    ums.insert(1);
    ums.insert(1);

    // 计数
    std::cout << "count(1): " << ums.count(1) << std::endl;  // 4

    // 查找
    auto it = ums.find(2);
    if (it != ums.end()) std::cout << "找到: " << *it << std::endl;

    // C++20 contains
    if (ums.contains(3)) { /* 存在 */ }

    // 遍历（无序）
    for (const auto& v : ums) std::cout << v << " ";
    std::cout << std::endl;

    // 大小
    std::cout << "size: " << ums.size() << std::endl;
}
```

### 2.2 删除操作

```cpp
void erase_demo() {
    std::unordered_multiset<int> ums = {1, 1, 1, 2, 2, 3};

    // erase(key)删除所有匹配值
    size_t removed = ums.erase(1);
    std::cout << "删除了 " << removed << " 个1" << std::endl;  // 3

    // 用迭代器删除只删一个
    ums.insert(1);
    auto it = ums.find(1);
    if (it != ums.end()) ums.erase(it);  // 只删除一个1

    // 删除范围
    ums = {1, 1, 2, 2, 3, 3};
    auto range = ums.equal_range(2);
    ums.erase(range.first, range.second);  // 删除所有2
}
```

### 2.3 性能分析

```cpp
/*
| 操作     | 平均时间 | 最坏时间 |
|---------|---------|---------|
| 插入     | O(1)    | O(n)    |
| 查找     | O(1)    | O(n)    |
| 删除     | O(1)    | O(n)    |
| 计数     | O(k)    | O(n)    |

k = 匹配元素数
最坏情况：所有元素哈希到同一桶
*/
```

### 2.4 实用示例：词频统计

```cpp
#include <string>
#include <sstream>

void word_frequency() {
    std::string text = "the quick brown fox jumps over the lazy dog the fox";
    std::istringstream iss(text);
    std::unordered_multiset<std::string> words;

    std::string word;
    while (iss >> word) words.insert(word);

    // 统计每个词的出现次数
    std::unordered_set<std::string> seen;
    for (const auto& w : words) {
        if (seen.insert(w).second)  // 首次出现
            std::cout << w << ": " << words.count(w) << std::endl;
    }
    // 更好的方案：用unordered_map<string, int>
}
```

## 三、注意事项与常见陷阱

1. **`erase(key)`删除所有匹配值**：`erase(iterator)`只删除单个元素，误用可能导致数据丢失
2. **元素顺序不确定**：不要依赖遍历顺序
3. **如果不需要重复，使用`unordered_set`更高效**
4. **自定义类型需要提供哈希函数和相等比较函数**
5. **rehash时所有迭代器失效**
6. **大多数场景用`unordered_map`做计数更合适**：multiset的主要用途是维护有序重复集合（用multiset）或快速存在检查（用unordered_set）
