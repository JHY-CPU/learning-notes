# map详解

## 一、概念说明

`std::map`是基于红黑树实现的有序键值对容器，键唯一，支持O(log n)的查找、插入和删除。元素按键的顺序自动排列。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <map>
#include <iostream>

int main() {
    std::map<std::string, int> scores;

    // 插入
    scores["Alice"] = 90;           // operator[]：不存在则插入
    scores.insert({"Bob", 85});     // insert：存在则不插入
    scores.emplace("Charlie", 95);  // emplace：原地构造

    // 访问
    std::cout << scores["Alice"] << std::endl;  // 90
    // 注意：scores["Unknown"]会插入{Unknown, 0}！

    // 安全访问
    try {
        std::cout << scores.at("Alice") << std::endl;   // 90
        std::cout << scores.at("Unknown") << std::endl; // 抛出out_of_range
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
    }

    // 查找
    auto it = scores.find("Bob");
    if (it != scores.end()) {
        std::cout << it->first << ": " << it->second << std::endl;
    }

    // 检查存在
    if (scores.count("Alice") > 0) { /* 存在 */ }

    // 删除
    scores.erase("Charlie");
}
```

### 2.2 遍历

```cpp
void iterate_map() {
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};

    // 按键顺序遍历（自动排序）
    for (const auto& [key, value] : m) {  // C++17结构化绑定
        std::cout << key << ": " << value << std::endl;
    }
    // 1: one
    // 2: two
    // 3: three

    // 迭代器遍历
    for (auto it = m.begin(); it != m.end(); ++it) {
        std::cout << it->first << " -> " << it->second << std::endl;
    }
}
```

### 2.3 自定义比较器

```cpp
// 降序排列
std::map<int, std::string, std::greater<int>> desc_map = {{1, "one"}, {2, "two"}};

// lambda比较器（C++20）
// auto cmp = [](int a, int b) { return a > b; };
// std::map<int, std::string, decltype(cmp)> custom_map(cmp);
```

## 三、注意事项与常见陷阱

- `operator[]`在键不存在时插入默认值，可能意外修改map
- `insert`不覆盖已有值，`operator[]`或`insert_or_assign`(C++17)会覆盖
- map的键是const的（`pair<const Key, Value>`）
- 遍历顺序按键排序（不是插入顺序）
- lower_bound/upper_bound/equal_range用于范围查找
- map不适合需要频繁修改键的场景
