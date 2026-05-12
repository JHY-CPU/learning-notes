# map详解

## 一、概念说明

`std::map`是基于红黑树实现的有序键值对容器（C++标准 §23.4.4.1），键唯一，支持O(log n)的查找、插入和删除。元素按键的顺序自动排列（默认`operator<`）。map的键是const的（`pair<const Key, Value>`），不能修改。

### 1.1 核心特性

| 特性 | 说明 |
|------|------|
| 底层结构 | 红黑树 |
| 键唯一性 | 是 |
| 查找复杂度 | O(log n) |
| 元素顺序 | 按键排序 |
| 迭代器类别 | 双向 |
| 引用稳定性 | 插入删除不失效 |

## 二、具体用法

### 2.1 初始化与基本操作

```cpp
#include <map>
#include <iostream>
#include <string>

int main() {
    // 初始化方式
    std::map<std::string, int> m1;                              // 空
    std::map<std::string, int> m2 = {{"a", 1}, {"b", 2}};     // 初始化列表
    std::map<std::string, int> m3(m2);                         // 拷贝
    std::map<std::string, int> m4(std::move(m2));              // 移动

    // 插入
    m1["alice"] = 90;           // operator[]：不存在则插入，存在则更新
    m1.insert({"bob", 85});     // insert：存在则不插入
    m1.emplace("charlie", 95);  // emplace：原地构造
    m1.insert_or_assign("alice", 95);  // C++17：存在则更新

    // 访问
    std::cout << m1["alice"] << std::endl;  // 95

    // 安全访问（不插入）
    try {
        std::cout << m1.at("alice") << std::endl;   // 95
        std::cout << m1.at("unknown") << std::endl;  // 抛出out_of_range
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
    }

    // 查找
    auto it = m1.find("bob");
    if (it != m1.end())
        std::cout << it->first << ": " << it->second << std::endl;

    // 检查存在
    if (m1.count("alice") > 0) { /* 存在 */ }
    if (m1.contains("alice")) { /* C++20 */ }

    // 删除
    m1.erase("charlie");       // 按键删除
    m1.erase(m1.begin());      // 按迭代器删除

    // 大小
    std::cout << "size: " << m1.size() << std::endl;
    std::cout << "empty: " << m1.empty() << std::endl;
}
```

### 2.2 遍历

```cpp
void iterate_map() {
    std::map<int, std::string> m = {{1, "one"}, {2, "two"}, {3, "three"}};

    // 结构化绑定（C++17，推荐）
    for (const auto& [key, value] : m) {
        std::cout << key << ": " << value << std::endl;
    }
    // 输出按key排序：1: one, 2: two, 3: three

    // 迭代器遍历
    for (auto it = m.begin(); it != m.end(); ++it)
        std::cout << it->first << " -> " << it->second << std::endl;

    // 反向遍历
    for (auto rit = m.rbegin(); rit != m.rend(); ++rit)
        std::cout << rit->first << ": " << rit->second << std::endl;
}
```

### 2.3 范围查询

```cpp
void range_query() {
    std::map<int, std::string> m = {{1, "a"}, {3, "c"}, {5, "e"}, {7, "g"}};

    // lower_bound: >= key 的第一个元素
    auto lo = m.lower_bound(3);   // 指向{3, "c"}

    // upper_bound: > key 的第一个元素
    auto hi = m.upper_bound(5);   // 指向{7, "g"}

    // equal_range: [lower_bound, upper_bound)
    auto [begin, end] = m.equal_range(3);

    // 范围遍历 [2, 6]
    auto range_lo = m.lower_bound(2);  // {3, "c"}
    auto range_hi = m.upper_bound(6);  // {7, "g"}
    for (auto it = range_lo; it != range_hi; ++it)
        std::cout << it->first << ": " << it->second << std::endl;
    // 输出: 3: c, 5: e
}
```

### 2.4 自定义比较器

```cpp
// 降序排列
std::map<int, std::string, std::greater<int>> desc = {{1, "one"}, {2, "two"}};

// lambda比较器（C++20）
// auto cmp = [](int a, int b) { return a > b; };
// std::map<int, std::string, decltype(cmp)> custom(cmp);

// 自定义结构体
struct CaseInsensitiveLess {
    bool operator()(const std::string& a, const std::string& b) const {
        return std::lexicographical_compare(
            a.begin(), a.end(), b.begin(), b.end(),
            [](char x, char y) { return std::tolower(x) < std::tolower(y); }
        );
    }
};
std::map<std::string, int, CaseInsensitiveLess> ci_map;
```

## 三、insert vs operator[] vs emplace

```cpp
/*
| 操作               | 键存在时           | 键不存在时          |
|-------------------|-------------------|-------------------|
| operator[]        | 返回引用，可修改    | 插入默认值并返回引用  |
| insert            | 不修改            | 插入新元素           |
| insert_or_assign  | 更新值（C++17）    | 插入新元素           |
| emplace           | 不修改            | 原地构造             |
| try_emplace       | 不修改（C++17）    | 原地构造             |
*/
```

## 四、注意事项与常见陷阱

1. **`operator[]`在键不存在时插入默认值**：可能意外修改map，使用`find`或`at`更安全
2. **`insert`不覆盖已有值**：返回`pair<iterator, bool>`，第二个元素表示是否插入成功
3. **map的键是const的**：`pair<const Key, Value>`，不能修改键
4. **遍历顺序按键排序**：不是插入顺序
5. **`lower_bound`/`upper_bound`/`equal_range`用于范围查找**：有序容器特有
6. **map不适合需要频繁修改键的场景**：需要erase+insert
7. **`try_emplace`（C++17）**：键存在时不构造参数，更高效
