# multimap详解

## 一、概念说明

`std::multimap`是允许重复键的有序关联容器（C++标准 §23.4.4.1）。与`map`不同，`multimap`没有`operator[]`（因为键不唯一，无法确定返回哪个值），需要用`equal_range`、`lower_bound`和`upper_bound`来查找相同键的所有元素。底层同样基于红黑树。

### 1.1 与map的核心差异

| 特性 | map | multimap |
|------|-----|---------|
| 键唯一性 | 唯一 | 可重复 |
| `operator[]` | 有 | 无 |
| `insert_or_assign` | 有 | 无 |
| `at()` | 有 | 无 |
| `find`返回 | 第一个匹配 | 第一个匹配 |
| `erase(key)` | 删除1个 | 删除所有匹配 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <map>
#include <iostream>
#include <string>

int main() {
    std::multimap<std::string, int> grades;

    // 插入（允许重复键）
    grades.insert({"Alice", 90});
    grades.insert({"Alice", 85});   // Alice有两个成绩
    grades.insert({"Alice", 92});   // Alice有三个成绩
    grades.insert({"Bob", 88});

    // emplace
    grades.emplace("Charlie", 78);

    // 遍历（按键排序，同键相邻）
    for (const auto& [name, score] : grades) {
        std::cout << name << ": " << score << std::endl;
    }
    // Alice: 85, Alice: 90, Alice: 92, Bob: 88, Charlie: 78

    // 计数
    std::cout << "Alice的数量: " << grades.count("Alice") << std::endl;  // 3

    // find返回第一个匹配
    auto it = grades.find("Alice");
    if (it != grades.end())
        std::cout << "第一个Alice: " << it->second << std::endl;  // 85
}
```

### 2.2 equal_range查找

```cpp
void find_all() {
    std::multimap<std::string, int> grades;
    grades.insert({"Alice", 90});
    grades.insert({"Alice", 85});
    grades.insert({"Alice", 92});
    grades.insert({"Bob", 88});

    // equal_range返回[first, last)范围
    auto [begin, end] = grades.equal_range("Alice");
    std::cout << "Alice的所有成绩:" << std::endl;
    for (auto it = begin; it != end; ++it) {
        std::cout << "  " << it->second << std::endl;
    }
    // 85, 90, 92（按key排序）

    // lower_bound / upper_bound等价
    auto lo = grades.lower_bound("Alice");  // 第一个>= "Alice"
    auto hi = grades.upper_bound("Alice");  // 第一个> "Alice"
    // [lo, hi) 等价于 equal_range("Alice")
}
```

### 2.3 删除操作

```cpp
void erase_demo() {
    std::multimap<std::string, int> m;
    m.insert({"Alice", 90});
    m.insert({"Alice", 85});
    m.insert({"Alice", 92});
    m.insert({"Bob", 88});

    // 删除所有Alice
    size_t removed = m.erase("Alice");
    std::cout << "删除了 " << removed << " 个元素" << std::endl;  // 3

    // 删除单个元素（通过迭代器）
    m.insert({"Alice", 90});
    m.insert({"Alice", 85});
    auto it = m.find("Alice");
    if (it != m.end()) m.erase(it);  // 只删除一个

    // 删除范围
    m.insert({"Alice", 92});
    auto [begin, end] = m.equal_range("Alice");
    m.erase(begin, end);  // 删除所有Alice
}
```

### 2.4 提取与合并（C++17）

```cpp
void extract_merge() {
    std::multimap<std::string, int> m1 = {{"a", 1}, {"b", 2}};
    std::multimap<std::string, int> m2 = {{"a", 3}, {"c", 4}};

    // merge：合并两个multimap
    m1.merge(m2);
    // m1: {a:1, a:3, b:2, c:4}
    // m2: 空（或部分元素，如果有比较器冲突）

    // extract：提取节点（C++17，无拷贝）
    auto node = m1.extract("a");
    if (!node.empty()) {
        node.mapped() = 100;  // 修改值
        m1.insert(std::move(node));
    }
}
```

## 三、实用示例：单词计数

```cpp
#include <map>
#include <string>
#include <iostream>
#include <sstream>

void word_count() {
    std::string text = "the quick brown fox jumps over the lazy dog the fox";
    std::istringstream iss(text);
    std::multimap<std::string, int> word_positions;

    std::string word;
    int pos = 0;
    while (iss >> word) {
        word_positions.insert({word, pos++});
    }

    // 打印每个单词的所有出现位置
    std::string target = "the";
    auto [begin, end] = word_positions.equal_range(target);
    std::cout << target << " 出现位置: ";
    for (auto it = begin; it != end; ++it)
        std::cout << it->second << " ";
    std::cout << std::endl;
}
```

## 四、注意事项与常见陷阱

1. **multimap没有`operator[]`**：键不唯一，无法确定返回哪个值
2. **使用`equal_range`查找所有相同键的元素**：这是multimap的核心操作
3. **`erase(key)`删除所有匹配键的元素**：用迭代器删除只删一个
4. **`find`返回第一个匹配元素的迭代器**：不保证是最早插入的
5. **元素按key排序**：相同key的元素相邻但不保证插入顺序
6. **如果键唯一，优先使用`map`**：接口更丰富
7. **C++17的`extract`可无拷贝移动节点**：跨容器转移时高效
