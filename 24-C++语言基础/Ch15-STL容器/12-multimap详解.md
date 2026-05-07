# multimap详解

## 一、概念说明

`std::multimap`是允许重复键的有序关联容器。与`map`不同，`multimap`没有`operator[]`，需要用`equal_range`、`lower_bound`和`upper_bound`来查找相同键的所有元素。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <map>
#include <iostream>

int main() {
    std::multimap<std::string, int> grades;

    // 插入（允许重复键）
    grades.insert({"Alice", 90});
    grades.insert({"Alice", 85});  // Alice有两个成绩
    grades.insert({"Bob", 88});

    // 遍历
    for (const auto& [name, score] : grades) {
        std::cout << name << ": " << score << std::endl;
    }
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

    // 计数
    std::cout << "Alice的数量: " << grades.count("Alice") << std::endl;  // 3
}
```

### 2.3 删除特定键

```cpp
void erase_demo() {
    std::multimap<std::string, int> m;
    m.insert({"Alice", 90});
    m.insert({"Alice", 85});
    m.insert({"Bob", 88});

    // 删除所有Alice
    size_t removed = m.erase("Alice");
    std::cout << "删除了 " << removed << " 个元素" << std::endl;
}
```

## 三、注意事项与常见陷阱

- multimap没有`operator[]`（键不唯一）
- 使用`equal_range`查找所有相同键的元素
- `erase(key)`删除所有匹配键的元素
- `find`返回第一个匹配元素的迭代器
- 元素按key排序，相同key的元素相邻
- 如果键唯一，优先使用`map`
