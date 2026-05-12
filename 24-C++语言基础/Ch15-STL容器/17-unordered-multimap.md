# unordered_multimap

## 一、概念说明

`std::unordered_multimap`是允许重复键的无序关联容器（C++标准 §23.5.4.1，C++11），基于哈希表实现。与`unordered_map`不同，它可以存储多个具有相同键的元素。适用场景包括：一对多映射、索引、标签系统等。

### 1.1 与相关容器的对比

| 特性 | unordered_map | unordered_multimap | multimap |
|------|--------------|-------------------|---------|
| 键唯一性 | 唯一 | 可重复 | 可重复 |
| 有序 | 否 | 否 | 是 |
| `operator[]` | 有 | 无 | 无 |
| 查找 | O(1) | O(1) | O(log n) |
| 范围查询 | 不支持 | 不支持 | 支持 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <unordered_map>
#include <iostream>
#include <string>

int main() {
    std::unordered_multimap<std::string, int> umm;

    // 插入（允许重复键）
    umm.insert({"apple", 1});
    umm.insert({"apple", 2});   // apple有两个值
    umm.insert({"apple", 3});   // apple有三个值
    umm.insert({"banana", 4});

    // emplace
    umm.emplace("cherry", 5);

    // 遍历
    for (const auto& [key, value] : umm) {
        std::cout << key << ": " << value << std::endl;
    }

    // 计数
    std::cout << "apple数量: " << umm.count("apple") << std::endl;  // 3

    // 查找（返回第一个匹配）
    auto it = umm.find("apple");
    if (it != umm.end())
        std::cout << "找到: " << it->first << " = " << it->second << std::endl;

    // 大小
    std::cout << "size: " << umm.size() << std::endl;
}
```

### 2.2 查找所有匹配键

```cpp
void find_all() {
    std::unordered_multimap<std::string, int> umm;
    umm.insert({"apple", 1});
    umm.insert({"apple", 2});
    umm.insert({"apple", 3});
    umm.insert({"banana", 4});

    // equal_range查找所有
    auto [begin, end] = umm.equal_range("apple");
    std::cout << "apple的所有值:" << std::endl;
    for (auto it = begin; it != end; ++it) {
        std::cout << "  " << it->second << std::endl;
    }

    // 也可用count + find循环
    auto count = umm.count("apple");
    auto it = umm.find("apple");
    for (size_t i = 0; i < count && it != umm.end(); ++i, ++it) {
        std::cout << it->second << " ";
    }
}
```

### 2.3 删除操作

```cpp
void erase_demo() {
    std::unordered_multimap<std::string, int> umm;
    umm.insert({"apple", 1});
    umm.insert({"apple", 2});
    umm.insert({"apple", 3});
    umm.insert({"banana", 4});

    // 删除所有apple
    size_t removed = umm.erase("apple");
    std::cout << "删除了 " << removed << " 个apple" << std::endl;  // 3

    // 重新插入
    umm.insert({"apple", 1});
    umm.insert({"apple", 2});

    // 删除单个元素
    auto it = umm.find("apple");
    if (it != umm.end()) umm.erase(it);

    // 删除范围
    umm.insert({"apple", 3});
    auto [begin, end] = umm.equal_range("apple");
    umm.erase(begin, end);
}
```

### 2.4 实用示例：倒排索引

```cpp
#include <vector>

void inverted_index() {
    std::unordered_multimap<std::string, int> index;

    // 文档内容
    std::vector<std::string> docs = {
        "apple banana",
        "apple cherry",
        "banana cherry",
        "apple banana cherry"
    };

    // 建立倒排索引
    for (size_t i = 0; i < docs.size(); ++i) {
        std::istringstream iss(docs[i]);
        std::string word;
        while (iss >> word)
            index.insert({word, static_cast<int>(i)});
    }

    // 查询"apple"出现在哪些文档
    auto [begin, end] = index.equal_range("apple");
    std::cout << "apple出现在文档: ";
    for (auto it = begin; it != end; ++it)
        std::cout << it->second << " ";
    std::cout << std::endl;
    // 0 1 3
}
```

## 三、注意事项与常见陷阱

1. **没有`operator[]`**：键不唯一，无法确定返回哪个值
2. **使用`equal_range`查找所有匹配键**：这是核心操作
3. **遍历顺序不确定**：不要依赖插入顺序
4. **如果键唯一，用`unordered_map`更简单**：接口更丰富
5. **rehash时所有迭代器失效**：大量插入时先reserve
6. **`erase(key)`删除所有匹配键**：用迭代器删除只删一个
