# unordered_multimap

## 一、概念说明

`std::unordered_multimap`是允许重复键的无序关联容器，基于哈希表实现。与`unordered_map`不同，它可以存储多个具有相同键的元素。

## 二、具体用法

```cpp
#include <unordered_map>
#include <iostream>

int main() {
    std::unordered_multimap<std::string, int> umm;

    // 插入（允许重复键）
    umm.insert({"apple", 1});
    umm.insert({"apple", 2});
    umm.insert({"banana", 3});

    // equal_range查找所有
    auto [begin, end] = umm.equal_range("apple");
    for (auto it = begin; it != end; ++it) {
        std::cout << it->first << ": " << it->second << std::endl;
    }

    // 计数
    std::cout << "apple数量: " << umm.count("apple") << std::endl;  // 2

    // 删除所有apple
    umm.erase("apple");
}
```

## 三、注意事项

- 没有`operator[]`（键不唯一）
- 使用`equal_range`查找所有匹配键
- 遍历顺序不确定
- 如果键唯一，用`unordered_map`更简单
