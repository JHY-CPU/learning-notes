# unordered_map详解

## 一、概念说明

`std::unordered_map`是基于哈希表实现的键值对容器，提供平均O(1)的查找、插入和删除。键不保证有序，通过哈希函数将键映射到桶中。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <unordered_map>
#include <iostream>

int main() {
    std::unordered_map<std::string, int> um;

    // 插入
    um["alice"] = 90;
    um.insert({"bob", 85});
    um.emplace("charlie", 95);

    // 访问
    std::cout << um["alice"] << std::endl;  // 90

    // 查找
    auto it = um.find("bob");
    if (it != um.end()) {
        std::cout << it->first << ": " << it->second << std::endl;
    }

    // 检查存在
    if (um.count("alice") > 0) { /* 存在 */ }
    if (um.contains("alice")) { /* C++20 */ }

    // 删除
    um.erase("charlie");

    // 遍历（顺序不确定）
    for (const auto& [k, v] : um) {
        std::cout << k << ": " << v << std::endl;
    }
}
```

### 2.2 哈希表信息

```cpp
void hash_info() {
    std::unordered_map<std::string, int> um = {{"a", 1}, {"b", 2}, {"c", 3}};

    // 桶信息
    std::cout << "bucket_count: " << um.bucket_count() << std::endl;
    std::cout << "load_factor: " << um.load_factor() << std::endl;
    std::cout << "max_load_factor: " << um.max_load_factor() << std::endl;

    // 设置最大负载因子
    um.max_load_factor(0.5);  // 更低的负载因子=更快但更耗内存
}
```

## 三、注意事项与常见陷阱

- 自定义类型作为键需要特化`std::hash`
- 遍历顺序不确定且可能随插入改变
- 最坏情况（哈希冲突）退化为O(n)
- `operator[]`在键不存在时插入默认值
- 不支持lower_bound/upper_bound（无序）
- 哈希冲突多时考虑调整桶数量或换用map
