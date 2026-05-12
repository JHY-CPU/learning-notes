# unordered_map详解

## 一、概念说明

`std::unordered_map`是基于哈希表实现的键值对容器（C++标准 §23.5.4.1，C++11），提供平均O(1)的查找、插入和删除。键不保证有序，通过哈希函数将键映射到桶（bucket）中。当负载因子超过阈值时自动rehash。

### 1.1 与map的对比

| 特性 | map | unordered_map |
|------|-----|--------------|
| 底层结构 | 红黑树 | 哈希表 |
| 查找复杂度 | O(log n) | 平均O(1)，最坏O(n) |
| 元素顺序 | 按键排序 | 无序 |
| 范围查询 | 支持 | 不支持 |
| 键要求 | 可比较 | 可哈希 |
| 内存开销 | 中等 | 较高（桶数组） |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <unordered_map>
#include <iostream>
#include <string>

int main() {
    std::unordered_map<std::string, int> um;

    // 插入
    um["alice"] = 90;           // operator[]
    um.insert({"bob", 85});     // insert
    um.emplace("charlie", 95);  // emplace
    um.try_emplace("dave", 80); // C++17：键存在不构造

    // 访问
    std::cout << um["alice"] << std::endl;  // 90
    // 注意：um["unknown"]会插入{unknown, 0}！

    // 安全访问
    try {
        std::cout << um.at("alice") << std::endl;
        std::cout << um.at("unknown") << std::endl;  // 抛异常
    } catch (const std::out_of_range& e) {
        std::cerr << e.what() << std::endl;
    }

    // 查找
    auto it = um.find("bob");
    if (it != um.end())
        std::cout << it->first << ": " << it->second << std::endl;

    // 检查存在
    if (um.count("alice") > 0) { /* 存在 */ }
    if (um.contains("alice")) { /* C++20 */ }

    // 删除
    um.erase("charlie");

    // 遍历（顺序不确定）
    for (const auto& [k, v] : um)
        std::cout << k << ": " << v << std::endl;

    std::cout << "size: " << um.size() << std::endl;
}
```

### 2.2 哈希表管理

```cpp
void hash_management() {
    std::unordered_map<std::string, int> um = {{"a", 1}, {"b", 2}, {"c", 3}};

    // 桶信息
    std::cout << "bucket_count: " << um.bucket_count() << std::endl;
    std::cout << "load_factor: " << um.load_factor() << std::endl;
    std::cout << "max_load_factor: " << um.max_load_factor() << std::endl;

    // 设置最大负载因子（默认1.0）
    um.max_load_factor(0.5);  // 更低=更快但更耗内存

    // 预留空间（避免多次rehash）
    um.reserve(1000);  // 预计存储1000个元素

    // 强制rehash
    um.rehash(100);  // 确保至少100个桶

    // 查看特定桶
    size_t bucket = um.bucket("a");
    std::cout << "a在桶" << bucket << std::endl;
    std::cout << "桶" << bucket << "的大小: " << um.bucket_size(bucket) << std::endl;
}
```

### 2.3 自定义哈希

```cpp
struct Point {
    int x, y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

// 方法1：特化std::hash
namespace std {
    template<> struct hash<Point> {
        size_t operator()(const Point& p) const {
            return std::hash<int>{}(p.x) ^ (std::hash<int>{}(p.y) << 1);
        }
    };
}

void custom_hash_demo() {
    std::unordered_map<Point, std::string> point_names;
    point_names[{1, 2}] = "A";
    point_names[{3, 4}] = "B";
}
```

### 2.4 性能优化

```cpp
void performance_tips() {
    // 1. 预分配避免rehash
    std::unordered_map<int, int> um;
    um.reserve(100000);

    // 2. 使用emplace避免拷贝
    um.emplace(42, 100);

    // 3. 使用try_emplace避免不必要的构造
    um.try_emplace(42, 200);  // 键存在则不构造

    // 4. 批量插入后调整负载因子
    // for (...) um.insert(...);
    // um.max_load_factor(1.0);
    // um.rehash(0);  // 收缩到合适大小
}
```

## 三、注意事项与常见陷阱

1. **自定义类型作为键需要特化`std::hash`**：还需要`operator==`
2. **遍历顺序不确定且可能随插入改变**：不要依赖顺序
3. **最坏情况（哈希冲突）退化为O(n)**：选择好的哈希函数很重要
4. **`operator[]`在键不存在时插入默认值**：可能意外修改容器
5. **不支持`lower_bound`/`upper_bound`**：无序容器没有范围查询
6. **哈希冲突多时**：考虑调整桶数量、换用map或改进哈希函数
7. **`reserve`预分配空间**：避免频繁rehash
