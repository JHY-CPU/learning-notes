# unordered_set详解

## 一、概念说明

`std::unordered_set`是基于哈希表的无序唯一元素集合（C++标准 §23.5.6.1，C++11），提供平均O(1)的查找、插入和删除。相比`set`的O(log n)，在数据量大时性能优势明显，但不保证元素顺序且不支持范围查询。

### 1.1 与set的对比

| 特性 | set | unordered_set |
|------|-----|--------------|
| 底层结构 | 红黑树 | 哈希表 |
| 查找 | O(log n) | 平均O(1) |
| 排序 | 自动排序 | 无序 |
| 范围查询 | 支持 | 不支持 |
| 迭代器 | 双向 | 前向 |
| 内存 | 中等 | 较高 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <unordered_set>
#include <iostream>

int main() {
    std::unordered_set<int> us = {5, 3, 1, 4, 2};

    // 插入
    auto [it, inserted] = us.insert(6);
    std::cout << "插入6: " << (inserted ? "成功" : "失败") << std::endl;

    // 重复插入被忽略
    auto [it2, ins2] = us.insert(3);
    std::cout << "插入3: " << (ins2 ? "成功" : "失败") << std::endl;  // 失败

    // 查找
    if (us.find(3) != us.end())
        std::cout << "找到3" << std::endl;

    // C++20 contains
    if (us.contains(3)) std::cout << "包含3" << std::endl;

    // 删除
    us.erase(4);

    // 遍历（无序）
    for (const auto& v : us) std::cout << v << " ";
    std::cout << std::endl;

    // 容量信息
    std::cout << "size: " << us.size() << std::endl;
    std::cout << "buckets: " << us.bucket_count() << std::endl;
    std::cout << "load_factor: " << us.load_factor() << std::endl;
}
```

### 2.2 集合运算

```cpp
#include <algorithm>

void set_operations() {
    std::unordered_set<int> a = {1, 2, 3, 4};
    std::unordered_set<int> b = {3, 4, 5, 6};

    // 手动实现集合运算
    // 并集
    std::unordered_set<int> union_set = a;
    union_set.insert(b.begin(), b.end());
    // {1, 2, 3, 4, 5, 6}

    // 交集
    std::unordered_set<int> intersect;
    for (const auto& x : a) {
        if (b.count(x)) intersect.insert(x);
    }
    // {3, 4}

    // 差集 A-B
    std::unordered_set<int> diff;
    for (const auto& x : a) {
        if (!b.count(x)) diff.insert(x);
    }
    // {1, 2}

    // 子集判断
    bool is_subset = true;
    for (const auto& x : a) {
        if (!b.count(x)) { is_subset = false; break; }
    }
}
```

### 2.3 实用示例：去重

```cpp
#include <vector>

void deduplicate() {
    std::vector<int> data = {1, 2, 3, 2, 1, 4, 3, 5};

    // 去重
    std::unordered_set<int> seen;
    std::vector<int> unique;
    for (int x : data) {
        if (seen.insert(x).second)  // 插入成功说明是新元素
            unique.push_back(x);
    }
    // unique: {1, 2, 3, 4, 5}（保持首次出现顺序）

    // 或者直接用set
    std::unordered_set<int> unique_set(data.begin(), data.end());
}
```

### 2.4 自定义类型

```cpp
struct Point {
    int x, y;
    bool operator==(const Point& other) const {
        return x == other.x && y == other.y;
    }
};

namespace std {
    template<> struct hash<Point> {
        size_t operator()(const Point& p) const {
            return std::hash<int>{}(p.x) ^ (std::hash<int>{}(p.y) << 1);
        }
    };
}

void custom_type_demo() {
    std::unordered_set<Point> points;
    points.insert({1, 2});
    points.insert({3, 4});
}
```

## 三、注意事项与常见陷阱

1. **元素顺序不保证**：不要依赖遍历顺序
2. **自定义类型需要特化`std::hash`**：还需要`operator==`
3. **不支持范围查询**：无序，没有`lower_bound`等
4. **比set更快（平均O(1) vs O(log n)）**：但最坏O(n)
5. **遍历性能取决于桶的数量和负载因子**
6. **`reserve`预分配空间**：避免频繁rehash
7. **大多数场景优先使用unordered_set**：除非需要有序遍历
