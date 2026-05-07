# unordered_set详解

## 一、概念说明

`std::unordered_set`是基于哈希表的无序唯一元素集合，提供平均O(1)的查找、插入和删除。

## 二、具体用法

```cpp
#include <unordered_set>
#include <iostream>

int main() {
    std::unordered_set<int> us = {5, 3, 1, 4, 2};

    // 插入
    auto [it, inserted] = us.insert(6);
    std::cout << "插入6: " << (inserted ? "成功" : "失败") << std::endl;

    // 查找
    if (us.find(3) != us.end()) {
        std::cout << "找到3" << std::endl;
    }

    // 删除
    us.erase(4);

    // 遍历（无序）
    for (const auto& v : us) std::cout << v << " ";
    std::cout << std::endl;

    // 容量信息
    std::cout << "size: " << us.size() << std::endl;
    std::cout << "buckets: " << us.bucket_count() << std::endl;
}
```

## 三、注意事项

- 元素顺序不保证，且可能随时间变化
- 自定义类型需要特化`std::hash`
- 不支持范围查询（无序）
- 比set更快（平均O(1) vs O(log n)）
- 遍历性能取决于桶的数量和负载因子
