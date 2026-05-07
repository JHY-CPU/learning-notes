# unordered_multiset

## 一、概念说明

`std::unordered_multiset`是允许重复元素的无序集合，基于哈希表实现。

## 二、具体用法

```cpp
#include <unordered_set>
#include <iostream>

int main() {
    std::unordered_multiset<int> ums = {5, 3, 1, 4, 2, 1, 1};

    // 插入重复元素
    ums.insert(1);

    // 计数
    std::cout << "count(1): " << ums.count(1) << std::endl;  // 4

    // 查找
    auto it = ums.find(2);
    if (it != ums.end()) std::cout << "找到: " << *it << std::endl;

    // 删除（erase(key)删除所有匹配值）
    ums.erase(1);

    // 删除单个元素
    auto it2 = ums.find(3);
    if (it2 != ums.end()) ums.erase(it2);
}
```

## 三、注意事项

- `erase(key)`删除所有匹配值
- 元素顺序不确定
- 如果不需要重复，使用`unordered_set`
