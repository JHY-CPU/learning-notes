# flat_map与flat_set（C++23）

## 一、概念说明

`std::flat_map`和`std::flat_set`是C++23引入的有序平坦容器，底层使用排序的连续存储（vector），提供与map/set相同的接口但更好的缓存性能。

## 二、具体用法

```cpp
// C++23 <flat_map> <flat_set>
#include <flat_map>
#include <flat_set>

void demo() {
    // flat_map：基于排序数组
    std::flat_map<std::string, int> fm = {{"alice", 90}, {"bob", 85}};
    fm["charlie"] = 95;

    // flat_set
    std::flat_set<int> fs = {5, 3, 1, 4, 2};
    fs.insert(6);

    // 优势：连续内存，缓存友好
    // 查找O(log n)，但实际更快（缓存命中率高）
    // 劣势：插入删除O(n)（需要移动元素）
}
```

## 三、与map/set对比

```cpp
/*
| 特性          | map/set          | flat_map/flat_set    |
|--------------|------------------|----------------------|
| 底层结构      | 红黑树           | 排序数组(vector)      |
| 查找          | O(log n)         | O(log n)             |
| 插入          | O(log n)         | O(n)                 |
| 删除          | O(log n)         | O(n)                 |
| 缓存友好      | 差               | 好                   |
| 内存开销      | 高(节点指针)     | 低                   |
| 迭代器稳定    | 是               | 否                   |
*/
```

## 三、注意事项

- 适用于读多写少的场景
- 批量插入时先收集再排序更高效
- C++23特性，需要新编译器支持
