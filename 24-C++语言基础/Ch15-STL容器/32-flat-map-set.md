# flat_map与flat_set（C++23）

## 一、概念说明

`std::flat_map`和`std::flat_set`是C++23引入的有序平坦容器（C++标准 §23.6.8），底层使用排序的连续存储（默认vector），提供与map/set相同的接口但更好的**缓存性能**。它们适用于读多写少的场景，查找性能优于红黑树实现的map/set。

### 1.1 设计动机

```
传统map/set的问题：
- 红黑树节点分散在堆上，缓存不友好
- 每个节点有额外指针开销

flat_map/flat_set的优势：
- 连续内存，缓存友好
- 无指针开销，内存紧凑
- 查找O(log n)但实际更快（CPU预取）

劣势：
- 插入删除O(n)（需要移动元素）
- 迭代器稳定性差
```

## 二、具体用法

### 2.1 基本操作

```cpp
// C++23 <flat_map> <flat_set>
#include <flat_map>
#include <flat_set>
#include <iostream>
#include <string>

int main() {
    // flat_map
    std::flat_map<std::string, int> fm = {{"alice", 90}, {"bob", 85}};
    fm["charlie"] = 95;
    fm.insert({"dave", 80});

    for (const auto& [k, v] : fm)
        std::cout << k << ": " << v << std::endl;

    // flat_set
    std::flat_set<int> fs = {5, 3, 1, 4, 2};
    fs.insert(6);

    for (int v : fs) std::cout << v << " ";  // 1 2 3 4 5 6
}
```

### 2.2 与map/set对比

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
| 范围查询      | 支持             | 支持                 |
*/
```

### 2.3 批量插入优化

```cpp
void batch_insert() {
    // 批量插入：先收集再排序更高效
    std::vector<std::pair<std::string, int>> data = {
        {"z", 1}, {"a", 2}, {"m", 3}, {"b", 4}
    };

    // 方法1：直接插入（每次O(n)）
    std::flat_map<std::string, int> fm1;
    for (const auto& [k, v] : data) fm1.insert({k, v});

    // 方法2：排序后批量构造（更高效）
    std::sort(data.begin(), data.end());
    std::flat_map<std::string, int> fm2(data.begin(), data.end());
}
```

### 2.4 自定义比较器

```cpp
void custom_compare() {
    // 降序
    std::flat_set<int, std::greater<int>> desc = {1, 2, 3};
    for (int v : desc) std::cout << v << " ";  // 3 2 1
}
```

## 三、适用场景分析

```cpp
/*
| 场景                          | 推荐容器          |
|------------------------------|------------------|
| 读多写少                      | flat_map/set     |
| 频繁插入删除                  | map/set          |
| 需要迭代器稳定                 | map/set          |
| 内存受限                      | flat_map/set     |
| 批量加载后只查询               | flat_map/set     |
| 需要频繁随机插入               | map/set          |
*/
```

## 四、注意事项与常见陷阱

1. **适用于读多写少的场景**：频繁插入删除时map/set更快
2. **批量插入时先收集再排序更高效**：避免每次O(n)插入
3. **C++23特性**：需要新编译器支持（GCC 14+, Clang 18+, MSVC 19.38+）
4. **迭代器稳定性差**：插入删除使所有迭代器失效
5. **底层容器可配置**：默认vector，可用deque等
6. **`extract`和`replace`可高效移动底层容器**（C++23）
