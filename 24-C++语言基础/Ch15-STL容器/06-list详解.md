# list详解

## 一、概念说明

`std::list`是双向链表容器（C++标准 §23.3.10），支持任意位置的O(1)插入删除（给定迭代器），但不支持随机访问。每个元素额外存储两个指针（前驱和后继）。list的核心优势在于**引用稳定性**：插入和删除操作不会使其他元素的迭代器、指针和引用失效。

### 1.1 与vector的核心差异

| 特性 | vector | list |
|------|--------|------|
| 内存布局 | 连续 | 非连续 |
| 随机访问 | O(1) | O(n) |
| 中间插入 | O(n) | O(1) |
| 引用稳定性 | 扩容时失效 | 稳定 |
| 缓存友好 | 高 | 低 |
| 每元素开销 | 无 | 2个指针 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <list>
#include <iostream>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};

    // 两端操作（O(1)）
    lst.push_front(0);
    lst.push_back(9);
    lst.emplace_front(-1);   // C++11
    lst.emplace_back(10);    // C++11
    lst.pop_front();
    lst.pop_back();

    // 遍历（不支持下标访问）
    for (const auto& v : lst) std::cout << v << " ";
    std::cout << std::endl;

    // 任意位置插入删除（O(1)，给定迭代器）
    auto it = lst.begin();
    std::advance(it, 2);     // 前进2步（O(n)操作！）
    lst.insert(it, 99);      // 在it前插入99
    lst.erase(it);           // 删除it指向的元素

    // size（C++11后是O(1)）
    std::cout << "size: " << lst.size() << std::endl;
}
```

### 2.2 特有操作

```cpp
void unique_ops() {
    std::list<int> lst1 = {1, 2, 3, 4, 5};
    std::list<int> lst2 = {6, 7, 8};

    // splice：O(1)转移元素（不拷贝）
    lst1.splice(lst1.end(), lst2);       // lst2全部转移到lst1末尾
    // lst1.splice(it, lst2, pos);        // 转移单个元素
    // lst1.splice(it, lst2, first, last); // 转移范围

    // remove：删除所有匹配值
    lst1.remove(3);  // 删除所有值为3的元素

    // remove_if：条件删除
    lst1.remove_if([](int x) { return x > 4; });

    // unique：删除连续重复元素（通常先sort）
    lst1.sort();
    lst1.unique();

    // sort：list特有的O(n log n)排序（不能用std::sort）
    lst1.sort();
    lst1.sort(std::greater<int>());  // 降序

    // merge：合并两个有序list（O(n)）
    std::list<int> a = {1, 3, 5};
    std::list<int> b = {2, 4, 6};
    a.merge(b);  // b变空，a有序

    // reverse：O(n)反转
    a.reverse();
}
```

### 2.3 引用稳定性

```cpp
void stable_references() {
    std::list<int> lst = {1, 2, 3, 4, 5};

    // 获取引用
    int& ref = lst.front();
    auto it = lst.begin();
    std::advance(it, 2);
    int& mid = *it;

    // 插入删除不影响其他元素的引用
    lst.push_back(6);       // ref和mid仍然有效
    lst.erase(lst.begin()); // mid仍然有效
    lst.insert(it, 99);     // mid仍然有效

    std::cout << ref << " " << mid << std::endl;  // 安全

    // 对比vector：
    // std::vector<int> v = {1, 2, 3};
    // int& r = v[0];
    // v.push_back(4);  // 如果扩容，r失效！
}
```

### 2.4 与std::sort的区别

```cpp
#include <algorithm>
#include <list>

void list_sort() {
    std::list<int> lst = {5, 3, 1, 4, 2};

    // std::sort(lst.begin(), lst.end());  // 编译错误！
    // list迭代器是双向的，不是随机访问的

    // 必须用list的成员函数sort
    lst.sort();  // O(n log n)，归并排序实现

    // std::list有自己特殊的算法实现
    // 因为可以O(1)交换节点（不需要移动元素）
}
```

## 三、性能分析

```cpp
/*
| 操作            | 时间复杂度 | 说明                    |
|----------------|-----------|------------------------|
| push_front/back| O(1)      | 分配节点+修改指针        |
| pop_front/back | O(1)      | 释放节点+修改指针        |
| insert         | O(1)      | 需要已定位到位置的迭代器   |
| erase          | O(1)      | 同上                    |
| find           | O(n)      | 必须顺序遍历             |
| sort           | O(n log n)| 成员函数，归并排序        |
| size           | O(1)      | C++11后                   |
| advance        | O(n)      | 非随机访问迭代器          |

实际性能注意：
- 缓存不友好：每个节点可能在内存不同位置
- 小对象时指针开销大：每个元素额外16字节（64位系统）
- 大多数场景vector更快，即使理论复杂度更差
*/
```

## 四、注意事项与常见陷阱

1. **不支持随机访问**：`std::distance(lst.begin(), it)`是O(n)
2. **`sort`是成员函数**：list迭代器不是随机访问的，不能用`std::sort`
3. **`splice`不会使迭代器失效**：只是修改指针，不移动元素
4. **每元素额外开销**：两个指针（前驱+后继），存储小对象时显著
5. **缓存不友好**：现代CPU的缓存架构使vector在大多数场景更快
6. **优势在于稳定性**：引用/指针稳定性和O(1)插入删除
7. **`std::distance`是O(n)**：获取两个迭代器间的距离需要遍历
8. **C++11后`size()`是O(1)**：早期标准中某些实现是O(n)
