# list详解

## 一、概念说明

`std::list`是双向链表容器，支持任意位置的O(1)插入删除（给定迭代器），但不支持随机访问。每个元素额外存储两个指针（前驱和后继）。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <list>
#include <iostream>

int main() {
    std::list<int> lst = {3, 1, 4, 1, 5};

    // 两端操作
    lst.push_front(0);
    lst.push_back(9);
    lst.pop_front();
    lst.pop_back();

    // 遍历（不支持下标访问）
    for (const auto& v : lst) std::cout << v << " ";
    std::cout << std::endl;

    // 插入删除（O(1)给定迭代器）
    auto it = lst.begin();
    std::advance(it, 2);
    lst.insert(it, 99);    // 在第3个位置前插入
    lst.erase(it);         // 删除第3个元素
}
```

### 2.2 特有操作

```cpp
void unique_ops() {
    std::list<int> lst1 = {1, 2, 3};
    std::list<int> lst2 = {4, 5, 6};

    // splice：O(1)转移元素
    lst1.splice(lst1.end(), lst2);  // lst2的所有元素转移到lst1末尾
    // lst2现在为空

    // remove：删除所有匹配值
    lst1.remove(3);  // 删除所有值为3的元素

    // remove_if：条件删除
    lst1.remove_if([](int x) { return x > 4; });

    // unique：删除连续重复元素
    lst1.sort();       // 先排序
    lst1.unique();     // 去重

    // sort：list特有的O(n log n)排序
    lst1.sort();
    lst1.sort(std::greater<int>());  // 降序

    // merge：合并两个有序list
    std::list<int> a = {1, 3, 5};
    std::list<int> b = {2, 4, 6};
    a.merge(b);  // b变空，a包含所有元素
}
```

### 2.3 引用稳定性

```cpp
void stable_references() {
    std::list<int> lst = {1, 2, 3, 4, 5};
    int& ref = lst.front();  // 获取引用

    lst.push_back(6);       // ref仍然有效
    lst.erase(lst.begin()); // ref仍然有效
    std::cout << ref << std::endl;  // 安全
}
```

## 三、注意事项与常见陷阱

- list不支持随机访问，`std::distance`是O(n)
- list的sort是成员函数（非`std::sort`），因为list迭代器不是随机访问
- splice操作不会使迭代器失效
- list的每个元素有额外开销（两个指针）
- 缓存不友好，大多数场景下vector更快
- list的优势在于引用/指针稳定性和O(1)插入删除
