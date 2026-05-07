# forward_list详解

## 一、概念说明

`std::forward_list`是C++11引入的单向链表，比list更省内存（每个元素只有一个指针），但只能单向遍历。它提供了特殊的插入删除接口（操作在指定位置之后）。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <forward_list>
#include <iostream>

int main() {
    std::forward_list<int> flst = {1, 2, 3, 4, 5};

    // 没有size()方法（为了O(1)操作）
    // 没有push_back（单向链表，无法O(1)尾部插入）
    // 没有operator[]

    flst.push_front(0);           // O(1)头部插入
    flst.pop_front();             // O(1)头部删除

    // 遍历
    for (const auto& v : flst) std::cout << v << " ";
    std::cout << std::endl;

    // 特殊的insert：在指定位置之后插入
    auto it = flst.begin();
    flst.insert_after(it, 99);  // 在第一个元素之后插入99

    // 特殊的erase：删除指定位置之后的元素
    flst.erase_after(it);       // 删除it之后的元素
}
```

### 2.2 特有操作

```cpp
void unique_ops() {
    std::forward_list<int> flst = {5, 3, 1, 4, 2};

    // sort
    flst.sort();  // O(n log n)

    // remove / remove_if
    flst.remove(3);
    flst.remove_if([](int x) { return x > 4; });

    // unique
    flst.unique();  // 删除连续重复

    // splice_after：转移元素
    std::forward_list<int> other = {10, 20};
    flst.splice_after(flst.begin(), other);

    // before_begin：获取首元素前的位置（用于在头部前插入）
    auto before = flst.before_begin();
    flst.insert_after(before, 0);
}
```

## 三、注意事项与常见陷阱

- 没有`size()`、`push_back()`、`operator[]`
- `insert_after`和`erase_after`操作的是指定位置之后的元素
- `before_begin()`返回的迭代器不能解引用
- 内存开销比list小（每个元素一个指针vs两个）
- 适用于只需要单向遍历的场景
- 缓存不友好，频繁遍历时vector通常更快
