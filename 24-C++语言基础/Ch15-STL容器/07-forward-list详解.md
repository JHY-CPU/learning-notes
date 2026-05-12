# forward_list详解

## 一、概念说明

`std::forward_list`是C++11引入的单向链表（C++标准 §23.3.9），比`list`更省内存（每个元素只有一个后继指针），但只能单向遍历。它提供了特殊的插入删除接口（`insert_after`/`erase_after`，操作在指定位置之后），以保持单向链表的高效性。forward_list的设计哲学是**最小化内存开销**，甚至故意省略了`size()`方法。

### 1.1 与list的对比

| 特性 | forward_list | list |
|------|-------------|------|
| 指针数/元素 | 1（后继） | 2（前驱+后继） |
| 遍历方向 | 单向 | 双向 |
| `size()` | 无（O(n)） | 有（O(1)，C++11后） |
| `push_back` | 无 | 有 |
| `operator[]` | 无 | 无 |
| 内存开销 | 最小 | 中等 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <forward_list>
#include <iostream>

int main() {
    std::forward_list<int> flst = {1, 2, 3, 4, 5};

    // 头部操作（O(1)）
    flst.push_front(0);       // 头部插入
    flst.pop_front();          // 头部删除

    // 没有push_back（单向链表，尾部插入需要遍历）
    // 没有size()（为了O(1)操作）
    // 没有operator[]

    // 遍历
    for (const auto& v : flst) std::cout << v << " ";
    std::cout << std::endl;

    // 特殊的insert：在指定位置之后插入
    auto it = flst.begin();
    flst.insert_after(it, 99);  // 在第一个元素之后插入99

    // 特殊的erase：删除指定位置之后的元素
    flst.erase_after(it);       // 删除it之后的元素

    // before_begin：获取首元素前的位置（用于在头部前插入）
    auto before = flst.before_begin();
    flst.insert_after(before, 0);  // 在头部插入0
}
```

### 2.2 特有操作

```cpp
void unique_ops() {
    std::forward_list<int> flst = {5, 3, 1, 4, 2};

    // sort（O(n log n)，归并排序）
    flst.sort();
    flst.sort(std::greater<int>());  // 降序

    // remove / remove_if
    flst.remove(3);  // 删除所有值为3的元素
    flst.remove_if([](int x) { return x > 4; });

    // unique：删除连续重复元素
    flst.sort();
    flst.unique();

    // splice_after：O(1)转移元素
    std::forward_list<int> other = {10, 20, 30};
    flst.splice_after(flst.begin(), other);  // other全部转移到flst

    // 计算size（手动）
    size_t count = std::distance(flst.begin(), flst.end());
    std::cout << "size: " << count << std::endl;
}
```

### 2.3 before_begin用法

```cpp
void before_begin_demo() {
    std::forward_list<int> flst = {2, 3, 4};

    // before_begin()返回首元素前的"哨兵"位置
    // 不能解引用，只能用于insert_after和erase_after
    auto before = flst.before_begin();

    // 在头部插入
    flst.insert_after(before, 1);   // {1, 2, 3, 4}

    // 查找并删除
    auto prev = flst.before_begin();
    for (auto it = flst.begin(); it != flst.end(); ++it) {
        if (*it == 3) {
            flst.erase_after(prev);  // 删除prev之后的元素
            break;
        }
        prev = it;
    }
}
```

### 2.4 与vector性能对比

```cpp
#include <vector>
#include <chrono>

void compare() {
    const int N = 100000;

    // forward_list头部插入
    auto start = std::chrono::high_resolution_clock::now();
    std::forward_list<int> flst;
    for (int i = 0; i < N; ++i) flst.push_front(i);
    auto end = std::chrono::high_resolution_clock::now();
    std::cout << "forward_list push_front: "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    // vector头部插入（O(n)，很慢）
    start = std::chrono::high_resolution_clock::now();
    std::vector<int> vec;
    for (int i = 0; i < N; ++i) vec.insert(vec.begin(), i);
    end = std::chrono::high_resolution_clock::now();
    std::cout << "vector insert(begin): "
              << std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count()
              << "ms" << std::endl;

    // 但vector顺序遍历通常快得多（缓存友好）
}
```

## 三、注意事项与常见陷阱

1. **没有`size()`、`push_back()`、`operator[]`**：这些操作在单向链表中效率低或语义不明确
2. **`insert_after`和`erase_after`**：操作的是指定位置之后的元素，不是之前
3. **`before_begin()`返回的迭代器不能解引用**：只能用于after操作
4. **内存开销比list小**：每个元素一个指针（8字节）vs 两个指针（16字节）
5. **缓存不友好**：频繁遍历时vector通常更快
6. **适用场景**：只需要单向遍历、频繁头部操作、内存受限的场景
7. **手动计算size**：`std::distance(begin, end)`是O(n)
