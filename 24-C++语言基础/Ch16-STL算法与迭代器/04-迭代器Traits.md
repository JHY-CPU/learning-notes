# 迭代器Traits

## 一、概念说明

`std::iterator_traits`是迭代器的类型萃取工具（C++标准 §24.4.1），提供迭代器的类型信息（值类型、差类型、类别等）。算法通过traits了解迭代器能力，选择最优实现（如`std::advance`对随机访问迭代器用O(1)实现，对双向迭代器用O(n)实现）。

### 1.1 traits提供的信息

| 类型别名 | 说明 |
|---------|------|
| `value_type` | 迭代器指向的值类型 |
| `difference_type` | 两个迭代器间的距离类型 |
| `pointer` | 指向值的指针类型 |
| `reference` | 值的引用类型 |
| `iterator_category` | 迭代器类别标签 |

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iterator>
#include <vector>
#include <list>
#include <type_traits>
#include <iostream>

int main() {
    using VecIter = std::vector<int>::iterator;
    using Traits = std::iterator_traits<VecIter>;

    // 获取类型信息
    static_assert(std::is_same_v<Traits::value_type, int>);
    static_assert(std::is_same_v<Traits::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Traits::iterator_category,
                                  std::random_access_iterator_tag>);

    // 检查list迭代器
    using ListIter = std::list<int>::iterator;
    using ListTraits = std::iterator_traits<ListIter>;
    static_assert(std::is_same_v<ListTraits::iterator_category,
                                  std::bidirectional_iterator_tag>);

    // 检查指针（指针也是迭代器）
    static_assert(std::is_same_v<std::iterator_traits<int*>::iterator_category,
                                  std::contiguous_iterator_tag>);  // C++20
}
```

### 2.2 通用函数模板

```cpp
// 获取迭代器的值类型（通用写法）
template <typename Iter>
typename std::iterator_traits<Iter>::value_type
get_value(Iter it) {
    return *it;
}

// 检查迭代器类别
template <typename Iter>
constexpr bool is_random_access_v = std::is_same_v<
    typename std::iterator_traits<Iter>::iterator_category,
    std::random_access_iterator_tag
>;

// 根据类别选择实现
template <typename Iter>
void advance_impl(Iter& it, typename std::iterator_traits<Iter>::difference_type n) {
    if constexpr (is_random_access_v<Iter>) {
        it += n;  // O(1)
    } else {
        // 双向迭代器的O(n)实现
        if (n > 0) while (n--) ++it;
        else while (n++) --it;
    }
}
```

### 2.3 自定义迭代器Traits

```cpp
struct MyIterator {
    // 必须定义这5个类型别名
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    int data = 0;
    int operator*() const { return data; }
    MyIterator& operator++() { ++data; return *this; }
    MyIterator operator++(int) { auto tmp = *this; ++data; return tmp; }
    bool operator==(const MyIterator& other) const { return data == other.data; }
    bool operator!=(const MyIterator& other) const { return data != other.data; }
};

void custom_iterator_demo() {
    // iterator_traits自动提取这些类型
    using Traits = std::iterator_traits<MyIterator>;
    static_assert(std::is_same_v<Traits::value_type, int>);
    static_assert(std::is_same_v<Traits::iterator_category,
                                  std::forward_iterator_tag>);
}
```

### 2.4 C++20 concepts替代

```cpp
#include <iterator>
#include <concepts>

// C++20用concepts替代部分traits检查
template<std::random_access_iterator Iter>
void fast_sort(Iter first, Iter last) {
    std::sort(first, last);  // 只接受随机访问迭代器
}

template<std::input_iterator Iter>
void process(Iter first, Iter last) {
    // 接受任何输入迭代器
    for (; first != last; ++first)
        std::cout << *first << " ";
}
```

## 三、注意事项

1. **所有标准迭代器都支持`iterator_traits`**：包括原生指针
2. **自定义迭代器需要定义5个类型别名**：`value_type`、`difference_type`、`pointer`、`reference`、`iterator_category`
3. **`iterator_category`决定了算法的实现方式**
4. **C++17后`std::iterator`基类已废弃**：直接定义typedef
5. **C++20用concepts做编译期约束**：更清晰的错误信息
6. **指针的traits是特化的**：自动正确识别为连续迭代器
