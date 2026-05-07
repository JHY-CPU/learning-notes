# 迭代器Traits

## 一、概念说明

`std::iterator_traits`是迭代器的类型萃取工具，提供迭代器的类型信息（值类型、差类型、类别等）。算法通过traits了解迭代器能力，选择最优实现。

## 二、具体用法

```cpp
#include <iterator>
#include <vector>
#include <type_traits>
#include <iostream>

int main() {
    using Iter = std::vector<int>::iterator;
    using Traits = std::iterator_traits<Iter>;

    // 获取迭代器关联的类型
    static_assert(std::is_same_v<Traits::value_type, int>);
    static_assert(std::is_same_v<Traits::difference_type, std::ptrdiff_t>);
    static_assert(std::is_same_v<Traits::iterator_category, std::random_access_iterator_tag>);

    // 获取值类型（通用写法）
    template <typename Iter>
    typename std::iterator_traits<Iter>::value_type
    get_value(Iter it) {
        return *it;
    }
}
```

### 2.1 自定义迭代器Traits

```cpp
struct MyIterator {
    using iterator_category = std::forward_iterator_tag;
    using value_type = int;
    using difference_type = std::ptrdiff_t;
    using pointer = int*;
    using reference = int&;

    int operator*() const { return 42; }
    MyIterator& operator++() { return *this; }
    bool operator!=(const MyIterator&) const { return false; }
};

// std::iterator_traits会自动提取这些类型
using Traits = std::iterator_traits<MyIterator>;
// Traits::value_type 是 int
```

## 三、注意事项

- 所有标准迭代器都支持`iterator_traits`
- 自定义迭代器需要定义5个typedef或从`iterator`继承
- `iterator_category`决定了算法的实现方式
- C++17后`std::iterator`基类已废弃，直接定义typedef
