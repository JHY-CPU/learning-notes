# pair与tuple

## 一、概念说明

`std::pair`存储两个值，`std::tuple`存储任意数量的值。它们常用于从函数返回多个值，以及作为map的元素类型。

## 二、具体用法

### 2.1 std::pair

```cpp
#include <utility>
#include <iostream>

int main() {
    // 创建
    std::pair<std::string, int> p1("age", 25);
    auto p2 = std::make_pair("score", 90);

    // 访问
    std::cout << p1.first << ": " << p1.second << std::endl;  // age: 25

    // 结构化绑定（C++17）
    auto [key, value] = p2;
    std::cout << key << ": " << value << std::endl;  // score: 90

    // 比较
    std::pair<int, int> a{1, 2}, b{1, 3};
    std::cout << (a < b) << std::endl;  // true（先比first再比second）
}
```

### 2.2 std::tuple

```cpp
#include <tuple>
#include <string>

void tuple_demo() {
    // 创建
    auto t1 = std::make_tuple(1, 3.14, "hello");
    std::tuple<int, double, std::string> t2(42, 2.72, "world");

    // 访问
    std::cout << std::get<0>(t1) << std::endl;  // 1
    std::cout << std::get<2>(t1) << std::endl;  // hello

    // 按类型访问（同类型不能重复）
    std::tuple<int, double, char> t3(1, 2.0, 'a');
    std::cout << std::get<double>(t3) << std::endl;  // 2.0

    // 结构化绑定
    auto [a, b, c] = t1;
    std::cout << a << " " << b << " " << c << std::endl;

    // tie解包
    int x; double y; std::string z;
    std::tie(x, y, z) = t2;

    // tuple_size
    std::cout << "size: " << std::tuple_size_v<decltype(t1)> << std::endl;  // 3
}
```

## 三、注意事项

- pair用于固定两个值，tuple用于多个值
- `std::get<I>`索引必须是编译期常量
- 结构化绑定是C++17最简洁的方式
- `std::tie`可以忽略某些值：`std::tie(a, std::ignore, c) = tuple`
- map的`value_type`是`pair<const Key, Value>`
