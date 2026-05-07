# array详解

## 一、概念说明

`std::array`是C++11引入的固定大小数组容器，封装了C风格数组，提供了STL接口（迭代器、size等），同时保持零开销抽象。大小在编译时确定。

## 二、具体用法

### 2.1 基本操作

```cpp
#include <array>
#include <iostream>

int main() {
    // 声明时必须指定大小
    std::array<int, 5> arr = {1, 2, 3, 4, 5};

    // STL接口
    std::cout << "size: " << arr.size() << std::endl;      // 5
    std::cout << "front: " << arr.front() << std::endl;    // 1
    std::cout << "back: " << arr.back() << std::endl;      // 5
    std::cout << "empty: " << arr.empty() << std::endl;    // 0

    // 访问
    arr[0] = 10;
    arr.at(1) = 20;  // 越界检查

    // 迭代器
    for (auto it = arr.begin(); it != arr.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;
}
```

### 2.2 与C数组互操作

```cpp
void c_compatibility() {
    std::array<int, 3> arr = {1, 2, 3};

    // 获取底层数据指针
    int* ptr = arr.data();

    // 传给C接口
    // void c_function(int* data, size_t len);
    // c_function(arr.data(), arr.size());

    // 可以直接初始化C风格数组
    int c_arr[3] = {1, 2, 3};
    // std::array<int, 3> std_arr = c_arr;  // C++17起支持
}
```

### 2.3 编译期操作

```cpp
#include <tuple>

void compile_time_ops() {
    // constexpr支持
    constexpr std::array<int, 3> arr = {1, 2, 3};
    constexpr int first = arr[0];  // 编译期访问

    // 结构化绑定（C++17）
    auto [a, b, c] = arr;
    std::cout << a << " " << b << " " << c << std::endl;

    // tuple接口
    auto elem = std::get<0>(arr);  // 编译期索引
}
```

## 三、注意事项与常见陷阱

- 大小是类型的一部分，`array<int,3>`和`array<int,4>`是不同类型
- 没有`push_back`等动态操作（大小固定）
- `data()`返回连续内存指针，可与C接口互操作
- 零开销：和C数组一样的性能
- 支持所有STL容器接口（迭代器、算法等）
- 可以作为函数返回值（C数组不行）
