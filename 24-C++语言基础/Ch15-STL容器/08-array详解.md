# array详解

## 一、概念说明

`std::array`是C++11引入的固定大小数组容器（C++标准 §23.3.2），封装了C风格数组，提供了完整的STL容器接口（迭代器、size等），同时保持**零开销抽象**——编译后与C数组完全相同。大小在编译时确定，是类型的一部分（`array<int,3>`和`array<int,4>`是不同类型）。

### 1.1 核心优势

| 特性 | C数组 | std::array |
|------|-------|-----------|
| 大小信息 | 隐式 | 编译期已知 |
| STL接口 | 无 | 完整 |
| 可拷贝 | 不可 | 可以 |
| 越界检查 | 无 | `at()`提供 |
| 作为返回值 | 不可 | 可以 |
| 性能 | 基准 | 零开销 |

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
    std::cout << "max_size: " << arr.max_size() << std::endl;  // 5
    std::cout << "front: " << arr.front() << std::endl;    // 1
    std::cout << "back: " << arr.back() << std::endl;      // 5
    std::cout << "empty: " << arr.empty() << std::endl;    // 0

    // 访问元素
    arr[0] = 10;
    arr.at(1) = 20;        // 越界检查，抛出out_of_range

    // 迭代器
    for (auto it = arr.begin(); it != arr.end(); ++it)
        std::cout << *it << " ";

    // 范围for
    for (const auto& v : arr)
        std::cout << v << " ";

    // 填充
    arr.fill(0);  // 所有元素设为0
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

    // C数组初始化array（C++17起支持）
    int c_arr[3] = {1, 2, 3};
    // std::array<int, 3> std_arr = c_arr;  // C++17 OK

    // 拷贝（C数组不能直接拷贝）
    std::array<int, 3> arr2 = arr;  // OK
}
```

### 2.3 编译期操作

```cpp
#include <tuple>

constexpr int compile_time_demo() {
    // constexpr支持
    constexpr std::array<int, 5> arr = {1, 2, 3, 4, 5};
    constexpr int first = arr[0];      // 编译期访问
    constexpr int sz = arr.size();     // 编译期获取大小

    // 编译期计算
    int sum = 0;
    for (size_t i = 0; i < arr.size(); ++i)
        sum += arr[i];
    return sum;
}

void structured_binding() {
    // 结构化绑定（C++17）
    std::array<int, 3> arr = {1, 2, 3};
    auto [a, b, c] = arr;
    std::cout << a << " " << b << " " << c << std::endl;

    // tuple接口
    auto elem = std::get<0>(arr);  // 编译期索引
}
```

### 2.4 作为函数参数和返回值

```cpp
// 作为返回值（C数组不行）
std::array<int, 3> make_array() {
    return {1, 2, 3};
}

// 作为参数（模板使大小灵活）
template<size_t N>
void process(const std::array<int, N>& arr) {
    for (const auto& v : arr)
        std::cout << v << " ";
}

// 引用传递避免拷贝
void modify(std::array<int, 5>& arr) {
    arr[0] = 42;
}

int main() {
    auto a = make_array();  // OK
    process(a);             // OK
}
```

## 三、与std::vector对比

```cpp
/*
| 特性          | array           | vector          |
|--------------|-----------------|-----------------|
| 大小          | 编译期固定       | 运行时动态       |
| 内存          | 栈上            | 堆上            |
| 性能          | 零开销          | 有额外开销       |
| 可扩展        | 否             | 是              |
| 适用场景      | 大小已知        | 大小未知         |
*/
```

## 四、注意事项与常见陷阱

1. **大小是类型的一部分**：`array<int,3>`和`array<int,4>`是不同类型，不能互相赋值
2. **没有`push_back`等动态操作**：大小固定，不支持增减元素
3. **`data()`返回连续内存指针**：可与C接口互操作
4. **零开销**：编译后与C数组性能完全相同
5. **所有STL容器接口**：迭代器、算法等都能正常使用
6. **可以作为函数返回值**：C数组不行
7. **未初始化行为与C数组相同**：`array<int, 5> a;`元素值未定义
8. **`at()`提供越界检查**：`operator[]`不检查
