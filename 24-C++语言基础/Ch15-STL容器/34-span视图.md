# span视图（C++20）

## 一、概念说明

`std::span`是C++20引入的非拥有视图，提供对连续数据序列的安全访问。它不管理内存，只引用已有的连续数据（数组、vector、C数组等）。

## 二、具体用法

```cpp
#include <span>
#include <vector>
#include <iostream>

// 函数参数使用span替代指针+大小
void print_span(std::span<const int> data) {
    for (int v : data) std::cout << v << " ";
    std::cout << std::endl;
}

int main() {
    // 从不同来源创建span
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {6, 7, 8, 9, 10};

    std::span<int> s1(arr);           // 从C数组
    std::span<int> s2(vec);           // 从vector
    std::span<int> s3(arr, 3);        // 固定大小

    // 动态大小span
    std::span<int> dyn(arr);          // 运行时确定大小

    // 编译期大小span
    std::span<int, 5> fixed(arr);     // 编译期大小5

    // 子视图
    auto sub = s1.subspan(1, 3);      // [2, 3, 4]

    // 使用
    print_span(s1);   // 1 2 3 4 5
    print_span(sub);  // 2 3 4

    // 访问
    std::cout << s1[0] << std::endl;  // 1
    std::cout << s1.front() << std::endl;
    std::cout << s1.back() << std::endl;
    std::cout << s1.size() << std::endl;  // 5
}
```

## 三、注意事项

- span不拥有数据，原始数据必须在其生命周期内有效
- span的大小为0时不调用data()
- 用于替代指针+大小的C风格接口
- 支持迭代器、范围for循环
- `span<T>` vs `span<const T>`类似指针语义
