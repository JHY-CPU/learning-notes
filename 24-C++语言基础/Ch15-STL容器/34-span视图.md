# span视图（C++20）

## 一、概念说明

`std::span`是C++20引入的非拥有视图（C++标准 §24.7），提供对连续数据序列的安全访问。它不管理内存，只引用已有的连续数据（数组、vector、C数组等）。span可以看作"安全版的指针+大小"，替代了传统的`void func(T* data, size_t size)`接口。

### 1.1 span vs 指针+大小

| 特性 | span | 指针+大小 |
|------|------|---------|
| 安全性 | 边界检查（可选） | 无 |
| 接口 | STL风格 | C风格 |
| 编译期大小 | 可选 | 无 |
| 子视图 | `subspan` | 手动计算 |
| 零开销 | 是 | 是 |

## 二、具体用法

### 2.1 基本操作

```cpp
#include <span>
#include <vector>
#include <array>
#include <iostream>

// 函数参数使用span替代指针+大小
void print_span(std::span<const int> data) {
    std::cout << "大小: " << data.size() << std::endl;
    for (int v : data) std::cout << v << " ";
    std::cout << std::endl;
}

int main() {
    // 从不同来源创建span（零拷贝）
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {6, 7, 8, 9, 10};
    std::array<int, 3> std_arr = {11, 12, 13};

    std::span<int> s1(arr);              // 从C数组
    std::span<int> s2(vec);              // 从vector
    std::span<int> s3(std_arr);          // 从array
    std::span<int, 5> fixed(arr);        // 编译期大小

    // 动态大小span
    std::span<int> dyn(arr);             // 运行时确定大小

    // 使用
    print_span(s1);   // 1 2 3 4 5
    print_span(s2);   // 6 7 8 9 10

    // 访问
    std::cout << s1[0] << std::endl;     // 1
    std::cout << s1.front() << std::endl; // 1
    std::cout << s1.back() << std::endl;  // 5
    std::cout << s1.size() << std::endl;  // 5
    std::cout << s1.size_bytes() << std::endl; // 20
}
```

### 2.2 子视图

```cpp
void subspan_demo() {
    int arr[] = {1, 2, 3, 4, 5, 6, 7, 8};
    std::span<int> s(arr);

    // subspan(offset, count)
    auto sub1 = s.subspan(2, 3);    // [3, 4, 5]
    auto sub2 = s.subspan(3);       // [4, 5, 6, 7, 8]（从offset到末尾）

    // first和last
    auto first3 = s.first(3);       // [1, 2, 3]
    auto last2 = s.last(2);         // [7, 8]

    // 修改视图会修改原数据
    sub1[0] = 99;
    std::cout << arr[2] << std::endl;  // 99

    // 范围for
    for (int v : sub1) std::cout << v << " ";  // 99 4 5
}
```

### 2.3 编译期大小span

```cpp
void fixed_size_span() {
    int arr[5] = {1, 2, 3, 4, 5};

    // 编译期大小span
    std::span<int, 5> fixed(arr);

    // 大小是类型的一部分
    std::cout << "extent: " << fixed.extent << std::endl;  // 5
    std::cout << "size: " << fixed.size() << std::endl;     // 5

    // dynamic_extent表示运行时大小
    static_assert(std::span<int>::extent == std::dynamic_extent);

    // 编译期大小可以优化（不需要存储size）
    std::cout << "sizeof dynamic span: " << sizeof(std::span<int>) << std::endl;
    std::cout << "sizeof fixed span: " << sizeof(std::span<int, 5>) << std::endl;
    // 固定大小span可能更小（不需要size字段）
}
```

### 2.4 as_bytes和as_writable_bytes

```cpp
#include <span>
#include <cstring>

void bytes_view() {
    int arr[] = {0x12345678, 0x9ABCDEF0};
    std::span<int> s(arr);

    // 只读字节视图
    auto bytes = std::as_bytes(s);
    std::cout << "字节数: " << bytes.size() << std::endl;  // 8

    // 可写字节视图
    auto writable = std::as_writable_bytes(s);
    writable[0] = 0xFF;  // 修改第一个字节
    std::cout << std::hex << arr[0] << std::endl;

    // 应用：序列化/反序列化、网络协议解析
}
```

## 三、实用示例

```cpp
#include <span>
#include <algorithm>

// 通用处理函数
template<typename T>
T sum(std::span<const T> data) {
    return std::accumulate(data.begin(), data.end(), T{0});
}

// 处理子数组
void process_middle(std::span<int> data) {
    if (data.size() < 3) return;
    auto middle = data.subspan(1, data.size() - 2);
    std::fill(middle.begin(), middle.end(), 0);
}

int main() {
    int arr[] = {1, 2, 3, 4, 5};
    std::vector<int> vec = {6, 7, 8};

    // 同一函数处理不同容器
    std::cout << sum<int>(arr) << std::endl;  // 15
    std::cout << sum<int>(vec) << std::endl;  // 21

    process_middle(arr);  // arr: {1, 0, 0, 0, 5}
}
```

## 四、注意事项与常见陷阱

1. **span不拥有数据**：原始数据必须在其生命周期内有效
2. **span的大小为0时`data()`可能返回nullptr**：不要解引用
3. **用于替代指针+大小的C风格接口**：更安全更清晰
4. **支持迭代器、范围for循环**：STL兼容
5. **`span<T>` vs `span<const T>`**：类似指针语义，const保护数据
6. **`subspan`、`first`、`last`创建子视图**：不拷贝数据
7. **编译期大小span更紧凑**：不需要存储size字段
8. **C++26可能引入`std::mdspan`**：多维span
