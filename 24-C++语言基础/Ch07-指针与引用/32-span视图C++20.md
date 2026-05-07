# span视图（C++20）

## 一、概念说明

`std::span`（C++20，`<span>`头文件）是连续序列的**非拥有视图**，类似`string_view`之于字符串。它可以包装C风格数组、`std::array`、`std::vector`等连续容器，提供统一的访问接口。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <span>

void print(std::span<const int> sp) {
    std::cout << "大小: " << sp.size() << " 内容: ";
    for (int x : sp) std::cout << x << " ";
    std::cout << std::endl;
}

// 从vector
std::vector<int> v = {1, 2, 3, 4, 5};
print(v);
// 输出: 大小: 5 内容: 1 2 3 4 5

// 从C风格数组
int arr[] = {10, 20, 30};
print(arr);
// 输出: 大小: 3 内容: 10 20 30

// 从initializer_list
print({7, 8, 9});
// 输出: 大小: 3 内容: 7 8 9
```

### 2.2 静态大小 span

```cpp
// 编译期已知大小
void process(std::span<int, 3> sp) {
    static_assert(sp.size() == 3);
    for (int x : sp) std::cout << x << " ";
    std::cout << std::endl;
}

int arr[3] = {1, 2, 3};
process(arr);
// 输出: 1 2 3
```

### 2.3 子视图

```cpp
std::vector<int> v = {1, 2, 3, 4, 5, 6, 7, 8};

// subspan：获取子序列
auto sub = std::span(v).subspan(2, 4);
for (int x : sub) std::cout << x << " ";
// 输出: 3 4 5 6

// first / last
auto first3 = std::span(v).first(3);
auto last2 = std::span(v).last(2);
```

### 2.4 修改数据

```cpp
void doubleAll(std::span<int> sp) {
    for (int& x : sp) x *= 2;
}

std::vector<int> v = {1, 2, 3, 4, 5};
doubleAll(v);
for (int x : v) std::cout << x << " ";
// 输出: 2 4 6 8 10
```

### 2.5 span vs 其他方式

```cpp
// span统一了不同的连续容器接口
template <typename Container>
void process(std::span<const int> data) {
    // 可接受 vector, array, 原生数组等
}
```

## 三、注意事项与常见陷阱

- `span`是非拥有视图，不延长底层数据生命周期
- `span`的`data()`可能返回`nullptr`（空span）
- 静态大小span（`span<T, N>`）在编译期检查大小
- `span`可以修改底层数据（除非用`span<const T>`）
- C++20之前可使用GSL库的`gsl::span`
