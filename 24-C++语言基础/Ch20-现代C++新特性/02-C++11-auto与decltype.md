# C++11 auto与decltype

## 一、概念说明

`auto`让编译器自动推导变量类型，`decltype`获取表达式的类型。它们是现代C++类型推导的基石。

- `auto`：从初始化表达式推导类型
- `decltype(expr)`：获取表达式的精确类型（含引用、const）

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

int main() {
    // auto推导
    auto i = 42;           // int
    auto d = 3.14;         // double
    auto s = "hello";      // const char*
    auto v = std::vector<int>{1, 2, 3}; // std::vector<int>

    // decltype获取类型
    decltype(i) j = 100;   // int
    decltype((i)) k = i;   // int&（因为(i)是左值表达式）

    // 类型检查
    std::cout << "i is int: " << std::is_same<decltype(i), int>::value << std::endl;
    std::cout << "d is double: " << std::is_same<decltype(d), double>::value << std::endl;
    std::cout << "k is int&: " << std::is_same<decltype(k), int&>::value << std::endl;

    // 遍历时简化
    std::vector<int> nums = {1, 2, 3};
    for (auto it = nums.begin(); it != nums.end(); ++it) {
        std::cout << *it << " ";
    }
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
i is int: 1
d is double: 1
k is int&: 1
1 2 3
```

## 二、具体用法

### 2.1 auto的推导规则

```cpp
#include <iostream>
#include <type_traits>

int main() {
    int x = 42;
    const int cx = 42;
    const int& cr = x;

    auto a1 = x;    // int（去掉引用和const）
    auto a2 = cx;   // int（去掉const）
    auto a3 = cr;   // int（去掉引用和const）
    auto& a4 = cx;  // const int&（保留const）

    std::cout << "a1: " << std::is_same<decltype(a1), int>::value << std::endl;
    std::cout << "a2: " << std::is_same<decltype(a2), int>::value << std::endl;
    std::cout << "a3: " << std::is_same<decltype(a3), int>::value << std::endl;
    std::cout << "a4: " << std::is_same<decltype(a4), const int&>::value << std::endl;

    // auto不能推导的场景
    // auto arr[] = {1, 2, 3};       // 错误：不能推导数组
    // auto func(int x);             // 错误：不能用于函数参数（C++14前）
    // std::vector<auto> v;          // 错误：不能用于模板参数

    return 0;
}
```

**输出：**
```
a1: 1
a2: 1
a3: 1
a4: 1
```

### 2.2 decltype的推导规则

```cpp
#include <iostream>
#include <type_traits>
#include <vector>

// decltype用于返回类型推导（C++11尾置返回类型）
template <typename Container>
auto getItem(Container& c, size_t i) -> decltype(c[i]) {
    return c[i];
}

int main() {
    std::vector<int> v = {1, 2, 3};
    // decltype(c[i]) 推导为 int&

    getItem(v, 0) = 42; // 因为返回引用，可以赋值
    std::cout << "v[0] = " << v[0] << std::endl;

    // decltype vs auto
    const int ci = 0;
    auto a = ci;          // int（去const）
    decltype(ci) di = 0;  // const int（保留const）

    std::cout << "auto去const: " << std::is_same<decltype(a), int>::value << std::endl;
    std::cout << "decltype保留const: " << std::is_same<decltype(di), const int>::value << std::endl;

    return 0;
}
```

**输出：**
```
v[0] = 42
auto去const: 1
decltype保留const: 1
```

## 三、注意事项与常见陷阱

- **`auto`去掉引用和const**：需要保留时用`auto&`或`const auto&`。
- **`decltype((x))`得到引用**：括号使其成为左值表达式。
- **`auto`不能用于函数参数**（C++14前）：C++14起可用于泛型lambda。
- **`auto x{1}`在C++11中推导为`initializer_list<int>`**：C++17改为`int`。
- **`decltype`在编译期求值**：不执行表达式。
