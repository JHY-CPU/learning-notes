# variant详解

## 一、概念说明

`std::variant`是C++17引入的类型安全联合体，可以在编译时确定可能的类型集合。相比`any`，它提供更高效的访问和编译时类型检查。

## 二、具体用法

```cpp
#include <variant>
#include <iostream>
#include <string>

int main() {
    // 存储不同类型（编译时确定）
    std::variant<int, double, std::string> v = 42;

    // 访问
    std::cout << std::get<int>(v) << std::endl;  // 42

    // 修改
    v = 3.14;
    std::cout << std::get<double>(v) << std::endl;  // 3.14

    // 安全访问
    auto* p = std::get_if<int>(&v);
    if (p) std::cout << *p << std::endl;
    else std::cout << "不是int" << std::endl;

    // 检查当前类型索引
    std::cout << "index: " << v.index() << std::endl;  // 1 (double)

    // visit访问
    v = std::string("hello");
    std::visit([](auto&& val) {
        std::cout << val << std::endl;
    }, v);
}
```

### 2.1 monostate处理默认构造

```cpp
#include <variant>

// variant的第一个类型必须可默认构造
// 如果都不行，用std::monostate
std::variant<std::monostate, std::string, std::vector<int>> v;  // 默认是monostate
```

### 2.2 类型安全的错误处理

```cpp
template <typename T, typename E = std::string>
using Result = std::variant<T, E>;

Result<int> divide(int a, int b) {
    if (b == 0) return std::string("除零");
    return a / b;
}
```

## 三、注意事项

- variant的大小是最大类型的大小+类型索引
- `std::get<T>`类型不匹配时抛出`bad_variant_access`
- `std::get_if<T>`返回指针，失败返回nullptr
- visit需要一个可调用对象处理所有可能类型
- variant不可持有引用类型，但可持有`reference_wrapper`
