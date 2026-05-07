# C++17 optional variant any

## 一、概念说明

C++17引入三种类型安全的容器：
- `std::optional<T>`：可能有值也可能没有（替代`unique_ptr`作可选值）
- `std::variant<T1, T2, ...>`：类型安全的联合体（替代`union`）
- `std::any`：可以持有任意类型

```cpp
#include <iostream>
#include <optional>
#include <variant>
#include <any>
#include <string>

// optional：可能失败的函数
std::optional<int> safeDivide(int a, int b) {
    if (b == 0) return std::nullopt;
    return a / b;
}

int main() {
    // optional
    if (auto result = safeDivide(10, 3)) {
        std::cout << "10/3=" << *result << std::endl;
    }
    if (auto result = safeDivide(10, 0); !result) {
        std::cout << "除零失败" << std::endl;
    }

    // variant
    std::variant<int, double, std::string> v = 42;
    std::cout << "int: " << std::get<int>(v) << std::endl;
    v = 3.14;
    std::cout << "double: " << std::get<double>(v) << std::endl;
    v = "hello";
    std::cout << "string: " << std::get<std::string>(v) << std::endl;

    // any
    std::any a = 42;
    std::cout << "any int: " << std::any_cast<int>(a) << std::endl;
    a = std::string("world");
    std::cout << "any string: " << std::any_cast<std::string>(a) << std::endl;

    return 0;
}
```

**输出：**
```
10/3=3
除零失败
int: 42
double: 3.14
string: hello
any int: 42
any string: world
```

## 二、具体用法

### 2.1 variant访问

```cpp
#include <iostream>
#include <variant>

struct Visitor {
    void operator()(int i) const { std::cout << "int: " << i << std::endl; }
    void operator()(double d) const { std::cout << "double: " << d << std::endl; }
    void operator()(const std::string& s) const { std::cout << "string: " << s << std::endl; }
};

int main() {
    std::variant<int, double, std::string> v = 3.14;

    // std::visit
    std::visit(Visitor{}, v);

    // C++17 lambda visit
    std::visit([](auto&& arg) {
        using T = std::decay_t<decltype(arg)>;
        if constexpr (std::is_same_v<T, int>)
            std::cout << "lambda int: " << arg << std::endl;
        else if constexpr (std::is_same_v<T, double>)
            std::cout << "lambda double: " << arg << std::endl;
    }, v);

    return 0;
}
```

**输出：**
```
double: 3.14
lambda double: 3.14
```

## 三、注意事项与常见陷阱

- **`optional`的`value()`在无值时抛`bad_optional_access`**：用`value_or()`提供默认值。
- **`variant`的第一个类型必须可默认构造**。
- **`std::get`类型错误抛`bad_variant_access`**：用`std::get_if`安全检查。
- **`any`使用`type()`检查类型**：`any_cast`失败抛`bad_any_cast`。
- **`optional`/`variant`比`unique_ptr`/`any`更高效**：不需要堆分配。
