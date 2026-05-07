# C++17 if constexpr

## 一、概念说明

`if constexpr`是编译期条件分支，在模板中根据类型特征选择不同的代码路径。不满足条件的分支在编译时被丢弃，不会产生编译错误。

```cpp
#include <iostream>
#include <type_traits>
#include <string>

template <typename T>
std::string toString(const T& value) {
    if constexpr (std::is_integral_v<T>) {
        return "整数: " + std::to_string(value);
    } else if constexpr (std::is_floating_point_v<T>) {
        return "浮点: " + std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "字符串: " + value;
    } else {
        return "未知类型";
    }
}

int main() {
    std::cout << toString(42) << std::endl;
    std::cout << toString(3.14) << std::endl;
    std::cout << toString(std::string("hello")) << std::endl;

    return 0;
}
```

**输出：**
```
整数: 42
浮点: 3.140000
字符串: hello
```

## 二、具体用法

### 2.1 SFINAE替代

```cpp
#include <iostream>
#include <type_traits>

// 不用if constexpr（SFINAE方式，复杂）
// template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
// auto process(T val) { ... }

// 用if constexpr（简洁）
template <typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>) {
        return val * 2;
    } else if constexpr (std::is_pointer_v<T>) {
        return *val * 2;
    } else {
        return val;
    }
}

int main() {
    std::cout << process(21) << std::endl;

    int x = 21;
    std::cout << process(&x) << std::endl;

    std::cout << process(3.14) << std::endl;

    return 0;
}
```

**输出：**
```
42
42
3.14
```

### 2.2 编译期递归终止

```cpp
#include <iostream>

template <typename T>
void print(T val) {
    std::cout << val << std::endl;
}

template <typename T, typename... Args>
void print(T first, Args... rest) {
    std::cout << first << ", ";
    if constexpr (sizeof...(rest) > 0) {
        print(rest...); // 只有rest不为空才展开
    } else {
        std::cout << std::endl;
    }
}

int main() {
    print(1, 2, 3, "hello");
    print(42);
    return 0;
}
```

**输出：**
```
1, 2, 3, hello,
42,
```

## 三、注意事项与常见陷阱

- **`if constexpr`的条件必须是编译期常量**。
- **被丢弃的分支不进行语义检查**：可以调用不存在的方法。
- **每个分支必须语法正确**：即使不被选中。
- **不能用在非模板函数中**（C++23前）：条件必须依赖模板参数。
- **`else if constexpr`是常见的模式**：替代SFINAE。
