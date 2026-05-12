# C++17 if constexpr

## 一、概念说明

`if constexpr`（C++17 §9.4.1）是编译期条件分支，在模板中根据类型特征选择不同的代码路径。不满足条件的分支在编译时被丢弃，不进行语义检查，不会产生编译错误。它是SFINAE的优雅替代方案。

### 1.1 if constexpr vs 普通if

| 特性 | 普通if | if constexpr |
|------|--------|-------------|
| 条件求值 | 运行时 | 编译时 |
| 分支丢弃 | 不丢弃 | 丢弃不满足的分支 |
| 编译要求 | 所有分支必须有效 | 被丢弃的分支可以无效 |
| 适用场景 | 通用 | 模板中 |

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

// 不用if constexpr（SFINAE方式，复杂且难读）
// template <typename T, std::enable_if_t<std::is_integral_v<T>>* = nullptr>
// auto process(T val) { return val * 2; }
// template <typename T, std::enable_if_t<std::is_pointer_v<T>>* = nullptr>
// auto process(T val) { return *val * 2; }

// 用if constexpr（简洁明了）
template <typename T>
auto process(T val) {
    if constexpr (std::is_integral_v<T>) {
        return val * 2;
    } else if constexpr (std::is_pointer_v<T>) {
        return *val * 2;
    } else if constexpr (std::is_floating_point_v<T>) {
        return val * 2.0;
    } else {
        static_assert(sizeof(T) == 0, "不支持的类型");
    }
}

int main() {
    std::cout << process(21) << std::endl;       // 整数

    int x = 21;
    std::cout << process(&x) << std::endl;       // 指针

    std::cout << process(3.14) << std::endl;     // 浮点

    return 0;
}
```

**输出：**
```
42
42
6.28
```

### 2.2 编译期递归终止

```cpp
#include <iostream>

// 使用if constexpr替代递归终止重载
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
    print(3.14, true, 'A');
    return 0;
}
```

**输出：**
```
1, 2, 3, hello
42
3.14, 1, A
```

### 2.3 类型相关的序列化

```cpp
#include <iostream>
#include <sstream>
#include <vector>
#include <type_traits>

template <typename T>
std::string serialize(const T& value) {
    if constexpr (std::is_arithmetic_v<T>) {
        return std::to_string(value);
    } else if constexpr (std::is_same_v<T, std::string>) {
        return "\"" + value + "\"";
    } else if constexpr (std::is_same_v<T, bool>) {
        return value ? "true" : "false";
    } else {
        // 被丢弃的分支：即使vector没有to_string也不会编译错误
        std::ostringstream oss;
        oss << "[";
        // 不能直接用value，因为T可能是int
        return oss.str() + "]";
    }
}

int main() {
    std::cout << serialize(42) << std::endl;
    std::cout << serialize(3.14) << std::endl;
    std::cout << serialize(std::string("hello")) << std::endl;
    std::cout << serialize(true) << std::endl;
    return 0;
}
```

**输出：**
```
42
3.140000
"hello"
true
```

## 三、注意事项与常见陷阱

1. **`if constexpr`的条件必须是编译期常量**：不能用运行时变量。
2. **被丢弃的分支不进行语义检查**：可以调用不存在的方法（但必须语法正确）。
3. **每个分支必须语法正确**：即使不被选中，也要能被解析。
4. **不能用在非模板函数中**（C++23前）：条件必须依赖模板参数。
5. **`else if constexpr`是常见模式**：替代SFINAE，更清晰。
6. **`static_assert(false)`在被丢弃的分支中不会触发**（C++23前需用依赖模板参数的方式）。
