# SFINAE基础

## 一、概念说明

SFINAE（Substitution Failure Is Not An Error）即"替换失败不是错误"。当编译器在模板参数替换过程中遇到无效类型或表达式时，不会立即报错，而是将该候选从重载集合中移除，转而选择其他候选。

SFINAE是C++元编程的重要技术，用于在编译期根据类型特征选择不同的函数实现。

## 二、具体用法

### 2.1 基本SFINAE

```cpp
#include <type_traits>

// 仅当T是整数类型时启用
template <typename T>
typename std::enable_if<std::is_integral<T>::value, T>::type
safeDivide(T a, T b) {
    return a / b;
}

// 仅当T是浮点类型时启用
template <typename T>
typename std::enable_if<std::is_floating_point<T>::value, T>::type
safeDivide(T a, T b) {
    return (b != 0) ? a / b : 0;
}

std::cout << safeDivide(10, 3) << std::endl;    // 输出: 3 (整数除法)
std::cout << safeDivide(10.0, 3.0) << std::endl;  // 输出: 3.33333
```

### 2.2 使用void_t检测类型特征（C++17）

```cpp
template <typename, typename = void>
struct has_size : std::false_type {};

template <typename T>
struct has_size<T, std::void_t<decltype(std::declval<T>().size())>>
    : std::true_type {};

static_assert(has_size<std::string>::value, "string has size()");
static_assert(!has_size<int>::value, "int has no size()");
```

### 2.3 C++20 concepts替代SFINAE

```cpp
// C++20更清晰的方式
template <typename T>
concept Integral = std::is_integral_v<T>;

template <Integral T>
T safeDivide(T a, T b) {
    return a / b;
}

std::cout << safeDivide(20, 4) << std::endl;  // 输出: 5
// safeDivide(2.5, 1.5);  // 编译错误：不满足Integral约束
```

### 2.4 常用类型特征

```cpp
std::cout << std::is_integral<int>::value << std::endl;        // 输出: 1
std::cout << std::is_pointer<int*>::value << std::endl;        // 输出: 1
std::cout << std::is_same<int, int32_t>::value << std::endl;   // 输出: 1 (通常)
std::cout << std::is_class<std::string>::value << std::endl;   // 输出: 1
```

## 三、注意事项与常见陷阱

- SFINAE只在模板参数替换的**直接上下文**中有效，函数体内的错误是硬错误
- `enable_if`的条件为false时产生替换失败，移除该候选
- C++20的`concept`和`requires`是SFINAE的更好替代
- SFINAE错误信息难以阅读，调试时注意区分硬错误和软失败
- `std::enable_if_t`是`typename std::enable_if<...>::type`的简写
