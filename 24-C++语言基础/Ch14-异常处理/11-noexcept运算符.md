# noexcept运算符

## 一、概念说明

`noexcept`运算符（注意与`noexcept`说明符的区别）是一个编译期一元运算符，用于检测表达式是否被声明为`noexcept`。它返回`bool`常量，可在模板和SFINAE中使用。

## 二、具体用法

### 2.1 基本检测

```cpp
#include <iostream>
#include <vector>
#include <type_traits>

void safe_func() noexcept {}
void risky_func() {}

int main() {
    std::cout << std::boolalpha;

    // 检测函数是否noexcept
    std::cout << "safe_func: " << noexcept(safe_func()) << std::endl;  // true
    std::cout << "risky_func: " << noexcept(risky_func()) << std::endl; // false

    // 检测表达式
    int x = 1;
    std::cout << "int移动: " << noexcept(std::move(x)) << std::endl;  // true

    // 检测类型操作
    std::cout << "vector析构: " << noexcept(std::declval<std::vector<int>>().~vector()) << std::endl;  // true
}
```

### 2.2 条件模板

```cpp
// 根据类型是否支持noexcept移动来选择策略
template <typename T>
void conditional_move(T& a, T& b) {
    if constexpr (noexcept(T(std::move(a)))) {
        std::cout << "使用移动语义" << std::endl;
        T temp = std::move(a);
        a = std::move(b);
        b = std::move(temp);
    } else {
        std::cout << "使用拷贝语义" << std::endl;
        T temp = a;
        a = b;
        b = temp;
    }
}
```

### 2.3 检测自定义类型

```cpp
struct NoThrowType {
    NoThrowType(NoThrowType&&) noexcept {}
    NoThrowType& operator=(NoThrowType&&) noexcept { return *this; }
};

struct MayThrowType {
    MayThrowType(MayThrowType&&) {}  // 可能抛异常
    MayThrowType& operator=(MayThrowType&&) { return *this; }
};

int main() {
    std::cout << std::boolalpha;
    std::cout << "NoThrowType移动: "
              << noexcept(NoThrowType(std::declval<NoThrowType>())) << std::endl;  // true
    std::cout << "MayThrowType移动: "
              << noexcept(MayThrowType(std::declval<MayThrowType>())) << std::endl; // false

    // 与type_traits配合
    std::cout << "is_nothrow_move_constructible: "
              << std::is_nothrow_move_constructible_v<NoThrowType> << std::endl;  // true
}
```

### 2.4 在SFINAE中使用

```cpp
// 检测swap是否noexcept
template <typename T, typename = void>
struct is_nothrow_swappable : std::false_type {};

template <typename T>
struct is_nothrow_swappable<T, std::enable_if_t<noexcept(std::swap(std::declval<T&>(), std::declval<T&>()))>>
    : std::true_type {};

int main() {
    std::cout << "int可noexcept交换: "
              << is_nothrow_swappable<int>::value << std::endl;  // true
}
```

## 三、注意事项与常见陷阱

- `noexcept`运算符的结果是编译期常量
- 运算符不会求值其操作数（只检查声明）
- 注意区分`noexcept`说明符（声明）和`noexcept`运算符（检测）
- 运算符可与`constexpr if`、`enable_if`等配合使用
- `noexcept(noexcept(expr))`是常见用法（说明符中使用运算符）
- 移动构造/赋值的noexcept状态影响容器行为
