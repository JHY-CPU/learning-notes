# Concepts概念（C++20）

## 一、概念说明

Concepts（概念）是C++20引入的模板约束机制，允许以声明式语法指定模板参数必须满足的条件。它替代了SFINAE+enable_if的复杂写法，提供了更清晰的约束语义和更好的编译错误信息。

## 二、具体用法

### 2.1 concept定义

```cpp
// 定义一个concept
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

// 使用concept约束模板
template <Numeric T>
T add(T a, T b) {
    return a + b;
}

// 等价写法：requires子句
template <typename T>
    requires Numeric<T>
T multiply(T a, T b) {
    return a * b;
}

int main() {
    std::cout << add(3, 4) << std::endl;       // 7
    std::cout << multiply(2.5, 3.0) << std::endl; // 7.5
    // add("a", "b");  // 编译错误：不满足Numeric约束
}
```

### 2.2 requires表达式

```cpp
// requires表达式：检测类型是否满足特定条件
template <typename T>
concept Container = requires(T c) {
    typename T::value_type;          // 有value_type类型
    typename T::iterator;            // 有iterator类型
    { c.begin() } -> std::same_as<typename T::iterator>; // begin()返回iterator
    { c.end() } -> std::same_as<typename T::iterator>;   // end()返回iterator
    { c.size() } -> std::convertible_to<std::size_t>;    // size()转为size_t
};

template <Container C>
void print_container(const C& c) {
    std::cout << "[ ";
    for (const auto& elem : c) {
        std::cout << elem << " ";
    }
    std::cout << "] size=" << c.size() << std::endl;
}

int main() {
    std::vector<int> v{1, 2, 3};
    print_container(v);  // [ 1 2 3 ] size=3
    // print_container(42);  // 编译错误：int不满足Container
}
```

### 2.3 简写与auto

```cpp
// concept用作auto的约束
Numeric auto square(Numeric auto x) {
    return x * x;
}

// 直接在函数参数中使用
void process(std::integral auto val) {
    std::cout << "整数: " << val << std::endl;
}

int main() {
    std::cout << square(5) << std::endl;    // 25
    std::cout << square(2.5) << std::endl;  // 6.25
    process(42);                             // 整数: 42
}
```

### 2.4 组合concept

```cpp
// 逻辑组合
template <typename T>
concept Printable = requires(std::ostream& os, T val) {
    { os << val } -> std::same_as<std::ostream&>;
};

template <typename T>
concept PrintableNumeric = Numeric<T> && Printable<T>;

template <PrintableNumeric T>
void safe_print(T val) {
    std::cout << "安全打印: " << val << std::endl;
}

int main() {
    safe_print(42);       // 安全打印: 42
    safe_print(3.14);     // 安全打印: 3.14
}
```

## 三、注意事项与常见陷阱

- `concept`本身是bool类型的编译期常量，可用于`if constexpr`
- requires表达式中的约束在花括号内写条件，每行一个约束
- `std::same_as`、`std::convertible_to`等是标准库预定义的concept
- 简写语法`Numeric auto`等价于`auto`加concept约束
- concept约束可以用于变量声明、函数参数、模板参数等
- 编译器在concept不满足时会给出精确的错误位置和原因
- concept不能递归定义（不能在concept的requires表达式中引用自身）
