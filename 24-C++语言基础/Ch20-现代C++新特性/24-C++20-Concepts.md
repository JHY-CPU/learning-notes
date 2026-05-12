# C++20 Concepts

## 一、概念说明

Concepts（概念，C++20 §13.5）是对模板参数的约束，替代SFINAE，提供更清晰的错误信息。定义在`<concepts>`头文件中。Concepts是C++20四大特性之一，极大地改善了模板编程的体验。

### 1.1 为什么需要Concepts

```
SFINAE的问题：
1. 错误信息冗长难懂
2. 代码复杂（enable_if、decltype等）
3. 约束表达不直观

Concepts的优势：
1. 清晰的约束语法
2. 错误信息直接指出不满足哪个约束
3. 可组合、可复用
```

```cpp
#include <iostream>
#include <concepts>
#include <vector>
#include <string>

// 定义concept
template <typename T>
concept Numeric = std::is_arithmetic_v<T>;

// 使用concept约束
template <Numeric T>
T add(T a, T b) {
    return a + b;
}

// requires表达式定义concept
template <typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template <Printable T>
void print(const T& val) {
    std::cout << val << std::endl;
}

int main() {
    std::cout << add(3, 4) << std::endl;
    std::cout << add(1.5, 2.5) << std::endl;
    // add("a", "b"); // 编译错误：不满足Numeric

    print(42);
    print("hello");

    return 0;
}
```

**输出：**
```
7
4
42
hello
```

## 二、具体用法

### 2.1 标准concepts

`<concepts>`头文件提供了一系列标准concepts：

```cpp
#include <iostream>
#include <concepts>
#include <vector>

// 常用标准concepts
// std::same_as<T, U>         T和U是同一类型
// std::derived_from<T, B>    T派生自B
// std::convertible_to<T, U>  T可转换为U
// std::integral<T>           T是整数类型
// std::floating_point<T>     T是浮点类型
// std::copyable<T>           T可拷贝
// std::movable<T>            T可移动
// std::default_initializable<T> T可默认构造
// std::invocable<F, Args...> F可调用
// std::predicate<P, Args...> P是谓词

template <std::integral T>
T gcd(T a, T b) {
    while (b) { T t = b; b = a % b; a = t; }
    return a;
}

template <typename T>
    requires std::copyable<T> && std::default_initializable<T>
T createDefault() {
    return T{};
}

int main() {
    std::cout << "gcd(48,18) = " << gcd(48, 18) << std::endl;
    auto d = createDefault<int>();
    std::cout << "default int: " << d << std::endl;
    return 0;
}
```

### 2.2 requires子句

```cpp
#include <iostream>
#include <vector>

// 简单requires
template <typename T>
    requires sizeof(T) >= 4
void process(T val) {
    std::cout << "处理(>=4字节): " << val << std::endl;
}

// requires表达式定义concept
template <typename T>
concept HasSize = requires(T t) {
    { t.size() } -> std::convertible_to<size_t>;
};

template <typename T>
concept HasPushBack = requires(T t, typename T::value_type v) {
    t.push_back(v);
};

// 组合concept
template <typename T>
concept Container = HasSize<T> && HasPushBack<T>;

template <Container T>
void printSize(const T& container) {
    std::cout << "大小: " << container.size() << std::endl;
}

int main() {
    process(42);
    process(3.14);

    std::vector<int> v = {1, 2, 3};
    printSize(v);
    // printSize(42); // 错误：int没有size()方法

    return 0;
}
```

### 2.3 concept的简写

```cpp
#include <iostream>
#include <concepts>

// auto简写
std::integral auto square(std::integral auto x) {
    return x * x;
}

// 多个concept约束
template <typename T>
    requires std::integral<T> && std::copyable<T>
T double_it(T x) {
    return x * 2;
}

int main() {
    std::cout << "square(5) = " << square(5) << std::endl;
    std::cout << "double_it(21) = " << double_it(21) << std::endl;
    // double_it(3.14); // 错误：不满足integral

    return 0;
}
```

**输出：**
```
square(5) = 25
double_it(21) = 42
```

## 三、注意事项与常见陷阱

1. **Concepts比SFINAE错误信息更清晰**：直接指出不满足哪个约束，而非一大堆模板实例化失败信息。
2. **`requires`子句可以出现在模板参数列表之后**：`template<typename T> requires Concept<T>`。
3. **多个约束用`&&`连接**：`requires ConceptA<T> && ConceptB<T>`。
4. **`auto`概念简写**：`Integral auto x = 42;`等价于`std::integral auto x = 42;`。
5. **需要C++20支持**：GCC 10+、Clang 10+、MSVC 2019 16.3+。
6. **concept不能递归定义**：concept表达式中不能引用自身。
