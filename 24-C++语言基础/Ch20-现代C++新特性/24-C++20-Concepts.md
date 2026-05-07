# C++20 Concepts

## 一、概念说明

Concepts（概念）是对模板参数的约束，替代SFINAE，提供更清晰的错误信息。定义在`<concepts>`头文件中。

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

// requires表达式
template <typename T>
concept Printable = requires(T t) {
    { std::cout << t } -> std::same_as<std::ostream&>;
};

template <Printable T>
void print(const T& val) {
    std::cout << val << std::endl;
}

int main() {
    std::cout << add(3, 4) << std::endl;       // OK
    std::cout << add(1.5, 2.5) << std::endl;   // OK
    // add("a", "b"); // 错误：不满足Numeric

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

```cpp
#include <iostream>
#include <concepts>

// 常用标准concepts
// std::same_as<T, U>      T和U是同一类型
// std::derived_from<T, B> T派生自B
// std::convertible_to<T, U> T可转换为U
// std::integral<T>        T是整数类型
// std::floating_point<T>  T是浮点类型
// std::copyable<T>        T可拷贝
// std::movable<T>         T可移动
// std::default_initializable<T> T可默认构造

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

**输出：**
```
gcd(48,18) = 6
default int: 0
```

### 2.2 requires子句

```cpp
#include <iostream>

// 简单requires
template <typename T>
    requires sizeof(T) >= 4
void process(T val) {
    std::cout << "处理: " << val << std::endl;
}

// requires表达式
template <typename T>
concept HasSize = requires(T t) {
    { t.size() } -> std::convertible_to<size_t>;
};

template <HasSize T>
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

**输出：**
```
处理: 42
处理: 3.14
大小: 3
```

## 三、注意事项与常见陷阱

- **Concepts比SFINAE错误信息更清晰**：直接指出不满足哪个约束。
- **`requires`子句可以出现在模板参数列表之后**。
- **多个约束用`&&`连接**：`requires ConceptA<T> && ConceptB<T>`。
- **`auto`概念简写**：`Integral auto x = 42;`等价于`std::integral auto`。
- **需要C++20支持**：GCC 10+、Clang 10+、MSVC 2019 16.3+。
