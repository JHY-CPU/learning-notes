# constexpr函数

## 一、概念说明

`constexpr`函数是可以在**编译期**求值的函数。编译器在满足条件时将函数在编译期执行，结果直接嵌入生成的代码中。C++11仅允许函数体含单个return语句，C++14/17/20逐步放宽限制。

`constexpr`函数**也可以**在运行时调用（取决于参数是否为编译期常量）。

## 二、具体用法

### 2.1 基本constexpr函数

```cpp
constexpr int square(int x) {
    return x * x;
}

// 编译期求值
constexpr int result = square(5);
static_assert(result == 25, "compile-time check");

// 运行时调用
int n = 10;
std::cout << square(n) << std::endl;  // 输出: 100
```

### 2.2 C++14放宽（循环和变量）

```cpp
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {  // C++14允许循环
        result *= i;
    }
    return result;
}

constexpr int val = factorial(6);
std::cout << "6! = " << val << std::endl;  // 输出: 6! = 720
```

### 2.3 C++17：if constexpr（编译期分支）

```cpp
template <typename T>
auto get_value(T t) {
    if constexpr (std::is_pointer_v<T>) {
        return *t;  // 指针类型：解引用
    } else {
        return t;   // 非指针：直接返回
    }
}

int x = 42;
std::cout << get_value(x) << std::endl;   // 输出: 42
std::cout << get_value(&x) << std::endl;  // 输出: 42
```

### 2.4 constexpr构造函数

```cpp
class Point {
    double x_, y_;
public:
    constexpr Point(double x, double y) : x_(x), y_(y) {}
    constexpr double x() const { return x_; }
    constexpr double y() const { return y_; }
};

constexpr Point origin(0, 0);
constexpr Point p(3.0, 4.0);
static_assert(p.x() == 3.0, "x should be 3");
```

## 三、注意事项与常见陷阱

- `constexpr`函数不保证一定在编译期执行，取决于参数和上下文
- 要求编译期结果时，用`constexpr`变量或`static_assert`接收
- C++20的`consteval`可以强制编译期执行
- `constexpr`函数中不能有`static`变量、`try-catch`、`goto`
- 编译期执行的函数参数也必须是编译期常量
