# C++11 constexpr

## 一、概念说明

`constexpr`指示表达式或函数可以在编译期求值。它让编译器在编译时计算常量表达式，提高运行时性能。

C++11的`constexpr`函数限制较多（只能有return语句），C++14/17/20逐步放宽。

```cpp
#include <iostream>

// C++11 constexpr函数（受限）
constexpr int square(int x) { return x * x; }

// 编译期常量
constexpr int result = square(5); // 编译期计算
// int arr[square(5)]; // 可以用作数组大小

int main() {
    constexpr int a = square(3);    // 编译期
    int n = 10;
    int b = square(n);              // 运行时（n不是常量表达式）

    std::cout << "square(3)=" << a << std::endl;
    std::cout << "square(10)=" << b << std::endl;
    std::cout << "result=" << result << std::endl;

    return 0;
}
```

**输出：**
```
square(3)=9
square(10)=100
result=25
```

## 二、具体用法

### 2.1 C++14 constexpr增强

```cpp
#include <iostream>

// C++14: constexpr函数可以有局部变量、循环、if
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; ++i) {
        int temp = b;
        b = a + b;
        a = temp;
    }
    return b;
}

int main() {
    constexpr int f5 = factorial(5);
    constexpr int fib10 = fibonacci(10);

    std::cout << "5! = " << f5 << std::endl;
    std::cout << "fib(10) = " << fib10 << std::endl;

    // 编译期生成查找表
    constexpr int table[] = {factorial(0), factorial(1), factorial(2),
                              factorial(3), factorial(4), factorial(5)};
    std::cout << "查找表: ";
    for (int v : table) std::cout << v << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
5! = 120
fib(10) = 55
查找表: 1 1 2 6 24 120
```

### 2.2 constexpr vs consteval vs constinit

| 关键字 | 含义 | 版本 |
|--------|------|------|
| `constexpr` | 可以在编译期求值 | C++11 |
| `consteval` | 必须在编译期求值 | C++20 |
| `constinit` | 变量在编译期初始化 | C++20 |

```cpp
#include <iostream>

// constexpr: 可以编译期或运行时
constexpr int add(int a, int b) { return a + b; }

// consteval: 必须编译期（C++20）
// consteval int mustCompile(int x) { return x * 2; }

int main() {
    constexpr int a = add(1, 2); // 编译期
    int n = 5;
    int b = add(n, 3);           // 运行时（也OK）

    std::cout << "a=" << a << ", b=" << b << std::endl;
    return 0;
}
```

**输出：**
```
a=3, b=8
```

## 三、注意事项与常见陷阱

- **`constexpr`函数不保证编译期执行**：取决于参数是否为常量表达式。
- **C++11 `constexpr`函数只能有一条return语句**：C++14放宽。
- **`constexpr`构造函数可以构造非常量对象**：反之不行。
- **虚函数不能是`constexpr`**（C++20前）。
- **`consteval`是C++20特性**：保证编译期执行。
