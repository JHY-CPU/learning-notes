# C++11 constexpr

## 一、概念说明

`constexpr`（C++11 §5.19, §7.1.5）指示表达式或函数可以在编译期求值。它让编译器在编译时计算常量表达式，提高运行时性能，支持编译期数组大小、模板参数等场景。

### 1.1 constexpr vs const

| 特性 | const | constexpr |
|------|-------|-----------|
| 求值时机 | 运行时或编译时 | 尽可能编译时 |
| 适用对象 | 变量 | 变量、函数、构造函数 |
| 修改性 | 不可修改 | 不可修改 |
| 数组大小 | 不保证 | 保证可用作数组大小 |

C++11的`constexpr`函数限制较多（只能有return语句），C++14/17/20逐步放宽。

```cpp
#include <iostream>

// C++11 constexpr函数（受限：单条return语句）
constexpr int square(int x) { return x * x; }

// 编译期常量
constexpr int result = square(5); // 编译期计算
int arr[square(5)];               // 可用作数组大小

int main() {
    constexpr int a = square(3);    // 编译期
    int n = 10;
    int b = square(n);              // 运行时（n不是常量表达式）

    std::cout << "square(3)=" << a << std::endl;
    std::cout << "square(10)=" << b << std::endl;
    std::cout << "result=" << result << std::endl;
    std::cout << "arr大小=" << sizeof(arr)/sizeof(int) << std::endl;

    return 0;
}
```

**输出：**
```
square(3)=9
square(10)=100
result=25
arr大小=25
```

## 二、具体用法

### 2.1 C++14 constexpr增强

C++14允许`constexpr`函数使用局部变量、循环、if语句等。

```cpp
#include <iostream>

// C++14: 循环和局部变量
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// constexpr数组操作
constexpr int sumArray(const int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    constexpr int f10 = factorial(10);
    constexpr int g = gcd(48, 18);

    std::cout << "10! = " << f10 << std::endl;
    std::cout << "gcd(48,18) = " << g << std::endl;

    constexpr int arr[] = {1, 2, 3, 4, 5};
    constexpr int total = sumArray(arr, 5);
    std::cout << "数组求和 = " << total << std::endl;

    return 0;
}
```

**输出：**
```
10! = 3628800
gcd(48,18) = 6
数组求和 = 15
```

### 2.2 编译期查找表

```cpp
#include <iostream>
#include <array>
#include <cmath>

// 编译期生成正弦查找表（泰勒级数）
constexpr double mySin(double x) {
    double term = x;
    double sum = x;
    for (int i = 1; i < 10; ++i) {
        term *= -x * x / ((2 * i) * (2 * i + 1));
        sum += term;
    }
    return sum;
}

template <int N>
constexpr std::array<double, N> makeSinTable() {
    std::array<double, N> table{};
    constexpr double pi = 3.14159265358979;
    for (int i = 0; i < N; ++i) {
        double angle = pi * i / (N - 1);
        table[i] = mySin(angle);
    }
    return table;
}

int main() {
    // 编译期生成正弦表
    constexpr auto sinTable = makeSinTable<360>();

    std::cout << "sin(0) = " << sinTable[0] << std::endl;
    std::cout << "sin(90) = " << sinTable[90] << std::endl;
    std::cout << "sin(180) = " << sinTable[180] << std::endl;
    std::cout << "sin(270) = " << sinTable[270] << std::endl;

    return 0;
}
```

**输出：**
```
sin(0) = 0
sin(90) = 1
sin(180) = 1.22465e-15
sin(270) = -1
```

### 2.3 constexpr构造函数

```cpp
#include <iostream>

class Point {
    double x_, y_;
public:
    constexpr Point(double x, double y) : x_(x), y_(y) {}
    constexpr double x() const { return x_; }
    constexpr double y() const { return y_; }
    constexpr double distanceFromOrigin() const {
        return x_ * x_ + y_ * y_; // 简化：返回平方
    }
};

int main() {
    constexpr Point p(3.0, 4.0);
    constexpr double d2 = p.distanceFromOrigin();
    std::cout << "p=(" << p.x() << "," << p.y() << ")" << std::endl;
    std::cout << "距离平方=" << d2 << std::endl;

    // 运行时使用
    Point q(1.0, 1.0);
    std::cout << "q距离平方=" << q.distanceFromOrigin() << std::endl;

    return 0;
}
```

**输出：**
```
p=(3,4)
距离平方=25
q距离平方=2
```

## 三、注意事项与常见陷阱

1. **`constexpr`函数不保证编译期执行**：取决于参数是否为常量表达式。
2. **C++11 `constexpr`函数只能有一条return语句**：C++14放宽到允许循环和局部变量。
3. **`constexpr`构造函数可以构造非常量对象**：反之不行。
4. **虚函数在C++20前不能是`constexpr`**：C++20放宽。
5. **`static_assert`可以用`constexpr`函数的结果**：验证编译期计算。
6. **`constexpr`对象的成员自动是`const`的**。
