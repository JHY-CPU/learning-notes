# 内联函数（inline）

## 一、概念说明

`inline`关键字（C++11标准 §7.1.2）具有两个语义：

1. **优化建议**：建议编译器将函数调用处替换为函数体，消除调用开销（只是建议，编译器可忽略）
2. **ODR豁免**：允许在多个翻译单元中定义相同的函数（放宽One Definition Rule），这是现代C++中更重要的语义

C++17起 `inline` 也适用于变量（inline variable），使全局常量的定义可以放在头文件中。

## 二、具体用法

### 2.1 基本内联函数

```cpp
// header.h — 在头文件中定义inline函数
inline int square(int x) {
    return x * x;
}

// 编译器可能将 square(5) 替换为 5 * 5（取决于优化级别）
int result = square(5);
cout << result << endl;  // 25
```

### 2.2 类内定义的成员函数自动内联

```cpp
#include <iostream>
using namespace std;

class Circle {
    double radius;
public:
    Circle(double r) : radius(r) {}

    // 类内定义的成员函数隐式inline
    double area() const { return 3.14159265 * radius * radius; }
    double circumference() const { return 2 * 3.14159265 * radius; }
};

// 等价写法：类外定义加inline
class Square {
    double side;
public:
    Square(double s) : side(s);
    double area() const;
};

inline double Square::area() const {
    return side * side;
}
```

### 2.3 constexpr隐含inline

```cpp
// constexpr函数自动具有inline属性
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}
// 等价于 inline constexpr int factorial(int n) { ... }

// C++17 inline变量
inline const int MAX_SIZE = 1024;  // 头文件中定义，不会多重定义
inline const char* APP_NAME = "MyApp";
```

### 2.4 头文件中的inline函数

```cpp
// ========== math_utils.h ==========
// 没有inline，放头文件会导致多重定义链接错误
// int add(int a, int b) { return a + b; }  // 错误！

// 使用inline，允许放头文件
inline int add(int a, int b) { return a + b; }  // 正确

// 匿名命名空间中的函数自动内部链接（另一种方案）
namespace {
    int internalHelper(int x) { return x * x; }
}
```

### 2.5 性能考量

```cpp
// 适合inline的小函数
inline int max(int a, int b) { return a > b ? a : b; }

// 不适合inline（编译器可能忽略inline建议）
inline void largeFunction() {
    // 几百行代码... 编译器不会内联
}

// 递归函数通常不会被内联
inline int fibonacci(int n) {
    if (n <= 1) return n;
    return fibonacci(n-1) + fibonacci(n-2);  // 递归调用无法内联
}
```

### 2.6 C++17 inline变量

```cpp
// ========== constants.h ==========
// C++17之前：需要extern声明+源文件定义
// extern const int MAX_BUFFER;
// ========== constants.cpp ==========
// const int MAX_BUFFER = 4096;

// C++17起：直接在头文件中定义
inline constexpr int MAX_BUFFER = 4096;
inline constexpr double PI = 3.14159265358979;

// 线程安全的inline变量
inline atomic<int> globalCounter{0};
```

## 三、注意事项与常见陷阱

1. **inline是建议而非命令**：编译器会自行决定是否内联，可能内联未标记的函数，也可能忽略inline
2. **递归函数、过大的函数通常不会被内联**：编译器优化器自行判断
3. **inline函数必须在每个使用它的翻译单元中定义**：通常放头文件
4. **过度内联会导致代码膨胀**：增大二进制体积，反而降低CPU缓存命中率
5. **现代编译器的优化器会自动内联小函数**：`-O2`下手动标记inline往往不必要
6. **inline不等同于static**：inline允许多个定义（ODR豁免），static产生多个副本（内部链接）
7. **调试模式下inline通常不生效**：方便单步调试
8. **C++17 inline变量解决了全局常量的ODR问题**：无需extern声明
