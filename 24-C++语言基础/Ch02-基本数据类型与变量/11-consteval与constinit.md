# consteval与constinit

## 一、概念说明

C++20引入了`consteval`和`constinit`两个新关键字。`consteval`声明**立即函数**，强制在编译期求值；`constinit`确保变量在**编译期初始化**，但之后可以修改。

## 二、具体用法

### 2.1 consteval立即函数

```cpp
#include <iostream>
using namespace std;

// consteval：必须在编译期求值，否则编译错误
consteval int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
    }
    return result;
}

// 对比constexpr：可以在编译期或运行时求值
constexpr int fib(int n) {
    if (n <= 1) return n;
    return fib(n - 1) + fib(n - 2);
}

int main() {
    // consteval函数必须编译期调用
    constexpr int f5 = factorial(5);  // OK：编译期求值
    cout << "5! = " << f5 << endl;

    // int n = 5;
    // int f = factorial(n);  // 错误：n不是编译期常量

    // 可以用字面量调用
    cout << "6! = " << factorial(6) << endl;

    // constexpr可以在运行时调用
    int x = 10;
    cout << "fib(10) = " << fib(x) << endl;

    return 0;
}
```

输出：
```
5! = 120
6! = 720
fib(10) = 55
```

### 2.2 constinit编译期初始化

```cpp
#include <iostream>
using namespace std;

// constinit：确保编译期初始化，但之后可以修改
constinit int globalCounter = 0;

constinit const char* appName = "MyApp";

// constinit可以用于静态存储期变量
constinit static int staticValue = 42;

void incrementCounter() {
    globalCounter++;  // OK：constinit变量可以修改
}

int main() {
    cout << "初始值: " << globalCounter << endl;
    incrementCounter();
    incrementCounter();
    cout << "修改后: " << globalCounter << endl;

    // constinit变量可以重新赋值
    staticValue = 100;
    cout << "staticValue: " << staticValue << endl;

    // 对比const：const变量不可修改
    const int constVar = 42;
    // constVar = 100;  // 错误

    // constinit const：编译期初始化 + 不可修改
    constinit const int fixed = 42;
    // fixed = 100;  // 错误：const不允许修改

    return 0;
}
```

输出：
```
初始值: 0
修改后: 2
staticValue: 100
```

### 2.3 解决静态初始化顺序问题

```cpp
#include <iostream>
using namespace std;

// 问题：不同编译单元的静态变量初始化顺序不确定
// 解决：使用constinit确保编译期初始化

// file1.cpp
constinit int configValue = 42;

// file2.cpp - 依赖configValue
// 传统方式可能访问未初始化的变量
// constinit确保configValue在编译期就已初始化

int getDoubleConfig() {
    return configValue * 2;
}

int main() {
    cout << "配置值: " << configValue << endl;
    cout << "双倍: " << getDoubleConfig() << endl;

    configValue = 100;
    cout << "修改后: " << getDoubleConfig() << endl;

    return 0;
}
```

输出：
```
配置值: 42
双倍: 84
修改后: 200
```

## 三、注意事项与常见陷阱

1. **consteval强制编译期**：运行时调用consteval函数会导致编译错误
2. **constinit不是const**：constinit只保证编译期初始化，变量仍可修改
3. **consteval vs constexpr**：consteval更严格（必须编译期），constexpr更灵活（可以运行时）
4. **constinit用于全局变量**：解决C++经典的"静态初始化顺序惨案"问题
5. **C++20特性**：需要`-std=c++20`，较老的编译器可能不支持
