# constexpr编译期常量

## 一、概念说明

`constexpr`表示"常量表达式"，要求值在**编译时**就能确定。C++11引入的`constexpr`比`const`更强，它可以修饰变量、函数和构造函数，支持编译期计算。

## 二、具体用法

### 2.1 constexpr变量

```cpp
#include <iostream>
#include <array>
using namespace std;

int main() {
    // constexpr变量：编译期确定值
    constexpr int maxSize = 100;
    constexpr double pi = 3.14159265358979;

    // 可以用于需要编译期常量的场景
    int arr1[maxSize];                // C风格数组大小
    array<int, maxSize> arr2;         // std::array大小

    // const vs constexpr
    int x = 42;
    const int c = x;          // OK：运行时初始化
    // constexpr int ce = x;  // 错误：x不是编译期常量
    constexpr int ce = 42;    // OK：字面量是编译期常量

    cout << "maxSize: " << maxSize << endl;
    cout << "arr2大小: " << arr2.size() << endl;

    return 0;
}
```

输出：
```
maxSize: 100
arr2大小: 100
```

### 2.2 constexpr函数

```cpp
#include <iostream>
using namespace std;

// constexpr函数：可以在编译期求值
constexpr int factorial(int n) {
    return (n <= 1) ? 1 : n * factorial(n - 1);
}

constexpr int fibonacci(int n) {
    if (n <= 1) return n;
    int a = 0, b = 1;
    for (int i = 2; i <= n; i++) {
        int temp = a + b;
        a = b;
        b = temp;
    }
    return b;
}

constexpr double circleArea(double radius) {
    return 3.14159265358979 * radius * radius;
}

int main() {
    // 编译期求值：结果嵌入到生成的机器码中
    constexpr int fact5 = factorial(5);
    constexpr int fib10 = fibonacci(10);
    constexpr double area = circleArea(5.0);

    cout << "5! = " << fact5 << endl;
    cout << "fib(10) = " << fib10 << endl;
    cout << "圆面积: " << area << endl;

    // 编译期用作数组大小
    int arr[factorial(4)];  // 数组大小24在编译期确定
    cout << "数组大小: " << sizeof(arr) / sizeof(arr[0]) << endl;

    // 运行时也可以调用constexpr函数
    int n = 7;
    cout << n << "! = " << factorial(n) << endl;

    return 0;
}
```

输出：
```
5! = 120
fib(10) = 55
圆面积: 78.5398
数组大小: 24
7! = 5040
```

### 2.3 constexpr与const的区别

```cpp
#include <iostream>
using namespace std;

// const：只表示"不可修改"，不要求编译期确定
// constexpr：要求必须在编译期能确定值

int getValue() { return 42; }

int main() {
    // const可以在运行时初始化
    int x = getValue();
    const int c = x;           // OK：运行时const

    // constexpr必须编译期确定
    constexpr int ce = 42;     // OK：字面量
    // constexpr int ce2 = x;  // 错误：x是运行时值

    // const函数 vs constexpr函数
    // const：成员函数不修改对象
    // constexpr：函数结果可以在编译期求值

    // C++17起，if-constexpr
    constexpr bool isDebug = true;
    if constexpr (isDebug) {
        cout << "调试模式（编译期决定）" << endl;
    }

    return 0;
}
```

输出：
```
调试模式（编译期决定）
```

### 2.4 constexpr构造函数

```cpp
#include <iostream>
using namespace std;

class Point {
    double x_, y_;
public:
    // constexpr构造函数
    constexpr Point(double x, double y) : x_(x), y_(y) {}

    constexpr double x() const { return x_; }
    constexpr double y() const { return y_; }

    constexpr double distanceFromOrigin() const {
        return x_ * x_ + y_ * y_;  // 简化版
    }
};

int main() {
    // 编译期创建对象
    constexpr Point p1(3.0, 4.0);
    constexpr Point origin(0.0, 0.0);

    // 编译期调用成员函数
    constexpr double dist = p1.distanceFromOrigin();

    cout << "p1 = (" << p1.x() << ", " << p1.y() << ")" << endl;
    cout << "到原点距离的平方: " << dist << endl;

    // 用于模板参数
    constexpr Point p2(1.0, 1.0);
    cout << "p2 = (" << p2.x() << ", " << p2.y() << ")" << endl;

    return 0;
}
```

输出：
```
p1 = (3, 4)
到原点距离的平方: 25
p2 = (1, 1)
```

## 三、注意事项与常见陷阱

1. **constexpr函数不一定在编译期求值**：只有在需要编译期常量的上下文中才会编译期求值
2. **C++14放宽了限制**：constexpr函数可以有局部变量、循环和条件语句
3. **不能在constexpr函数中使用动态内存**：`new`/`delete`不允许（C++20有所放宽）
4. **隐式inline**：constexpr函数隐式具有inline属性
5. **consteval（C++20）**：如果需要强制编译期求值，使用`consteval`而非`constexpr`
