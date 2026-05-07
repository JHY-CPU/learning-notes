# 常量与const

## 一、概念说明

`const`关键字用于声明**不可修改**的变量（常量）。它是一种类型修饰符，告诉编译器该对象在初始化后不应被更改。`const`是C++中最重要的关键字之一，广泛用于函数参数、成员函数和变量声明。

## 二、具体用法

### 2.1 const变量

```cpp
#include <iostream>
using namespace std;

int main() {
    // const变量必须初始化，之后不能修改
    const int maxSize = 100;
    const double pi = 3.14159265358979;
    const string appName = "我的应用";

    // maxSize = 200;  // 编译错误：不能修改const变量

    cout << "最大大小: " << maxSize << endl;
    cout << "圆周率: " << pi << endl;
    cout << "应用名: " << appName << endl;

    // const可以是运行时确定的值
    int x;
    // cin >> x;
    x = 42;
    const int runtimeConst = x;  // 运行时常量

    cout << "运行时常量: " << runtimeConst << endl;

    return 0;
}
```

输出：
```
最大大小: 100
圆周率: 3.14159
应用名: 我的应用
运行时常量: 42
```

### 2.2 const指针（三种形式）

```cpp
#include <iostream>
using namespace std;

int main() {
    int a = 10, b = 20;

    // 形式1：指向常量的指针（数据不可改，指针可改）
    const int* p1 = &a;
    // *p1 = 30;  // 错误：不能通过p1修改指向的数据
    p1 = &b;     // 可以：指针本身可以改

    // 形式2：常量指针（指针不可改，数据可改）
    int* const p2 = &a;
    *p2 = 30;    // 可以：可以修改指向的数据
    // p2 = &b;  // 错误：指针本身不可改

    // 形式3：指向常量的常量指针（都不可改）
    const int* const p3 = &a;
    // *p3 = 40;  // 错误
    // p3 = &b;   // 错误

    // 从右往左读的技巧
    // const int* p  → p is a pointer to const int
    // int* const p  → p is a const pointer to int

    cout << "p1指向: " << *p1 << endl;
    cout << "p2指向: " << *p2 << endl;
    cout << "p3指向: " << *p3 << endl;

    return 0;
}
```

输出：
```
p1指向: 20
p2指向: 30
p3指向: 30
```

### 2.3 const引用

```cpp
#include <iostream>
#include <string>
using namespace std;

// const引用参数：避免拷贝，同时防止修改
void printInfo(const string& name, const int& age) {
    // name = "修改";  // 编译错误：不能修改const引用
    cout << "姓名: " << name << ", 年龄: " << age << endl;
}

// 返回const引用避免拷贝
const string& getMax(const string& a, const string& b) {
    return (a > b) ? a : b;
}

int main() {
    string name = "张三";
    int age = 25;

    // const引用可以绑定到临时值
    const int& ref = 42;  // 延长临时对象的生命周期
    cout << "const引用绑定字面量: " << ref << endl;

    // const引用可以绑定到不同类型的值
    double pi = 3.14;
    const int& intRef = pi;  // 绑定到临时int值
    cout << "const引用绑定转换值: " << intRef << endl;

    printInfo(name, age);

    return 0;
}
```

输出：
```
const引用绑定字面量: 42
const引用绑定转换值: 3
姓名: 张三, 年龄: 25
```

### 2.4 const成员函数

```cpp
#include <iostream>
#include <string>
using namespace std;

class Circle {
private:
    double radius;
    mutable int accessCount = 0;  // mutable允许在const函数中修改

public:
    Circle(double r) : radius(r) {}

    // const成员函数：不能修改成员变量（mutable除外）
    double getArea() const {
        accessCount++;  // mutable成员可以修改
        return 3.14159 * radius * radius;
    }

    double getRadius() const {
        return radius;
    }

    // 非const成员函数：可以修改成员变量
    void setRadius(double r) {
        radius = r;
    }

    int getAccessCount() const { return accessCount; }
};

void printCircle(const Circle& c) {
    // 只能调用const成员函数
    cout << "半径: " << c.getRadius()
         << ", 面积: " << c.getArea() << endl;
    // c.setRadius(10);  // 错误：不能在const对象上调用非const函数
}

int main() {
    Circle c(5.0);

    printCircle(c);
    c.setRadius(7.0);
    printCircle(c);

    cout << "访问次数: " << c.getAccessCount() << endl;

    return 0;
}
```

输出：
```
半径: 5, 面积: 78.5398
半径: 7, 面积: 153.938
访问次数: 2
```

### 2.5 const与宏的对比

```cpp
#include <iostream>
using namespace std;

// #define方式（不推荐）
#define MAX_SIZE_DEFINE 1024

// const方式（推荐）
const int MAX_SIZE_CONST = 1024;
constexpr int MAX_SIZE_CONSTEXPR = 1024;  // C++11编译期常量

int main() {
    // const优势：有类型、有作用域、可调试
    int arr1[MAX_SIZE_DEFINE];   // OK
    int arr2[MAX_SIZE_CONST];    // OK

    // const在调试器中可见，宏不可见
    cout << "宏定义: " << MAX_SIZE_DEFINE << endl;
    cout << "const: " << MAX_SIZE_CONST << endl;
    cout << "constexpr: " << MAX_SIZE_CONSTEXPR << endl;

    return 0;
}
```

输出：
```
宏定义: 1024
const: 1024
constexpr: 1024
```

## 三、注意事项与常见陷阱

1. **const必须初始化**：`const int x;`是错误的，必须在声明时给出初始值
2. **const指针从右往左读**：`const int* p`是指向const int的指针，`int* const p`是const指针
3. **const成员函数重载**：同一个函数可以有const和非const两个版本
4. **const正确性**：函数不需要修改参数时，参数应声明为`const`，这是重要的编程习惯
5. **mutable关键字**：允许成员在const函数中被修改，用于缓存等场景
