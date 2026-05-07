# 类型转换 static_cast

## 一、概念说明

`static_cast`是C++中最常用的类型转换运算符，在**编译时**执行类型检查。它用于相关的类型之间的转换，比C风格转换更安全、意图更明确。

## 二、具体用法

### 2.1 基本数值转换

```cpp
#include <iostream>
using namespace std;

int main() {
    // 整型与浮点型转换
    double pi = 3.14159;
    int truncated = static_cast<int>(pi);    // 截断小数部分
    cout << "3.14159 -> int: " << truncated << endl;

    // 整型之间的转换
    int big = 300;
    char small = static_cast<char>(big);     // 可能溢出
    cout << "300 -> char: " << (int)small << endl;  // 溢出

    // enum与int
    enum Color { Red, Green, Blue };
    int colorVal = static_cast<int>(Green);
    Color c = static_cast<Color>(2);
    cout << "Green -> int: " << colorVal << endl;
    cout << "2 -> Color: " << c << endl;

    // void* 与其他指针类型
    int x = 42;
    void* vp = &x;
    int* ip = static_cast<int*>(vp);
    cout << "void* -> int*: " << *ip << endl;

    return 0;
}
```

输出：
```
3.14159 -> int: 3
300 -> char: 44
Green -> int: 1
2 -> Color: 2
void* -> int*: 42
```

### 2.2 类层次结构中的转换

```cpp
#include <iostream>
using namespace std;

class Base {
public:
    virtual void show() { cout << "Base::show()" << endl; }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void show() override { cout << "Derived::show()" << endl; }
    void extra() { cout << "Derived::extra()" << endl; }
};

int main() {
    Derived d;
    Base* bp = &d;

    // 向上转型（派生类→基类）：安全，隐式即可
    Base& br = d;

    // 向下转型（基类→派生类）：需要显式转换
    // 注意：static_cast不做运行时检查
    Derived* dp = static_cast<Derived*>(bp);
    dp->show();
    dp->extra();

    // 水平转型（兄弟类之间）：不安全
    // static_cast不允许不相关的类之间转换

    // const转换（用const_cast更明确）
    const int cx = 42;
    int* px = const_cast<int*>(&cx);  // 用const_cast

    return 0;
}
```

输出：
```
Derived::show()
Derived::extra()
```

### 2.3 运算符转换

```cpp
#include <iostream>
using namespace std;

class Fraction {
    int num, den;
public:
    Fraction(int n, int d) : num(n), den(d) {}

    // 类型转换运算符
    explicit operator double() const {
        return static_cast<double>(num) / den;
    }

    explicit operator int() const {
        return num / den;
    }
};

int main() {
    Fraction f(3, 4);

    // 使用static_cast调用转换运算符
    double d = static_cast<double>(f);
    int i = static_cast<int>(f);

    cout << "3/4 as double: " << d << endl;
    cout << "3/4 as int: " << i << endl;

    // 普通数值转换
    long long ll = 123456789012345LL;
    int narrowed = static_cast<int>(ll);
    cout << "long long -> int: " << narrowed << endl;

    return 0;
}
```

输出：
```
3/4 as double: 0.75
3/4 as int: 0
long long -> int: 123456789012345（取决于实现，可能截断）
```

### 2.4 static_cast的安全性

```cpp
#include <iostream>
using namespace std;

int main() {
    // static_cast在编译时检查类型关系

    // 合法转换
    int a = static_cast<int>(3.14);           // OK
    void* vp = static_cast<void*>(&a);        // OK
    int* ip = static_cast<int*>(vp);          // OK

    // 编译时错误
    // int* p = static_cast<int*>(new double(3.14));  // 错误：不相关类型
    // string s = static_cast<string>(42);            // 错误

    // 运行时不安全但编译通过的转换
    Base* b = nullptr;
    // Derived* d = static_cast<Derived*>(b);  // 编译通过但运行时危险

    cout << "转换完成" << endl;

    return 0;
}
```

输出：
```
转换完成
```

## 三、注意事项与常见陷阱

1. **不检查运行时类型**：向下转型时如果基类指针实际不指向派生类对象，`static_cast`不会报错，但访问派生类成员是未定义行为
2. **数值溢出不报错**：大类型转小类型可能截断，`static_cast`不提供溢出检查
3. **禁止不相关类型**：无法在无关的指针/引用类型之间转换（如`int*`转`double*`）
4. **优先使用static_cast**：比C风格转换更安全，编译器会检查类型关系
5. **向下转型用dynamic_cast**：如果需要运行时类型检查，使用`dynamic_cast`
