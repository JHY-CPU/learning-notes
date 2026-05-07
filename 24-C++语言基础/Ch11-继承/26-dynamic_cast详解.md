# dynamic_cast详解

## 一、概念说明

`dynamic_cast`用于**安全的向下转型**（downcast）和**交叉转型**（crosscast）。它在运行时检查类型是否合法，失败时返回`nullptr`（指针）或抛出`std::bad_cast`（引用）。

## 二、具体用法

### 2.1 安全向下转型

```cpp
#include <iostream>

class Base {
public:
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void derivedOnly() { std::cout << "Derived特有方法" << std::endl; }
};

class Other : public Base {};

int main() {
    Base* b1 = new Derived();
    Base* b2 = new Other();

    // 安全向下转型
    Derived* d1 = dynamic_cast<Derived*>(b1);
    if (d1) d1->derivedOnly();  // OK

    Derived* d2 = dynamic_cast<Derived*>(b2);
    if (!d2) std::cout << "b2不是Derived类型" << std::endl;

    delete b1;
    delete b2;
    return 0;
}
```

**输出：**
```
Derived特有方法
b2不是Derived类型
```

### 2.2 交叉转型

```cpp
#include <iostream>

class A { public: virtual ~A() = default; };
class B { public: virtual ~B() = default; };
class C : public A, public B {};

int main() {
    C* c = new C();
    A* a = c;                      // 向上转型
    B* b = dynamic_cast<B*>(a);    // 交叉转型：A* → B*
    if (b) std::cout << "交叉转型成功" << std::endl;
    delete c;
    return 0;
}
```

**输出：**
```
交叉转型成功
```

## 三、注意事项与常见陷阱

- `dynamic_cast`需要虚函数（运行时类型检查依赖RTTI）
- 转型失败：指针返回`nullptr`，引用抛出`std::bad_cast`
- 交叉转型只在多继承中需要
- `dynamic_cast`有运行时开销，频繁调用影响性能
- 优先使用虚函数多态避免`dynamic_cast`
