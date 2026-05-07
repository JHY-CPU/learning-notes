# 虚函数表（vtable）

## 一、概念说明

**虚函数表**（vtable）是编译器为含虚函数的类生成的静态函数指针数组。每个对象包含一个指向vtable的指针（vptr）。调用虚函数时通过vptr查vtable找到实际函数地址。

### 1.1 内存布局

```
对象:  [vptr] → [vtable]
                  [0]: &Derived::foo
                  [1]: &Derived::bar
                  [2]: &Derived::~Derived
       [成员1]
       [成员2]
```

## 二、具体用法

### 2.1 vptr和vtable验证

```cpp
#include <iostream>

class Base {
public:
    virtual void f() { std::cout << "Base::f" << std::endl; }
    virtual void g() { std::cout << "Base::g" << std::endl; }
    int data = 42;
};

class Derived : public Base {
public:
    void f() override { std::cout << "Derived::f" << std::endl; }
    void g() override { std::cout << "Derived::g" << std::endl; }
};

int main() {
    Base b;
    Derived d;

    // 对象的第一个成员是vptr
    std::cout << "sizeof(Base) = " << sizeof(b) << std::endl;
    // 包含vptr(8字节) + data(4字节) + 对齐

    // 通过函数指针调用验证vtable
    using FuncPtr = void (*)();
    // 对象地址即vptr地址
    void** vptr = *reinterpret_cast<void***>(&d);
    FuncPtr f = reinterpret_cast<FuncPtr>(vptr[0]);
    f();  // 输出Derived::f
    return 0;
}
```

**输出（示例）：**
```
sizeof(Base) = 16
Derived::f
```

## 三、注意事项与常见陷阱

- vptr在构造过程中逐步设置（构造期间虚函数非多态）
- 每个含虚函数的类至少增加一个vptr的大小（通常8字节）
- vtable是只读的，存储在程序的只读数据段
- 多继承可能有多个vptr
- `final`类的虚调用可以被devirtualize（消除查表）
