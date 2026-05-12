# 虚函数表（vtable）

## 一、概念说明

**虚函数表**（vtable, virtual table）是编译器为含虚函数的类生成的静态函数指针数组。每个对象包含一个指向vtable的指针（vptr）。调用虚函数时通过vptr查vtable找到实际函数地址，实现运行时多态。

### 1.1 内存布局

```
对象:  [vptr] → [vtable]
                  [0]: &Derived::foo
                  [1]: &Derived::bar
                  [2]: &Derived::~Derived
       [成员1]
       [成员2]
```

### 1.2 vtable组成

| 项目 | 说明 |
|------|------|
| 虚函数指针 | 按声明顺序排列 |
| RTTI指针 | `type_info`信息 |
| 偏移量 | 多继承中的偏移调整 |
| 虚基类偏移 | 虚继承中的定位信息 |

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

    // 验证vptr位于对象起始位置
    void** vptr_b = *reinterpret_cast<void***>(&b);
    void** vptr_d = *reinterpret_cast<void***>(&d);

    using FuncPtr = void (*)();

    // 通过vtable调用虚函数
    std::cout << "Derived的vtable[0]: ";
    reinterpret_cast<FuncPtr>(vptr_d[0])();  // Derived::f

    std::cout << "Derived的vtable[1]: ";
    reinterpret_cast<FuncPtr>(vptr_d[1])();  // Derived::g

    return 0;
}
```

### 2.2 vtable在继承中的变化

```cpp
#include <iostream>

class Base {
public:
    virtual void f() { std::cout << "Base::f" << std::endl; }
    int base_val = 1;
};

class Middle : public Base {
public:
    void f() override { std::cout << "Middle::f" << std::endl; }
    virtual void g() { std::cout << "Middle::g" << std::endl; }  // 新增虚函数
    int mid_val = 2;
};

class Derived : public Middle {
public:
    void f() override { std::cout << "Derived::f" << std::endl; }
    void g() override { std::cout << "Derived::g" << std::endl; }
    virtual void h() { std::cout << "Derived::h" << std::endl; }  // 新增
    int der_val = 3;
};

int main() {
    std::cout << "sizeof(Base) = " << sizeof(Base) << std::endl;
    std::cout << "sizeof(Middle) = " << sizeof(Middle) << std::endl;
    std::cout << "sizeof(Derived) = " << sizeof(Derived) << std::endl;

    // Derived的vtable: [Derived::f, Derived::g, Derived::h, ~Derived]
    // 只有一份vptr（单继承）

    Derived d;
    void** vptr = *reinterpret_cast<void***>(&d);
    using FuncPtr = void (*)();

    std::cout << "vtable[0]: "; reinterpret_cast<FuncPtr>(vptr[0])();  // f
    std::cout << "vtable[1]: "; reinterpret_cast<FuncPtr>(vptr[1])();  // g
    std::cout << "vtable[2]: "; reinterpret_cast<FuncPtr>(vptr[2])();  // h

    return 0;
}
```

### 2.3 sizeof的影响

```cpp
#include <iostream>

struct NoVirtual {
    int x, y;
};

struct WithVirtual {
    virtual void f() {}
    int x, y;
};

struct MultiVirtual {
    virtual void f() {}
    virtual void g() {}
    virtual void h() {}
    int x, y;
};

int main() {
    std::cout << "NoVirtual: " << sizeof(NoVirtual) << std::endl;    // 8
    std::cout << "WithVirtual: " << sizeof(WithVirtual) << std::endl; // 16 (vptr=8 + 8)
    std::cout << "MultiVirtual: " << sizeof(MultiVirtual) << std::endl; // 16 (vptr还是一个)
    // 多个虚函数只增加一个vptr，vtable中多个条目
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **vptr在构造过程中逐步设置**：构造期间虚函数非多态，调用当前类版本
2. **每个含虚函数的类至少增加一个vptr的大小**：通常8字节（64位系统）
3. **vtable是只读的**：存储在程序的只读数据段
4. **多继承可能有多个vptr**：每个含虚函数的基类子对象一个
5. **`final`类的虚调用可以被devirtualize**：编译器优化消除查表
6. **vtable布局是编译器实现细节**：不同编译器可能不同
