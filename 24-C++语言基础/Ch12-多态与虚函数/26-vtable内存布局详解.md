# vtable内存布局详解

## 一、概念说明

vtable是编译器为每个含虚函数的类生成的**静态函数指针表**（C++标准 §10.3，实现定义）。每个对象通过vptr指向其对应类的vtable。多继承时可能有多个vptr/vtable。vtable布局是编译器实现细节，但了解它有助于理解多态的开销和优化。

### 1.1 vtable内容

| 条目 | 说明 |
|------|------|
| RTTI信息 | `type_info`指针 |
| 虚函数地址 | 按声明顺序 |
| 虚基类偏移 | 虚继承时 |
| offset-to-top | 多继承时的偏移调整 |

## 二、具体用法

### 2.1 单继承vtable布局

```cpp
#include <iostream>

class Base {
public:
    virtual void f() { std::cout << "Base::f" << std::endl; }
    virtual void g() { std::cout << "Base::g" << std::endl; }
    virtual ~Base() = default;
    int baseData = 1;
};

class Derived : public Base {
public:
    void f() override { std::cout << "Derived::f" << std::endl; }
    void g() override { std::cout << "Derived::g" << std::endl; }
    virtual void h() { std::cout << "Derived::h" << std::endl; }
    int derivedData = 2;
};

int main() {
    Derived d;
    std::cout << "对象大小: " << sizeof(d) << " 字节" << std::endl;

    // 单继承：只有一个vptr
    // vtable布局：[RTTI, &Derived::f, &Derived::g, &Derived::h, &~Derived]

    void** vtable = *reinterpret_cast<void***>(&d);
    using Func = void(*)();

    std::cout << "vtable[0] (RTTI信息，非函数): " << vtable[0] << std::endl;
    std::cout << "vtable[1]: "; reinterpret_cast<Func>(vtable[1])();  // f
    std::cout << "vtable[2]: "; reinterpret_cast<Func>(vtable[2])();  // g
    std::cout << "vtable[3]: "; reinterpret_cast<Func>(vtable[3])();  // h

    return 0;
}
```

### 2.2 多继承vtable布局

```cpp
#include <iostream>

class A {
public:
    virtual void aFunc() { std::cout << "A::aFunc" << std::endl; }
    virtual ~A() = default;
    int a_val = 1;
};

class B {
public:
    virtual void bFunc() { std::cout << "B::bFunc" << std::endl; }
    virtual ~B() = default;
    int b_val = 2;
};

class C : public A, public B {
public:
    void aFunc() override { std::cout << "C::aFunc" << std::endl; }
    void bFunc() override { std::cout << "C::bFunc" << std::endl; }
    int c_val = 3;
};

int main() {
    C c;
    std::cout << "sizeof(A) = " << sizeof(A) << std::endl;  // vptr + int + padding
    std::cout << "sizeof(B) = " << sizeof(B) << std::endl;
    std::cout << "sizeof(C) = " << sizeof(C) << std::endl;  // A + B + int

    // 多继承：两个vptr
    // C的内存布局：[A子对象: vptr_A, a_val] [B子对象: vptr_B, b_val] [c_val]

    A* pa = &c;
    B* pb = &c;

    std::cout << "&c  = " << (void*)&c << std::endl;
    std::cout << "pa  = " << (void*)pa << std::endl;   // A子对象起始
    std::cout << "pb  = " << (void*)pb << std::endl;   // B子对象起始（不同！）

    pa->aFunc();  // C::aFunc（通过A的vtable）
    pb->bFunc();  // C::bFunc（通过B的vtable）

    return 0;
}
```

### 2.3 sizeof分析

```cpp
#include <iostream>

struct NoVirt { int x, y; };
struct OneVirt { virtual void f() {} int x, y; };
struct MultiVirt { virtual void f() {} virtual void g() {} int x, y; };
struct MultiInherit : NoVirt { virtual void f() {} };

int main() {
    std::cout << "NoVirt:     " << sizeof(NoVirt) << std::endl;      // 8
    std::cout << "OneVirt:    " << sizeof(OneVirt) << std::endl;     // 16 (vptr=8 + 8)
    std::cout << "MultiVirt:  " << sizeof(MultiVirt) << std::endl;   // 16 (一个vptr)
    std::cout << "MultiInherit: " << sizeof(MultiInherit) << std::endl; // 16

    // 关键结论：
    // 1. 无论多少虚函数，只增加一个vptr
    // 2. vptr大小 = 指针大小（64位系统=8字节）
    // 3. vtable本身只在类级别存在一份，不在每个对象中

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **vtable布局是编译器实现细节**：不同编译器（MSVC/GCC/Clang）可能不同
2. **vptr通常位于对象起始位置**：方便编译器生成代码
3. **vtable包含虚函数地址和RTTI信息**：不能只关注函数地址
4. **多继承中每个含虚函数的基类子对象有自己的vptr**：增加对象大小
5. **`final`允许编译器内联虚调用（devirtualization）**：消除查表
6. **虚继承的vtable更复杂**：增加虚基类偏移信息
