# vtable内存布局详解

## 一、概念说明

vtable是编译器为每个含虚函数的类生成的**静态函数指针表**。每个对象通过vptr指向其对应类的vtable。多继承时可能有多个vptr/vtable。

## 二、具体用法

### 2.1 查看vtable布局

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
    virtual void h() { std::cout << "Derived::h" << std::endl; }
    int derivedData = 2;
};

int main() {
    Derived d;
    std::cout << "对象大小: " << sizeof(d) << " 字节" << std::endl;
    // 通常: vptr(8) + baseData(4) + derivedData(4) + padding = 24

    // 通过vptr调用函数
    void** vtable = *reinterpret_cast<void***>(&d);
    using Func = void(*)();

    std::cout << "vtable[0]: "; reinterpret_cast<Func>(vtable[0])();  // f
    std::cout << "vtable[1]: "; reinterpret_cast<Func>(vtable[1])();  // g
    std::cout << "vtable[2]: "; reinterpret_cast<Func>(vtable[2])();  // h
    return 0;
}
```

**输出（示例）：**
```
对象大小: 16 字节
vtable[0]: Derived::f
vtable[1]: Base::g
vtable[2]: Derived::h
```

## 三、注意事项与常见陷阱

- vtable布局是编译器实现细节，不同编译器不同
- vptr通常位于对象起始位置
- vtable包含虚函数地址和RTTI信息
- 多继承中每个含虚函数的基类子对象有自己的vptr
- `final`允许编译器内联虚调用（devirtualization）
