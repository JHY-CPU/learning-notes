# override关键字

## 一、概念说明

C++11引入`override`关键字显式标记虚函数的重写。如果签名不匹配基类的虚函数，编译器会报错，避免"本想重写实际是隐藏"的bug。

## 二、具体用法

### 2.1 override防止错误

```cpp
#include <iostream>

class Base {
public:
    virtual void foo(int x) { std::cout << "Base::foo(" << x << ")" << std::endl; }
    virtual void bar() const { std::cout << "Base::bar()" << std::endl; }
};

class Derived : public Base {
public:
    // 正确重写
    void foo(int x) override {
        std::cout << "Derived::foo(" << x << ")" << std::endl;
    }

    // const不匹配 → 编译错误（有override时）
    // void bar() override {}  // 编译错误！

    // 正确：签名完全匹配
    void bar() const override {
        std::cout << "Derived::bar()" << std::endl;
    }
};

int main() {
    Derived d;
    Base* b = &d;
    b->foo(42);
    b->bar();
    return 0;
}
```

**输出：**
```
Derived::foo(42)
Derived::bar()
```

## 三、注意事项与常见陷阱

- **始终**使用`override`标记重写的虚函数
- `override`不是关键字（在其他上下文中可作标识符），但应视为关键字使用
- 签名必须完全匹配：参数类型、const/volatile、引用限定符
- 返回类型需协变（Covariant）才能构成有效重写
- 没有`override`时签名不匹配只会导致隐藏而非错误
