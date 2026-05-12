# override关键字

## 一、概念说明

C++11引入`override`关键字（C++标准 §10.3）显式标记虚函数的重写。如果签名不匹配基类的任何虚函数，编译器会报错。这避免了"本想重写实际是隐藏"的经典bug——在没有`override`时，签名不匹配只会导致函数隐藏而非错误。

### 1.1 override的作用

| 场景 | 无override | 有override |
|------|-----------|-----------|
| 签名匹配 | 编译通过，多态 | 编译通过，多态 |
| 签名不匹配 | 编译通过，**隐藏**（bug） | **编译错误**（捕获bug） |
| const不匹配 | 编译通过，隐藏 | 编译错误 |
| 返回类型不协变 | 编译通过，隐藏 | 编译错误 |

## 二、具体用法

### 2.1 override防止签名错误

```cpp
#include <iostream>

class Base {
public:
    virtual void foo(int x) {
        std::cout << "Base::foo(int " << x << ")" << std::endl;
    }
    virtual void bar() const {
        std::cout << "Base::bar() const" << std::endl;
    }
    virtual int getValue() const { return 0; }
    virtual ~Base() = default;
};

class DerivedGood : public Base {
public:
    // 正确：签名完全匹配
    void foo(int x) override {
        std::cout << "DerivedGood::foo(" << x << ")" << std::endl;
    }
    void bar() const override {
        std::cout << "DerivedGood::bar() const" << std::endl;
    }
    int getValue() const override { return 42; }
};

class DerivedBad : public Base {
public:
    // 以下如果没有override会编译通过但行为错误：

    // void foo(double x) override {}  // 编译错误！参数类型不匹配
    // void bar() override {}           // 编译错误！缺少const
    // double getValue() const override { return 3.14; }  // 编译错误！返回类型不协变

    // 没有override时的隐藏bug：
    void foo(double x) {  // 隐藏了Base::foo(int)，不是重写
        std::cout << "DerivedBad::foo(double)" << std::endl;
    }
};

int main() {
    DerivedGood dg;
    Base* pb = &dg;
    pb->foo(42);      // DerivedGood::foo(42)
    pb->bar();        // DerivedGood::bar() const

    DerivedBad db;
    Base* pb2 = &db;
    pb2->foo(42);     // Base::foo(int) — DerivedBad的foo(double)是隐藏不是重写！

    return 0;
}
```

### 2.2 override与引用限定符

```cpp
#include <iostream>

class Base {
public:
    virtual void process() & {  // 左值限定
        std::cout << "Base::process() &" << std::endl;
    }
    virtual void process() && {  // 右值限定
        std::cout << "Base::process() &&" << std::endl;
    }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void process() & override {  // 必须匹配引用限定符
        std::cout << "Derived::process() &" << std::endl;
    }
    void process() && override {
        std::cout << "Derived::process() &&" << std::endl;
    }
};

int main() {
    Derived d;
    Base& ref = d;
    ref.process();           // Derived::process() &

    Base&& rref = Derived();
    rref.process();          // Derived::process() &&

    return 0;
}
```

### 2.3 override与协变返回类型

```cpp
#include <iostream>

class Base {
public:
    virtual Base* clone() const {
        std::cout << "Base::clone()" << std::endl;
        return new Base(*this);
    }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    // 协变返回类型：Derived* 是 Base* 的派生类
    Derived* clone() const override {
        std::cout << "Derived::clone()" << std::endl;
        return new Derived(*this);
    }
};

int main() {
    Derived d;
    Derived* copy = d.clone();  // 返回Derived*，无需转换
    delete copy;

    Base* bp = &d;
    Base* copy2 = bp->clone();  // 返回Derived*，自动转为Base*
    delete copy2;

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **始终使用`override`标记重写的虚函数**：这是现代C++的最佳实践
2. **`override`不是关键字**：是标识符的特殊上下文技术，但应视为关键字
3. **签名必须完全匹配**：参数类型、const/volatile、引用限定符
4. **返回类型需协变才能构成有效重写**：派生类返回类型必须是基类返回类型的派生
5. **没有`override`时签名不匹配只会导致隐藏**：非常难以发现的bug
6. **`override`和`final`可组合使用**：`void f() override final`
