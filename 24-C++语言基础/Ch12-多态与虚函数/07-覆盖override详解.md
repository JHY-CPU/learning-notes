# 覆盖（override）详解

## 一、概念说明

**覆盖**（Override，C++标准 §10.3）指派生类重新定义基类的虚函数，实现运行时多态。重写要求函数签名完全匹配：参数类型、const/volatile限定符、引用限定符。返回类型可以是协变的。

### 1.1 覆盖 vs 隐藏

| 特性 | 覆盖（Override） | 隐藏（Hide） |
|------|-----------------|-------------|
| 函数类型 | 虚函数 | 任何函数 |
| 签名要求 | 完全匹配 | 只需同名 |
| 绑定方式 | 动态绑定 | 静态绑定 |
| 基类同名函数 | 仍可通过基类调用 | 被完全遮蔽 |
| 使用`override` | 标记 | 不适用 |

## 二、具体用法

### 2.1 完整签名匹配

```cpp
#include <iostream>

class Base {
public:
    virtual void func(int) const { std::cout << "Base" << std::endl; }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    // 必须签名完全匹配：参数+const
    void func(int) const override {
        std::cout << "Derived" << std::endl;
    }

    // 以下都会编译错误（有override时）：
    // void func(double) const override {}        // 参数类型不同
    // void func(int) override {}                  // 缺少const
    // void func(int) const volatile override {}   // 多了volatile
};

int main() {
    Derived d;
    Base* b = &d;
    b->func(42);  // Derived
    return 0;
}
```

### 2.2 引用限定符覆盖

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
    void process() & override {
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

### 2.3 隐藏与覆盖的对比

```cpp
#include <iostream>

class Base {
public:
    virtual void foo(int) { std::cout << "Base::foo(int)" << std::endl; }
    void bar(int) { std::cout << "Base::bar(int)" << std::endl; }
    virtual ~Base() = default;
};

class Derived : public Base {
public:
    void foo(double) {  // 隐藏Base::foo(int)！不是override
        std::cout << "Derived::foo(double)" << std::endl;
    }
    void bar(double) {  // 隐藏Base::bar(int)
        std::cout << "Derived::bar(double)" << std::endl;
    }
};

int main() {
    Derived d;
    Base* p = &d;

    p->foo(42);    // Base::foo(int) — Derived::foo(double)是隐藏不是覆盖
    p->bar(42);    // Base::bar(int) — 静态绑定

    d.foo(42);     // Derived::foo(double) — 隐藏，int提升为double
    // d.foo(42) 如果想调用Base::foo(int)，需要 d.Base::foo(42)

    return 0;
}
```

## 三、注意事项与常见陷阱

1. **始终使用`override`关键字标记重写的虚函数**：现代C++最佳实践
2. **`const`和引用限定符必须匹配**：这些是签名的一部分
3. **返回类型可以是协变的**：派生类返回`Derived*`替代`Base*`
4. **没有`override`时签名不匹配只会隐藏而非报错**：极难发现的bug
5. **函数遮蔽（hiding）和覆盖（overriding）不同**：遮蔽是名称查找行为
6. **`override`不是关键字**：是标识符的特殊上下文，但应视为关键字
