# final关键字

## 一、概念说明

C++11引入`final`关键字（C++标准 §10.3/§9），可以修饰**类**（禁止被继承）和**虚函数**（禁止被重写）。`final`帮助编译器进行**去虚化**（devirtualization）优化，并增强设计约束，防止不合理的扩展。

### 1.1 final的两种用途

| 修饰对象 | 语法 | 效果 | 编译器优化 |
|----------|------|------|-----------|
| 类 | `class X final {}` | 禁止被继承 | 可将虚调用变为直接调用 |
| 虚函数 | `void f() final` | 禁止被重写 | 同上 |

## 二、具体用法

### 2.1 final类

```cpp
#include <iostream>

// final类：不能被继承
class Widget final {
public:
    virtual void show() { std::cout << "Widget::show()" << std::endl; }
    virtual void render() { std::cout << "Widget::render()" << std::endl; }
    virtual ~Widget() = default;
};

// 编译错误！Widget是final
// class SpecialWidget : public Widget {};

int main() {
    Widget w;
    w.show();
    w.render();

    // 编译器知道Widget没有派生类，
    // 可以将虚调用优化为直接调用
    Widget* p = &w;
    p->show();   // 可被devirtualize
    p->render(); // 可被devirtualize

    return 0;
}
```

### 2.2 final虚函数

```cpp
#include <iostream>

class Base {
public:
    virtual void process() final {  // 不能被重写
        std::cout << "Base::process()" << std::endl;
    }
    virtual void handle() {  // 可以被重写
        std::cout << "Base::handle()" << std::endl;
    }
    virtual void init() {
        std::cout << "Base::init()" << std::endl;
    }
};

class Middle : public Base {
    // void process() override {}  // 编译错误！process是final

    void handle() override {  // OK
        std::cout << "Middle::handle()" << std::endl;
    }
    void init() final {  // Middle禁止进一步重写init
        std::cout << "Middle::init()" << std::endl;
    }
};

class Leaf : public Middle {
    void handle() override {  // OK：handle不是final
        std::cout << "Leaf::handle()" << std::endl;
    }
    // void init() override {}  // 编译错误！init在Middle中是final
    // void process() override {}  // 编译错误！process在Base中是final
};

int main() {
    Leaf leaf;
    Base* pb = &leaf;

    pb->process();  // Base::process() — final保证
    pb->handle();   // Leaf::handle()
    pb->init();     // Middle::init()

    return 0;
}
```

### 2.3 final与编译器优化

```cpp
#include <iostream>

// final类允许devirtualization
class Sealed final {
public:
    virtual int compute() const { return 42; }
    virtual ~Sealed() = default;
};

// 非final类，编译器不确定是否有派生类
class Open {
public:
    virtual int compute() const { return 42; }
    virtual ~Open() = default;
};

void testSealed(Sealed* s) {
    // 编译器知道Sealed没有派生类
    // s->compute() 可被优化为直接调用
    int result = s->compute();
}

void testOpen(Open* o) {
    // 编译器不确定Open是否有派生类
    // o->compute() 通常保留为虚调用
    int result = o->compute();
}
```

## 三、注意事项与常见陷阱

1. **`final`类不能作为基类**：但可以正常使用其所有功能
2. **`final`虚函数在当前类仍可实现**：只是禁止派生类重写
3. **`final`允许编译器devirtualization**：消除虚函数调用开销
4. **`final`应谨慎使用**：过度使用会限制合理的扩展性
5. **`final`和`override`可以组合使用**：`void f() override final`
6. **`final`不是关键字**：是标识符的特殊上下文用法（技术上可用作变量名，但不应这样做）
