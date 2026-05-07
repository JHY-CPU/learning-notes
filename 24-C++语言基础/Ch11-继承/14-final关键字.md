# final关键字

## 一、概念说明

C++11引入`final`关键字，可以修饰**类**（禁止被继承）和**虚函数**（禁止被重写）。`final`帮助编译器进行优化（devirtualization）并增强设计约束。

## 二、具体用法

### 2.1 final类

```cpp
#include <iostream>

class Widget final {  // 不能被继承
public:
    void show() { std::cout << "Widget::show()" << std::endl; }
};

// class SpecialWidget : public Widget {};  // 编译错误！Widget是final

int main() {
    Widget w;
    w.show();
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
    virtual void handle() {
        std::cout << "Base::handle()" << std::endl;
    }
};

class Derived : public Base {
    // void process() override {}  // 编译错误！process是final
    void handle() override {  // OK
        std::cout << "Derived::handle()" << std::endl;
    }
};

int main() {
    Derived d;
    d.process();
    d.handle();
    return 0;
}
```

**输出：**
```
Base::process()
Derived::handle()
```

## 三、注意事项与常见陷阱

- `final`类不能作为基类，但可以正常使用
- `final`虚函数在当前类仍可实现，只是不能被派生类重写
- `final`允许编译器进行devirtualization优化
- `final`应谨慎使用，过度使用会限制扩展性
