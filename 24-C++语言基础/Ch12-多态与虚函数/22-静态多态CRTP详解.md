# 静态多态CRTP详解

## 一、概念说明

**CRTP**（Curiously Recurring Template Pattern）通过模板实现编译期多态，避免虚函数的运行时开销。基类通过`static_cast<Derived*>(this)`调用派生类方法。CRTP是C++中实现静态多态和Mixin-in的核心技术。

### 1.1 CRTP基本结构

```cpp
template<typename Derived>
class Base {
public:
    void interface() {
        static_cast<Derived*>(this)->implementation();  // 编译期绑定
    }
};

class Derived : public Base<Derived> {  // 将自己作为模板参数
public:
    void implementation() { /* 具体实现 */ }
};
```

## 二、具体用法

### 2.1 CRTP实现静态多态

```cpp
#include <iostream>

template<typename Derived>
class Shape {
public:
    double area() const {
        return static_cast<const Derived*>(this)->areaImpl();
    }
    void describe() const {
        std::cout << "面积: " << area() << std::endl;
    }
};

class Circle : public Shape<Circle> {
    double r;
public:
    Circle(double r) : r(r) {}
    double areaImpl() const { return 3.14159 * r * r; }
};

class Rectangle : public Shape<Rectangle> {
    double w, h;
public:
    Rectangle(double w, double h) : w(w), h(h) {}
    double areaImpl() const { return w * h; }
};

// 通用函数模板
template<typename T>
void printArea(const Shape<T>& shape) {
    shape.describe();  // 编译期绑定
}

int main() {
    Circle c(5);
    Rectangle r(3, 4);
    printArea(c);
    printArea(r);
    return 0;
}
```

### 2.2 CRTP实现Mixin-in

```cpp
#include <iostream>
#include <string>

template<typename Derived>
class Printable {
public:
    void print() const {
        std::cout << static_cast<const Derived*>(this)->toString() << std::endl;
    }
};

template<typename Derived>
class Comparable {
public:
    bool operator>(const Derived& rhs) const {
        return static_cast<const Derived*>(this)->compareTo(rhs) > 0;
    }
    bool operator>=(const Derived& rhs) const {
        return static_cast<const Derived*>(this)->compareTo(rhs) >= 0;
    }
    bool operator<(const Derived& rhs) const {
        return static_cast<const Derived*>(this)->compareTo(rhs) < 0;
    }
    bool operator<=(const Derived& rhs) const {
        return static_cast<const Derived*>(this)->compareTo(rhs) <= 0;
    }
    bool operator==(const Derived& rhs) const {
        return static_cast<const Derived*>(this)->compareTo(rhs) == 0;
    }
};

class Number : public Printable<Number>, public Comparable<Number> {
    int value;
public:
    Number(int v) : value(v) {}
    std::string toString() const { return "Number(" + std::to_string(value) + ")"; }
    int compareTo(const Number& o) const { return value - o.value; }
};

int main() {
    Number a(10), b(5), c(10);
    a.print();  // Number(10)
    std::cout << "a > b: " << (a > b) << std::endl;   // 1
    std::cout << "a == c: " << (a == c) << std::endl;  // 1
    std::cout << "b < a: " << (b < a) << std::endl;   // 1
    return 0;
}
```

### 2.3 CRTP计数器

```cpp
#include <iostream>

template<typename Derived>
class Counter {
    static int count;
    static int totalCount;
public:
    Counter() { ++count; ++totalCount; }
    ~Counter() { --count; }

    static int getInstanceCount() { return count; }
    static int getTotalCreated() { return totalCount; }
};

template<typename Derived>
int Counter<Derived>::count = 0;

template<typename Derived>
int Counter<Derived>::totalCount = 0;

class Widget : public Counter<Widget> {};
class Gadget : public Counter<Gadget> {};

int main() {
    Widget w1, w2;
    Gadget g1;

    std::cout << "Widget实例: " << Widget::getInstanceCount() << std::endl;  // 2
    std::cout << "Gadget实例: " << Gadget::getInstanceCount() << std::endl;  // 1
    std::cout << "Widget总创建: " << Widget::getTotalCreated() << std::endl; // 2

    {
        Widget w3;
        std::cout << "Widget实例: " << Widget::getInstanceCount() << std::endl; // 3
    }
    std::cout << "Widget实例: " << Widget::getInstanceCount() << std::endl;  // 2

    return 0;
}
```

### 2.4 CRTP实现链式调用

```cpp
#include <iostream>

template<typename Derived>
class Builder {
public:
    Derived& self() { return static_cast<Derived&>(*this); }

    Derived& withName(const std::string& n) {
        self().name = n;
        return self();
    }
};

class Config : public Builder<Config> {
public:
    std::string name;
    int value = 0;

    Config& withValue(int v) {
        value = v;
        return *this;
    }

    void show() const {
        std::cout << name << ": " << value << std::endl;
    }
};

int main() {
    Config cfg;
    cfg.withName("设置").withValue(42).show();
    // 设置: 42
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **CRTP不能放入异构容器**：`Shape<Circle>`和`Shape<Rectangle>`是不同类型
2. **派生类必须在使用前完整定义**：不能前向声明
3. **CRTP支持链式方法（Fluent API）**：`static_cast<Derived*>(this)`返回正确类型
4. **编译期绑定允许更好的内联优化**：消除虚函数开销
5. **CRTP可以与虚函数混合使用**：根据需要选择
6. **CRTP是C++20 Concepts的前驱**：约束模板参数的方式
