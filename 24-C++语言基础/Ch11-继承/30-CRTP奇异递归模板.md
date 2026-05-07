# CRTP奇异递归模板

## 一、概念说明

**CRTP**（Curiously Recurring Template Pattern）是将派生类作为模板参数传递给基类的模式。它在编译期实现多态，避免虚函数的运行时开销。

## 二、具体用法

### 2.1 基本CRTP

```cpp
#include <iostream>

// 基类模板：派生类作为参数
template<typename Derived>
class Counter {
    static int count;
public:
    Counter() { ++count; }
    Counter(const Counter&) { ++count; }
    ~Counter() { --count; }
    static int getCount() { return count; }
};

template<typename Derived>
int Counter<Derived>::count = 0;

// 派生类将自己传给基类
class Widget : public Counter<Widget> {
    int id;
public:
    Widget(int id) : id(id) {}
};

class Gadget : public Counter<Gadget> {};

int main() {
    Widget w1(1), w2(2);
    Gadget g1;
    std::cout << "Widget: " << Widget::getCount() << std::endl;  // 2
    std::cout << "Gadget: " << Gadget::getCount() << std::endl;  // 1
    return 0;
}
```

**输出：**
```
Widget: 2
Gadget: 1
```

### 2.2 静态多态（编译期多态）

```cpp
#include <iostream>

template<typename Derived>
class Shape {
public:
    void draw() const {
        // 编译期绑定：static_cast到派生类
        static_cast<const Derived*>(this)->drawImpl();
    }
};

class Circle : public Shape<Circle> {
public:
    void drawImpl() const { std::cout << "绘制圆形" << std::endl; }
};

class Rectangle : public Shape<Rectangle> {
public:
    void drawImpl() const { std::cout << "绘制矩形" << std::endl; }
};

template<typename T>
void render(const Shape<T>& shape) {
    shape.draw();
}

int main() {
    Circle c;
    Rectangle r;
    render(c);  // 编译期解析为Circle::drawImpl
    render(r);  // 编译期解析为Rectangle::drawImpl
    return 0;
}
```

**输出：**
```
绘制圆形
绘制矩形
```

## 三、注意事项与常见陷阱

- CRTP是编译期技术，不支持运行时多态
- 派生类必须完整定义后才能使用（不能前向声明后用）
- CRTP常用于计数器、单例、mixin-in等功能
- 与虚函数不同，CRTP不能放入异构容器
- CRTP比虚函数调用更快（编译期内联，无vtable查找）
