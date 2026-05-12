# Visitor模式

## 一、概念说明

**Visitor模式**（GoF设计模式）通过**双重分发**（Double Dispatch）实现对不同类型元素的不同操作，而不需要修改元素类。它将操作从数据结构中分离出来，适合"元素类型稳定但操作多变"的场景。

### 1.1 双重分发

```
第一次分发：element.accept(visitor) → 根据element类型选择accept
第二次分发：visitor.visit(*this) → 根据visitor类型选择visit重载
```

## 二、具体用法

### 2.1 经典Visitor实现

```cpp
#include <iostream>
#include <vector>
#include <memory>

class Circle;
class Rectangle;

// Visitor接口
class ShapeVisitor {
public:
    virtual void visit(Circle&) = 0;
    virtual void visit(Rectangle&) = 0;
    virtual ~ShapeVisitor() = default;
};

// 元素接口
class Shape {
public:
    virtual void accept(ShapeVisitor& v) = 0;
    virtual ~Shape() = default;
};

class Circle : public Shape {
public:
    double radius = 5;
    void accept(ShapeVisitor& v) override { v.visit(*this); }
};

class Rectangle : public Shape {
public:
    double w = 3, h = 4;
    void accept(ShapeVisitor& v) override { v.visit(*this); }
};

// 具体Visitor
class AreaVisitor : public ShapeVisitor {
public:
    void visit(Circle& c) override {
        std::cout << "圆面积: " << 3.14159 * c.radius * c.radius << std::endl;
    }
    void visit(Rectangle& r) override {
        std::cout << "矩形面积: " << r.w * r.h << std::endl;
    }
};

class DrawVisitor : public ShapeVisitor {
public:
    void visit(Circle& c) override {
        std::cout << "绘制圆(r=" << c.radius << ")" << std::endl;
    }
    void visit(Rectangle& r) override {
        std::cout << "绘制矩形(" << r.w << "x" << r.h << ")" << std::endl;
    }
};

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Rectangle>());

    AreaVisitor areaCalc;
    DrawVisitor drawCalc;

    std::cout << "=== 计算面积 ===" << std::endl;
    for (auto& s : shapes) s->accept(areaCalc);

    std::cout << "\n=== 绘制 ===" << std::endl;
    for (auto& s : shapes) s->accept(drawCalc);

    return 0;
}
```

### 2.2 C++17 std::variant替代

```cpp
#include <iostream>
#include <vector>
#include <variant>

struct Circle { double radius; };
struct Rectangle { double w, h; };
struct Triangle { double a, b, c; };

using Shape = std::variant<Circle, Rectangle, Triangle>;

// 使用std::visit替代Visitor模式
struct AreaVisitor {
    double operator()(const Circle& c) const {
        return 3.14159 * c.radius * c.radius;
    }
    double operator()(const Rectangle& r) const {
        return r.w * r.h;
    }
    double operator()(const Triangle& t) const {
        double s = (t.a + t.b + t.c) / 2;
        return std::sqrt(s * (s-t.a) * (s-t.b) * (s-t.c));
    }
};

int main() {
    std::vector<Shape> shapes = {Circle{5}, Rectangle{3, 4}, Triangle{3, 4, 5}};

    double total = 0;
    for (const auto& s : shapes)
        total += std::visit(AreaVisitor{}, s);

    std::cout << "总面积: " << total << std::endl;
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **添加新元素类型需要修改所有Visitor接口**：违反开闭原则（对元素）
2. **访问者模式适合元素类型稳定但操作多变的场景**：反之用策略模式
3. **`accept`方法实现双重分发**：C++没有内建的双重分发
4. **C++17的`std::variant`+`std::visit`可替代**：更简洁，无需虚函数
5. **Visitor可以累积状态**：如计算总面积、收集所有元素
6. **Acyclic Visitor模式可以避免修改所有Visitor**：通过dynamic_cast检查
