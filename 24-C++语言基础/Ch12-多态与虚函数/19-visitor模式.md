# Visitor模式

## 一、概念说明

**Visitor模式**通过**双重分发**（Double Dispatch）实现对不同类型元素的不同操作，而不需要修改元素类。它将操作从数据结构中分离出来。

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

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Rectangle>());

    AreaVisitor areaCalc;
    for (auto& s : shapes) s->accept(areaCalc);
    return 0;
}
```

**输出：**
```
圆面积: 78.5398
矩形面积: 12
```

## 三、注意事项与常见陷阱

- 添加新元素类型需要修改所有Visitor接口
- 访问者模式适合元素类型稳定但操作多变的场景
- `accept`方法实现双重分发
- 可用`std::variant` + `std::visit`替代（C++17）
