# 运行时类型信息（RTTI）

## 一、概念说明

**RTTI**（Run-Time Type Information，C++标准 §7.6.1.6/§7.6.1.7）允许在运行时查询对象的实际类型。主要机制是`typeid`运算符和`type_info`类。RTTI需要类至少有一个虚函数才能获取动态类型信息。

### 1.1 RTTI机制

| 机制 | 用途 | 头文件 |
|------|------|--------|
| `typeid`运算符 | 获取类型信息 | `<typeinfo>` |
| `type_info`类 | 存储类型信息 | `<typeinfo>` |
| `dynamic_cast` | 安全类型转换 | 内建 |

## 二、具体用法

### 2.1 typeid运算符

```cpp
#include <iostream>
#include <typeinfo>

class Base {
public:
    virtual ~Base() = default;  // 需要虚函数才能获取动态类型
};

class Derived : public Base {};
class Other : public Base {};

int main() {
    Base* b1 = new Derived();
    Base* b2 = new Other();

    // typeid返回type_info引用
    std::cout << "b1动态类型: " << typeid(*b1).name() << std::endl;
    std::cout << "b2动态类型: " << typeid(*b2).name() << std::endl;
    std::cout << "Base静态类型: " << typeid(Base).name() << std::endl;

    // 比较实际类型
    if (typeid(*b1) == typeid(Derived))
        std::cout << "b1实际是Derived" << std::endl;

    if (typeid(*b2) != typeid(Derived))
        std::cout << "b2不是Derived" << std::endl;

    delete b1;
    delete b2;
    return 0;
}
```

### 2.2 type_info成员

```cpp
#include <iostream>
#include <typeinfo>
#include <unordered_map>

int main() {
    const std::type_info& ti1 = typeid(int);
    const std::type_info& ti2 = typeid(double);
    const std::type_info& ti3 = typeid(int);

    std::cout << "int:     name=" << ti1.name()
              << " hash=" << ti1.hash_code() << std::endl;
    std::cout << "double:  name=" << ti2.name()
              << " hash=" << ti2.hash_code() << std::endl;

    // 比较
    std::cout << "int == int: " << (ti1 == ti3) << std::endl;     // 1
    std::cout << "int == double: " << (ti1 == ti2) << std::endl;  // 0

    // before()定义了排序关系
    std::cout << "int before double: " << ti1.before(ti2) << std::endl;

    // hash_code可用于unordered_map
    std::unordered_map<size_t, std::string> typeMap;
    typeMap[ti1.hash_code()] = "整数类型";
    typeMap[ti2.hash_code()] = "浮点类型";

    return 0;
}
```

### 2.3 没有虚函数时的typeid行为

```cpp
#include <iostream>
#include <typeinfo>

class BaseNoVirtual {
public:
    int x;
    // 没有虚函数
};

class DerivedNoVirtual : public BaseNoVirtual {
    int y;
};

class BaseWithVirtual {
public:
    virtual ~BaseWithVirtual() = default;
};

class DerivedWithVirtual : public BaseWithVirtual {};

int main() {
    // 没有虚函数：typeid只看静态类型
    DerivedNoVirtual dnv;
    BaseNoVirtual* pnv = &dnv;
    std::cout << "无虚函数: " << typeid(*pnv).name() << std::endl;
    // 输出BaseNoVirtual（静态类型），不是DerivedNoVirtual！

    // 有虚函数：typeid看动态类型
    DerivedWithVirtual dwv;
    BaseWithVirtual* pwv = &dwv;
    std::cout << "有虚函数: " << typeid(*pwv).name() << std::endl;
    // 输出DerivedWithVirtual（动态类型）

    return 0;
}
```

### 2.4 RTTI的应用场景

```cpp
#include <iostream>
#include <typeinfo>
#include <vector>
#include <memory>

class Shape {
public:
    virtual ~Shape() = default;
};

class Circle : public Shape {};
class Rectangle : public Shape {};
class Triangle : public Shape {};

// 类型统计
void typeStats(const std::vector<std::unique_ptr<Shape>>& shapes) {
    int circles = 0, rects = 0, tris = 0;
    for (const auto& s : shapes) {
        if (typeid(*s) == typeid(Circle)) ++circles;
        else if (typeid(*s) == typeid(Rectangle)) ++rects;
        else if (typeid(*s) == typeid(Triangle)) ++tris;
    }
    std::cout << "圆=" << circles << " 矩形=" << rects
              << " 三角=" << tris << std::endl;
}

int main() {
    std::vector<std::unique_ptr<Shape>> shapes;
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Rectangle>());
    shapes.push_back(std::make_unique<Circle>());
    shapes.push_back(std::make_unique<Triangle>());

    typeStats(shapes);
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **没有虚函数的类，`typeid`只看静态类型**：`Base* p = &derived; typeid(*p)` 返回Base类型
2. **`type_info::name()`返回实现定义的名称**：可能被mangled（可用`abi::__cxa_demangle`解码）
3. **RTTI有运行时开销**：某些嵌入式系统禁用RTTI（`-fno-rtti`）
4. **优先使用虚函数多态**：只在必要时用`typeid`
5. **`type_info`不能拷贝、赋值**：只能通过引用使用
6. **`hash_code()`（C++11）可用于哈希容器**：但不同运行可能不同
