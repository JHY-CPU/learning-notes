# RTTI详解

## 一、概念说明

**RTTI**（Run-Time Type Information，C++标准 §7.6.1.6/§7.6.1.7）提供运行时类型检查能力，包括`typeid`运算符和`dynamic_cast`。RTTI需要至少一个虚函数才能工作（依赖vtable中的RTTI信息）。

### 1.1 RTTI组件

| 组件 | 功能 | 头文件 |
|------|------|--------|
| `typeid` | 获取类型信息 | `<typeinfo>` |
| `type_info` | 存储/比较类型信息 | `<typeinfo>` |
| `dynamic_cast` | 安全类型转换 | 内建 |
| `std::type_index` | 可哈希的类型包装 | `<typeindex>` |

## 二、具体用法

### 2.1 typeid类型识别

```cpp
#include <iostream>
#include <typeinfo>

class Base { public: virtual ~Base() = default; };
class Derived : public Base {};
class Other : public Base {};

int main() {
    Derived d;
    Other o;
    Base* p1 = &d;
    Base* p2 = &o;

    // 有虚函数时，typeid返回动态类型
    std::cout << "p1动态类型: " << typeid(*p1).name() << std::endl;   // Derived
    std::cout << "p2动态类型: " << typeid(*p2).name() << std::endl;   // Other

    // 静态类型
    std::cout << "p1静态类型: " << typeid(p1).name() << std::endl;    // Base*

    // 类型比较
    if (typeid(*p1) == typeid(Derived))
        std::cout << "p1指向Derived" << std::endl;

    if (typeid(*p1) != typeid(*p2))
        std::cout << "p1和p2指向不同类型" << std::endl;

    return 0;
}
```

### 2.2 type_info成员

```cpp
#include <iostream>
#include <typeinfo>
#include <typeindex>
#include <unordered_map>

int main() {
    const std::type_info& ti_int = typeid(int);
    const std::type_info& ti_double = typeid(double);

    std::cout << "int:     name=" << ti_int.name()
              << " hash=" << ti_int.hash_code() << std::endl;
    std::cout << "double:  name=" << ti_double.name()
              << " hash=" << ti_double.hash_code() << std::endl;

    // 比较
    std::cout << "int == int: " << (typeid(int) == typeid(int)) << std::endl;      // 1
    std::cout << "int == double: " << (typeid(int) == typeid(double)) << std::endl; // 0

    // 使用type_index作为map的键
    std::unordered_map<std::type_index, std::string> typeNames;
    typeNames[std::type_index(typeid(int))] = "整数";
    typeNames[std::type_index(typeid(double))] = "浮点数";

    std::cout << typeid(int).name() << " = " << typeNames[typeid(int)] << std::endl;

    return 0;
}
```

### 2.3 没有虚函数时的行为

```cpp
#include <iostream>
#include <typeinfo>

class BaseNoVirtual {
public:
    int x;
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

### 2.4 RTTI的实际应用

```cpp
#include <iostream>
#include <typeinfo>
#include <vector>
#include <memory>

class Shape { public: virtual ~Shape() = default; };
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
    typeStats(shapes);
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **没有虚函数时`typeid`只看静态类型**：必须有虚函数才能获取动态类型
2. **RTTI有运行时开销**：嵌入式系统可能禁用RTTI（`-fno-rtti`）
3. **`type_info::name()`返回实现定义的名称**：可能被mangled
4. **优先使用虚函数多态**：只在必要时用RTTI
5. **`std::type_index`可用于哈希容器**：比`type_info::hash_code()`更方便
6. **`type_info`不能拷贝、赋值**：只能通过引用使用
