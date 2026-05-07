# 静态多态CRTP详解

## 一、概念说明

**CRTP**通过模板实现编译期多态，避免虚函数的运行时开销。基类通过`static_cast<Derived*>(this)`调用派生类方法。

## 二、具体用法

### 2.1 CRTP实现mixin-in

```cpp
#include <iostream>

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
    bool operator<=(const Derived& rhs) const {
        return !(static_cast<const Derived*>(this)->compareTo(rhs) > 0);
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
    Number a(10), b(5);
    a.print(); b.print();
    std::cout << "a > b: " << (a > b) << std::endl;
    return 0;
}
```

**输出：**
```
Number(10)
Number(5)
a > b: 1
```

## 三、注意事项与常见陷阱

- CRTP不能放入异构容器
- 派生类必须在使用前完整定义
- CRTP支持链式方法（Fluent API）
- 编译期绑定允许更好的内联优化
