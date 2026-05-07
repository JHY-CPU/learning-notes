# RTTI详解

## 一、概念说明

**RTTI**（Run-Time Type Information）提供运行时类型检查能力，包括`typeid`运算符和`dynamic_cast`。RTTI需要至少一个虚函数才能工作。

## 二、具体用法

### 2.1 typeid类型识别

```cpp
#include <iostream>
#include <typeinfo>

class Base { public: virtual ~Base() = default; };
class Derived : public Base {};

int main() {
    Derived d;
    Base* p = &d;

    // 有虚函数时，typeid返回动态类型
    std::cout << typeid(*p).name() << std::endl;   // Derived
    std::cout << typeid(p).name() << std::endl;    // Base*（静态类型）

    // 类型比较
    if (typeid(*p) == typeid(Derived))
        std::cout << "p指向Derived" << std::endl;

    return 0;
}
```

### 2.2 运行时类型检查

```cpp
#include <iostream>

class Animal { public: virtual ~Animal() = default; };
class Dog : public Animal { public: void bark() { std::cout << "汪!" << std::endl; } };
class Cat : public Animal { public: void meow() { std::cout << "喵!" << std::endl; } };

void handle(Animal* a) {
    if (auto* dog = dynamic_cast<Dog*>(a))
        dog->bark();
    else if (auto* cat = dynamic_cast<Cat*>(a))
        cat->meow();
}

int main() {
    Dog d; Cat c;
    handle(&d);
    handle(&c);
    return 0;
}
```

**输出：**
```
汪!
喵!
```

## 三、注意事项与常见陷阱

- 没有虚函数时`typeid`只看静态类型
- RTTI有运行时开销（编译器选项可禁用）
- `type_info::name()`返回实现定义的名称
- 优先使用虚函数多态，必要时才用RTTI
