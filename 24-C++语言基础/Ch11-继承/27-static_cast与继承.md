# static_cast与继承

## 一、概念说明

`static_cast`用于编译期的类型转换，在继承体系中用于**向上转型**（总是安全）和**向下转型**（不检查，程序员负责安全）。

## 二、具体用法

### 2.1 static_cast在继承中的使用

```cpp
#include <iostream>

class Base {
public:
    virtual ~Base() = default;
    virtual void show() { std::cout << "Base" << std::endl; }
};

class Derived : public Base {
public:
    void show() override { std::cout << "Derived" << std::endl; }
    void derivedMethod() { std::cout << "Derived特有" << std::endl; }
};

int main() {
    Derived d;
    Base& b = d;

    // 向上转型：static_cast总是安全
    Base* pb = static_cast<Base*>(&d);

    // 向下转型：程序员确认安全（无运行时检查）
    Derived* pd = static_cast<Derived*>(pb);
    pd->derivedMethod();  // OK

    Base b2;
    Derived* pd2 = static_cast<Derived*>(&b2);  // 危险！但编译通过
    // pd2->derivedMethod();  // 未定义行为
    return 0;
}
```

**输出：**
```
Derived特有
```

## 三、注意事项与常见陷阱

- `static_cast`向下转型**不检查**安全性，使用前必须确认类型
- `static_cast`比`dynamic_cast`快（无运行时检查）
- 向上转型不需要任何cast，编译器自动完成
- `static_cast`不能移除`const`（用`const_cast`）
- 如果不确定类型，使用`dynamic_cast`并检查结果
