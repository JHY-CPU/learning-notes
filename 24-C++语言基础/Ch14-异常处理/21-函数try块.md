# 函数try块

## 一、概念说明

函数try块（Function Try Block）将整个函数体包裹在try-catch中，主要用于构造函数中捕获成员初始化时的异常。语法为在函数体前写`try`，在参数列表后写`catch`。

## 二、具体用法

### 2.1 构造函数try块

```cpp
#include <iostream>
#include <stdexcept>

class Member {
    int value;
public:
    Member(int v) : value(v) {
        if (v < 0) throw std::invalid_argument("Member值不能为负");
        std::cout << "Member(" << v << ")" << std::endl;
    }
};

class Container {
    Member m1;
    Member m2;
public:
    // 构造函数try块：捕获成员初始化异常
    Container(int a, int b)
    try : m1(a), m2(b) {
        // 构造函数体
        std::cout << "Container构造完成" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Container构造失败: " << e.what() << std::endl;
        // 注意：这里会自动重新抛出异常
        // 无法"修复"并继续构造
    }
};

int main() {
    try {
        Container c(1, -2);  // m2初始化失败
    } catch (const std::exception& e) {
        std::cerr << "main捕获: " << e.what() << std::endl;
    }
}
```

```output
Member(1)
Container构造失败: Member值不能为负
main捕获: Member值不能为负
```

### 2.2 普通函数try块

```cpp
int divide(int a, int b)
try {
    if (b == 0) throw std::runtime_error("除零");
    return a / b;
} catch (const std::exception& e) {
    std::cerr << "divide错误: " << e.what() << std::endl;
    return -1;  // 返回错误值
}

int main() {
    int result = divide(10, 0);
    std::cout << "结果: " << result << std::endl;  // -1
}
```

### 2.3 析构函数try块

```cpp
class Resource {
public:
    ~Resource() noexcept(false) {
        throw std::runtime_error("析构异常");
    }
};

class Manager {
    Resource res;
public:
    ~Manager()
    try {
        // res在这里析构
    } catch (const std::exception& e) {
        std::cerr << "Manager析构异常: " << e.what() << std::endl;
        // 仍然会调用std::terminate
    }
};
```

### 2.4 基类构造异常

```cpp
class Base {
public:
    Base() {
        throw std::runtime_error("Base构造失败");
    }
};

class Derived : public Base {
public:
    Derived()
    try : Base() {
        std::cout << "Derived构造" << std::endl;
    } catch (const std::exception& e) {
        std::cerr << "Derived构造失败: " << e.what() << std::endl;
        // 基类构造失败，Derived也不会被构造
    }
};
```

## 三、注意事项与常见陷阱

- 构造函数try块的catch无法阻止异常传播（对象已无法构造）
- 函数try块中的catch可以返回值（普通函数）
- 构造函数try块中不能使用`return`语句
- 析构函数try块中捕获异常后仍会`std::terminate`
- 成员初始化时的异常只能通过函数try块捕获
- 函数try块增加了代码复杂度，仅在必要时使用
