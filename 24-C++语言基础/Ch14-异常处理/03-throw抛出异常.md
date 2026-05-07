# throw抛出异常

## 一、概念说明

`throw`关键字用于抛出异常对象。抛出的异常对象会被复制（或移动）到异常处理机制中。抛出类型和捕获类型的匹配规则遵循C++的类型转换规则。

## 二、具体用法

### 2.1 抛出不同类型的异常

```cpp
#include <iostream>
#include <stdexcept>
#include <string>

void demo_throw() {
    // 抛出标准异常
    throw std::runtime_error("运行时错误");

    // 抛出自定义类型
    // throw std::string("自定义错误");

    // 抛出基本类型（不推荐）
    // throw 42;
    // throw "错误消息";
}

int main() {
    try {
        demo_throw();
    } catch (const std::exception& e) {
        std::cerr << e.what() << std::endl;
    }
}
```

```output
运行时错误
```

### 2.2 按值抛出 vs 按指针抛出

```cpp
// 推荐：按值抛出（异常对象被复制）
void good_throw() {
    throw std::runtime_error("按值抛出");
}

// 不推荐：按指针抛出（需要手动管理内存）
void bad_throw() {
    throw new std::runtime_error("按指针抛出");  // 内存泄漏风险
}

int main() {
    try {
        good_throw();
    } catch (const std::exception& e) {
        std::cout << "捕获: " << e.what() << std::endl;
    }

    // 按指针捕获需要delete
    try {
        bad_throw();
    } catch (const std::exception* e) {
        std::cerr << e->what() << std::endl;
        delete e;  // 必须手动删除！
    }
}
```

### 2.3 重新抛出

```cpp
void inner() {
    throw std::runtime_error("内部错误");
}

void middle() {
    try {
        inner();
    } catch (const std::runtime_error& e) {
        std::cerr << "middle记录: " << e.what() << std::endl;
        throw;  // 重新抛出当前异常（不指定对象）
    }
}

int main() {
    try {
        middle();
    } catch (const std::exception& e) {
        std::cerr << "main捕获: " << e.what() << std::endl;
    }
}
```

```output
middle记录: 内部错误
main捕获: 内部错误
```

### 2.4 抛出自定义对象

```cpp
struct MyError {
    int code;
    std::string message;
    MyError(int c, const std::string& msg) : code(c), message(msg) {}
};

void risky() {
    throw MyError(404, "资源未找到");
}

int main() {
    try {
        risky();
    } catch (const MyError& e) {
        std::cout << "错误码: " << e.code << ", 消息: " << e.message << std::endl;
    }
}
```

```output
错误码: 404, 消息: 资源未找到
```

## 三、注意事项与常见陷阱

- 应按值抛出异常对象，不要按指针抛出（避免内存管理问题）
- `throw;`（无参数）重新抛出当前异常，保留原始类型和信息
- 异常对象在抛出时被复制，因此应避免抛出大型对象
- 不要在析构函数中抛出异常（可能导致双重异常和`std::terminate`）
- 抛出的异常对象必须可复制（C++11后支持移动语义）
- `noexcept`函数中抛出异常会导致`std::terminate`
