# try-catch基本用法

## 一、概念说明

`try`块包含可能抛出异常的代码，`catch`块用于捕获和处理异常。可以有多个`catch`块处理不同类型的异常，`catch(...)`可以捕获所有异常。

## 二、具体用法

### 2.1 基本try-catch

```cpp
#include <iostream>
#include <stdexcept>

int main() {
    try {
        int value = -1;
        if (value < 0) {
            throw std::invalid_argument("值不能为负数");
        }
        std::cout << "值: " << value << std::endl;
    } catch (const std::invalid_argument& e) {
        std::cerr << "无效参数: " << e.what() << std::endl;
    }
    return 0;
}
```

```output
无效参数: 值不能为负数
```

### 2.2 多个catch块

```cpp
void risky_operation(int choice) {
    switch (choice) {
        case 1: throw std::runtime_error("运行时错误");
        case 2: throw std::logic_error("逻辑错误");
        case 3: throw 42;  // 抛出int
        default: std::cout << "正常执行" << std::endl;
    }
}

int main() {
    for (int i = 1; i <= 3; ++i) {
        try {
            risky_operation(i);
        } catch (const std::runtime_error& e) {
            std::cerr << "[Runtime] " << e.what() << std::endl;
        } catch (const std::logic_error& e) {
            std::cerr << "[Logic] " << e.what() << std::endl;
        } catch (int code) {
            std::cerr << "[Error Code] " << code << std::endl;
        }
    }
    return 0;
}
```

```output
[Runtime] 运行时错误
[Logic] 逻辑错误
[Error Code] 42
```

### 2.3 catch(...)捕获所有异常

```cpp
void process() {
    try {
        throw std::string("未知错误");
    } catch (const std::exception& e) {
        std::cerr << "标准异常: " << e.what() << std::endl;
    } catch (...) {
        std::cerr << "捕获到未知类型的异常" << std::endl;
    }
}

int main() {
    process();  // 捕获到未知类型的异常
}
```

### 2.4 异常捕获顺序

```cpp
// catch按声明顺序匹配，更具体的类型放在前面
try {
    throw std::runtime_error("test");
} catch (const std::runtime_error& e) {  // 先匹配（更具体）
    std::cout << "runtime_error" << std::endl;
} catch (const std::exception& e) {       // 后匹配（更通用）
    std::cout << "exception" << std::endl;
} catch (...) {                           // 最后兜底
    std::cout << "unknown" << std::endl;
}
```

## 三、注意事项与常见陷阱

- 引用捕获（`const&`）避免对象切片和额外拷贝
- 捕获顺序必须从具体到通用，否则具体类型会被基类catch屏蔽
- `catch(...)`应放在最后，且通常需要重新抛出或记录
- 未捕获的异常会导致`std::terminate`被调用
- catch参数可以不命名（如果不需要使用异常对象）
- 空的catch块（`catch(...) {}`）会静默吞掉异常，通常不推荐
