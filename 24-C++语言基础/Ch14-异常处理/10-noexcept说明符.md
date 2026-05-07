# noexcept说明符

## 一、概念说明

`noexcept`是C++11引入的说明符，用于标记函数不会抛出异常。它替代了已废弃的`throw()`异常规范，为编译器优化提供信息，并在运行时保证函数的异常行为。

## 二、具体用法

### 2.1 基本用法

```cpp
#include <iostream>
#include <stdexcept>

// 保证不抛异常的函数
void safe_function() noexcept {
    std::cout << "此函数不会抛异常" << std::endl;
}

// 可能抛异常的函数（默认）
void risky_function() {
    throw std::runtime_error("出错了");
}

// 如果noexcept函数抛出异常，std::terminate被调用
void will_terminate() noexcept {
    throw std::runtime_error("在noexcept中抛异常");
    // 程序终止，不会进行栈展开
}

int main() {
    safe_function();  // OK
    try {
        risky_function();
    } catch (...) {
        std::cout << "捕获异常" << std::endl;
    }
    // will_terminate();  // 调用std::terminate
}
```

### 2.2 条件noexcept

```cpp
// 条件noexcept：根据表达式决定是否noexcept
template <typename T>
void swap_if_nothrow(T& a, T& b) noexcept(std::is_nothrow_move_constructible_v<T>) {
    T temp = std::move(a);
    a = std::move(b);
    b = std::move(temp);
}

// 自定义类型的noexcept规范
class MyType {
    int* data;
public:
    // 移动构造标记为noexcept（推荐）
    MyType(MyType&& other) noexcept : data(other.data) {
        other.data = nullptr;
    }

    // 移动赋值标记为noexcept
    MyType& operator=(MyType&& other) noexcept {
        delete data;
        data = other.data;
        other.data = nullptr;
        return *this;
    }

    // 析构函数隐式noexcept
    ~MyType() { delete data; }
};
```

### 2.3 noexcept运算符

```cpp
// noexcept(expr) 检查表达式是否不会抛异常
void demo_noexcept_operator() {
    std::cout << std::boolalpha;

    // 检查函数是否noexcept
    std::cout << "safe_function noexcept: "
              << noexcept(safe_function()) << std::endl;  // true

    // 检查表达式是否noexcept
    int a = 1, b = 2;
    std::cout << "int移动noexcept: "
              << noexcept(std::move(a)) << std::endl;  // true

    // 检查类型操作
    std::cout << "vector移动noexcept: "
              << noexcept(std::move(std::vector<int>{})) << std::endl;  // true
}
```

### 2.4 移动构造与noexcept

```cpp
// 容器在扩容时优先使用noexcept移动构造
class Widget {
    std::string name;
    std::vector<int> data;
public:
    // 标记为noexcept，vector扩容时使用移动而非拷贝
    Widget(Widget&& other) noexcept
        : name(std::move(other.name)), data(std::move(other.data)) {}

    Widget& operator=(Widget&& other) noexcept {
        name = std::move(other.name);
        data = std::move(other.data);
        return *this;
    }
};

// std::vector::push_back 在扩容时：
// - 如果元素的移动构造是noexcept，使用移动
// - 否则使用拷贝（保证强异常安全）
```

## 三、注意事项与常见陷阱

- `noexcept`在运行时生效：违反时调用`std::terminate`
- 析构函数默认隐式`noexcept`
- 移动构造和移动赋值应尽量标记为`noexcept`
- `std::move_if_noexcept`根据条件选择移动或拷贝
- `noexcept`让编译器生成更高效的代码
- 过度使用`noexcept`可能导致难以调试的终止
- C++11的`throw()`已被废弃，用`noexcept`替代
