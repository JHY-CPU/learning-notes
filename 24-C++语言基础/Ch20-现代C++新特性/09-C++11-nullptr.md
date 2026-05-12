# C++11 nullptr

## 一、概念说明

`nullptr`（C++11 §4.10, §2.14.7）是C++11引入的空指针字面量，类型为`std::nullptr_t`。它替代了传统的`NULL`（通常是`0`或`((void*)0)`），解决了函数重载时的歧义问题。

### 1.1 NULL的问题

在C++中，`NULL`通常定义为整数0。当存在整数和指针的重载时，`NULL`会匹配整数版本，而非期望的指针版本，导致意外行为。

```cpp
#include <iostream>

void func(int) {
    std::cout << "func(int)" << std::endl;
}

void func(char*) {
    std::cout << "func(char*)" << std::endl;
}

void func(bool) {
    std::cout << "func(bool)" << std::endl;
}

int main() {
    // NULL的歧义问题
    // func(NULL); // 可能调用func(int)或func(bool)，不是期望的func(char*)！

    // nullptr明确表示空指针
    func(nullptr);  // 调用func(char*)

    // nullptr是独立类型
    auto p = nullptr;
    std::cout << "sizeof(nullptr): " << sizeof(nullptr) << std::endl;
    std::cout << "p == nullptr: " << (p == nullptr) << std::endl;

    // nullptr可以隐式转换为任何指针类型
    int* ip = nullptr;
    double* dp = nullptr;
    void* vp = nullptr;

    std::cout << "ip is null: " << (ip == nullptr) << std::endl;

    return 0;
}
```

**输出：**
```
func(char*)
sizeof(nullptr): 8
p == nullptr: 1
ip is null: 1
```

## 二、具体用法

### 2.1 nullptr vs NULL vs 0

```cpp
#include <iostream>
#include <type_traits>

int main() {
    // nullptr是std::nullptr_t类型
    std::cout << "nullptr type: " << typeid(nullptr).name() << std::endl;
    std::cout << "0 type: " << typeid(0).name() << std::endl;
    std::cout << "NULL type: " << typeid(NULL).name() << std::endl;

    // 类型检查
    std::cout << "nullptr is nullptr_t: "
              << std::is_same<decltype(nullptr), std::nullptr_t>::value << std::endl;

    // nullptr可转换为bool
    bool b = nullptr; // false
    std::cout << "bool(nullptr): " << b << std::endl;

    // nullptr不能转换为整数
    // int x = nullptr; // 编译错误

    // 但可以与整数0比较
    int* p = nullptr;
    // if (p == 0) {} // OK但不推荐
    if (p == nullptr) {} // 推荐

    return 0;
}
```

**输出：**
```
nullptr type: decltype(nullptr)
0 type: int
NULL type: decltype(nullptr)   （取决于实现，可能是long）
nullptr is nullptr_t: 1
bool(nullptr): 0
```

### 2.2 模板中的nullptr优势

```cpp
#include <iostream>
#include <type_traits>

// 模板中区分指针和整数
template <typename T>
void process(T value) {
    if constexpr (std::is_pointer_v<T>) {
        if (value == nullptr)
            std::cout << "空指针" << std::endl;
        else
            std::cout << "指针: " << *value << std::endl;
    } else if constexpr (std::is_integral_v<T>) {
        std::cout << "整数: " << value << std::endl;
    }
}

int main() {
    int x = 42;
    int* p = &x;
    int* null_ptr = nullptr;

    process(p);          // 指针
    process(null_ptr);   // 空指针
    process(42);         // 整数
    process(nullptr);    // 空指针（nullptr可匹配T*）

    return 0;
}
```

**输出：**
```
指针: 42
空指针
整数: 42
空指针
```

### 2.3 nullptr与智能指针

```cpp
#include <iostream>
#include <memory>

int main() {
    std::unique_ptr<int> p1 = nullptr;
    std::shared_ptr<int> p2 = nullptr;

    std::cout << "p1 == nullptr: " << (p1 == nullptr) << std::endl;
    std::cout << "p2 == nullptr: " << (p2 == nullptr) << std::endl;

    p1 = std::make_unique<int>(42);
    std::cout << "赋值后 p1 == nullptr: " << (p1 == nullptr) << std::endl;

    p1 = nullptr; // 释放并置空
    std::cout << "重置后 p1 == nullptr: " << (p1 == nullptr) << std::endl;

    return 0;
}
```

**输出：**
```
p1 == nullptr: 1
p2 == nullptr: 1
赋值后 p1 == nullptr: 0
重置后 p1 == nullptr: 1
```

## 三、注意事项与常见陷阱

1. **始终使用`nullptr`代替`NULL`或`0`**：避免重载歧义，意图更清晰。
2. **`nullptr`不是整数类型**：不能用于算术运算，不能赋值给整数变量。
3. **`sizeof(nullptr)`等于`sizeof(void*)`**：通常是4（32位）或8（64位）字节。
4. **`nullptr`可以与所有指针类型比较**：但不能与整数（除了字面量0）比较。
5. **模板中`nullptr`比`NULL`更安全**：`NULL`可能是整数类型，导致模板推导意外。
6. **C++11起应全面使用`nullptr`**：包括初始化空指针。
