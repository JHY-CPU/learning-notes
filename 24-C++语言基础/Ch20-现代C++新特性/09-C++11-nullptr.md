# C++11 nullptr

## 一、概念说明

`nullptr`是C++11引入的空指针字面量，类型为`std::nullptr_t`。它替代了传统的`NULL`（通常是`0`或`((void*)0)`），解决了重载歧义问题。

```cpp
#include <iostream>

void func(int) {
    std::cout << "func(int)" << std::endl;
}

void func(char*) {
    std::cout << "func(char*)" << std::endl;
}

int main() {
    // NULL的歧义问题
    // func(NULL); // 可能调用func(int)，不是期望的！

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

### 2.1 与NULL和0的区别

```cpp
#include <iostream>
#include <type_traits>

int main() {
    // NULL可能是0或(void*)0
    #ifdef NULL
    std::cout << "NULL已定义" << std::endl;
    #endif

    // nullptr是std::nullptr_t类型
    std::cout << "nullptr type: " << typeid(nullptr).name() << std::endl;
    std::cout << "0 type: " << typeid(0).name() << std::endl;

    // nullptr可转换为bool
    bool b = nullptr; // false
    std::cout << "bool(nullptr): " << b << std::endl;

    // 模板中nullptr的优势
    // template<typename T> void f(T* p);
    // f(0);       // 错误：0不是指针
    // f(NULL);    // 可能错误：NULL可能是整数
    // f(nullptr); // 正确：nullptr是指针类型

    return 0;
}
```

**输出：**
```
NULL已定义
nullptr type: decltype(nullptr)
0 type: int
bool(nullptr): 0
```

## 三、注意事项与常见陷阱

- **始终使用`nullptr`代替`NULL`或`0`**：避免重载歧义。
- **`nullptr`不是整数类型**：不能用于算术运算。
- **`sizeof(nullptr)`等于`sizeof(void*)`**：通常是4或8字节。
- **`nullptr`可以与所有指针类型比较**：但不能与整数比较（除了0）。
- **模板特化中`nullptr`很有用**：可以特化指针类型。
