# C++20 constinit

## 一、概念说明

`constinit`确保变量在编译时初始化，解决C++的静态初始化顺序问题（Static Initialization Order Fiasco）。它不暗示`const`，变量仍然可以修改。

```cpp
#include <iostream>

constexpr int computeValue() { return 42; }

// constinit: 编译时初始化，但不是const
constinit int globalValue = computeValue();

// 对比const：编译时常量，不可修改
constexpr int constValue = 100;

// constinit避免静态初始化顺序问题
constinit int* ptr = nullptr;

void setPtr(int* p) {
    ptr = p; // 可以修改constinit变量
}

int main() {
    std::cout << "globalValue = " << globalValue << std::endl;
    globalValue = 100; // 可以修改（不是const）
    std::cout << "修改后: " << globalValue << std::endl;

    int local = 99;
    setPtr(&local);
    std::cout << "*ptr = " << *ptr << std::endl;

    return 0;
}
```

**输出：**
```
globalValue = 42
修改后: 100
*ptr = 99
```

## 二、具体用法

### 2.1 静态初始化顺序问题

```cpp
#include <iostream>

// 问题：谁先初始化？A的构造可能使用B，但B还没初始化
// extern int globalB;
// int globalA = globalB + 1; // 未定义行为

// 解决方案：用constinit保证编译时初始化
constinit int globalB = 42;
constinit int globalA = globalB + 1; // 安全：都是编译时

// 延迟初始化模式
constinit int* cachedPtr = nullptr;

int& getCached() {
    if (!cachedPtr) {
        cachedPtr = new int(99);
    }
    return *cachedPtr;
}

int main() {
    std::cout << "globalA = " << globalA << std::endl;
    std::cout << "cached = " << getCached() << std::endl;
    return 0;
}
```

**输出：**
```
globalA = 43
cached = 99
```

## 三、注意事项与常见陷阱

- **`constinit`不意味着`const`**：变量可以修改。
- **`constinit`只应用于静态/线程存储期变量**：局部变量不能用。
- **`constinit`必须有初始化器**：且必须是常量表达式。
- **`constinit`与`constexpr`不同**：`constexpr`暗示`const`，`constinit`不暗示。
- **适合替代函数内的`static`变量**：避免首次调用时的初始化开销。
