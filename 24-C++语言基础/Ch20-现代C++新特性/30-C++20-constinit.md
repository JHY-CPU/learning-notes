# C++20 constinit

## 一、概念说明

`constinit`（C++20 §9.2.4）确保变量在编译时初始化，解决C++的静态初始化顺序问题（Static Initialization Order Fiasco）。与`constexpr`不同，`constinit`不暗示`const`，变量仍然可以在运行时修改。

### 1.1 constinit vs constexpr vs const

| 特性 | constinit | constexpr | const |
|------|-----------|-----------|-------|
| 编译时初始化 | 强制 | 强制 | 不保证 |
| 运行时可修改 | 是 | 否 | 否 |
| 适用对象 | 静态/线程局部变量 | 任意 | 任意 |
| 引入版本 | C++20 | C++11 | C++98 |

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
    globalValue = 200; // 可以修改（不是const）
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
修改后: 200
*ptr = 99
```

## 二、具体用法

### 2.1 解决静态初始化顺序问题

```cpp
#include <iostream>

// 问题场景：跨翻译单元的静态变量
// file1.cpp: extern int globalB;
// file2.cpp: int globalB = 42;
// file1.cpp: int globalA = globalB + 1; // 未定义行为！

// 解决方案：用constinit保证编译时初始化
constinit int globalB = 42;
constinit int globalA = globalB + 1; // 安全：都是编译时

// 延迟初始化模式
constinit int* cachedPtr = nullptr;

int& getCached() {
    if (!cachedPtr) {
        cachedPtr = new int(99); // 运行时初始化
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

### 2.2 线程安全的单例

```cpp
#include <iostream>

class Singleton {
    Singleton() { std::cout << "Singleton构造" << std::endl; }
public:
    static Singleton& instance() {
        // constinit保证inst被零初始化（或编译期初始化）
        static Singleton inst;
        return inst;
    }

    void doWork() { std::cout << "工作" << std::endl; }
};

// constinit配合函数内的static
constinit int callCount = 0;

void incrementCall() {
    // constinit变量在编译时初始化，首次调用时不会执行初始化代码
    // 对比：普通static变量首次调用时初始化
    ++callCount;
}

int main() {
    incrementCall();
    incrementCall();
    std::cout << "callCount = " << callCount << std::endl;

    Singleton::instance().doWork();

    return 0;
}
```

### 2.3 模块化初始化

```cpp
#include <iostream>

// 全局配置
constinit int g_logLevel = 1; // 编译期初始化
constinit bool g_debug = false;

void configure(int level, bool debug) {
    g_logLevel = level;  // 运行时修改
    g_debug = debug;
}

void log(const char* msg) {
    if (g_debug) {
        std::cout << "[DEBUG] " << msg << std::endl;
    }
}

int main() {
    log("启动"); // g_debug=false，不输出

    configure(3, true);
    log("配置完成"); // 输出

    std::cout << "logLevel=" << g_logLevel << std::endl;
    return 0;
}
```

## 三、注意事项与常见陷阱

1. **`constinit`不意味着`const`**：变量可以修改，只是初始化必须在编译时。
2. **`constinit`只应用于静态/线程局部变量**：局部变量不能用`constinit`。
3. **`constinit`必须有初始化器**：且必须是常量表达式。
4. **`constinit`与`constexpr`不同**：`constexpr`暗示`const`（不可修改），`constinit`不暗示。
5. **适合替代函数内的`static`变量**：避免首次调用时的初始化开销。
6. **`constinit`可以用于非const变量**：这是它与`constexpr`的核心区别。
