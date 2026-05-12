# C++20 consteval

## 一、概念说明

`consteval`（C++20 §9.2.5）声明立即函数（Immediate Function），必须在编译期求值。与`constexpr`不同，`constexpr`函数可以运行时执行，`consteval`必须编译时执行，否则编译失败。

### 1.1 consteval vs constexpr

| 特性 | constexpr | consteval |
|------|-----------|-----------|
| 可以运行时执行 | 是 | 否 |
| 必须编译时执行 | 否 | 是 |
| 引入版本 | C++11 | C++20 |
| 适用场景 | 可选编译期计算 | 强制编译期计算 |

```cpp
#include <iostream>

// consteval: 必须编译期求值
consteval int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) result *= i;
    return result;
}

// consteval: 编译期哈希
consteval uint64_t hash(const char* str) {
    uint64_t h = 0;
    while (*str) {
        h = h * 31 + static_cast<uint64_t>(*str++);
    }
    return h;
}

int main() {
    constexpr int f5 = factorial(5); // OK：编译期
    // int n = 5;
    // int fn = factorial(n); // 编译错误：n不是常量表达式

    std::cout << "5! = " << f5 << std::endl;

    // 编译期哈希：可用于switch-case
    constexpr auto h = hash("hello");
    std::cout << "hash(\"hello\") = " << h << std::endl;

    return 0;
}
```

**输出：**
```
5! = 120
hash("hello") = 99162322
```

## 二、具体用法

### 2.1 编译期哈希与switch

```cpp
#include <iostream>
#include <string_view>

consteval uint32_t hash(std::string_view str) {
    uint32_t h = 5381;
    for (char c : str) {
        h = ((h << 5) + h) + static_cast<uint32_t>(c); // djb2
    }
    return h;
}

void handleCommand(std::string_view cmd) {
    // 编译期生成switch表
    switch (hash(cmd)) {
        case hash("start"):
            std::cout << "执行: 启动" << std::endl;
            break;
        case hash("stop"):
            std::cout << "执行: 停止" << std::endl;
            break;
        case hash("restart"):
            std::cout << "执行: 重启" << std::endl;
            break;
        default:
            std::cout << "未知命令: " << cmd << std::endl;
    }
}

int main() {
    handleCommand("start");
    handleCommand("stop");
    handleCommand("unknown");
    return 0;
}
```

**输出：**
```
执行: 启动
执行: 停止
未知命令: unknown
```

### 2.2 consteval构造函数

```cpp
#include <iostream>

class CompileTimeBuffer {
    int data[100];
    int size_;
public:
    consteval CompileTimeBuffer(int size) : size_(size) {
        for (int i = 0; i < size; ++i) data[i] = i;
    }
    constexpr int getSize() const { return size_; }
    constexpr int operator[](int i) const { return data[i]; }
};

int main() {
    // 必须在编译期构造
    constexpr CompileTimeBuffer buf(10);
    std::cout << "size=" << buf.getSize() << std::endl;
    std::cout << "buf[5]=" << buf[5] << std::endl;

    return 0;
}
```

### 2.3 consteval vs constexpr vs constinit

```cpp
#include <iostream>

consteval int mustCompile(int x) { return x * 2; }
constexpr int mayCompile(int x) { return x * 2; }
constinit int globalInit = 42; // 编译期初始化，但不是const

int main() {
    constexpr int a = mustCompile(21);  // OK：编译期
    constexpr int b = mayCompile(21);   // OK：编译期

    int n = 21;
    // int c = mustCompile(n);  // 错误：必须编译期
    int d = mayCompile(n);             // OK：运行时

    std::cout << a << " " << b << " " << d << std::endl;

    globalInit = 100; // 可以修改（constinit不意味着const）
    std::cout << "globalInit=" << globalInit << std::endl;

    return 0;
}
```

**输出：**
```
42 42 42
globalInit=100
```

## 三、注意事项与常见陷阱

1. **`consteval`函数不能用运行时参数调用**：否则编译失败。
2. **`consteval`函数可以调用`constexpr`函数**：但反过来不行。
3. **适合编译期字符串处理、哈希、校验等**：确保计算在编译时完成。
4. **需要C++20支持**：GCC 11+、Clang 14+、MSVC 19.28+。
5. **`consteval`构造函数保证编译期构造**：不能在运行时调用。
6. **`consteval`函数不能是虚函数**：C++20限制。
