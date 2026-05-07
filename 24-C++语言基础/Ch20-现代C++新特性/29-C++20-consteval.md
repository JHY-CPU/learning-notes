# C++20 consteval

## 一、概念说明

`consteval`声明立即函数（immediate function），必须在编译期求值。与`constexpr`不同，`constexpr`函数可以运行时执行，`consteval`必须编译时执行。

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
        h = h * 31 + *str++;
    }
    return h;
}

int main() {
    constexpr int f5 = factorial(5); // OK：编译期
    // int n = 5;
    // int fn = factorial(n); // 错误：n不是常量表达式

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

### 2.1 consteval vs constexpr vs constinit

| 关键字 | 含义 | 可运行时 | 必须编译时 | 用途 |
|--------|------|---------|-----------|------|
| `constexpr` | 可以编译时求值 | 是 | 否 | 常量函数 |
| `consteval` | 必须编译时求值 | 否 | 是 | 立即函数 |
| `constinit` | 编译时初始化 | - | 初始化时 | 避免静态初始化顺序问题 |

```cpp
#include <iostream>

consteval int mustCompile(int x) { return x * 2; }
constexpr int mayCompile(int x) { return x * 2; }

int main() {
    constexpr int a = mustCompile(21);  // OK
    constexpr int b = mayCompile(21);   // OK

    // int n = 21;
    // int c = mustCompile(n);  // 错误：必须编译期
    int d = mayCompile(21);             // OK：运行时

    std::cout << a << " " << b << " " << d << std::endl;
    return 0;
}
```

**输出：**
```
42 42 42
```

## 三、注意事项与常见陷阱

- **`consteval`函数不能用运行时参数调用**。
- **`consteval`函数可以调用`constexpr`函数**：但反过来不行。
- **`consteval`适合编译期字符串处理、哈希、校验等**。
- **`consteval`是C++20特性**：GCC 11+、Clang 14+。
- **`consteval`构造函数保证编译期构造**。
