# consteval函数（C++20）

## 一、概念说明

`consteval`是C++20引入的关键字，用于声明**立即函数**（immediate function）。与`constexpr`不同，`consteval`函数**必须**在编译期求值，如果无法在编译期执行则编译错误。

`consteval`确保函数**永远**不会在运行时执行，适用于需要强制编译期计算的场景。

## 二、具体用法

### 2.1 基本consteval

```cpp
consteval int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

// 编译期使用：正常
constexpr int val = factorial(5);
static_assert(val == 120);

// 运行时调用：编译错误！
// int n = 5;
// int x = factorial(n);  // 错误：参数不是常量表达式
```

### 2.2 consteval vs constexpr

```cpp
constexpr int squareConstexpr(int x) { return x * x; }
consteval int squareConsteval(int x) { return x * x; }

int n = 10;

// constexpr可以在运行时调用
int a = squareConstexpr(n);    // 正确：运行时调用

// consteval不能在运行时调用
// int b = squareConsteval(n);  // 编译错误
int b = squareConsteval(10);   // 正确：字面量是编译期常量
```

### 2.3 编译期字符串哈希

```cpp
consteval uint32_t hashString(const char* str) {
    uint32_t hash = 5381;
    while (*str) {
        hash = ((hash << 5) + hash) + static_cast<uint32_t>(*str++);
    }
    return hash;
}

// 编译期计算哈希值
constexpr auto h = hashString("hello");
static_assert(h == hashString("hello"));
std::cout << h << std::endl;  // 输出: 261238937 (示例值)
```

### 2.4 编译期保证

```cpp
consteval int mustBeCompileTime(int x) {
    return x * 2 + 1;
}

// 保证此函数的返回值一定是编译期已知的
template <size_t N>
struct Array {
    int data[mustBeCompileTime(N)];  // 确保编译期计算
};
```

## 三、注意事项与常见陷阱

- `consteval`函数的所有调用都必须在编译期完成
- `consteval`不能用于虚函数（虚函数需要运行时调度）
- `consteval`函数可以调用`constexpr`函数，反之不行
- `consteval`函数不能通过函数指针调用（指针值在运行时才确定）
- 适用于需要强制编译期求值的场景，如模板参数、数组大小等
