# constexpr函数模板

## 一、概念说明

`constexpr`函数模板结合了编译期求值和泛型编程的能力。`constexpr`函数可以在编译时求值（当参数是常量表达式时），而模板提供了类型泛化。两者结合使得编写通用的编译期计算成为可能。

## 二、具体用法

### 2.1 基本constexpr函数模板

```cpp
template <typename T>
constexpr T power(T base, int exp) {
    T result = 1;
    for (int i = 0; i < exp; ++i) result *= base;
    return result;
}

int main() {
    // 编译期求值
    constexpr int p1 = power(2, 10);
    constexpr double p2 = power(2.5, 3);
    std::cout << "2^10 = " << p1 << std::endl;    // 2^10 = 1024
    std::cout << "2.5^3 = " << p2 << std::endl;   // 2.5^3 = 15.625

    // 运行时也可用
    int x = 5;
    std::cout << power(x, 3) << std::endl;  // 125
}
```

### 2.2 编译期数组处理

```cpp
template <typename T, std::size_t N>
constexpr T array_sum(const T (&arr)[N]) {
    T sum = 0;
    for (std::size_t i = 0; i < N; ++i) sum += arr[i];
    return sum;
}

template <typename T, std::size_t N>
constexpr T array_max(const T (&arr)[N]) {
    T max_val = arr[0];
    for (std::size_t i = 1; i < N; ++i)
        if (arr[i] > max_val) max_val = arr[i];
    return max_val;
}

int main() {
    constexpr int data[] = {3, 1, 4, 1, 5, 9, 2, 6};
    constexpr int sum = array_sum(data);
    constexpr int max_val = array_max(data);
    std::cout << "sum=" << sum << ", max=" << max_val << std::endl;  // sum=31, max=9
}
```

### 2.3 constexpr与if constexpr结合

```cpp
template <typename T>
constexpr T absolute(T val) {
    if constexpr (std::is_unsigned_v<T>) {
        return val;  // 无符号数不需要取绝对值
    } else {
        return val < 0 ? -val : val;
    }
}

int main() {
    constexpr int a = absolute(-42);
    constexpr unsigned int b = absolute(42U);
    std::cout << a << std::endl;  // 42
    std::cout << b << std::endl;  // 42
}
```

### 2.4 编译期类型特征检测

```cpp
template <typename T>
constexpr std::size_t type_size() {
    if constexpr (std::is_same_v<T, char>) return 1;
    else if constexpr (std::is_same_v<T, short>) return 2;
    else if constexpr (std::is_same_v<T, int>) return 4;
    else if constexpr (std::is_same_v<T, double>) return 8;
    else return sizeof(T);
}

int main() {
    constexpr auto s1 = type_size<int>();
    constexpr auto s2 = type_size<double>();
    std::cout << "int: " << s1 << ", double: " << s2 << std::endl;  // int: 4, double: 8
}
```

## 三、注意事项与常见陷阱

- C++14起`constexpr`函数可以包含局部变量、循环和条件语句
- 函数必须在编译时可求值才能成为常量表达式（参数也是常量时）
- `constexpr`函数也可以在运行时调用
- C++20允许`constexpr`函数中使用`try-catch`和动态分配
- `constexpr`函数模板不能是虚函数
- 编译器会检查`constexpr`函数是否真的能在编译期求值
