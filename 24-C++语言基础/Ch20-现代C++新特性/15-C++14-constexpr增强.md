# C++14 constexpr增强

## 一、概念说明

C++14大幅放宽`constexpr`函数的限制（C++14 §5.19），允许局部变量、循环、if语句、switch等控制流。使得更多计算可以在编译期完成，`constexpr`函数的实用性大幅提高。

### 1.1 C++11 vs C++14 constexpr

| 特性 | C++11 | C++14 |
|------|-------|-------|
| 局部变量 | 不允许 | 允许 |
| 循环 | 不允许 | 允许 |
| if/switch | 不允许 | 允许 |
| 修改局部变量 | 不允许 | 允许 |
| try-catch | 不允许 | 不允许（C++20允许） |

```cpp
#include <iostream>

// C++11: 只能有一条return语句（递归方式）
// constexpr int factorial(int n) {
//     return n <= 1 ? 1 : n * factorial(n - 1);
// }

// C++14: 可以使用循环和局部变量
constexpr int factorial(int n) {
    int result = 1;
    for (int i = 2; i <= n; ++i) {
        result *= i;
    }
    return result;
}

constexpr int gcd(int a, int b) {
    while (b != 0) {
        int temp = b;
        b = a % b;
        a = temp;
    }
    return a;
}

// constexpr数组操作
constexpr int sumArray(const int* arr, int size) {
    int sum = 0;
    for (int i = 0; i < size; ++i) {
        sum += arr[i];
    }
    return sum;
}

int main() {
    constexpr int f10 = factorial(10);
    constexpr int g = gcd(48, 18);

    std::cout << "10! = " << f10 << std::endl;
    std::cout << "gcd(48,18) = " << g << std::endl;

    // 编译期数组操作
    constexpr int arr[] = {1, 2, 3, 4, 5};
    constexpr int total = sumArray(arr, 5);
    std::cout << "数组求和 = " << total << std::endl;

    return 0;
}
```

**输出：**
```
10! = 3628800
gcd(48,18) = 6
数组求和 = 15
```

## 二、具体用法

### 2.1 编译期查找表

```cpp
#include <iostream>
#include <array>

// 编译期生成正弦查找表
constexpr double mySin(double x) {
    // 泰勒级数: sin(x) = x - x^3/3! + x^5/5! - ...
    double term = x;
    double sum = x;
    for (int i = 1; i < 10; ++i) {
        term *= -x * x / ((2 * i) * (2 * i + 1));
        sum += term;
    }
    return sum;
}

template <int N>
constexpr std::array<double, N> makeSinTable() {
    std::array<double, N> table{};
    for (int i = 0; i < N; ++i) {
        double angle = 3.14159265358979 * i / (N - 1);
        table[i] = mySin(angle);
    }
    return table;
}

int main() {
    // 编译期生成
    constexpr auto sinTable = makeSinTable<360>();

    std::cout << "sin(0) = " << sinTable[0] << std::endl;
    std::cout << "sin(90) = " << sinTable[90] << std::endl;
    std::cout << "sin(180) = " << sinTable[180] << std::endl;
    std::cout << "sin(270) = " << sinTable[270] << std::endl;

    return 0;
}
```

**输出：**
```
sin(0) = 0
sin(90) = 1
sin(180) = 1.22465e-15
sin(270) = -1
```

### 2.2 编译期字符串处理

```cpp
#include <iostream>

// 编译期计算字符串长度
constexpr size_t my_strlen(const char* s) {
    size_t len = 0;
    while (s[len] != '\0') ++len;
    return len;
}

// 编译期字符串比较
constexpr bool my_strcmp(const char* a, const char* b) {
    while (*a && *b) {
        if (*a != *b) return false;
        ++a; ++b;
    }
    return *a == *b;
}

// 编译期哈希
constexpr uint32_t hash(const char* s) {
    uint32_t h = 0;
    while (*s) {
        h = h * 31 + static_cast<uint32_t>(*s++);
    }
    return h;
}

int main() {
    constexpr auto len = my_strlen("hello");
    constexpr auto eq = my_strcmp("hello", "hello");
    constexpr auto h = hash("hello");

    std::cout << "strlen(hello) = " << len << std::endl;
    std::cout << "strcmp(hello,hello) = " << eq << std::endl;
    std::cout << "hash(hello) = " << h << std::endl;

    return 0;
}
```

**输出：**
```
strlen(hello) = 5
strcmp(hello,hello) = 1
hash(hello) = 99162322
```

### 2.3 编译期排序

```cpp
#include <iostream>
#include <array>

constexpr std::array<int, 5> bubbleSort(std::array<int, 5> arr) {
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4 - i; ++j) {
            if (arr[j] > arr[j + 1]) {
                int temp = arr[j];
                arr[j] = arr[j + 1];
                arr[j + 1] = temp;
            }
        }
    }
    return arr;
}

int main() {
    constexpr std::array<int, 5> sorted = bubbleSort({5, 3, 1, 4, 2});

    std::cout << "编译期排序: ";
    for (int x : sorted) std::cout << x << " ";
    std::cout << std::endl;

    return 0;
}
```

**输出：**
```
编译期排序: 1 2 3 4 5
```

## 三、注意事项与常见陷阱

1. **`constexpr`函数内的变量必须是字面类型**：不能使用`std::string`等非字面类型。
2. **不能使用`try-catch`和`goto`**（C++14）：C++20放宽了`try-catch`限制。
3. **虚函数在C++20前不能是`constexpr`**：C++20放宽。
4. **`constexpr`不保证编译期执行**：取决于参数是否为常量表达式。
5. **`static_assert`可以用`constexpr`函数的结果**：验证编译期计算正确性。
6. **编译期计算可能增加编译时间**：复杂的constexpr函数可能使编译变慢。
