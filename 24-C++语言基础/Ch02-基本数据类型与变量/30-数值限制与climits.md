# 数值限制与climits

## 一、概念说明

C++提供了两种获取数值类型限制的方式：C风格的`<climits>`/`<cfloat>`宏和C++风格的`<limits>`模板。它们提供类型的最大值、最小值、精度等信息。

## 二、具体用法

### 2.1 C风格头文件

```cpp
#include <iostream>
#include <climits>  // 整型限制
#include <cfloat>   // 浮点型限制
using namespace std;

int main() {
    // 整型限制
    cout << "===== 整型限制 =====" << endl;
    cout << "char范围: " << CHAR_MIN << " ~ " << CHAR_MAX << endl;
    cout << "signed char: " << SCHAR_MIN << " ~ " << SCHAR_MAX << endl;
    cout << "unsigned char: 0 ~ " << UCHAR_MAX << endl;
    cout << "short: " << SHRT_MIN << " ~ " << SHRT_MAX << endl;
    cout << "int: " << INT_MIN << " ~ " << INT_MAX << endl;
    cout << "long: " << LONG_MIN << " ~ " << LONG_MAX << endl;
    cout << "long long: " << LLONG_MIN << " ~ " << LLONG_MAX << endl;

    // 浮点型限制
    cout << "\n===== 浮点型限制 =====" << endl;
    cout << "float有效位: " << FLT_DIG << endl;
    cout << "float最大值: " << FLT_MAX << endl;
    cout << "float最小正规值: " << FLT_MIN << endl;
    cout << "float epsilon: " << FLT_EPSILON << endl;
    cout << "double有效位: " << DBL_DIG << endl;
    cout << "double最大值: " << DBL_MAX << endl;
    cout << "double epsilon: " << DBL_EPSILON << endl;

    return 0;
}
```

输出（典型值）：
```
===== 整型限制 =====
char范围: -128 ~ 127
signed char: -128 ~ 127
unsigned char: 0 ~ 255
short: -32768 ~ 32767
int: -2147483648 ~ 2147483647
long: -9223372036854775808 ~ 9223372036854775807
long long: -9223372036854775808 ~ 9223372036854775807

===== 浮点型限制 =====
float有效位: 6
float最大值: 3.40282e+38
float最小正规值: 1.17549e-38
float epsilon: 1.19209e-07
double有效位: 15
double最大值: 1.79769e+308
double epsilon: 2.22045e-16
```

### 2.2 C++风格 numeric_limits

```cpp
#include <iostream>
#include <limits>
#include <cstdint>
using namespace std;

template<typename T>
void printLimits(const char* name) {
    cout << "===== " << name << " =====" << endl;
    cout << "  大小: " << sizeof(T) << " 字节" << endl;
    cout << "  最小值: " << numeric_limits<T>::min() << endl;
    cout << "  最大值: " << numeric_limits<T>::max() << endl;
    cout << "  是否有符号: " << numeric_limits<T>::is_signed << endl;
    cout << "  是否为整数: " << numeric_limits<T>::is_integer << endl;

    if constexpr (numeric_limits<T>::is_iec559) {
        cout << "  精度位数: " << numeric_limits<T>::digits10 << endl;
        cout << "  epsilon: " << numeric_limits<T>::epsilon() << endl;
        cout << "  无穷大: " << numeric_limits<T>::infinity() << endl;
    }
    cout << endl;
}

int main() {
    printLimits<int>("int");
    printLimits<unsigned int>("unsigned int");
    printLimits<float>("float");
    printLimits<double>("double");
    printLimits<int64_t>("int64_t");

    // 实用查询
    cout << "int可以表示的十进制位数: "
         << numeric_limits<int>::digits10 << endl;
    cout << "double是否符合IEEE754: "
         << numeric_limits<double>::is_iec559 << endl;

    return 0;
}
```

输出：
```
===== int =====
  大小: 4 字节
  最小值: -2147483648
  最大值: 2147483647
  是否有符号: 1
  是否为整数: 1

===== float =====
  大小: 4 字节
  最小值: 1.17549e-38
  最大值: 3.40282e+38
  是否有符号: 1
  是否为整数: 0
  精度位数: 6
  epsilon: 1.19209e-07
  无穷大: inf
```

### 2.3 实际应用场景

```cpp
#include <iostream>
#include <limits>
#include <algorithm>
using namespace std;

// 查找数组最小值（正确处理边界）
int findMin(const int arr[], int size) {
    int minVal = numeric_limits<int>::max();  // 初始化为最大值
    for (int i = 0; i < size; i++) {
        minVal = min(minVal, arr[i]);
    }
    return minVal;
}

// 检查加法是否溢出
bool safeAdd(int a, int b, int& result) {
    if (b > 0 && a > numeric_limits<int>::max() - b) {
        return false;  // 上溢
    }
    if (b < 0 && a < numeric_limits<int>::min() - b) {
        return false;  // 下溢
    }
    result = a + b;
    return true;
}

int main() {
    int arr[] = {42, -10, 99, 0, 7};
    cout << "最小值: " << findMin(arr, 5) << endl;

    int result;
    if (safeAdd(2000000000, 1000000000, result)) {
        cout << "2000000000 + 1000000000 = " << result << endl;
    } else {
        cout << "2000000000 + 1000000000 = 溢出！" << endl;
    }

    if (safeAdd(100, 200, result)) {
        cout << "100 + 200 = " << result << endl;
    }

    return 0;
}
```

输出：
```
最小值: -10
2000000000 + 1000000000 = 溢出！
100 + 200 = 300
```

## 三、注意事项与常见陷阱

1. **优先使用numeric_limits**：C++的`<limits>`模板比C宏更安全，支持泛型编程
2. **数值溢出检查**：运算前用`numeric_limits`检查是否会发生溢出
3. **浮点epsilon**：`epsilon()`是最小的使`1.0 + eps != 1.0`的值，用于浮点比较
4. **min()对无符号类型**：`numeric_limits<unsigned>::min()`是0而非负数
5. **平台依赖**：数值限制随平台变化，不要硬编码具体值
