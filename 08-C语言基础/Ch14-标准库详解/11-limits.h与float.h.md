# limits.h 与 float.h - 类型限制

## 1. 概述

`<limits.h>` 定义了整数类型的限制，`<float.h>` 定义了浮点类型的特性。这些宏使得程序能够了解当前平台上各类型的具体范围，编写可移植的代码。

## 2. limits.h - 整数类型限制

### 2.1 字符类型

```c
#include <limits.h>
#include <stdio.h>

int main(void) {
    // char类型
    printf("CHAR_BIT  = %d (每字节的位数)\n", CHAR_BIT);    // 通常为8
    printf("CHAR_MIN  = %d\n", CHAR_MIN);  // 有符号char最小值
    printf("CHAR_MAX  = %d\n", CHAR_MAX);  // 有符号char最大值

    // signed char
    printf("SCHAR_MIN = %d\n", SCHAR_MIN);  // -128
    printf("SCHAR_MAX = %d\n", SCHAR_MAX);  // 127

    // unsigned char
    printf("UCHAR_MAX = %u\n", UCHAR_MAX);  // 255

    return 0;
}
```

### 2.2 整数类型

```c
#include <limits.h>
#include <stdio.h>

int main(void) {
    // short (短整型)
    printf("SHRT_MIN  = %d\n", SHRT_MIN);   // -32768
    printf("SHRT_MAX  = %d\n", SHRT_MAX);   // 32767
    printf("USHRT_MAX = %u\n", USHRT_MAX);  // 65535

    // int (整型)
    printf("INT_MIN   = %d\n", INT_MIN);    // -2147483648
    printf("INT_MAX   = %d\n", INT_MAX);    // 2147483647
    printf("UINT_MAX  = %u\n", UINT_MAX);   // 4294967295

    // long (长整型)
    printf("LONG_MIN  = %ld\n", LONG_MIN);
    printf("LONG_MAX  = %ld\n", LONG_MAX);
    printf("ULONG_MAX = %lu\n", ULONG_MAX);

    // long long (C99)
    printf("LLONG_MIN  = %lld\n", LLONG_MIN);
    printf("LLONG_MAX  = %lld\n", LLONG_MAX);
    printf("ULLONG_MAX = %llu\n", ULLONG_MAX);

    return 0;
}
```

### 2.3 MB_LEN_MAX

```c
#include <limits.h>
#include <stdio.h>

int main(void) {
    // 多字节字符的最大字节数
    printf("MB_LEN_MAX = %d\n", MB_LEN_MAX);  // 通常为4(UTF-8)

    return 0;
}
```

## 3. float.h - 浮点类型限制

### 3.1 浮点类型宏一览

```c
#include <float.h>
#include <stdio.h>

int main(void) {
    printf("=== float (单精度) ===\n");
    printf("FLT_MANT_DIG    = %d\n", FLT_MANT_DIG);     // 尾数位数
    printf("FLT_DIG         = %d\n", FLT_DIG);          // 十进制精度(6)
    printf("FLT_MIN_EXP     = %d\n", FLT_MIN_EXP);
    printf("FLT_MAX_EXP     = %d\n", FLT_MAX_EXP);
    printf("FLT_MIN_10_EXP  = %d\n", FLT_MIN_10_EXP);   // 最小十进制指数
    printf("FLT_MAX_10_EXP  = %d\n", FLT_MAX_10_EXP);   // 最大十进制指数
    printf("FLT_MIN         = %e\n", FLT_MIN);          // 最小正正规数
    printf("FLT_MAX         = %e\n", FLT_MAX);          // 最大值
    printf("FLT_EPSILON     = %e\n", FLT_EPSILON);      // 1.0与下一个数的差

    printf("\n=== double (双精度) ===\n");
    printf("DBL_MANT_DIG    = %d\n", DBL_MANT_DIG);
    printf("DBL_DIG         = %d\n", DBL_DIG);          // 15
    printf("DBL_MIN         = %e\n", DBL_MIN);
    printf("DBL_MAX         = %e\n", DBL_MAX);
    printf("DBL_EPSILON     = %e\n", DBL_EPSILON);

    printf("\n=== long double (扩展精度) ===\n");
    printf("LDBL_MANT_DIG   = %d\n", LDBL_MANT_DIG);
    printf("LDBL_DIG        = %d\n", LDBL_DIG);         // 通常18或33
    printf("LDBL_MIN        = %Le\n", LDBL_MIN);
    printf("LDBL_MAX        = %Le\n", LDBL_MAX);
    printf("LDBL_EPSILON    = %Le\n", LDBL_EPSILON);

    return 0;
}
```

### 3.2 舍入模式（C99）

```c
#include <float.h>
#include <stdio.h>

int main(void) {
    // FLT_ROUNDS: 浮点加法的舍入模式
    // -1: 不确定
    //  0: 向零舍入
    //  1: 向最近舍入
    //  2: 向正无穷舍入
    //  3: 向负无穷舍入
    printf("FLT_ROUNDS = %d\n", FLT_ROUNDS);

    // C99: 舍入模式宏
    #ifdef FLT_EVAL_METHOD
    printf("FLT_EVAL_METHOD = %d\n", FLT_EVAL_METHOD);
    // 0: 不使用额外的精度和范围
    // 1: float和double使用double的精度
    // 2: 所有浮点类型使用long double的精度
    #endif

    // C11: 十进制有效位数
    #ifdef DECIMAL_DIG
    printf("DECIMAL_DIG = %d\n", DECIMAL_DIG);
    #endif

    return 0;
}
```

## 4. 实用技巧

### 4.1 确定类型范围

```c
#include <stdio.h>
#include <limits.h>
#include <float.h>

// 使用宏打印类型信息
#define PRINT_TYPE_INFO(type, fmt_unsigned) \
    printf("sizeof(%s) = %zu\n", #type, sizeof(type))

#define PRINT_INT_LIMITS(type, fmt, min, max) \
    printf("%s: [% "fmt", %"fmt"]\n", #type, min, max)

int main(void) {
    // 确定各类型大小
    printf("=== 类型大小 ===\n");
    PRINT_TYPE_INFO(char,);
    PRINT_TYPE_INFO(short,);
    PRINT_TYPE_INFO(int,);
    PRINT_TYPE_INFO(long,);
    PRINT_TYPE_INFO(long long,);
    PRINT_TYPE_INFO(float,);
    PRINT_TYPE_INFO(double,);
    PRINT_TYPE_INFO(long double,);

    // 确定整数范围
    printf("\n=== 整数范围 ===\n");
    printf("char:        [%d, %d]\n", CHAR_MIN, CHAR_MAX);
    printf("short:       [%d, %d]\n", SHRT_MIN, SHRT_MAX);
    printf("int:         [%d, %d]\n", INT_MIN, INT_MAX);
    printf("long:        [%ld, %ld]\n", LONG_MIN, LONG_MAX);
    printf("long long:   [%lld, %lld]\n", LLONG_MIN, LLONG_MAX);

    return 0;
}
```

### 4.2 安全的数值运算

```c
#include <stdio.h>
#include <limits.h>
#include <stdbool.h>

// 安全的整数加法（检查溢出）
bool safe_add(int a, int b, int *result) {
    if ((b > 0 && a > INT_MAX - b) ||
        (b < 0 && a < INT_MIN - b)) {
        return false;  // 会溢出
    }
    *result = a + b;
    return true;
}

// 安全的整数乘法
bool safe_multiply(int a, int b, int *result) {
    if (a == 0 || b == 0) {
        *result = 0;
        return true;
    }
    if ((a > 0 && b > 0 && a > INT_MAX / b) ||
        (a > 0 && b < 0 && b < INT_MIN / a) ||
        (a < 0 && b > 0 && a < INT_MIN / b) ||
        (a < 0 && b < 0 && a < INT_MAX / b)) {
        return false;  // 会溢出
    }
    *result = a * b;
    return true;
}

int main(void) {
    int result;

    // 测试安全加法
    if (safe_add(1000000000, 1000000000, &result)) {
        printf("1000000000 + 1000000000 = %d\n", result);
    } else {
        printf("加法溢出!\n");
    }

    if (safe_add(INT_MAX, 1, &result)) {
        printf("INT_MAX + 1 = %d\n", result);
    } else {
        printf("INT_MAX + 1 溢出!\n");
    }

    // 测试安全乘法
    if (safe_multiply(100000, 10000, &result)) {
        printf("100000 * 10000 = %d\n", result);
    } else {
        printf("乘法溢出!\n");
    }

    return 0;
}
```

### 4.3 浮点精度检测

```c
#include <stdio.h>
#include <float.h>
#include <math.h>

// 检查浮点数是否"足够接近"零
int is_nearly_zero(double x) {
    return fabs(x) < DBL_EPSILON;
}

// 相对误差比较
int nearly_equal(double a, double b, double rel_eps) {
    double diff = fabs(a - b);
    double largest = fabs(a) > fabs(b) ? fabs(a) : fabs(b);
    return diff <= largest * rel_eps;
}

int main(void) {
    // 浮点精度问题
    double a = 0.1 + 0.2;
    double b = 0.3;

    printf("0.1 + 0.2 = %.20f\n", a);
    printf("0.3       = %.20f\n", b);
    printf("相等: %s\n", a == b ? "是" : "否");

    // 使用epsilon比较
    printf("近似相等: %s\n",
           nearly_equal(a, b, DBL_EPSILON * 10) ? "是" : "否");

    // 确定有效精度
    printf("\nfloat 精度: %d 位十进制数字\n", FLT_DIG);
    printf("double 精度: %d 位十进制数字\n", DBL_DIG);
    printf("long double 精度: %d 位十进制数字\n", LDBL_DIG);

    return 0;
}
```

## 5. 各平台典型值参考

```
常见平台的整数限制（64位系统）:
=====================================
char:        -128 ~ 127
short:       -32768 ~ 32767
int:         -2147483648 ~ 2147483647
long:        取决于系统模型
  - ILP32 (32位): -2147483648 ~ 2147483647
  - LP64 (64位):  -9223372036854775808 ~ 9223372036854775807
long long:   -9223372036854775808 ~ 9223372036854775807

常见平台的浮点限制:
=====================================
float:       精度6位,  范围 ~1e-38 ~ 1e+38
double:      精度15位, 范围 ~1e-308 ~ 1e+308
long double: 精度18-33位
```

## 6. 重要注意事项

> **要点一**：`char` 是否有符号是实现定义的，可移植代码应使用 `signed char` 或 `unsigned char` 明确指定。

> **要点二**：`int` 的大小至少是16位，但现代系统上通常是32位。

> **要点三**：`long` 在32位Windows和64位Windows上都是32位，但在64位Linux上是64位（LP64模型）。

> **要点四**：`long long` 保证至少64位。

> **要点五**：`FLT_EPSILON` 是1.0与大于1.0的最小可表示值之间的差，不是最小可表示的正数。

> **要点六**：`DBL_DIG` 表示十进制有效位数，`DBL_MANT_DIG` 表示二进制尾数位数。

> **要点七**：浮点数的比较不能直接用 `==`，应使用带容差的比较。

> **要点八**：整数运算溢出是未定义行为，应在运算前检查是否可能溢出。
