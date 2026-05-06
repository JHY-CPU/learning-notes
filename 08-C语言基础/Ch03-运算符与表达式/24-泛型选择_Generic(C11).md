# 24 - 泛型选择 _Generic（C11）

## 一、概述

C11引入了 `_Generic` 关键字，它提供了一种**编译时类型选择**机制，允许根据表达式的类型选择不同的表达式。这使得C语言可以实现类似其他语言中泛型（Generics）的功能。

### 语法格式

```c
_Generic(表达式,
    类型1: 表达式1,
    类型2: 表达式2,
    ...
    default: 默认表达式
)
```

**工作原理**：编译时检查"表达式"的类型，匹配到对应类型时返回该类型对应的表达式的值。

## 二、基本用法

```c
#include <stdio.h>

int main(void) {
    int i = 42;
    double d = 3.14;
    char c = 'A';

    // 根据类型选择不同的格式化字符串
    #define TYPE_NAME(x) _Generic((x), \
        int: "int", \
        double: "double", \
        char: "char", \
        default: "unknown" \
    )

    printf("i 的类型: %s\n", TYPE_NAME(i));  // "int"
    printf("d 的类型: %s\n", TYPE_NAME(d));  // "double"
    printf("c 的类型: %s\n", TYPE_NAME(c));  // "int"（char被提升为int！）

    return 0;
}
```

> **注意**：`char` 和 `short` 类型在表达式中会被**整型提升**为 `int`，所以在 `_Generic` 中需要特别注意。

## 三、实现类型安全的泛型函数

### 3.1 泛型打印

```c
#include <stdio.h>
#include <math.h>

// 泛型打印宏
#define PRINT(x) _Generic((x), \
    int: printf("int: %d\n", x), \
    long: printf("long: %ld\n", x), \
    double: printf("double: %f\n", x), \
    float: printf("float: %f\n", x), \
    char*: printf("string: %s\n", x), \
    default: printf("unknown type\n") \
)

int main(void) {
    PRINT(42);          // int: 42
    PRINT(3.14);        // double: 3.140000
    PRINT(3.14f);       // float: 3.140000
    PRINT("hello");     // string: hello
    PRINT(100L);        // long: 100

    return 0;
}
```

### 3.2 泛型数学函数

```c
#include <stdio.h>
#include <math.h>

// 浮点类型使用 fabs，整数类型使用自定义实现
static inline int abs_int(int x) { return x < 0 ? -x : x; }
static inline long abs_long(long x) { return x < 0 ? -x : x; }

#define ABS(x) _Generic((x), \
    int: abs_int(x), \
    long: abs_long(x), \
    float: fabsf(x), \
    double: fabs(x), \
    long double: fabsl(x) \
)

int main(void) {
    printf("abs(-5) = %d\n", ABS(-5));           // 5（int版本）
    printf("abs(-5L) = %ld\n", ABS(-5L));        // 5（long版本）
    printf("abs(-3.14) = %f\n", ABS(-3.14));     // 3.140000（double版本）
    printf("abs(-2.5f) = %f\n", ABS(-2.5f));     // 2.500000（float版本）

    return 0;
}
```

### 3.3 泛型平方函数

```c
#include <stdio.h>

static inline int square_int(int x) { return x * x; }
static inline double square_double(double x) { return x * x; }
static inline float square_float(float x) { return x * x; }

#define SQUARE(x) _Generic((x), \
    int: square_int(x), \
    double: square_double(x), \
    float: square_float(x), \
    default: square_double((double)(x)) \
)

int main(void) {
    printf("%d\n", SQUARE(5));        // 25
    printf("%f\n", SQUARE(2.5));      // 6.250000
    printf("%f\n", SQUARE(2.5f));     // 6.250000

    return 0;
}
```

## 四、与 typeof 配合使用

```c
// 注意：typeof 是GCC扩展，不是C标准
// C23将引入 typeof

// 使用 _Generic 实现泛型交换
#define SWAP(a, b) do { \
    _Generic((a), \
        int: swap_int, \
        double: swap_double, \
        float: swap_float \
    )(&(a), &(b)); \
} while (0)

static inline void swap_int(int *a, int *b) {
    int tmp = *a; *a = *b; *b = tmp;
}
static inline void swap_double(double *a, double *b) {
    double tmp = *a; *a = *b; *b = tmp;
}
static inline void swap_float(float *a, float *b) {
    float tmp = *a; *a = *b; *b = tmp;
}
```

## 五、整型提升的注意事项

```c
#include <stdio.h>

// char 和 short 会被提升为 int！
#define TYPE_NAME(x) _Generic((x), \
    char: "char",       // 永远不会匹配！
    short: "short",     // 永远不会匹配！
    int: "int",         // char和short都会匹配到这里
    double: "double", \
    default: "other" \
)

int main(void) {
    char c = 'A';
    short s = 10;

    printf("c 的类型: %s\n", TYPE_NAME(c));  // "int"（不是"char"！）
    printf("s 的类型: %s\n", TYPE_NAME(s));  // "int"（不是"short"！）

    // 解决方案：使用赋值来避免提升
    #define TYPE_NAME_SAFE(x) _Generic(&(x), \
        char*: "char", \
        short*: "short", \
        int*: "int", \
        double*: "double", \
        default: "other" \
    )
    // 通过对变量取地址来避免整型提升

    return 0;
}
```

## 六、实际应用场景

### 6.1 类型安全的 printf 包装

```c
#include <stdio.h>

#define PRINTF_FMT(x) _Generic((x), \
    int: "%d", \
    long: "%ld", \
    double: "%f", \
    char*: "%s", \
    void*: "%p", \
    default: "%p" \
)

// 使用方式
// printf(PRINTF_FMT(x), x);
// printf(PRINTF_FMT(y), y);
```

### 6.2 类型检查

```c
#include <stdio.h>

// 编译时类型检查
#define IS_FLOATING(x) _Generic((x), \
    float: 1, \
    double: 1, \
    long double: 1, \
    default: 0 \
)

int main(void) {
    int i = 42;
    double d = 3.14;

    if (IS_FLOATING(i)) {
        printf("i 是浮点类型\n");
    } else {
        printf("i 不是浮点类型\n");  // 输出这个
    }

    if (IS_FLOATING(d)) {
        printf("d 是浮点类型\n");  // 输出这个
    }

    return 0;
}
```

## 七、关键要点

1. `_Generic` 是C11引入的**编译时**类型选择机制。
2. 语法：`_Generic(expr, type1: expr1, type2: expr2, default: default_expr)`。
3. 匹配是在编译时完成的，没有运行时开销。
4. 注意**整型提升**：`char`、`short` 在表达式中会提升为 `int`。
5. 可以用来实现类型安全的泛型宏和泛型函数。
6. `_Generic` 的每个分支必须是**有效的表达式**（即使不会被选择的分支也必须语法正确）。
7. 常用于实现类型安全的打印、数学函数、比较函数等。
8. C23将引入 `typeof` 关键字，与 `_Generic` 配合使用将更加强大。
