# tgmath.h - 泛型数学宏（C99）

## 1. 概述

`<tgmath.h>`（C99引入）提供了泛型数学宏，自动根据参数类型选择正确的数学函数版本（`float`、`double`、`long double` 或复数版本），简化了数学函数的使用。

## 2. 工作原理

`<tgmath.h>` 中的宏使用 `_Generic` 选择表达式（C11）或编译器内建机制，在编译时根据参数类型选择合适的函数：

```c
// 以sqrt为例，tgmath.h中的实现类似于：
#define sqrt(x) _Generic((x), \
    float: sqrtf, \
    double: sqrt, \
    long double: sqrtl, \
    float complex: csqrtf, \
    double complex: csqrt, \
    long double complex: csqrtl \
)(x)
```

## 3. 可用的泛型宏

### 3.1 三角函数

```c
#include <tgmath.h>

sin     cos     tan
asin    acos    atan
sinh    cosh    tanh
asinh   acosh   atanh
atan2
```

### 3.2 指数与对数

```c
exp     exp2    expm1
log     log2    log10   log1p
```

### 3.3 幂与根

```c
pow     sqrt    cbrt    hypot
```

### 3.4 取整与取余

```c
ceil    floor   round   trunc
fmod    remainder   nearbyint   rint
```

### 3.5 其他

```c
fabs    fmax    fmin    fdim    fma
copysign    nan
```

### 3.6 复数相关

```c
carg    conj    cimag   creal   cproj
```

## 4. 使用示例

### 4.1 自动类型选择

```c
#include <stdio.h>
#include <tgmath.h>

int main(void) {
    // sqrt 自动选择版本
    float f = sqrt(2.0f);          // 调用 sqrtf
    double d = sqrt(2.0);          // 调用 sqrt
    long double ld = sqrt(2.0L);   // 调用 sqrtl

    printf("sqrt(2.0f) = %f\n", f);
    printf("sqrt(2.0)  = %f\n", d);
    printf("sqrt(2.0L) = %Lf\n", ld);

    // sin 同样自动选择
    float sf = sin(1.0f);          // sinf
    double sd = sin(1.0);          // sin
    long double sld = sin(1.0L);   // sinl

    printf("sin(1.0f) = %f\n", sf);
    printf("sin(1.0)  = %f\n", sd);
    printf("sin(1.0L) = %Lf\n", sld);

    return 0;
}
```

### 4.2 复数支持

```c
#include <stdio.h>
#include <tgmath.h>
#include <complex.h>

int main(void) {
    // sqrt 自动处理复数
    double complex z = sqrt(-1.0 + 0.0*I);
    printf("sqrt(-1) = %f + %fi\n", creal(z), cimag(z));

    // exp 处理复数（欧拉公式）
    double complex euler = exp(I * 3.14159265358979);
    printf("e^(i*pi) = %f + %fi\n",
           creal(euler), cimag(euler));

    // log 处理复数
    double complex log_z = log(1.0 + I);
    printf("log(1+i) = %f + %fi\n",
           creal(log_z), cimag(log_z));

    // pow 处理复数
    double complex pow_z = pow(1.0 + I, 2.0);
    printf("(1+i)^2 = %f + %fi\n",
           creal(pow_z), cimag(pow_z));

    return 0;
}
```

### 4.3 编写泛型数学函数

```c
#include <stdio.h>
#include <tgmath.h>

// 使用tgmath宏编写的泛型函数
// 自动支持float、double、long double
#define DISTANCE_2D(x1, y1, x2, y2) \
    sqrt(((x2)-(x1))*((x2)-(x1)) + ((y2)-(y1))*((y2)-(y1)))

// 泛型二次方程求解
#define QUADRATIC_DISCRIMINANT(a, b, c) \
    ((b)*(b) - 4*(a)*(c))

int main(void) {
    // float版本
    float fd = DISTANCE_2D(0.0f, 0.0f, 3.0f, 4.0f);
    printf("float距离: %f\n", fd);  // 5.0

    // double版本
    double dd = DISTANCE_2D(0.0, 0.0, 3.0, 4.0);
    printf("double距离: %f\n", dd);  // 5.0

    // long double版本
    long double ld = DISTANCE_2D(0.0L, 0.0L, 3.0L, 4.0L);
    printf("long double距离: %Lf\n", ld);  // 5.0

    // 二次方程判别式
    double disc = QUADRATIC_DISCRIMINANT(1.0, 2.0, -3.0);
    printf("判别式: %f\n", disc);  // 16.0
    if (disc > 0) {
        double root = sqrt(disc);
        printf("平方根: %f\n", root);
    }

    return 0;
}
```

## 5. 重要注意事项

> **要点一**：泛型宏在编译时确定调用的函数版本，没有运行时开销。

> **要点二**：如果参数类型不匹配任何选项，行为取决于编译器实现。

> **要点三**：`tgmath.h` 中的宏可能会与同名的函数指针产生冲突。

> **要点四**：对于混合类型的参数（如 `float` 和 `double`），具体选择哪个版本由实现定义。

> **要点五**：`tgmath.h` 是基于 `<math.h>` 和 `<complex.h>` 构建的。

> **要点六**：使用 `tgmath.h` 时，仍需包含 `<math.h>` 或 `<complex.h>` 来使用其中的类型和常量。

> **要点七**：并非所有编译器都完美支持 `tgmath.h`，某些旧编译器可能有兼容性问题。
