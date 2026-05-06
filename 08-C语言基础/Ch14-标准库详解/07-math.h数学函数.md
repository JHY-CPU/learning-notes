# math.h - 数学函数

## 1. 概述

`<math.h>` 提供了丰富的数学运算函数，包括三角函数、指数对数函数、取整函数等。C99对数学库进行了大幅扩展，增加了 `float` 和 `long double` 版本以及许多新函数。

## 2. 数学常量

```c
#include <math.h>

// C99定义了数学宏（不是常量）
// M_PI, M_E 等不是标准的，但在大多数实现中可用
// 安全做法：自己定义

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

#ifndef M_E
#define M_E 2.71828182845904523536
#endif
```

## 3. 三角函数

```c
#include <math.h>

double sin(double x);    // 正弦
double cos(double x);    // 余弦
double tan(double x);    // 正切
double asin(double x);   // 反正弦
double acos(double x);   // 反余弦
double atan(double x);   // 反正切
double atan2(double y, double x);  // 四象限反正切
```

### C99 新增 float/long double 版本

```c
// float版本（后缀f）
float sinf(float x);
float cosf(float x);
float tanf(float x);

// long double版本（后缀l）
double sinl(long double x);
double cosl(long double x);
double tanl(long double x);
```

```c
#include <stdio.h>
#include <math.h>

#define PI 3.14159265358979323846

int main(void) {
    // 基本三角函数
    printf("sin(0) = %f\n", sin(0));           // 0
    printf("sin(PI/6) = %f\n", sin(PI / 6));   // 0.5
    printf("sin(PI/2) = %f\n", sin(PI / 2));   // 1.0
    printf("cos(0) = %f\n", cos(0));           // 1.0
    printf("cos(PI) = %f\n", cos(PI));         // -1.0
    printf("tan(PI/4) = %f\n", tan(PI / 4));   // 1.0

    // 反三角函数
    printf("asin(0.5) = %f 弧度\n", asin(0.5));        // PI/6
    printf("asin(0.5) = %f 度\n", asin(0.5) * 180/PI); // 30度
    printf("acos(0.5) = %f 度\n", acos(0.5) * 180/PI); // 60度
    printf("atan(1.0) = %f 度\n", atan(1.0) * 180/PI); // 45度

    // atan2 - 四象限反正切（推荐使用）
    printf("atan2(1, 1) = %f 度\n", atan2(1, 1) * 180/PI);   // 45
    printf("atan2(1, -1) = %f 度\n", atan2(1, -1) * 180/PI);  // 135
    printf("atan2(-1, -1) = %f 度\n", atan2(-1, -1) * 180/PI);// -135
    printf("atan2(-1, 1) = %f 度\n", atan2(-1, 1) * 180/PI);  // -45

    return 0;
}
```

## 4. 双曲函数

```c
double sinh(double x);   // 双曲正弦
double cosh(double x);   // 双曲余弦
double tanh(double x);   // 双曲正切
double asinh(double x);  // 反双曲正弦 (C99)
double acosh(double x);  // 反双曲余弦 (C99)
double atanh(double x);  // 反双曲正切 (C99)
```

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("sinh(0) = %f\n", sinh(0));   // 0
    printf("cosh(0) = %f\n", cosh(0));   // 1
    printf("tanh(1) = %f\n", tanh(1));   // 0.761594

    // 双曲函数的关系: cosh²(x) - sinh²(x) = 1
    double x = 2.0;
    double identity = cosh(x) * cosh(x) - sinh(x) * sinh(x);
    printf("cosh²(2) - sinh²(2) = %f\n", identity);  // 1.0

    return 0;
}
```

## 5. 指数与对数函数

```c
double exp(double x);     // e^x
double exp2(double x);    // 2^x (C99)
double log(double x);     // 自然对数 ln(x)
double log2(double x);    // 以2为底 (C99)
double log10(double x);   // 常用对数 lg(x)
double log1p(double x);   // ln(1+x) (C99)，精度更高
double pow(double x, double y);  // x^y
double sqrt(double x);    // 平方根
double cbrt(double x);    // 立方根 (C99)
```

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    // 指数函数
    printf("e^1 = %f\n", exp(1));       // 2.718282
    printf("e^0 = %f\n", exp(0));       // 1.0
    printf("2^10 = %f\n", exp2(10));    // 1024.0

    // 对数函数
    printf("ln(e) = %f\n", log(exp(1)));  // 1.0
    printf("log2(1024) = %f\n", log2(1024));  // 10.0
    printf("log10(1000) = %f\n", log10(1000));  // 3.0

    // log1p - 计算 ln(1+x)，当x很小时精度更高
    double small = 1e-15;
    printf("log(1 + 1e-15) = %.20f\n", log(1 + small));
    printf("log1p(1e-15) = %.20f\n", log1p(small));

    // 幂函数
    printf("2^3 = %f\n", pow(2, 3));    // 8.0
    printf("sqrt(16) = %f\n", sqrt(16)); // 4.0
    printf("cbrt(27) = %f\n", cbrt(27)); // 3.0
    printf("sqrt(2) = %f\n", sqrt(2));   // 1.414214

    // 复合计算
    // 计算 log₂(e)
    printf("log₂(e) = %f\n", log2(exp(1)));  // 1.442695

    return 0;
}
```

## 6. 取整与取余函数

```c
double ceil(double x);    // 向上取整
double floor(double x);   // 向下取整
double trunc(double x);   // 截断小数部分 (C99)
double round(double x);   // 四舍五入 (C99)
long lround(double x);    // 四舍五入返回long (C99)
long long llround(double x);  // C99
double rint(double x);    // 按当前舍入模式取整 (C99)
double nearbyint(double x);  // C99
double fmod(double x, double y);  // 浮点取余
double remainder(double x, double y);  // IEEE余数 (C99)
double remquo(double x, double y, int *quo);  // C99
```

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    double vals[] = {2.3, 2.5, 2.7, -2.3, -2.5, -2.7};
    int n = 6;

    printf("值      ceil    floor   trunc   round\n");
    printf("------ ------- ------- ------- -------\n");
    for (int i = 0; i < n; i++) {
        double v = vals[i];
        printf("%6.1f %7.1f %7.1f %7.1f %7.1f\n",
               v, ceil(v), floor(v), trunc(v), round(v));
    }
    // 注意 round 的行为:
    // 2.5 -> 3, -2.5 -> -3 (远离零的方向)

    // fmod - 浮点取余
    printf("\nfmod(5.3, 2.0) = %f\n", fmod(5.3, 2.0));  // 1.3
    printf("fmod(-5.3, 2.0) = %f\n", fmod(-5.3, 2.0));  // -1.3

    // remainder - IEEE 754余数
    printf("remainder(5.3, 2.0) = %f\n", remainder(5.3, 2.0));  // -0.7

    return 0;
}
```

## 7. 绝对值与最值函数

```c
double fabs(double x);        // 绝对值
double fmax(double x, double y);  // 最大值 (C99)
double fmin(double x, double y);  // 最小值 (C99)
double fdim(double x, double y);  // 正差 (C99)
double copysign(double x, double y);  // 复制符号 (C99)
```

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    printf("fabs(-3.14) = %f\n", fabs(-3.14));   // 3.14
    printf("fmax(3.5, 2.8) = %f\n", fmax(3.5, 2.8));  // 3.5
    printf("fmin(3.5, 2.8) = %f\n", fmin(3.5, 2.8));  // 2.8

    // fdim: max(x-y, 0)
    printf("fdim(5.0, 3.0) = %f\n", fdim(5.0, 3.0));  // 2.0
    printf("fdim(3.0, 5.0) = %f\n", fdim(3.0, 5.0));  // 0.0

    // copysign: 取x的绝对值，y的符号
    printf("copysign(3.14, -1.0) = %f\n", copysign(3.14, -1.0));  // -3.14
    printf("copysign(-3.14, 1.0) = %f\n", copysign(-3.14, 1.0));  // 3.14

    return 0;
}
```

## 8. 浮点数分类（C99）

```c
#include <math.h>
#include <stdio.h>

int main(void) {
    double values[] = {0.0, -0.0, 1.0/0.0, -1.0/0.0, 0.0/0.0, 1e-300};
    const char *names[] = {"0.0", "-0.0", "Inf", "-Inf", "NaN", "1e-300"};

    for (int i = 0; i < 6; i++) {
        double v = values[i];
        printf("%s:\n", names[i]);
        printf("  isnan:    %d\n", isnan(v));
        printf("  isinf:    %d\n", isinf(v));
        printf("  isfinite: %d\n", isfinite(v));
        printf("  isnormal: %d\n", isnormal(v));
        printf("  signbit:  %d\n", signbit(v));
        printf("\n");
    }

    return 0;
}
```

## 9. 融合乘加（FMA）- C99

```c
double fma(double x, double y, double z);  // x * y + z（一次舍入）
```

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    // fma: 计算 x*y+z，只做一次舍入（更精确）
    double x = 1.0000000000000002;
    double y = 1.0000000000000002;
    double z = -1.0;

    double result1 = x * y + z;  // 两次舍入
    double result2 = fma(x, y, z);  // 一次舍入

    printf("x*y + z = %.20f\n", result1);
    printf("fma(x,y,z) = %.20f\n", result2);

    return 0;
}
```

## 10. 特殊值处理

```c
#include <stdio.h>
#include <math.h>

int main(void) {
    // 无穷大和NaN的生成
    double inf = 1.0 / 0.0;    // +Infinity
    double neginf = -1.0 / 0.0; // -Infinity
    double nan_val = 0.0 / 0.0; // NaN

    printf("inf = %f\n", inf);         // inf
    printf("-inf = %f\n", neginf);      // -inf
    printf("NaN = %f\n", nan_val);      // nan

    // NaN的特性：任何比较都返回false
    printf("NaN == NaN: %d\n", nan_val == nan_val);  // 0 (false!)
    printf("NaN != NaN: %d\n", nan_val != nan_val);  // 1 (true!)

    // 判断特殊值
    printf("isinf(inf): %d\n", isinf(inf));         // 非零
    printf("isnan(NaN): %d\n", isnan(nan_val));     // 非零
    printf("isfinite(1.0): %d\n", isfinite(1.0));   // 非零

    // 使用宏判断
    printf("isinf: %d\n", isinf(inf));           // 正无穷返回+1，负无穷返回-1
    printf("fpclassify(NaN): %d\n", fpclassify(nan_val));  // FP_NAN

    return 0;
}
```

## 11. 重要注意事项

> **要点一**：三角函数的参数是弧度，不是角度。角度转弧度：`弧度 = 角度 * PI / 180`。

> **要点二**：`asin` 和 `acos` 的参数范围是 [-1, 1]，超出范围会返回 NaN 并设置 `errno`。

> **要点三**：`pow(0, 0)` 返回 1，这与数学定义一致。

> **要点四**：`sqrt` 对负数参数返回 NaN，如果需要复数平方根，使用 `<complex.h>`。

> **要点五**：浮点数精度有限，不要直接用 `==` 比较浮点数。

> **要点六**：`round` 的行为是"半远离零"：`round(2.5) = 3`，`round(-2.5) = -3`。

> **要点七**：`NaN` 的所有比较操作都返回 false，包括 `NaN == NaN`。

> **要点八**：C99提供了 `float`（后缀f）和 `long double`（后缀l）版本的函数，如 `sinf`、`sinl`。
