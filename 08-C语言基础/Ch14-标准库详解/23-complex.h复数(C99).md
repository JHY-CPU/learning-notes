# complex.h - 复数运算（C99）

## 1. 概述

`<complex.h>`（C99引入）提供了复数类型和复数运算函数，支持复数的算术运算、三角函数、指数对数等。

## 2. 复数类型

```c
#include <complex.h>

// 复数类型
float complex       // 单精度复数
double complex      // 双精度复数（最常用）
long double complex // 扩展精度复数

// 类型别名
float _Complex
double _Complex
long double _Complex
```

## 3. 基本操作

### 3.1 创建复数

```c
#include <stdio.h>
#include <complex.h>

int main(void) {
    // 创建复数的三种方式

    // 方式1: 使用I宏（推荐）
    double complex z1 = 3.0 + 4.0 * I;

    // 方式2: 使用CMPLX宏（C11）
    double complex z2 = CMPLX(3.0, 4.0);

    // 方式3: 直接赋值实部虚部
    double complex z3;
    z3 = 3.0 + 4.0 * I;

    // 提取实部和虚部
    printf("z1 = %f + %fi\n", creal(z1), cimag(z1));
    printf("z2 = %f + %fi\n", creal(z2), cimag(z2));

    // 纯虚数
    double complex pure_imag = 5.0 * I;
    printf("纯虚数: %f + %fi\n",
           creal(pure_imag), cimag(pure_imag));

    // 负数虚部
    double complex z4 = 2.0 - 3.0 * I;
    printf("z4 = %f + %fi\n", creal(z4), cimag(z4));

    return 0;
}
```

### 3.2 基本算术运算

```c
#include <stdio.h>
#include <complex.h>

int main(void) {
    double complex a = 3.0 + 4.0 * I;
    double complex b = 1.0 + 2.0 * I;

    // 四则运算
    double complex sum = a + b;    // (4+6i)
    double complex diff = a - b;   // (2+2i)
    double complex prod = a * b;   // (3*1-4*2) + (3*2+4*1)i = (-5+10i)
    double complex quot = a / b;   // 复数除法

    printf("a = %f + %fi\n", creal(a), cimag(a));
    printf("b = %f + %fi\n", creal(b), cimag(b));
    printf("a + b = %f + %fi\n", creal(sum), cimag(sum));
    printf("a - b = %f + %fi\n", creal(diff), cimag(diff));
    printf("a * b = %f + %fi\n", creal(prod), cimag(prod));
    printf("a / b = %f + %fi\n", creal(quot), cimag(quot));

    // 共轭复数
    double complex conj_a = conj(a);
    printf("conj(a) = %f + %fi\n",
           creal(conj_a), cimag(conj_a));

    // 模（绝对值）
    double mod = cabs(a);
    printf("|a| = %f\n", mod);  // sqrt(9+16) = 5

    // 辐角（相位）
    double phase = carg(a);
    printf("arg(a) = %f 弧度\n", phase);

    return 0;
}
```

## 4. 复数函数

### 4.1 三角函数

```c
#include <stdio.h>
#include <complex.h>

#define PI 3.14159265358979323846

int main(void) {
    double complex z = 1.0 + 1.0 * I;

    // 三角函数
    printf("sin(z) = %f + %fi\n",
           creal(csin(z)), cimag(csin(z)));
    printf("cos(z) = %f + %fi\n",
           creal(ccos(z)), cimag(ccos(z)));
    printf("tan(z) = %f + %fi\n",
           creal(ctan(z)), cimag(ctan(z)));

    // 反三角函数
    printf("asin(0.5+0.5i) = %f + %fi\n",
           creal(casin(0.5 + 0.5*I)),
           cimag(casin(0.5 + 0.5*I)));
    printf("acos(0.5+0.5i) = %f + %fi\n",
           creal(cacos(0.5 + 0.5*I)),
           cimag(cacos(0.5 + 0.5*I)));
    printf("atan(1+i) = %f + %fi\n",
           creal(catan(1.0 + I)),
           cimag(catan(1.0 + I)));

    return 0;
}
```

### 4.2 指数与对数

```c
#include <stdio.h>
#include <complex.h>

#define PI 3.14159265358979323846

int main(void) {
    // 欧拉公式: e^(i*theta) = cos(theta) + i*sin(theta)
    double theta = PI / 4;  // 45度
    double complex euler = cexp(I * theta);
    printf("e^(i*pi/4) = %f + %fi\n",
           creal(euler), cimag(euler));
    printf("cos(pi/4) + i*sin(pi/4) = %f + %fi\n",
           cos(theta), sin(theta));

    // 复数对数
    double complex z = 1.0 + I;
    double complex log_z = clog(z);
    printf("log(1+i) = %f + %fi\n",
           creal(log_z), cimag(log_z));

    // 复数幂
    double complex pow_result = cpow(1.0 + I, 2.0);
    printf("(1+i)^2 = %f + %fi\n",
           creal(pow_result), cimag(pow_result));

    // 复数平方根
    double complex sqrt_result = csqrt(-1.0);
    printf("sqrt(-1) = %f + %fi\n",
           creal(sqrt_result), cimag(sqrt_result));

    // sqrt(-4) = 2i
    sqrt_result = csqrt(-4.0);
    printf("sqrt(-4) = %f + %fi\n",
           creal(sqrt_result), cimag(sqrt_result));

    return 0;
}
```

## 5. 实用示例

### 5.1 复数多项式求值

```c
#include <stdio.h>
#include <complex.h>

// 霍纳法则求多项式值
// P(z) = a_n*z^n + a_{n-1}*z^{n-1} + ... + a_1*z + a_0
double complex poly_eval(double complex z, double *coeffs, int degree) {
    double complex result = coeffs[degree];
    for (int i = degree - 1; i >= 0; i--) {
        result = result * z + coeffs[i];
    }
    return result;
}

int main(void) {
    // P(z) = z^2 + 1，根应该是 +/- i
    double coeffs[] = {1.0, 0.0, 1.0};  // 1 + 0*z + 1*z^2

    double complex test_points[] = {
        I, -I, 1.0 + I, 0.5
    };

    for (int i = 0; i < 4; i++) {
        double complex val = poly_eval(test_points[i], coeffs, 2);
        printf("P(%f+%fi) = %f + %fi\n",
               creal(test_points[i]), cimag(test_points[i]),
               creal(val), cimag(val));
    }

    return 0;
}
```

### 5.2 信号处理中的应用

```c
#include <stdio.h>
#include <complex.h>
#include <math.h>

#define PI 3.14159265358979323846
#define N 8

// 简单的DFT（离散傅里叶变换）
void dft(double complex *input, double complex *output, int n) {
    for (int k = 0; k < n; k++) {
        output[k] = 0;
        for (int j = 0; j < n; j++) {
            double angle = -2.0 * PI * k * j / n;
            output[k] += input[j] * cexp(I * angle);
        }
    }
}

int main(void) {
    // 创建一个简单的信号
    double complex signal[N];
    double complex spectrum[N];

    // 信号: cos(2*pi*f*t), f = 1/T
    for (int i = 0; i < N; i++) {
        signal[i] = cos(2.0 * PI * i / N);
    }

    printf("输入信号:\n");
    for (int i = 0; i < N; i++) {
        printf("  signal[%d] = %f\n", i, creal(signal[i]));
    }

    // DFT
    dft(signal, spectrum, N);

    printf("\n频谱:\n");
    for (int k = 0; k < N; k++) {
        double mag = cabs(spectrum[k]);
        printf("  |X[%d]| = %f\n", k, mag);
    }

    return 0;
}
```

## 6. 重要注意事项

> **要点一**：`I` 宏可能与代码中其他地方使用的 `I` 冲突，可使用 `#undef I` 或使用 `_Imaginary_I`。

> **要点二**：C99支持复数和虚数（`_Imaginary`）两种类型，虚数是实部为0的特殊复数。

> **要点三**：`cabs` 返回复数的模（绝对值），等价于 `sqrt(re*re + im*im)`。

> **要点四**：`carg` 返回复数的辐角（主值），范围是 `[-pi, pi]`。

> **要点五**：`cproj` 返回黎曼球面上的投影。

> **要点六**：所有复数函数都有 `float`（前缀c，后缀f）和 `long double`（前缀c，后缀l）版本。

> **要点七**：复数运算可能涉及无穷大和NaN，遵循IEEE 754扩展的复数运算规则。

> **要点八**：`<tgmath.h>` 提供了泛型宏，可以自动选择 `float`/`double`/`long double` 版本的复数函数。
