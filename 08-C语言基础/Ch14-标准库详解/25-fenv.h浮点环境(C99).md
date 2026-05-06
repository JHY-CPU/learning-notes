# fenv.h - 浮点环境（C99）

## 1. 概述

`<fenv.h>`（C99引入）提供了访问和控制浮点环境的函数，包括舍入模式、浮点异常状态等。

## 2. 核心类型

```c
#include <fenv.h>

fenv_t      // 浮点环境类型
fexcept_t   // 浮点异常标志集合类型
```

## 3. 舍入模式

### 3.1 可用的舍入模式

```c
#include <fenv.h>

FE_TONEAREST    // 向最近舍入（默认）
FE_UPWARD       // 向正无穷舍入
FE_DOWNWARD     // 向负无穷舍入
FE_TOWARDZERO   // 向零舍入
```

```c
#include <stdio.h>
#include <fenv.h>
#include <math.h>

int main(void) {
    // 获取当前舍入模式
    int mode = fegetround();
    printf("当前舍入模式: %d\n", mode);

    double val = 2.5;

    // 设置舍入模式并测试
    printf("\n不同舍入模式下的 round(2.5):\n");

    fesetround(FE_TONEAREST);
    printf("  FE_TONEAREST: %f\n", round(val));

    fesetround(FE_UPWARD);
    printf("  FE_UPWARD:    %f\n", round(val));

    fesetround(FE_DOWNWARD);
    printf("  FE_DOWNWARD:  %f\n", round(val));

    fesetround(FE_TOWARDZERO);
    printf("  FE_TOWARDZERO: %f\n", round(val));

    // 恢复默认
    fesetround(FE_TONEAREST);

    return 0;
}
```

### 3.2 舍入模式对计算的影响

```c
#include <stdio.h>
#include <fenv.h>

int main(void) {
    double a = 1.0;
    double b = 3.0;

    printf("1.0 / 3.0 在不同舍入模式下:\n");

    // 向零舍入
    fesetround(FE_TOWARDZERO);
    printf("  向零: %.20f\n", a / b);

    // 向正无穷舍入
    fesetround(FE_UPWARD);
    printf("  向上: %.20f\n", a / b);

    // 向负无穷舍入
    fesetround(FE_DOWNWARD);
    printf("  向下: %.20f\n", a / b);

    // 向最近舍入
    fesetround(FE_TONEAREST);
    printf("  最近: %.20f\n", a / b);

    return 0;
}
```

## 4. 浮点异常

### 4.1 异常类型

```c
#include <fenv.h>

FE_DIVBYZERO     // 除以零
FE_INEXACT       // 结果不精确
FE_INVALID       // 无效操作
FE_OVERFLOW      // 上溢
FE_UNDERFLOW     // 下溢
FE_ALL_EXCEPT    // 所有异常
```

### 4.2 异常操作

```c
#include <stdio.h>
#include <fenv.h>
#include <math.h>

#pragma STDC FENV_ACCESS ON

int main(void) {
    // 清除所有异常标志
    feclearexcept(FE_ALL_EXCEPT);

    // 执行可能触发异常的计算
    double result = 1.0 / 0.0;
    printf("1.0 / 0.0 = %f\n", result);

    // 检查异常
    if (fetestexcept(FE_DIVBYZERO)) {
        printf("检测到除以零异常\n");
    }

    if (fetestexcept(FE_INVALID)) {
        printf("检测到无效操作异常\n");
    }

    // 清除并测试其他异常
    feclearexcept(FE_ALL_EXCEPT);

    double tiny = 1e-300 * 1e-300;  // 下溢
    printf("下溢结果: %e\n", tiny);

    if (fetestexcept(FE_UNDERFLOW)) {
        printf("检测到下溢异常\n");
    }

    // 获取并清除所有异常
    feclearexcept(FE_ALL_EXCEPT);

    double huge = 1e300 * 1e300;  // 上溢
    printf("上溢结果: %f\n", huge);

    if (fetestexcept(FE_OVERFLOW)) {
        printf("检测到上溢异常\n");
    }

    // 不精确异常
    feclearexcept(FE_ALL_EXCEPT);

    double imprecise = 1.0 / 3.0;
    (void)imprecise;

    if (fetestexcept(FE_INEXACT)) {
        printf("检测到不精确异常\n");
    }

    return 0;
}
```

## 5. 浮点环境的保存与恢复

```c
#include <stdio.h>
#include <fenv.h>

#pragma STDC FENV_ACCESS ON

int main(void) {
    fenv_t saved_env;

    // 保存当前浮点环境
    fegetenv(&saved_env);

    // 修改环境
    fesetround(FE_UPWARD);
    feclearexcept(FE_ALL_EXCEPT);

    printf("修改后的舍入模式: %d\n", fegetround());

    // 恢复环境
    fesetenv(&saved_env);

    printf("恢复后的舍入模式: %d\n", fegetround());

    return 0;
}
```

## 6. 重要注意事项

> **要点一**：使用浮点环境控制时，应在代码前添加 `#pragma STDC FENV_ACCESS ON`，告诉编译器不要优化掉浮点环境相关操作。

> **要点二**：`fegetround` 返回当前舍入模式的宏值，`fesetround` 设置舍入模式。

> **要点三**：异常标志是"粘性"的，一旦设置不会自动清除，需要调用 `feclearexcept`。

> **要点四**：`fetestexcept` 检查指定的异常标志是否被设置。

> **要点五**：`FE_ALL_EXCEPT` 是所有异常标志的按位或。

> **要点六**：并非所有实现都支持所有舍入模式和异常类型，使用前应检查宏是否定义。

> **要点七**：浮点环境操作可能有性能开销，通常只在需要精确浮点控制的场景中使用。

> **要点八**：`feholdexcept` 可以保存环境并清除异常，`feupdateenv` 可以恢复环境并保留当前异常。
