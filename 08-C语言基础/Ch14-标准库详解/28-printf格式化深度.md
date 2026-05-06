# printf 格式化深度解析

## 1. 概述

`printf` 的格式字符串语法非常强大，包含格式说明符、标志、宽度、精度和长度修饰符。本章详细解析所有格式化选项。

## 2. 格式说明符完整参考

### 2.1 通用语法

```
%[flags][width][.precision][length]specifier
```

### 2.2 说明符一览

| 说明符 | 类型 | 输出格式 |
|--------|------|----------|
| `%d`, `%i` | int | 有符号十进制 |
| `%u` | unsigned int | 无符号十进制 |
| `%o` | unsigned int | 无符号八进制 |
| `%x` | unsigned int | 无符号十六进制（小写） |
| `%X` | unsigned int | 无符号十六进制（大写） |
| `%f`, `%F` | double | 十进制浮点 |
| `%e`, `%E` | double | 科学计数法 |
| `%g`, `%G` | double | 自动选择 %f 或 %e |
| `%a`, `%A` | double | 十六进制浮点（C99） |
| `%c` | int | 单个字符 |
| `%s` | char* | 字符串 |
| `%p` | void* | 指针地址 |
| `%n` | int* | 已输出字符数 |
| `%%` | - | 百分号 |

## 3. 标志（Flags）

```c
#include <stdio.h>

int main(void) {
    int val = 42;

    // - 左对齐
    printf("|%-10d|\n", val);    // |42        |

    // + 总是显示符号
    printf("|%+d|\n", val);      // |+42|
    printf("|%+d|\n", -val);     // |-42|

    // 空格 正数前加空格
    printf("|% d|\n", val);      // | 42|
    printf("|% d|\n", -val);     // |-42|

    // 0 零填充
    printf("|%010d|\n", val);    // |0000000042|

    // # 替换形式
    printf("%#x\n", 255);        // 0xff
    printf("%#X\n", 255);        // 0XFF
    printf("%#o\n", 255);        // 0377
    printf("%#f\n", 3.0);        // 3.000000（确保小数点）
    printf("%#g\n", 3.0);        // 3.00000（确保小数点和尾随零）

    return 0;
}
```

### 标志组合

```c
#include <stdio.h>

int main(void) {
    // 组合使用标志
    printf("|%+-10d|\n", 42);    // |+42       |
    printf("|% 010d|\n", 42);    // | 000000042|
    printf("|%#-10x|\n", 255);   // |0xff      |

    // 零填充与左对齐冲突时，-优先
    printf("|%-010d|\n", 42);    // |42        |

    return 0;
}
```

## 4. 宽度（Width）

```c
#include <stdio.h>

int main(void) {
    int val = 42;

    // 固定宽度
    printf("|%10d|\n", val);     // |        42| 右对齐，宽度10

    // 宽度不足时自动扩展
    printf("|%2d|\n", 12345);    // |12345|

    // 使用*从参数获取宽度
    printf("|%*d|\n", 10, val);  // |        42|

    // 负宽度相当于左对齐标志
    printf("|%*d|\n", -10, val); // |42        |

    return 0;
}
```

## 5. 精度（Precision）

```c
#include <stdio.h>

int main(void) {
    // 对于浮点数：小数位数
    printf("%.2f\n", 3.14159);    // 3.14
    printf("%.0f\n", 3.14159);    // 3
    printf("%.10f\n", 3.14159);   // 3.1415900000

    // 对于整数：最小数字位数
    printf("%.5d\n", 42);         // 00042
    printf("%.5d\n", 123456);     // 123456（不截断）

    // 对于字符串：最大字符数
    printf("%.5s\n", "Hello, World!"); // Hello
    printf("%.0s\n", "Hello");         // （空字符串）

    // 精度为0且值为0
    printf("%.0d\n", 0);          // （空）
    printf("%.0o\n", 0);          // （空）
    printf("%#.0o\n", 0);         // 0（#保留前导零）

    // 使用*从参数获取精度
    printf("%.*f\n", 3, 3.14159);  // 3.142

    // 宽度和精度同时使用
    printf("%10.2f\n", 3.14159);  //       3.14
    printf("%-10.2f|\n", 3.14159); // 3.14      |

    return 0;
}
```

## 6. 长度修饰符

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    // h: short
    printf("%hd\n", (short)12345);

    // hh: signed char / unsigned char
    printf("%hhu\n", (unsigned char)255);

    // l: long
    printf("%ld\n", 123456789L);

    // ll: long long
    printf("%lld\n", 123456789012345LL);

    // j: intmax_t
    printf("%jd\n", (intmax_t)12345);

    // z: size_t
    printf("%zu\n", sizeof(int));

    // t: ptrdiff_t
    int arr[10];
    printf("%td\n", &arr[5] - &arr[0]);  // 5

    // L: long double
    printf("%Lf\n", 3.14159L);

    return 0;
}
```

## 7. 特殊格式

### 7.1 %n - 输出字符计数

```c
#include <stdio.h>

int main(void) {
    int count;
    printf("Hello%n, World!\n", &count);
    printf("已输出 %d 个字符\n", count);  // 5

    // 使用 %hn, %hhn, %ln, %lln 指定类型
    short scount;
    printf("Test%hn\n", &scount);

    return 0;
}
```

### 7.2 %p - 指针格式

```c
#include <stdio.h>

int main(void) {
    int val = 42;
    int *ptr = &val;

    printf("指针: %p\n", (void*)ptr);

    // NULL指针
    int *null_ptr = NULL;
    printf("空指针: %p\n", (void*)null_ptr);

    return 0;
}
```

### 7.3 %g/%G - 自动选择

```c
#include <stdio.h>

int main(void) {
    // %g 选择 %f 或 %e 中较短的
    printf("%g\n", 0.00001234);   // 1.234e-05
    printf("%g\n", 123.456);      // 123.456
    printf("%g\n", 1234567.0);    // 1.23457e+06

    // 默认精度为6位有效数字
    printf("%g\n", 12345678.0);   // 1.23457e+07

    // 去除尾随零
    printf("%g\n", 3.14000);      // 3.14

    // %G 使用大写E
    printf("%G\n", 1234567.0);    // 1.23457E+06

    return 0;
}
```

## 8. 可变宽度和精度

```c
#include <stdio.h>

int main(void) {
    // 使用*动态指定宽度
    int width = 15;
    int precision = 3;
    double pi = 3.14159265358979;

    printf("%*.*f\n", width, precision, pi);  //           3.142

    // 动态宽度用于对齐表格
    printf("%-*s %*d\n", 10, "Name", 5, 100);
    printf("%-*s %*d\n", 10, "LongerName", 5, 200);

    return 0;
}
```

## 9. 安全格式化

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 使用snprintf安全格式化
    char buf[20];
    int written = snprintf(buf, sizeof(buf), "Value: %d", 12345);
    printf("buf = \"%s\", written = %d\n", buf, written);

    // 检查是否被截断
    if (written >= (int)sizeof(buf)) {
        printf("输出被截断\n");
    }

    // 计算需要的大小
    char *dynamic = NULL;
    int needed = snprintf(NULL, 0, "Name: %s, Age: %d", "Alice", 25);
    dynamic = malloc(needed + 1);
    if (dynamic) {
        snprintf(dynamic, needed + 1, "Name: %s, Age: %d", "Alice", 25);
        printf("%s\n", dynamic);
        free(dynamic);
    }

    return 0;
}
```

## 10. 重要注意事项

> **要点一**：格式字符串中的普通字符直接输出，`%%` 输出百分号。

> **要点二**：`%n` 存储已输出字符数到指针位置，存在安全风险，某些环境禁用。

> **要点三**：`printf` 返回输出的字符数（不含终止符），出错时返回负值。

> **要点四**：格式说明符的数量和类型必须与实际参数匹配。

> **要点五**：精度对字符串是截断，对整数是填充，对浮点数是小数位数。

> **要点六**：`snprintf` 的返回值是"应该"输出的字符数，即使被截断也如此。

> **要点七**：`%a` 和 `%A`（C99）输出十六进制浮点数，便于精确表示。

> **要点八**：`%g` 在精度范围内会去除尾随零和不必要的小数点。
