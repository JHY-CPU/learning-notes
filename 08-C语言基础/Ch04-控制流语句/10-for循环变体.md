# for 循环变体

## 1. 概述

C语言的 for 循环语法非常灵活，三个表达式（初始化、条件、更新）都可以省略或变化，形成多种变体。掌握这些变体能让你写出更灵活的循环代码。

## 2. 省略表达式

### 2.1 省略初始化

当循环变量已经在前面定义并初始化时：

```c
#include <stdio.h>

int main(void) {
    int i = 0;

    for (; i < 10; i++) {   // 省略初始化
        printf("%d ", i);
    }
    // 输出: 0 1 2 3 4 5 6 7 8 9

    return 0;
}
```

### 2.2 省略条件

省略条件意味着条件永远为真，形成**无限循环**：

```c
// 无限循环——需要 break 来退出
for (int i = 0; ; i++) {
    if (i >= 10) {
        break;
    }
    printf("%d ", i);
}
```

### 2.3 省略更新

更新操作在循环体内完成：

```c
#include <stdio.h>

int main(void) {
    // 打印2的幂次
    for (int i = 1; i < 1024; ) {
        printf("%d ", i);
        i *= 2;           // 在循环体内更新
    }
    // 输出: 1 2 4 8 16 32 64 128 256 512

    return 0;
}
```

### 2.4 全部省略——无限循环

```c
// 三个表达式全部省略——无条件的无限循环
for (;;) {
    printf("按Ctrl+C退出\n");
    // 必须用 break、return 或 exit() 退出
}
```

`for (;;)` 与 `while (1)` 等价，是C语言中最地道的无限循环写法。很多编译器会将 `for (;;)` 优化为无条件跳转，不会每次判断条件。

## 3. 多变量 for 循环

### 3.1 双变量遍历

使用逗号运算符在 for 中操作多个变量：

```c
#include <stdio.h>

int main(void) {
    // 两端向中间逼近
    for (int i = 0, j = 9; i < j; i++, j--) {
        printf("i=%d, j=%d\n", i, j);
    }
    // 输出:
    // i=0, j=9
    // i=1, j=8
    // i=2, j=7
    // i=3, j=6
    // i=4, j=5

    return 0;
}
```

### 3.2 同时遍历两个数组

```c
#include <stdio.h>

int main(void) {
    int a[] = {1, 2, 3, 4, 5};
    int b[] = {10, 20, 30, 40, 50};
    int n = 5;

    for (int i = 0; i < n; i++) {
        printf("%d + %d = %d\n", a[i], b[i], a[i] + b[i]);
    }

    return 0;
}
```

### 3.3 双指针法

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char str[] = "Hello";
    int len = strlen(str);

    // 反转字符串
    for (int i = 0, j = len - 1; i < j; i++, j--) {
        char temp = str[i];
        str[i] = str[j];
        str[j] = temp;
    }

    printf("反转后: %s\n", str);
    // 输出: 反转后: olleH

    return 0;
}
```

## 4. C99 声明特性

### 4.1 在 for 中声明变量

C99 标准允许在 for 的初始化部分声明变量：

```c
#include <stdio.h>

int main(void) {
    // i 只在 for 循环内可见
    for (int i = 0; i < 5; i++) {
        printf("i = %d\n", i);
    }

    // printf("%d\n", i);  // 错误：i 在此处不可见

    return 0;
}
```

### 4.2 声明多个变量

```c
// 两个变量都在 for 中声明（类型必须相同或兼容）
for (int i = 0, j = 10; i < j; i++, j--) {
    printf("i=%d, j=%d\n", i, j);
}

// 注意：不能声明不同类型
// for (int i = 0, double d = 1.0; ...)  // 错误
```

### 4.3 混合声明和已有变量

```c
int i;          // 外部声明

for (i = 0; i < 5; i++) {
    // 使用外部的 i
}

// i 在循环外仍然可用
printf("i = %d\n", i);  // 输出: i = 5
```

## 5. 复杂更新表达式

### 5.1 不规则步长

```c
#include <stdio.h>

int main(void) {
    // 打印奇数
    for (int i = 1; i < 20; i += 2) {
        printf("%d ", i);
    }
    printf("\n");
    // 输出: 1 3 5 7 9 11 13 15 17 19

    // 打印2的幂
    for (int i = 1; i <= 1024; i *= 2) {
        printf("%d ", i);
    }
    printf("\n");
    // 输出: 1 2 4 8 16 32 64 128 256 512 1024

    return 0;
}
```

### 5.2 逗号运算符的更新

```c
#include <stdio.h>

int main(void) {
    // 同时更新两个变量
    for (int i = 0, j = 0; i + j < 20; i++, j += 2) {
        printf("i=%d, j=%d, sum=%d\n", i, j, i + j);
    }

    return 0;
}
```

## 6. for 循环的特殊情况

### 6.1 循环体为空

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char str[] = "Hello";

    // 计算字符串长度（仅演示，实际用 strlen）
    int len;
    for (len = 0; str[len] != '\0'; len++)
        ;    // 空循环体，分号不能省略

    printf("长度 = %d\n", len);
    // 输出: 长度 = 5

    return 0;
}
```

### 6.2 循环体内声明变量

```c
for (int i = 0; i < 5; i++) {
    int temp = i * i;        // 每次循环都重新声明和初始化
    printf("%d ", temp);
}
// temp 在此处不可见
```

## 7. 综合示例

### 7.1 打印菱形

```c
#include <stdio.h>

int main(void) {
    int n = 5;  // 上半部分行数

    // 上半部分
    for (int i = 1; i <= n; i++) {
        for (int j = 0; j < n - i; j++) {
            printf(" ");
        }
        for (int j = 0; j < 2 * i - 1; j++) {
            printf("*");
        }
        printf("\n");
    }

    // 下半部分
    for (int i = n - 1; i >= 1; i--) {
        for (int j = 0; j < n - i; j++) {
            printf(" ");
        }
        for (int j = 0; j < 2 * i - 1; j++) {
            printf("*");
        }
        printf("\n");
    }

    return 0;
}
```

### 7.2 求最大公约数（辗转相除法）

```c
#include <stdio.h>

int main(void) {
    int a = 48, b = 18;

    // 用 for 循环实现辗转相除
    for (int temp; b != 0; ) {
        temp = b;
        b = a % b;
        a = temp;
    }

    printf("GCD = %d\n", a);
    // 输出: GCD = 6

    return 0;
}
```

## 8. 要点总结

1. for 的三个表达式都可以省略，但分号不能省略
2. `for (;;)` 是地道的无限循环写法
3. 逗号运算符支持多变量初始化和更新
4. C99 允许在 for 的初始化部分声明变量，该变量只在循环内可见
5. 选择合适的循环变体可以使代码更加简洁和清晰
6. 循环体为空时，分号作为占位符不能省略

## 9. 练习题

1. 用双变量 for 循环判断一个字符串是否是回文
2. 用 `for(;;)` 和 break 实现用户交互菜单
3. 用 for 循环打印如下图案（数字三角形）：
   ```
   1
   12
   123
   1234
   12345
   ```
