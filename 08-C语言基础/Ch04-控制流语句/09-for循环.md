# for 循环

## 1. for 循环概述

`for` 循环是C语言中最常用的循环结构，特别适合**已知循环次数**的场景。它将循环的初始化、条件判断、变量更新三要素集中在一条语句中，结构紧凑。

## 2. 基本语法

```c
for (初始化表达式; 条件表达式; 更新表达式) {
    // 循环体
}
```

执行顺序：

```
① 初始化表达式（只执行一次）
        |
        ▼
② 条件表达式 ←──────────┐
      /     \           |
    真       假         |
    |         |         |
    ▼         ▼         |
③ 执行循环体  退出循环   |
    |                    |
④ 更新表达式 ───────────┘
```

**详细执行流程：**
1. 执行**初始化表达式**（仅一次）
2. 计算**条件表达式**
3. 如果条件为真（非零），执行循环体
4. 执行**更新表达式**
5. 回到第2步
6. 如果条件为假（零），退出循环

## 3. 基本示例

### 3.1 简单计数

```c
#include <stdio.h>

int main(void) {
    // 打印1到10
    for (int i = 1; i <= 10; i++) {
        printf("%d ", i);
    }
    printf("\n");
    // 输出: 1 2 3 4 5 6 7 8 9 10

    return 0;
}
```

等价的 while 写法：

```c
int i = 1;            // 初始化
while (i <= 10) {     // 条件
    printf("%d ", i);
    i++;              // 更新
}
```

### 3.2 求和

```c
#include <stdio.h>

int main(void) {
    int sum = 0;

    for (int i = 1; i <= 100; i++) {
        sum += i;
    }

    printf("1+2+...+100 = %d\n", sum);
    // 输出: 1+2+...+100 = 5050
    return 0;
}
```

### 3.3 九九乘法表

```c
#include <stdio.h>

int main(void) {
    for (int i = 1; i <= 9; i++) {
        for (int j = 1; j <= i; j++) {
            printf("%d×%d=%-4d", j, i, i * j);
        }
        printf("\n");
    }

    return 0;
}
```

## 4. for 循环的三个表达式

### 4.1 初始化表达式

```c
// 声明并初始化变量（C99）
for (int i = 0; i < 10; i++) { /* ... */ }

// 使用已有变量
int i;
for (i = 0; i < 10; i++) { /* ... */ }

// 多个初始化（用逗号分隔）
for (int i = 0, j = 10; i < j; i++, j--) {
    printf("i=%d, j=%d\n", i, j);
}

// 空初始化
int i = 0;
for (; i < 10; i++) { /* ... */ }
```

### 4.2 条件表达式

```c
// 常规条件
for (int i = 0; i < 10; i++) { /* ... */ }

// 复合条件
for (int i = 0; i < 10 && !error; i++) { /* ... */ }

// 空条件（永远为真，等价于死循环）
for (int i = 0; ; i++) { /* ... */ }

// 函数调用作为条件
for (int i = 0; is_valid(i); i++) { /* ... */ }
```

### 4.3 更新表达式

```c
// 递增
for (int i = 0; i < 10; i++) { /* ... */ }

// 递减
for (int i = 10; i > 0; i--) { /* ... */ }

// 按步长增减
for (int i = 0; i < 100; i += 5) { /* ... */ }  // 0, 5, 10, ..., 95

// 乘除变化
for (int i = 1; i < 1024; i *= 2) { /* ... */ }  // 1, 2, 4, 8, ..., 512

// 多变量更新
for (int i = 0, j = 9; i < j; i++, j--) { /* ... */ }

// 空更新（需要在循环体内手动更新）
for (int i = 0; i < 10; ) {
    /* ... */
    i += 2;  // 在循环体内更新
}
```

## 5. for 循环的常见模式

### 5.1 遍历数组

```c
#include <stdio.h>

int main(void) {
    int arr[] = {10, 20, 30, 40, 50};
    int n = sizeof(arr) / sizeof(arr[0]);

    for (int i = 0; i < n; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    return 0;
}
```

### 5.2 字符串遍历

```c
#include <stdio.h>

int main(void) {
    char str[] = "Hello, World!";

    for (int i = 0; str[i] != '\0'; i++) {
        printf("str[%d] = '%c' (ASCII: %d)\n", i, str[i], str[i]);
    }

    return 0;
}
```

### 5.3 累乘

```c
#include <stdio.h>

int main(void) {
    int n = 10;
    long long factorial = 1;

    for (int i = 1; i <= n; i++) {
        factorial *= i;
    }

    printf("%d! = %lld\n", n, factorial);
    // 输出: 10! = 3628800

    return 0;
}
```

### 5.4 查找

```c
#include <stdio.h>

int main(void) {
    int arr[] = {3, 7, 1, 9, 2, 8, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    int target = 9;
    int found_index = -1;

    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            found_index = i;
            break;
        }
    }

    if (found_index != -1) {
        printf("找到%d，在索引%d处\n", target, found_index);
    } else {
        printf("未找到%d\n", target);
    }

    return 0;
}
```

## 6. for 与 while 的等价性

```c
// for 循环
for (初始化; 条件; 更新) {
    循环体;
}

// 等价的 while 循环
初始化;
while (条件) {
    循环体;
    更新;
}
```

**选择建议：**
- 循环次数明确时，用 `for` 更清晰
- 循环次数不确定时，用 `while` 更自然
- 读取输入、搜索等场景，用 `while`
- 遍历数组、计数等场景，用 `for`

## 7. 常见错误

### 7.1 差一错误

```c
// 错误：循环11次（0到10）
for (int i = 0; i <= 10; i++) { /* ... */ }

// 正确：循环10次（0到9）
for (int i = 0; i < 10; i++) { /* ... */ }
```

### 7.2 在循环条件后加分号

```c
// 错误：空循环体，只打印一次
for (int i = 0; i < 5; i++);    // 分号使循环体为空
    printf("%d\n", i);           // 这行只执行一次

// 正确
for (int i = 0; i < 5; i++) {
    printf("%d\n", i);
}
```

### 7.3 循环变量作用域（C99之前）

```c
// C89：变量必须在循环外声明
int i;
for (i = 0; i < 10; i++) {
    // ...
}
// 循环结束后 i 仍然可用
printf("%d\n", i);  // 输出 10

// C99：变量在循环内声明，循环结束后不可用
for (int i = 0; i < 10; i++) {
    // ...
}
// printf("%d\n", i);  // 编译错误：i未声明
```

## 8. 要点总结

1. for 循环将初始化、条件、更新三要素集中在一条语句中
2. 执行顺序：初始化 → 条件判断 → 循环体 → 更新 → 条件判断 → ...
3. 三个表达式都可以省略，但分号不能省略
4. for 循环特别适合已知循环次数的场景
5. C99 允许在 for 的初始化部分声明变量
6. 逗号运算符可以用于多变量初始化和更新

## 9. 练习题

1. 用 for 循环计算 1! + 2! + 3! + ... + n!
2. 用 for 循环打印一个由 `*` 组成的直角三角形（5行）
3. 用 for 循环判断一个数是否为素数
4. 用 for 循环求 Fibonacci 数列的前20项
