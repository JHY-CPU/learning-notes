# while 循环

## 1. while 循环概述

`while` 循环是最基本的循环结构，属于**当型循环**：每次执行循环体之前先判断条件，条件为真则执行循环体，条件为假则退出循环。

## 2. 基本语法

```c
while (条件表达式) {
    // 循环体：条件为真时重复执行
}
```

执行流程：

```
    条件表达式
    /        \
  非零(真)    零(假)
    |           |
    ▼           ▼
 执行循环体    继续后续代码
    |
    └──→ 回到条件表达式
```

**特点：**
- 先判断条件，再执行循环体
- 如果初始条件为假，循环体一次也不执行
- 循环体内必须有使条件趋于假的操作，否则成为死循环

## 3. 基本示例

### 3.1 简单计数

```c
#include <stdio.h>

int main(void) {
    int i = 1;              // 初始化

    while (i <= 5) {        // 条件判断
        printf("%d ", i);
        i++;                // 更新：使条件趋于假
    }
    printf("\n");

    // 输出: 1 2 3 4 5
    return 0;
}
```

### 3.2 求和计算

```c
#include <stdio.h>

int main(void) {
    int n = 1;
    int sum = 0;

    while (n <= 100) {
        sum += n;      // 累加
        n++;           // 更新计数器
    }

    printf("1到100的和 = %d\n", sum);
    // 输出: 1到100的和 = 5050

    return 0;
}
```

## 4. 循环三要素

每个循环都需要考虑三个要素：

| 要素 | 说明 | 示例 |
|------|------|------|
| **初始化** | 循环开始前的准备工作 | `int i = 1;` |
| **条件** | 决定是否继续循环 | `i <= 100` |
| **更新** | 修改循环变量，使条件趋于假 | `i++` |

缺少任何一个要素，循环都无法正常工作。

## 5. 循环条件

### 5.1 常见条件形式

```c
// 比较运算
while (count < 10)    { /* ... */ }
while (x != 0)        { /* ... */ }
while (flag == 1)     { /* ... */ }

// 逻辑组合
while (x > 0 && x < 100) { /* ... */ }
while (!done && attempts < 3) { /* ... */ }

// 变量本身作为条件（非零为真）
while (count)  { /* ... */ }    // count != 0 的简写
while (*ptr)   { /* ... */ }    // ptr指向非'\0'字符

// 函数返回值
while ((ch = getchar()) != EOF) { /* ... */ }
```

### 5.2 循环次数

循环次数与条件的关系：

```c
// 循环10次：i 从 0 到 9
int i = 0;
while (i < 10) {
    // 循环体
    i++;
}

// 循环10次：i 从 1 到 10
int i = 1;
while (i <= 10) {
    // 循环体
    i++;
}

// 循环5次：i 从 5 到 1
int i = 5;
while (i >= 1) {
    // 循环体
    i--;
}
```

## 6. 读取输入直到结束

while 循环非常适合处理不确定长度的输入：

```c
#include <stdio.h>

int main(void) {
    int num;
    int sum = 0;
    int count = 0;

    printf("请输入若干整数(输入非数字结束):\n");

    while (scanf("%d", &num) == 1) {
        sum += num;
        count++;
    }

    if (count > 0) {
        printf("共读入%d个数，总和 = %d，平均 = %.2f\n",
               count, sum, (double)sum / count);
    } else {
        printf("没有读入任何数字\n");
    }

    return 0;
}
```

`scanf("%d", &num) == 1` 会一直为真，直到遇到非数字输入或文件结束（EOF）。

## 7. 死循环

### 7.1 什么是死循环

如果循环条件永远为真，循环永远不会结束，称为**死循环**（Infinite Loop）。

```c
// 最常见的死循环写法
while (1) {
    // 永远执行，除非遇到 break 或 return
}

// 等价写法
while (true) {   // 需要 #include <stdbool.h>
    // ...
}
```

### 7.2 意外的死循环

```c
// 错误1：忘记更新循环变量
int i = 0;
while (i < 10) {
    printf("%d\n", i);
    // 忘记了 i++
}

// 错误2：更新方向错误
int i = 0;
while (i < 10) {
    printf("%d\n", i);
    i--;    // i越来越小，永远不会 >= 10
}

// 错误3：浮点数比较不精确
double x = 0.0;
while (x != 1.0) {    // 由于浮点误差，可能永远不等于1.0
    x += 0.1;
}
```

### 7.3 有意的死循环

死循环并非总是坏事。很多程序的核心就是一个死循环：

```c
#include <stdio.h>

int main(void) {
    int choice;

    while (1) {
        printf("\n===== 菜单 =====\n");
        printf("1. 添加\n");
        printf("2. 删除\n");
        printf("3. 查询\n");
        printf("0. 退出\n");
        printf("请选择: ");
        scanf("%d", &choice);

        if (choice == 0) {
            break;          // 用户选择退出
        }

        switch (choice) {
            case 1: printf("执行添加操作\n"); break;
            case 2: printf("执行删除操作\n"); break;
            case 3: printf("执行查询操作\n"); break;
            default: printf("无效选择\n"); break;
        }
    }

    printf("程序结束\n");
    return 0;
}
```

## 8. while 循环的变体

### 8.1 哨兵值循环

用一个特殊值（哨兵值）标记输入结束：

```c
#include <stdio.h>

int main(void) {
    int score;
    int total = 0;
    int count = 0;

    printf("请输入分数(-1结束):\n");

    scanf("%d", &score);
    while (score != -1) {       // -1是哨兵值
        total += score;
        count++;
        scanf("%d", &score);   // 读取下一个
    }

    if (count > 0) {
        printf("平均分: %.2f\n", (double)total / count);
    }

    return 0;
}
```

### 8.2 带标志变量的循环

```c
#include <stdio.h>
#include <stdbool.h>

int main(void) {
    int data[] = {3, 7, 1, 9, 2, 8, 5};
    int target = 9;
    bool found = false;
    int i = 0;

    while (i < 7 && !found) {
        if (data[i] == target) {
            found = true;
        }
        i++;
    }

    if (found) {
        printf("找到了%d，在位置%d\n", target, i - 1);
    } else {
        printf("未找到%d\n", target);
    }

    return 0;
}
```

## 9. 要点总结

1. `while` 是先判断后执行的循环，循环体可能一次也不执行
2. 循环三要素：初始化、条件、更新，缺一不可
3. 循环条件可以是任意表达式，非零为真，零为假
4. 必须确保循环变量能够被更新，否则会成为死循环
5. 死循环可以配合 `break` 使用，实现菜单等交互式程序
6. while 适合不确定循环次数的场景（如读取输入、搜索等）

## 10. 练习题

1. 用 while 循环计算 n!（阶乘）
2. 用 while 循环翻转一个整数（如 12345 → 54321）
3. 用 while 循环实现数字猜大小游戏的主循环
4. 用 while 循环统计输入字符串中某个字符出现的次数
