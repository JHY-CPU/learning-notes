# do-while 循环

## 1. do-while 概述

`do-while` 循环属于**直到型循环**：先执行一次循环体，然后再判断条件。与 `while` 的最大区别在于，**do-while 至少会执行一次循环体**。

## 2. 基本语法

```c
do {
    // 循环体：至少执行一次
} while (条件表达式);   // 注意末尾的分号！
```

执行流程：

```
 执行循环体（第一次无条件执行）
    |
    ▼
 条件表达式
    /        \
  非零(真)    零(假)
    |           |
    ▼           ▼
 回到循环体   继续后续代码
```

**与 while 的对比：**

```
while:  判断 → 执行 → 判断 → 执行 → ...
do-while: 执行 → 判断 → 执行 → 判断 → ...
```

## 3. 基本示例

### 3.1 简单计数

```c
#include <stdio.h>

int main(void) {
    int i = 1;

    do {
        printf("%d ", i);
        i++;
    } while (i <= 5);

    printf("\n");
    // 输出: 1 2 3 4 5
    return 0;
}
```

### 3.2 至少执行一次的效果

```c
#include <stdio.h>

int main(void) {
    // while 版本：条件一开始为假，循环体不执行
    int x = 10;
    while (x < 5) {
        printf("while循环体\n");   // 不会输出
    }

    // do-while 版本：即使条件为假，循环体也执行一次
    int y = 10;
    do {
        printf("do-while循环体\n");   // 会输出一次
    } while (y < 5);

    return 0;
}
```

## 4. do-while 的典型使用场景

### 4.1 输入验证

这是 do-while 最经典的使用场景：反复要求用户输入，直到输入合法。

```c
#include <stdio.h>

int main(void) {
    int num;

    do {
        printf("请输入1-10之间的整数: ");
        scanf("%d", &num);
    } while (num < 1 || num > 10);

    printf("你输入的是: %d\n", num);
    return 0;
}
```

为什么用 do-while？因为至少要提示用户输入一次，如果输入不合法再提示，这正好符合 do-while 的逻辑。

### 4.2 带验证的菜单

```c
#include <stdio.h>

int main(void) {
    int choice;

    do {
        printf("\n===== 菜单 =====\n");
        printf("1. 新建文件\n");
        printf("2. 打开文件\n");
        printf("3. 保存文件\n");
        printf("0. 退出\n");
        printf("请选择: ");
        scanf("%d", &choice);

        switch (choice) {
            case 1: printf("新建文件\n"); break;
            case 2: printf("打开文件\n"); break;
            case 3: printf("保存文件\n"); break;
            case 0: printf("再见\n"); break;
            default: printf("无效选择，请重新输入\n"); break;
        }
    } while (choice != 0);

    return 0;
}
```

### 4.3 游戏主循环

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    srand((unsigned)time(NULL));
    int target = rand() % 100 + 1;
    int guess;
    int attempts = 0;

    printf("猜数字游戏！我想了一个1-100之间的数字。\n");

    do {
        printf("请输入你的猜测: ");
        scanf("%d", &guess);
        attempts++;

        if (guess > target) {
            printf("太大了！\n");
        } else if (guess < target) {
            printf("太小了！\n");
        } else {
            printf("恭喜你猜对了！共用了%d次\n", attempts);
        }
    } while (guess != target);

    return 0;
}
```

## 5. do-while 与 while 的选择

| 场景 | 推荐 |
|------|------|
| 至少执行一次（输入验证、菜单） | do-while |
| 可能一次也不执行（条件判断在前） | while |
| 不确定循环次数（读取文件等） | while |
| 先操作后判断（游戏回合） | do-while |

### 5.1 相互转换

```c
// while 版本
int x = 0;
while (x < 5) {
    printf("%d ", x);
    x++;
}

// 等价的 do-while 版本
int x = 0;
if (x < 5) {            // 先判断一次
    do {
        printf("%d ", x);
        x++;
    } while (x < 5);
}
```

注意：while 转换为 do-while 需要额外的初始判断，因为 do-while 总是至少执行一次。

## 6. 常见陷阱

### 6.1 忘记分号

```c
// 错误：while 后面没有分号
do {
    printf("%d\n", i);
    i++;
} while (i < 10)    // 缺少分号！

// 正确
do {
    printf("%d\n", i);
    i++;
} while (i < 10);   // 注意分号
```

### 6.2 不更新循环变量

```c
// 死循环
int i = 0;
do {
    printf("%d\n", i);
    // 忘记了 i++
} while (i < 10);
```

### 6.3 误用 do-while

```c
// 不适合的场景：可能不需要执行循环体
// 用户输入可能已经是合法的

// 用 while 更自然
while (data_is_invalid(input)) {
    input = get_new_input();
}

// 用 do-while 不太自然
do {
    input = get_new_input();
} while (data_is_invalid(input));
// 问题：第一次获取的 input 被丢弃了
```

## 7. 宏中的 do-while(0) 惯用法

这是一个高级技巧，在宏定义中经常使用：

```c
// 没有 do-while 的宏——在 if-else 中会出问题
#define SWAP(a, b)  \
    int temp = a;   \
    a = b;          \
    b = temp;

// 使用时：
if (x > y)
    SWAP(x, y);     // 展开后有多条语句，else会匹配错误
else
    printf("无需交换\n");

// 用 do-while(0) 包裹的宏——安全
#define SWAP(a, b) do { \
    int temp = (a);     \
    (a) = (b);          \
    (b) = temp;         \
} while (0)

// 使用时：
if (x > y)
    SWAP(x, y);     // 安全，do-while作为一个整体
else
    printf("无需交换\n");
```

`do-while(0)` 的作用是将多条语句封装为一个"语句"，同时需要末尾加分号，使宏的使用方式与普通函数调用一致。

## 8. 要点总结

1. `do-while` 先执行循环体，再判断条件，**至少执行一次**
2. `while` 后面有分号，这是与 `while` 语法的一个区别
3. do-while 最适合"先操作后验证"的场景：输入验证、菜单、游戏主循环
4. 选择循环结构的原则：需要至少执行一次就用 do-while，否则用 while
5. do-while(0) 是C语言宏定义中一个重要的惯用法

## 9. 练习题

1. 用 do-while 实现一个密码输入程序，最多允许输入3次
2. 用 do-while 实现求一个正整数的各位数字之和
3. 用 do-while 编写程序，反复读取温度值，直到输入 -999 为止，计算平均温度
4. 用 do-while(0) 惯用法编写一个安全的宏，用于打印调试信息
