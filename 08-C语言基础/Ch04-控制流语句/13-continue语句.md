# continue 语句

## 1. continue 概述

`continue` 语句用于**跳过当前循环迭代中剩余的语句**，直接进入下一次循环的条件判断。与 `break` 的"终止循环"不同，`continue` 只是"跳过本次"。

## 2. 语法

```c
continue;
```

## 3. 在各种循环中的行为

### 3.1 在 while 循环中

```c
while (条件) {
    语句A;
    continue;    // 跳到条件判断，语句B不会执行
    语句B;
}
```

**警告**：在 while 中使用 continue 要特别小心，如果更新语句在 continue 之后，可能导致死循环：

```c
// 危险的死循环！
int i = 0;
while (i < 10) {
    if (i == 5) {
        i++;          // 必须在 continue 前更新！
        continue;
    }
    printf("%d ", i);
    i++;              // 当 i==5 时这行不会执行
}

// 更好的写法
int i = 0;
while (i < 10) {
    if (i != 5) {     // 用 if 代替 continue
        printf("%d ", i);
    }
    i++;              // 更新总是在最后
}
```

### 3.2 在 for 循环中

for 循环中使用 continue 更安全，因为更新表达式总会执行：

```c
for (int i = 0; i < 10; i++) {
    if (i == 5) {
        continue;    // 跳过 i==5 的处理，但 i++ 仍然会执行
    }
    printf("%d ", i);
}
// 输出: 0 1 2 3 4 6 7 8 9 (跳过了5)
```

for 循环的执行流程：
```
for (init; cond; update) {
    body;
    continue; → 直接跳到 update，然后判断 cond
}
```

### 3.3 在 do-while 循环中

```c
int i = 0;
do {
    i++;
    if (i == 5) {
        continue;    // 跳到条件判断
    }
    printf("%d ", i);
} while (i < 10);
// 输出: 1 2 3 4 6 7 8 9 10
```

## 4. continue 与 break 的对比

```c
// break：终止整个循环
for (int i = 0; i < 10; i++) {
    if (i == 5) break;
    printf("%d ", i);
}
// 输出: 0 1 2 3 4

// continue：跳过本次迭代
for (int i = 0; i < 10; i++) {
    if (i == 5) continue;
    printf("%d ", i);
}
// 输出: 0 1 2 3 4 6 7 8 9
```

```
break:    遇到5 → 终止整个循环
continue: 遇到5 → 跳过5，继续6,7,8,9
```

## 5. 常见使用场景

### 5.1 跳过特定值

```c
#include <stdio.h>

int main(void) {
    // 打印1-10中不是3的倍数的数
    for (int i = 1; i <= 10; i++) {
        if (i % 3 == 0) {
            continue;    // 跳过3的倍数
        }
        printf("%d ", i);
    }
    printf("\n");
    // 输出: 1 2 4 5 7 8 10

    return 0;
}
```

### 5.2 数据过滤

```c
#include <stdio.h>

int main(void) {
    // 只处理正数，跳过负数和零
    int data[] = {3, -1, 7, 0, 5, -3, 8};
    int n = sizeof(data) / sizeof(data[0]);
    int sum = 0;

    for (int i = 0; i < n; i++) {
        if (data[i] <= 0) {
            continue;    // 跳过非正数
        }
        sum += data[i];
    }

    printf("正数之和 = %d\n", sum);
    // 输出: 正数之和 = 23

    return 0;
}
```

### 5.3 跳过空行或无效输入

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char lines[][50] = {
        "Hello",
        "",
        "World",
        "   ",
        "C Language"
    };
    int n = 5;

    for (int i = 0; i < n; i++) {
        // 跳过空行（或只含空格的行）
        int is_blank = 1;
        for (int j = 0; lines[i][j] != '\0'; j++) {
            if (lines[i][j] != ' ' && lines[i][j] != '\t') {
                is_blank = 0;
                break;
            }
        }
        if (is_blank) {
            continue;
        }

        printf("有效行: \"%s\"\n", lines[i]);
    }

    return 0;
}
```

### 5.4 嵌套循环中的 continue

```c
#include <stdio.h>

int main(void) {
    // 打印乘法表，跳过结果大于20的组合
    for (int i = 1; i <= 5; i++) {
        for (int j = 1; j <= 5; j++) {
            if (i * j > 20) {
                continue;    // 跳过本次内层迭代
            }
            printf("%2d ", i * j);
        }
        printf("\n");
    }

    return 0;
}
```

## 6. 使用 continue 的注意事项

### 6.1 避免过度使用

过度使用 continue 会降低代码可读性：

```c
// 不推荐：continue 过多
for (int i = 0; i < n; i++) {
    if (a) continue;
    if (b) continue;
    if (c) continue;
    do_something();
}

// 推荐：使用正向条件
for (int i = 0; i < n; i++) {
    if (!a && !b && !c) {
        do_something();
    }
}
```

### 6.2 避免在 while 中产生死循环

```c
// 危险写法
int i = 0;
while (i < 10) {
    if (i % 2 == 0) {
        continue;    // i==0 时死循环！
    }
    i++;
}

// 安全写法
int i = 0;
while (i < 10) {
    if (i % 2 == 0) {
        i++;             // 先更新再 continue
        continue;
    }
    i++;
}
```

## 7. 要点总结

1. `continue` 跳过当前迭代的剩余语句，进入下一次迭代
2. 在 for 循环中，continue 之后更新表达式仍然会执行
3. 在 while/do-while 中，continue 后要确保循环变量能被更新
4. continue 与 break 的区别：continue 跳过本次，break 终止整个循环
5. 典型场景：数据过滤、跳过无效输入、忽略特定值
6. 避免过度使用 continue，必要时用正向条件替代

## 8. 练习题

1. 打印1到100中所有不能被3也不能被5整除的数
2. 读取一组整数，跳过负数，计算正数的平均值
3. 在嵌套循环中用 continue 打印一个跳过对角线元素的矩阵
