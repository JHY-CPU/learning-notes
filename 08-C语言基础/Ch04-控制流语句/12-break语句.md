# break 语句

## 1. break 概述

`break` 语句用于**立即终止**当前所在的循环（while、do-while、for）或 switch 语句，将控制权转移到循环或 switch 之后的语句。

## 2. 语法

```c
break;
```

`break` 不能单独使用，必须出现在以下结构中：
- `for` 循环体内
- `while` 循环体内
- `do-while` 循环体内
- `switch` 语句中

## 3. 在各循环中的使用

### 3.1 在 while 循环中

```c
#include <stdio.h>

int main(void) {
    int i = 0;

    while (1) {          // 无限循环
        if (i >= 5) {
            break;       // 当 i >= 5 时跳出循环
        }
        printf("%d ", i);
        i++;
    }
    printf("\n循环结束\n");

    // 输出: 0 1 2 3 4
    //      循环结束

    return 0;
}
```

### 3.2 在 do-while 循环中

```c
#include <stdio.h>

int main(void) {
    int num;

    do {
        printf("输入正数(0退出): ");
        scanf("%d", &num);

        if (num == 0) {
            break;       // 输入0时跳出
        }

        printf("你输入的是: %d\n", num);
    } while (1);

    printf("已退出循环\n");
    return 0;
}
```

### 3.3 在 for 循环中

```c
#include <stdio.h>

int main(void) {
    // 在数组中查找第一个负数
    int arr[] = {3, 7, 2, -1, 8, 5};
    int n = sizeof(arr) / sizeof(arr[0]);
    int found = -1;

    for (int i = 0; i < n; i++) {
        if (arr[i] < 0) {
            found = i;
            break;       // 找到第一个就停止
        }
    }

    if (found != -1) {
        printf("第一个负数在索引 %d 处: %d\n", found, arr[found]);
    } else {
        printf("没有找到负数\n");
    }

    return 0;
}
```

## 4. break 与条件的关系

`break` 通常与 `if` 配合使用，在满足特定条件时提前退出循环：

```c
// 模式1：满足条件时退出
while (condition1) {
    if (condition2) {
        break;
    }
    // 其他处理
}

// 模式2：遇到特殊值时退出
for (int i = 0; i < n; i++) {
    if (arr[i] == SENTINEL) {
        break;
    }
    process(arr[i]);
}
```

## 5. break 只跳出一层循环

在嵌套循环中，`break` 只跳出**当前所在的那一层**循环：

```c
#include <stdio.h>

int main(void) {
    for (int i = 0; i < 3; i++) {
        printf("外层 i=%d\n", i);

        for (int j = 0; j < 5; j++) {
            if (j == 2) {
                break;    // 只跳出内层for
            }
            printf("  内层 j=%d\n", j);
        }
    }

    return 0;
}
```

输出：
```
外层 i=0
  内层 j=0
  内层 j=1
外层 i=1
  内层 j=0
  内层 j=1
外层 i=2
  内层 j=0
  内层 j=1
```

外层循环仍然完整执行了3次。

## 6. 跳出多层循环

### 6.1 使用标志变量

```c
#include <stdio.h>

int main(void) {
    int matrix[3][4] = {
        {1, 5, 3, 8},
        {2, 7, 4, 6},
        {9, 0, 3, 5}
    };
    int target = 7;
    int found = 0;

    for (int i = 0; i < 3 && !found; i++) {
        for (int j = 0; j < 4 && !found; j++) {
            if (matrix[i][j] == target) {
                printf("找到 %d 在 [%d][%d]\n", target, i, j);
                found = 1;  // 设置标志，两层循环都会停止
            }
        }
    }

    return 0;
}
```

### 6.2 使用 goto

```c
for (int i = 0; i < 3; i++) {
    for (int j = 0; j < 4; j++) {
        if (matrix[i][j] == target) {
            goto found;
        }
    }
}
found:
    printf("搜索结束\n");
```

### 6.3 使用函数

```c
void find_element(int matrix[][4], int rows, int cols, int target) {
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            if (matrix[i][j] == target) {
                printf("找到在 [%d][%d]\n", i, j);
                return;    // return 跳出整个函数
            }
        }
    }
    printf("未找到\n");
}
```

## 7. break 在 switch 中

break 在 switch 语句中用于防止 fall-through：

```c
switch (value) {
    case 1:
        printf("Case 1\n");
        break;         // 跳出switch
    case 2:
        printf("Case 2\n");
        break;
    default:
        printf("Default\n");
        break;
}
```

注意：switch 中的 break 不会影响外层的循环。

```c
for (int i = 0; i < 5; i++) {
    switch (i) {
        case 2:
            break;     // 只跳出switch，不跳出for
    }
    printf("%d ", i);  // 仍然会执行
}
// 输出: 0 1 2 3 4
```

## 8. 常见使用场景

### 8.1 查找算法

```c
// 线性查找——找到即停止
int linear_search(int arr[], int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return i;  // 找到了，返回索引
        }
    }
    return -1;  // 未找到
}
```

### 8.2 输入验证

```c
int attempts = 0;
while (1) {
    char input[100];
    printf("输入密码: ");
    scanf("%s", input);

    if (strcmp(input, PASSWORD) == 0) {
        printf("密码正确\n");
        break;    // 验证通过，退出
    }

    attempts++;
    if (attempts >= 3) {
        printf("尝试次数过多\n");
        break;    // 超过次数，退出
    }

    printf("密码错误，剩余%d次\n", 3 - attempts);
}
```

### 8.3 菜单系统

```c
while (1) {
    display_menu();
    int choice = get_choice();

    if (choice == 0) {
        break;    // 用户选择退出
    }

    process_choice(choice);
}
```

## 9. 要点总结

1. `break` 立即终止当前循环或 switch，跳到后续语句
2. `break` 只能跳出**一层**循环
3. 跳出多层循环：标志变量、goto、函数 return
4. break 常与 if 配合，在满足条件时提前退出
5. 查找、输入验证、菜单系统是 break 的典型应用场景
6. switch 中的 break 不影响外层循环

## 10. 练习题

1. 用 while + break 实现：读取输入直到遇到 "quit" 字符串
2. 用嵌套循环 + 标志变量：在二维数组中查找目标值，找到后停止所有循环
3. 用 break 实现安全的密码验证（最多3次机会）
