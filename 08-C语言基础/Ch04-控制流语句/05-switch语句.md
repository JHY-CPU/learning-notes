# switch 语句

## 1. switch 概述

`switch` 语句是C语言提供的另一种多分支选择结构，专门用于**根据一个整数表达式的值**选择不同的执行路径。相比多个 else-if，switch 在处理等值判断时更加清晰高效。

## 2. 基本语法

```c
switch (表达式) {
    case 常量1:
        // 表达式 == 常量1 时执行
        语句组1;
        break;
    case 常量2:
        // 表达式 == 常量2 时执行
        语句组2;
        break;
    // ... 更多case ...
    default:
        // 没有匹配的case时执行
        默认语句组;
        break;
}
```

## 3. 重要规则

1. **switch 后的表达式**必须是整数类型（`int`、`char`、`short`、`long`、枚举等），不能是浮点数或字符串
2. **case 后的值**必须是整型常量表达式（编译时能确定的值）
3. **每个 case 的值必须唯一**，不能重复
4. **break** 用于跳出 switch 结构
5. **default** 是可选的，处理没有匹配的情况

## 4. 基本示例

```c
#include <stdio.h>

int main(void) {
    int day;

    printf("请输入星期几(1-7): ");
    scanf("%d", &day);

    switch (day) {
        case 1:
            printf("星期一\n");
            break;
        case 2:
            printf("星期二\n");
            break;
        case 3:
            printf("星期三\n");
            break;
        case 4:
            printf("星期四\n");
            break;
        case 5:
            printf("星期五\n");
            break;
        case 6:
            printf("星期六\n");
            break;
        case 7:
            printf("星期日\n");
            break;
        default:
            printf("输入无效\n");
            break;
    }

    return 0;
}
```

## 5. break 语句的作用

`break` 用于终止 switch 结构的执行，跳出到 switch 之后的语句。

### 5.1 有 break 的情况

```c
int x = 2;

switch (x) {
    case 1:
        printf("one\n");
        break;
    case 2:
        printf("two\n");
        break;    // 匹配case 2，输出"two"后跳出
    case 3:
        printf("three\n");
        break;
}
// 输出: two
```

### 5.2 没有 break 的情况（fall-through）

```c
int x = 2;

switch (x) {
    case 1:
        printf("one\n");
    case 2:
        printf("two\n");     // 匹配case 2，输出"two"
                             // 没有break，继续向下执行
    case 3:
        printf("three\n");   // 继续执行！
}
// 输出:
// two
// three
```

**重要**：如果忘记写 break，程序会继续执行下一个 case 的代码，这通常不是你想要的行为。

## 6. default 分支

`default` 处理所有未被 case 匹配的情况：

```c
switch (grade) {
    case 'A':
        printf("优秀\n");
        break;
    case 'B':
        printf("良好\n");
        break;
    case 'C':
        printf("及格\n");
        break;
    default:
        printf("不及格或无效输入\n");
        break;
}
```

**最佳实践**：
- 始终包含 default 分支，即使是处理"不可能发生"的情况
- 如果你确信所有情况都已覆盖，可以在 default 中放入断言或错误提示
- default 不一定放在最后，但放在最后最清晰

## 7. case 标签的特殊性

### 7.1 case 只是入口点

case 不会自动形成代码块，它只是一个**跳转标签**：

```c
switch (x) {
    case 1:
        printf("进入case 1\n");
        // 执行完后会继续向下到case 2
    case 2:
        printf("进入case 2\n");
        break;
}
```

### 7.2 case 中声明变量

在 case 中声明变量时要小心，因为 case 只是标签，不会形成新的作用域：

```c
switch (x) {
    case 1:
        // 可能导致编译错误或变量作用域混乱
        int y = 10;   // 声明在case标签后
        printf("%d\n", y);
        break;
    case 2:
        // 这里y的作用域不明确
        printf("%d\n", y);  // y是什么？
        break;
}

// 正确做法：用花括号创建块作用域
switch (x) {
    case 1: {
        int y = 10;   // y只在这个块内可见
        printf("%d\n", y);
        break;
    }
    case 2: {
        int y = 20;   // 这是另一个y
        printf("%d\n", y);
        break;
    }
}
```

## 8. 多个 case 共享代码

```c
#include <stdio.h>

int main(void) {
    int month;

    printf("请输入月份: ");
    scanf("%d", &month);

    switch (month) {
        case 3:
        case 4:
        case 5:
            printf("春季\n");
            break;
        case 6:
        case 7:
        case 8:
            printf("夏季\n");
            break;
        case 9:
        case 10:
        case 11:
            printf("秋季\n");
            break;
        case 12:
        case 1:
        case 2:
            printf("冬季\n");
            break;
        default:
            printf("无效月份\n");
            break;
    }

    return 0;
}
```

多个 case 连续排列（中间没有 break）意味着它们共享同一段处理代码。这是一种常见且有用的模式。

## 9. switch 与 if-else 的对比

```c
// 使用 if-else
if (ch == 'a' || ch == 'e' || ch == 'i' || ch == 'o' || ch == 'u') {
    printf("元音\n");
} else if (ch == 'A' || ch == 'E' || ch == 'I' || ch == 'O' || ch == 'U') {
    printf("大写元音\n");
} else {
    printf("辅音\n");
}

// 使用 switch（更清晰）
switch (ch) {
    case 'a': case 'e': case 'i': case 'o': case 'u':
        printf("元音\n");
        break;
    case 'A': case 'E': case 'I': case 'O': case 'U':
        printf("大写元音\n");
        break;
    default:
        printf("辅音\n");
        break;
}
```

### 选择指南

| 使用场景 | 推荐 |
|----------|------|
| 判断一个变量是否等于多个常量值之一 | switch |
| 判断条件涉及范围（如 `x > 10`） | if-else |
| 判断条件涉及浮点数 | if-else |
| 判断条件涉及多个不同变量 | if-else |
| 等值判断且case数量较多（>4个） | switch |

## 10. 简单计算器示例

```c
#include <stdio.h>

int main(void) {
    double a, b;
    char op;

    printf("请输入表达式 (如 3 + 4): ");
    scanf("%lf %c %lf", &a, &op, &b);

    switch (op) {
        case '+':
            printf("%.2f + %.2f = %.2f\n", a, b, a + b);
            break;
        case '-':
            printf("%.2f - %.2f = %.2f\n", a, b, a - b);
            break;
        case '*':
            printf("%.2f * %.2f = %.2f\n", a, b, a * b);
            break;
        case '/':
            if (b == 0) {
                printf("错误：除数不能为零\n");
            } else {
                printf("%.2f / %.2f = %.2f\n", a, b, a / b);
            }
            break;
        default:
            printf("不支持的运算符: %c\n", op);
            break;
    }

    return 0;
}
```

## 11. 要点总结

1. switch 的表达式必须是整数类型，case 值必须是整型常量
2. break 用于跳出 switch，**忘记 break 会导致 fall-through**
3. default 处理未匹配的情况，建议始终包含
4. 多个 case 可以共享同一段代码（堆叠 case）
5. 在 case 中声明变量时，建议用花括号创建块作用域
6. 等值判断用 switch 更清晰，范围判断用 if-else

## 12. 练习题

1. 编写程序，输入1-7的数字，输出对应的星期几，输入其他数字提示错误
2. 使用 switch 实现简单的菜单系统（1-添加，2-删除，3-查询，4-退出）
3. 输入一个算术运算符和两个操作数，计算结果（综合计算器）
