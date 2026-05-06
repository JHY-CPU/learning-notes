# if-else 语句

## 1. if-else 概述

`if-else` 语句实现了**双分支选择结构**：条件为真时执行一段代码，条件为假时执行另一段代码。与单分支 `if` 不同，`if-else` 总会执行其中一个分支。

## 2. 基本语法

```c
if (表达式) {
    // 表达式为真时执行
} else {
    // 表达式为假时执行
}
```

执行流程：

```
        表达式
       /      \
     真        假
      ↓         ↓
  语句块A    语句块B
      \       /
       ↓     ↓
     后续语句
```

## 3. 基本示例

```c
#include <stdio.h>

int main(void) {
    int temperature;

    printf("请输入当前温度: ");
    scanf("%d", &temperature);

    if (temperature >= 30) {
        printf("天气炎热，建议穿短袖\n");
    } else {
        printf("天气凉爽，建议穿长袖\n");
    }

    return 0;
}
```

## 4. 嵌套 if-else

if-else 中可以嵌套另一组 if-else，形成多层判断：

```c
#include <stdio.h>

int main(void) {
    int age;

    printf("请输入年龄: ");
    scanf("%d", &age);

    if (age >= 0) {
        if (age < 18) {
            printf("未成年人\n");
        } else {
            if (age < 60) {
                printf("成年人\n");
            } else {
                printf("老年人\n");
            }
        }
    } else {
        printf("年龄不能为负数\n");
    }

    return 0;
}
```

### 嵌套结构说明

```
age >= 0 ?
├─ 是 ──→ age < 18 ?
│         ├─ 是 ──→ "未成年人"
│         └─ 否 ──→ age < 60 ?
│                   ├─ 是 ──→ "成年人"
│                   └─ 否 ──→ "老年人"
└─ 否 ──→ "年龄不能为负数"
```

## 5. 悬空 else 问题

### 5.1 什么是悬空 else

当if和else的数目不匹配时，`else` 到底和哪个 `if` 匹配？这就是**悬空else（Dangling Else）** 问题。

```c
#include <stdio.h>

int main(void) {
    int x = 0;
    int y = 1;

    if (x == 0)
        if (y == 1)
            printf("x=0, y=1\n");
    else
        printf("x!=0\n");    // 这个else和哪个if配对？

    return 0;
}
```

### 5.2 else 匹配规则

**C语言规定：else 总是与最近的、尚未配对的 if 匹配。**

上面的代码等价于：

```c
if (x == 0) {
    if (y == 1) {
        printf("x=0, y=1\n");
    } else {                // else与内层的if匹配！
        printf("x!=0\n");
    }
}
```

所以上面的 `printf("x!=0\n")` 实际上表示的是 `y != 1` 的情况，而不是 `x != 0`。

### 5.3 如何避免悬空 else 问题

**方法1：始终使用花括号**

```c
// 明确意图——else与外层if匹配
if (x == 0) {
    if (y == 1) {
        printf("x=0, y=1\n");
    }
} else {
    printf("x!=0\n");
}
```

**方法2：即使只有一条语句也加花括号**

```c
// 良好的编码习惯
if (condition) {
    do_something();
} else {
    do_other();
}
```

### 5.4 经典陷阱示例

```c
// 这段代码的意图可能是"如果没登录，且尝试次数超过3次，显示错误"
// 但由于悬空else，实际行为与预期不同

if (logged_in == 0)
    if (attempts > 3)
        printf("账户已锁定\n");
else                                // 歧义！
    printf("请先登录\n");

// 修正版本
if (logged_in == 0) {
    if (attempts > 3) {
        printf("账户已锁定\n");
    }
} else {
    printf("请先登录\n");
}
```

## 6. if-else 的实际应用

### 6.1 绝对值计算

```c
int abs_value(int x) {
    if (x >= 0) {
        return x;
    } else {
        return -x;
    }
}
```

### 6.2 闰年判断

```c
#include <stdio.h>

int main(void) {
    int year;

    printf("请输入年份: ");
    scanf("%d", &year);

    if (year % 4 == 0) {
        if (year % 100 == 0) {
            if (year % 400 == 0) {
                printf("%d是闰年\n", year);
            } else {
                printf("%d不是闰年\n", year);
            }
        } else {
            printf("%d是闰年\n", year);
        }
    } else {
        printf("%d不是闰年\n", year);
    }

    return 0;
}
```

闰年规则：能被4整除但不能被100整除，或者能被400整除。

### 6.3 字符分类

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    char ch;

    printf("请输入一个字符: ");
    ch = getchar();

    if (isalpha(ch)) {
        if (isupper(ch)) {
            printf("'%c'是大写字母\n", ch);
        } else {
            printf("'%c'是小写字母\n", ch);
        }
    } else {
        if (isdigit(ch)) {
            printf("'%c'是数字\n", ch);
        } else {
            printf("'%c'是其他字符\n", ch);
        }
    }

    return 0;
}
```

### 6.4 三元运算符简化

简单的if-else可以用三元运算符 `?:` 替代：

```c
// if-else 写法
int max;
if (a > b) {
    max = a;
} else {
    max = b;
}

// 三元运算符写法（等价）
int max = (a > b) ? a : b;
```

三元运算符适合简单的赋值操作，但不要在复杂逻辑中使用，否则可读性会降低。

## 7. 要点总结

1. `if-else` 是双分支结构，总会执行两个分支中的一个
2. **else 总是与最近的、未配对的 if 匹配**——这是悬空else问题的根源
3. 始终使用花括号可以彻底避免悬空else问题
4. 嵌套if-else可以实现复杂的多条件判断，但不宜嵌套太深（一般不超过3层）
5. 简单的if-else赋值可以用三元运算符简化

## 8. 练习题

1. 输入两个整数，输出其中较大的一个（使用if-else）
2. 判断一个年份是否为闰年（简化为一条if-else表达式）
3. 输入一个字符，判断它是字母、数字还是其他字符
4. 编写函数 `sign(int x)`，x为正返回1，为负返回-1，为0返回0
