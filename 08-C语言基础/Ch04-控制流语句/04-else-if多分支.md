# else-if 多分支

## 1. else-if 概述

当需要判断多个条件时，可以使用 `else if` 实现**多分支选择结构**。`else if` 实质上是多个 `if-else` 的嵌套展开形式，但写起来更加清晰。

## 2. 基本语法

```c
if (表达式1) {
    // 表达式1为真时执行
} else if (表达式2) {
    // 表达式1为假，表达式2为真时执行
} else if (表达式3) {
    // 表达式1、2为假，表达式3为真时执行
} else {
    // 所有表达式都为假时执行
}
```

执行流程：

```
表达式1 → 真 → 块1 ─────────→ 后续
    │ 假
    ▼
表达式2 → 真 → 块2 ─────────→ 后续
    │ �假
    ▼
表达式3 → 真 → 块3 ─────────→ 后续
    │ 假
    ▼
  else块 ──────────────────→ 后续
```

**关键特性：**
- 条件从上到下依次判断，**一旦某个条件为真，执行对应块后跳出整个结构**
- 只会执行**一个**分支
- `else` 块是可选的

## 3. 成绩等级判定

```c
#include <stdio.h>

int main(void) {
    int score;

    printf("请输入成绩(0-100): ");
    scanf("%d", &score);

    if (score < 0 || score > 100) {
        printf("成绩无效\n");
    } else if (score >= 90) {
        printf("优秀 (A)\n");
    } else if (score >= 80) {
        printf("良好 (B)\n");
    } else if (score >= 70) {
        printf("中等 (C)\n");
    } else if (score >= 60) {
        printf("及格 (D)\n");
    } else {
        printf("不及格 (F)\n");
    }

    return 0;
}
```

### 条件判断过程

假设输入 `score = 85`：

```
score >= 90? → 85 >= 90? → 假 → 继续
score >= 80? → 85 >= 80? → 真 → 输出"良好(B)" → 跳出
(后续条件不再判断)
```

**注意**：这里的条件顺序非常重要。如果把 `score >= 60` 放在最前面，那么所有60分以上的成绩都会被判定为"及格"。正是因为条件从大到小排列，才能正确分级。

## 4. 与嵌套 if-else 的等价性

```c
// else-if 写法
if (score >= 90) {
    printf("优秀\n");
} else if (score >= 80) {
    printf("良好\n");
} else {
    printf("其他\n");
}

// 等价的嵌套 if-else 写法
if (score >= 90) {
    printf("优秀\n");
} else {
    if (score >= 80) {
        printf("良好\n");
    } else {
        printf("其他\n");
    }
}
```

else-if 是嵌套if-else的语法糖，使代码更加扁平化，可读性更好。

## 5. 优先级判断示例

```c
#include <stdio.h>

int main(void) {
    int priority;

    printf("请输入优先级(1-5): ");
    scanf("%d", &priority);

    if (priority == 1) {
        printf("【紧急】立即处理\n");
    } else if (priority == 2) {
        printf("【高】尽快处理\n");
    } else if (priority == 3) {
        printf("【中】正常处理\n");
    } else if (priority == 4) {
        printf("【低】有空再处理\n");
    } else if (priority == 5) {
        printf("【极低】可以忽略\n");
    } else {
        printf("无效的优先级\n");
    }

    return 0;
}
```

## 6. 分段函数计算

```c
#include <stdio.h>

int main(void) {
    double x, y;

    printf("请输入x的值: ");
    scanf("%lf", &x);

    // 分段函数:
    // y = -x      当 x < 0
    // y = x*x     当 0 <= x < 1
    // y = 2*x-1   当 1 <= x < 2
    // y = x+3     当 x >= 2

    if (x < 0) {
        y = -x;
    } else if (x < 1) {
        y = x * x;
    } else if (x < 2) {
        y = 2 * x - 1;
    } else {
        y = x + 3;
    }

    printf("f(%.2f) = %.2f\n", x, y);
    return 0;
}
```

## 7. 多条件组合判断

```c
#include <stdio.h>

int main(void) {
    int age;
    char gender;
    double height;

    printf("请输入年龄、性别(M/F)、身高(cm): ");
    scanf("%d %c %lf", &age, &gender, &height);

    if (age < 6) {
        printf("学龄前儿童\n");
    } else if (age < 18) {
        printf("未成年");
        if (gender == 'M' || gender == 'm') {
            printf("，男性");
        } else {
            printf("，女性");
        }
        if (height >= 170) {
            printf("，身高较高");
        }
        printf("\n");
    } else if (age < 60) {
        printf("成年人\n");
    } else {
        printf("老年人\n");
    }

    return 0;
}
```

## 8. 常见错误

### 8.1 条件顺序错误

```c
// 错误：85分会被第二个条件捕获，永远不会到达"优秀"
if (score >= 60) {
    printf("及格\n");
} else if (score >= 80) {   // 永远不会执行到这里
    printf("良好\n");
} else if (score >= 90) {   // 永远不会执行到这里
    printf("优秀\n");
}

// 正确：从大到小排列
if (score >= 90) {
    printf("优秀\n");
} else if (score >= 80) {
    printf("良好\n");
} else if (score >= 60) {
    printf("及格\n");
}
```

### 8.2 忘记 else 分支

```c
// 如果需要处理所有情况，不要忘记else
if (age < 18) {
    printf("未成年\n");
} else if (age < 60) {
    printf("成年\n");
}
// 缺少else：age >= 60 的情况没有处理
```

### 8.3 条件互斥

else-if 隐含了条件的互斥性：后续分支的条件暗含前面所有条件都为假。

```c
// x = 85 的执行路径:
// score >= 90? 假(前面条件为假)
// score >= 80? 真 → 输出"良好"
```

## 9. 与 switch 的对比

| 特性 | else-if | switch |
|------|---------|--------|
| 条件类型 | 任意表达式（布尔） | 整数/字符/枚举 |
| 范围判断 | 支持（如 `x >= 80`） | 不支持 |
| 等值判断 | 支持 | 更简洁 |
| 性能 | 顺序判断，O(n) | 编译器可优化为跳转表 |

对于等值判断，如果条件很多（如超过4个），switch 语句通常更高效、更清晰。

## 10. 要点总结

1. `else-if` 是多分支选择结构，从上到下依次判断条件
2. 只执行第一个为真的分支，然后跳出整个结构
3. 条件的排列顺序至关重要——范围大的条件应放在后面
4. `else` 块是可选的，但建议在需要处理所有情况时加上
5. else-if 本质上是嵌套 if-else 的扁平化写法
6. 对于等值判断且条件较多时，考虑使用 switch 语句

## 11. 练习题

1. 输入一个月份（1-12），输出该月份属于哪个季节（春/夏/秋/冬）
2. 输入收入金额，根据税率表计算应纳税额（使用分段税率）
3. 判断一个点 (x, y) 在平面直角坐标系中的位置（哪个象限或坐标轴上）
