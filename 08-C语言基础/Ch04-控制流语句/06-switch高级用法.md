# switch 高级用法

## 1. Fall-through 行为

### 1.1 什么是 Fall-through

当一个 case 分支没有 `break` 语句时，程序会继续执行下一个 case 的代码，这种行为称为 **fall-through（穿透）**。

```c
#include <stdio.h>

int main(void) {
    int x = 1;

    switch (x) {
        case 1:
            printf("执行case 1\n");
            // 没有break，穿透到case 2
        case 2:
            printf("执行case 2\n");
            // 没有break，穿透到case 3
        case 3:
            printf("执行case 3\n");
            break;
    }

    // 输出:
    // 执行case 1
    // 执行case 2
    // 执行case 3

    return 0;
}
```

### 1.2 有意为之的 Fall-through

Fall-through 不全是坏事。在某些场景下，它是有意设计的：

**场景1：多条件共享代码**

```c
// 判断某个月份有多少天（简化版，忽略闰年）
int days;
switch (month) {
    case 1: case 3: case 5: case 7:
    case 8: case 10: case 12:
        days = 31;
        break;
    case 4: case 6: case 9: case 11:
        days = 30;
        break;
    case 2:
        days = 28;
        break;
    default:
        days = -1;
        break;
}
```

**场景2：累积效果**

```c
// 统计元音字母的个数
int count = 0;
for (int i = 0; str[i] != '\0'; i++) {
    switch (str[i]) {
        case 'a': case 'A':
            count++;
            break;
        case 'e': case 'E':
            count++;
            break;
        // ... 其他元音
    }
}

// 可以用 fall-through 简化
for (int i = 0; str[i] != '\0'; i++) {
    switch (str[i]) {
        case 'a': case 'e': case 'i': case 'o': case 'u':
        case 'A': case 'E': case 'I': case 'O': case 'U':
            count++;
            break;
    }
}
```

### 1.3 意外的 Fall-through（常见Bug）

```c
// 程序员可能忘记了 break
switch (operation) {
    case OP_READ:
        read_data();
        // 缺少break！！！
    case OP_WRITE:
        write_data();  // 读操作后也会执行写操作！
        break;
}
```

**防御方法**：有意的 fall-through 应加注释说明

```c
switch (x) {
    case 1:
        handle_one();
        /* fall through */    // 明确标注这是有意的穿透
    case 2:
        handle_two();
        break;
}
```

## 2. switch 中声明变量

### 2.1 问题：变量作用域

```c
switch (x) {
    case 1:
        int a = 10;       // 这个声明在整个switch中都可见
        printf("%d\n", a);
        break;
    case 2:
        // a 在这里仍然可见，但它的值未初始化
        printf("%d\n", a);  // 未定义行为！
        break;
}
```

### 2.2 解决方案：使用花括号创建块

```c
switch (x) {
    case 1: {
        int a = 10;
        printf("%d\n", a);   // 正确：a在这里可用
        break;
    }
    case 2: {
        int a = 20;          // 独立的变量a
        printf("%d\n", a);
        break;
    }
}
```

### 2.3 C99 及之后的改进

在较新的C标准中，case 标签后的声明可以立即跟随初始化：

```c
switch (x) {
    case 1:
        int a = 10;    // C99允许，但不推荐
        printf("%d\n", a);
        break;
}
```

即使允许，也建议使用花括号来明确作用域。

## 3. 枚举与 switch 配合

### 3.1 基本配合

枚举类型和 switch 是天然搭档，用有意义的名称替代魔法数字：

```c
#include <stdio.h>

// 定义枚举
typedef enum {
    MONDAY,
    TUESDAY,
    WEDNESDAY,
    THURSDAY,
    FRIDAY,
    SATURDAY,
    SUNDAY
} Weekday;

int main(void) {
    Weekday today = WEDNESDAY;

    switch (today) {
        case MONDAY:
            printf("今天是星期一，新的一周开始了\n");
            break;
        case TUESDAY:
            printf("今天是星期二\n");
            break;
        case WEDNESDAY:
            printf("今天是星期三，一周过半\n");
            break;
        case THURSDAY:
            printf("今天是星期四\n");
            break;
        case FRIDAY:
            printf("今天是星期五，即将周末\n");
            break;
        case SATURDAY:
        case SUNDAY:
            printf("今天是周末，好好休息\n");
            break;
    }

    return 0;
}
```

### 3.2 与命令行状态机配合

```c
#include <stdio.h>

typedef enum {
    STATE_IDLE,
    STATE_RUNNING,
    STATE_PAUSED,
    STATE_STOPPED,
    STATE_ERROR
} State;

typedef enum {
    CMD_START,
    CMD_PAUSE,
    CMD_RESUME,
    CMD_STOP,
    CMD_RESET
} Command;

State handle_command(State current, Command cmd) {
    switch (current) {
        case STATE_IDLE:
            switch (cmd) {
                case CMD_START:
                    printf("系统启动\n");
                    return STATE_RUNNING;
                case CMD_RESET:
                    printf("系统重置\n");
                    return STATE_IDLE;
                default:
                    printf("当前状态下不能执行该命令\n");
                    return current;
            }
            break;

        case STATE_RUNNING:
            switch (cmd) {
                case CMD_PAUSE:
                    printf("系统暂停\n");
                    return STATE_PAUSED;
                case CMD_STOP:
                    printf("系统停止\n");
                    return STATE_STOPPED;
                default:
                    printf("当前状态下不能执行该命令\n");
                    return current;
            }
            break;

        case STATE_PAUSED:
            switch (cmd) {
                case CMD_RESUME:
                    printf("系统恢复\n");
                    return STATE_RUNNING;
                case CMD_STOP:
                    printf("系统停止\n");
                    return STATE_STOPPED;
                default:
                    printf("当前状态下不能执行该命令\n");
                    return current;
            }
            break;

        case STATE_STOPPED:
            switch (cmd) {
                case CMD_RESET:
                    printf("系统重置\n");
                    return STATE_IDLE;
                default:
                    printf("当前状态下不能执行该命令\n");
                    return current;
            }
            break;

        default:
            return current;
    }
}
```

## 4. switch 中的技巧

### 4.1 嵌套 switch

```c
#include <stdio.h>

int main(void) {
    char suit = 'H';   // 花色
    char rank = 'A';   // 牌面

    printf("扑克牌: ");
    switch (suit) {
        case 'H':
            printf("红心");
            break;
        case 'D':
            printf("方块");
            break;
        case 'C':
            printf("梅花");
            break;
        case 'S':
            printf("黑桃");
            break;
    }

    switch (rank) {
        case 'A':
            printf("A\n");
            break;
        case 'K':
            printf("K\n");
            break;
        case 'Q':
            printf("Q\n");
            break;
        case 'J':
            printf("J\n");
            break;
        default:
            printf("%c\n", rank);
            break;
    }

    return 0;
}
```

### 4.2 利用 fall-through 实现范围判断

虽然 switch 本身不支持范围，但可以通过一些技巧来实现：

```c
#include <stdio.h>

int main(void) {
    int score = 85;

    switch (score / 10) {
        case 10:
        case 9:
            printf("优秀\n");
            break;
        case 8:
            printf("良好\n");
            break;
        case 7:
            printf("中等\n");
            break;
        case 6:
            printf("及格\n");
            break;
        default:
            printf("不及格\n");
            break;
    }

    return 0;
}
```

通过 `score / 10` 将范围值转换为离散值，从而可以使用 switch。

## 5. 要点总结

1. Fall-through 是 switch 的默认行为：没有 break 会继续执行下一个 case
2. 有意的 fall-through 应加注释，避免被误认为是 bug
3. 在 case 中声明变量时，用花括号创建块作用域
4. 枚举类型与 switch 配合使用，可读性最佳
5. 可以通过整除等方式将范围判断转化为等值判断，从而使用 switch
6. 编译器可能会对 switch 生成跳转表，性能优于连续的 else-if

## 6. 练习题

1. 使用 enum 和 switch 实现一个简单的四则运算计算器
2. 用 switch 实现一个字符频率统计程序（统计 a-z 的出现次数）
3. 实现一个简单的状态机：输入流可以是数字、运算符或等号，用 switch 处理
