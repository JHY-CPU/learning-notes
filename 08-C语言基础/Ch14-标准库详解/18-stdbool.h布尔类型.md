# stdbool.h - 布尔类型（C99）

## 1. 概述

`<stdbool.h>`（C99引入）定义了布尔类型 `bool` 和常量 `true`、`false`。在C23中，`bool`、`true`、`false` 已成为关键字，不再需要此头文件。

## 2. 基本定义

```c
#include <stdbool.h>

// stdbool.h 中的定义：
#define bool    _Bool
#define true    1
#define false   0
```

## 3. 基本用法

```c
#include <stdio.h>
#include <stdbool.h>

int main(void) {
    // 声明布尔变量
    bool is_ready = true;
    bool is_done = false;

    printf("is_ready = %s\n", is_ready ? "true" : "false");
    printf("is_done = %s\n", is_done ? "true" : "false");

    // 布尔值的本质是整数
    printf("true 的值: %d\n", true);   // 1
    printf("false 的值: %d\n", false); // 0
    printf("sizeof(bool): %zu\n", sizeof(bool));  // 通常1

    // 任何非零值转换为true
    bool result = 42;
    printf("42 转bool: %d\n", result);  // 1

    result = 0;
    printf("0 转bool: %d\n", result);   // 0

    result = -1;
    printf("-1 转bool: %d\n", result);  // 1

    return 0;
}
```

## 4. 布尔运算

```c
#include <stdio.h>
#include <stdbool.h>

int main(void) {
    bool a = true;
    bool b = false;

    // 逻辑运算
    printf("true && false = %d\n", a && b);   // 0 (false)
    printf("true || false = %d\n", a || b);   // 1 (true)
    printf("!true = %d\n", !a);                // 0 (false)
    printf("!false = %d\n", !b);               // 1 (true)

    // 比较运算返回布尔值
    bool cmp1 = (5 > 3);
    bool cmp2 = (10 == 20);
    bool cmp3 = ('A' != 'B');

    printf("5 > 3: %d\n", cmp1);    // 1
    printf("10 == 20: %d\n", cmp2);  // 0
    printf("A != B: %d\n", cmp3);   // 1

    // 布尔值用于条件
    bool has_permission = true;
    if (has_permission) {
        printf("允许访问\n");
    } else {
        printf("拒绝访问\n");
    }

    return 0;
}
```

## 5. 实用示例

### 5.1 标志位管理

```c
#include <stdio.h>
#include <stdbool.h>

// 使用布尔值管理状态标志
typedef struct {
    bool is_connected;
    bool is_authenticated;
    bool is_admin;
    bool has_error;
} ConnectionState;

void process_connection(ConnectionState *state) {
    if (!state->is_connected) {
        printf("未连接\n");
        return;
    }

    if (!state->is_authenticated) {
        printf("未认证\n");
        return;
    }

    if (state->has_error) {
        printf("有错误发生\n");
        return;
    }

    printf("连接正常%s\n", state->is_admin ? "（管理员）" : "");
}

int main(void) {
    ConnectionState state = {true, true, false, false};
    process_connection(&state);

    state.has_error = true;
    process_connection(&state);

    return 0;
}
```

### 5.2 返回布尔值的函数

```c
#include <stdio.h>
#include <stdbool.h>
#include <ctype.h>

// 检查字符串是否是有效的标识符
bool is_valid_identifier(const char *str) {
    if (str == NULL || str[0] == '\0') {
        return false;
    }

    // 首字符必须是字母或下划线
    if (!isalpha(str[0]) && str[0] != '_') {
        return false;
    }

    // 其余字符可以是字母、数字或下划线
    for (int i = 1; str[i] != '\0'; i++) {
        if (!isalnum(str[i]) && str[i] != '_') {
            return false;
        }
    }

    return true;
}

// 检查年份是否是闰年
bool is_leap_year(int year) {
    return (year % 4 == 0 && year % 100 != 0) || (year % 400 == 0);
}

// 检查数字是否是素数
bool is_prime(int n) {
    if (n < 2) return false;
    if (n < 4) return true;
    if (n % 2 == 0 || n % 3 == 0) return false;

    for (int i = 5; i * i <= n; i += 6) {
        if (n % i == 0 || n % (i + 2) == 0) {
            return false;
        }
    }
    return true;
}

int main(void) {
    // 标识符检查
    printf("标识符检查:\n");
    printf("  \"hello\": %s\n", is_valid_identifier("hello") ? "有效" : "无效");
    printf("  \"2abc\": %s\n", is_valid_identifier("2abc") ? "有效" : "无效");
    printf("  \"_private\": %s\n", is_valid_identifier("_private") ? "有效" : "无效");

    // 闰年检查
    printf("\n闰年检查:\n");
    int years[] = {2000, 1900, 2024, 2023};
    for (int i = 0; i < 4; i++) {
        printf("  %d年: %s\n", years[i],
               is_leap_year(years[i]) ? "闰年" : "平年");
    }

    // 素数检查
    printf("\n素数检查:\n");
    for (int i = 2; i <= 20; i++) {
        if (is_prime(i)) {
            printf("  %d ", i);
        }
    }
    printf("\n");

    return 0;
}
```

## 6. C23 中的变化

```c
// C23: bool, true, false 成为真正的关键字
// 不再需要 #include <stdbool.h>

// C23代码（不需要stdbool.h）
int main(void) {
    bool flag = true;  // 直接使用
    if (flag) {
        // ...
    }
    return 0;
}
```

## 7. 重要注意事项

> **要点一**：`bool` 实际上是 `_Bool` 类型，占用1字节（至少），值为0或1。

> **要点二**：任何非零值赋给 `bool` 变量都会转换为 `true`（1）。

> **要点三**：C语言的 `bool` 与C++的 `bool` 类似但不完全相同。

> **要点四**：在C23之前，`bool`、`true`、`false` 是宏定义，可以用 `#undef` 取消。

> **要点五**：`sizeof(bool)` 通常是1，但标准只要求它至少能存储0和1。

> **要点六**：使用 `bool` 类型比用 `int` 作为标志更清晰、更能表达意图。

> **要点七**：关系运算符（`<`、`>`、`==` 等）的结果类型是 `int`（值为0或1），不是 `bool`。

> **要点八**：如果需要兼容C89，应使用 `int` 配合 `0` 和 `1` 代替 `bool`。
