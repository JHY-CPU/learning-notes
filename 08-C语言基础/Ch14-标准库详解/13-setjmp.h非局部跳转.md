# setjmp.h - 非局部跳转

## 1. 概述

`<setjmp.h>` 提供了非局部跳转机制，允许程序从深层嵌套的函数调用中直接跳转到指定位置。这可以用于实现简单的异常处理、错误恢复等。

## 2. 核心函数

```c
#include <setjmp.h>

int setjmp(jmp_buf env);      // 设置跳转点，返回0
void longjmp(jmp_buf env, int val);  // 跳转到setjmp位置
```

- `jmp_buf`：保存程序执行状态的数组类型
- `setjmp`：保存当前环境到 `env`，首次调用返回0
- `longjmp`：恢复 `env` 中保存的环境，`setjmp` 会返回 `val`（非0）

## 3. 基本用法

### 3.1 简单的跳转示例

```c
#include <stdio.h>
#include <setjmp.h>

jmp_buf jump_buffer;

void deep_function(void) {
    printf("进入深层函数\n");
    printf("即将跳转...\n");
    longjmp(jump_buffer, 42);  // 跳转回setjmp位置
    printf("这行不会被执行\n");  // 永远不会到这里
}

int main(void) {
    int val = setjmp(jump_buffer);

    if (val == 0) {
        printf("setjmp返回0，正常执行\n");
        deep_function();
    } else {
        printf("从longjmp返回，val = %d\n", val);
    }

    printf("程序继续\n");
    return 0;
}
// 输出:
// setjmp返回0，正常执行
// 进入深层函数
// 即将跳转...
// 从longjmp返回，val = 42
// 程序继续
```

## 4. 模拟异常处理

### 4.1 基本异常处理框架

```c
#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>

// 异常代码定义
typedef enum {
    NO_ERROR = 0,
    ERROR_DIVISION_BY_ZERO,
    ERROR_NULL_POINTER,
    ERROR_OUT_OF_MEMORY,
    ERROR_FILE_NOT_FOUND
} ErrorCode;

// 异常上下文
static jmp_buf exception_env;
static int current_error = NO_ERROR;

// 抛出异常
#define THROW(code) do { \
    current_error = code; \
    longjmp(exception_env, code); \
} while(0)

// 捕获异常（在main或调用函数中使用）
#define TRY if ((current_error = setjmp(exception_env)) == 0)
#define CATCH else

// 模拟可能出错的操作
int safe_divide(int a, int b) {
    if (b == 0) {
        THROW(ERROR_DIVISION_BY_ZERO);
    }
    return a / b;
}

void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        THROW(ERROR_OUT_OF_MEMORY);
    }
    return ptr;
}

const char *error_to_string(ErrorCode code) {
    switch (code) {
        case NO_ERROR:              return "无错误";
        case ERROR_DIVISION_BY_ZERO: return "除以零";
        case ERROR_NULL_POINTER:    return "空指针";
        case ERROR_OUT_OF_MEMORY:   return "内存不足";
        case ERROR_FILE_NOT_FOUND:  return "文件未找到";
        default:                    return "未知错误";
    }
}

int main(void) {
    TRY {
        // 可能抛出异常的代码
        printf("开始计算...\n");

        int result = safe_divide(10, 2);
        printf("10 / 2 = %d\n", result);

        // 这会触发异常
        result = safe_divide(10, 0);
        printf("这行不会执行: %d\n", result);

    } CATCH {
        printf("捕获异常: %s (代码: %d)\n",
               error_to_string(current_error), current_error);
    }

    // 异常被捕获后，程序可以继续执行
    printf("程序继续正常运行\n");

    // 另一个异常处理
    TRY {
        int *data = safe_malloc(1000000000000ULL);
        free(data);
    } CATCH {
        printf("捕获异常: %s\n", error_to_string(current_error));
    }

    return 0;
}
```

### 4.2 嵌套异常处理

```c
#include <stdio.h>
#include <setjmp.h>

jmp_buf outer_env, inner_env;

void inner_function(void) {
    printf("  进入内层函数\n");
    longjmp(inner_env, 1);
    printf("  内层函数：不会执行\n");
}

void outer_function(void) {
    printf("进入外层函数\n");
    if (setjmp(inner_env) == 0) {
        inner_function();
    } else {
        printf("外层捕获内层跳转\n");
    }
    printf("外层函数继续\n");
    longjmp(outer_env, 2);
}

int main(void) {
    if (setjmp(outer_env) == 0) {
        outer_function();
    } else {
        printf("main捕获外层跳转\n");
    }
    printf("程序结束\n");
    return 0;
}
```

## 5. 资源管理注意

```c
#include <stdio.h>
#include <setjmp.h>
#include <stdlib.h>

jmp_buf env;

void risky_operation(void) {
    // 在栈上分配的资源在longjmp后无法清理
    FILE *fp = fopen("test.txt", "r");
    if (fp == NULL) {
        longjmp(env, 1);  // 注意：文件未打开，没问题
    }

    // 这里longjmp会导致文件未关闭！
    // 使用goto或特殊处理来避免此问题
    longjmp(env, 2);  // 危险：fp未关闭！

    fclose(fp);  // 永远不会执行
}

// 更好的做法：使用清理模式
int safe_risky_operation(void) {
    FILE *fp = fopen("test.txt", "r");
    int result = 0;

    if (fp == NULL) {
        result = -1;
        goto cleanup;
    }

    // 操作...
    if (/* 某个错误条件 */) {
        result = -2;
        goto cleanup;
    }

cleanup:
    if (fp != NULL) fclose(fp);
    return result;
}
```

## 6. 重要注意事项

> **要点一**：`setjmp` 的返回值只能用于判断是否从 `longjmp` 返回，不能赋值给变量后在其他地方使用。

> **要点二**：`setjmp` 只能在简单表达式（如 `if`、`switch`、`for` 的条件部分）中直接调用。

> **要点三**：调用 `setjmp` 的函数返回后，对应的 `jmp_buf` 不再有效。

> **要点四**：`longjmp` 之后，局部变量的值是未定义的（除了 `volatile` 变量和未被修改的变量）。

> **要点五**：`longjmp` 会跳过正常的函数返回过程，可能导致资源泄漏（打开的文件、分配的内存等）。

> **要点六**：`longjmp` 不能跳转到已经返回的函数。

> **要点七**：在信号处理函数中调用 `longjmp` 的行为是未定义的（除非信号是由 `raise` 产生的）。

> **要点八**：现代C++使用异常机制替代了 `setjmp/longjmp`，C语言中也可以考虑使用错误码返回值替代。
