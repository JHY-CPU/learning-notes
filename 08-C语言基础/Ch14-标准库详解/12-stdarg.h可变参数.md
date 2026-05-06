# stdarg.h - 可变参数函数

## 1. 概述

`<stdarg.h>` 提供了处理可变参数函数的宏。可变参数函数接受不定数量的参数，如 `printf`、`scanf` 等。这些宏允许函数访问未指定数量和类型的参数。

## 2. 核心宏

```c
#include <stdarg.h>

va_list         // 参数列表类型
void va_start(va_list ap, last);  // 初始化参数列表
type va_arg(va_list ap, type);    // 获取下一个参数
void va_end(va_list ap);          // 清理参数列表
void va_copy(va_list dest, va_list src);  // 复制参数列表 (C99)
```

## 3. 基本用法

### 3.1 简单的可变参数函数

```c
#include <stdio.h>
#include <stdarg.h>

// 求n个整数的和
int sum(int count, ...) {
    va_list args;
    va_start(args, count);  // 初始化，count是最后一个固定参数

    int total = 0;
    for (int i = 0; i < count; i++) {
        int value = va_arg(args, int);  // 获取下一个int参数
        total += value;
    }

    va_end(args);  // 清理
    return total;
}

int main(void) {
    printf("sum(3, 1, 2, 3) = %d\n", sum(3, 1, 2, 3));       // 6
    printf("sum(5, 10, 20, 30, 40, 50) = %d\n",
           sum(5, 10, 20, 30, 40, 50));                       // 150

    return 0;
}
```

### 3.2 使用结束标记

```c
#include <stdio.h>
#include <stdarg.h>

// 使用-1作为结束标记
int sum_until_negative(int first, ...) {
    va_list args;
    va_start(args, first);

    int total = first;
    int value;

    while ((value = va_arg(args, int)) != -1) {
        total += value;
    }

    va_end(args);
    return total;
}

// 使用NULL作为结束标记（字符串）
void print_all(const char *first, ...) {
    va_list args;
    va_start(args, first);

    const char *str = first;
    while (str != NULL) {
        printf("%s ", str);
        str = va_arg(args, const char *);
    }
    printf("\n");

    va_end(args);
}

int main(void) {
    printf("sum: %d\n", sum_until_negative(1, 2, 3, 4, 5, -1));

    print_all("Hello", "World", "C语言", NULL);

    return 0;
}
```

## 4. 实用示例

### 4.1 自定义 printf

```c
#include <stdio.h>
#include <stdarg.h>

// 简化的格式化输出函数
void my_printf(const char *format, ...) {
    va_list args;
    va_start(args, format);

    for (const char *p = format; *p != '\0'; p++) {
        if (*p != '%') {
            putchar(*p);
            continue;
        }

        p++;  // 跳过 '%'
        switch (*p) {
            case 'd':
                printf("%d", va_arg(args, int));
                break;
            case 'f':
                printf("%f", va_arg(args, double));
                break;
            case 's': {
                const char *s = va_arg(args, const char *);
                printf("%s", s ? s : "(null)");
                break;
            }
            case 'c':
                putchar(va_arg(args, int));  // char提升为int
                break;
            case '%':
                putchar('%');
                break;
            default:
                putchar('%');
                putchar(*p);
                break;
        }
    }

    va_end(args);
}

int main(void) {
    my_printf("姓名: %s, 年龄: %d, 分数: %f\n",
              "张三", 25, 95.5);
    my_printf("百分号: %%\n");

    return 0;
}
```

### 4.2 带格式的日志函数

```c
#include <stdio.h>
#include <stdarg.h>
#include <time.h>

typedef enum {
    LOG_DEBUG,
    LOG_INFO,
    LOG_WARN,
    LOG_ERROR
} LogLevel;

const char *level_names[] = {"DEBUG", "INFO", "WARN", "ERROR"};

void log_message(LogLevel level, const char *fmt, ...) {
    // 获取当前时间
    time_t now = time(NULL);
    struct tm *t = localtime(&now);
    char time_buf[20];
    strftime(time_buf, sizeof(time_buf), "%H:%M:%S", t);

    // 输出日志前缀
    fprintf(stderr, "[%s] [%s] ", time_buf, level_names[level]);

    // 输出格式化消息
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);

    fprintf(stderr, "\n");
}

// 便捷宏
#define LOG_D(fmt, ...) log_message(LOG_DEBUG, fmt, ##__VA_ARGS__)
#define LOG_I(fmt, ...) log_message(LOG_INFO,  fmt, ##__VA_ARGS__)
#define LOG_W(fmt, ...) log_message(LOG_WARN,  fmt, ##__VA_ARGS__)
#define LOG_E(fmt, ...) log_message(LOG_ERROR, fmt, ##__VA_ARGS__)

int main(void) {
    LOG_D("调试信息: 变量x = %d", 42);
    LOG_I("程序启动成功");
    LOG_W("磁盘空间不足: 剩余 %d%%", 15);
    LOG_E("无法打开文件: %s", "config.txt");

    return 0;
}
```

### 4.3 字符串格式化

```c
#include <stdio.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>

// 安全的字符串格式化，自动分配内存
char *format_string(const char *fmt, ...) {
    va_list args1, args2;

    // 先计算需要的大小
    va_start(args1, fmt);
    va_copy(args2, args1);  // 复制一份

    int size = vsnprintf(NULL, 0, fmt, args1) + 1;
    va_end(args1);

    if (size <= 0) {
        va_end(args2);
        return NULL;
    }

    // 分配内存并格式化
    char *result = malloc(size);
    if (result != NULL) {
        vsnprintf(result, size, fmt, args2);
    }

    va_end(args2);
    return result;
}

int main(void) {
    // 自动格式化并分配字符串
    char *msg = format_string("ID: %d, Name: %s, Score: %.2f",
                               1001, "张三", 95.5);
    if (msg != NULL) {
        printf("%s\n", msg);
        free(msg);
    }

    return 0;
}
```

### 4.4 可变参数的数学运算

```c
#include <stdio.h>
#include <stdarg.h>
#include <float.h>

// 找出n个double中的最大值
double max_value(int count, ...) {
    if (count <= 0) return -DBL_MAX;

    va_list args;
    va_start(args, count);

    double max = va_arg(args, double);
    for (int i = 1; i < count; i++) {
        double val = va_arg(args, double);
        if (val > max) max = val;
    }

    va_end(args);
    return max;
}

// 计算平均值
double average(int count, ...) {
    if (count <= 0) return 0.0;

    va_list args;
    va_start(args, count);

    double sum = 0.0;
    for (int i = 0; i < count; i++) {
        sum += va_arg(args, double);
    }

    va_end(args);
    return sum / count;
}

int main(void) {
    printf("max(3.1, 2.4, 5.6, 1.2) = %.1f\n",
           max_value(4, 3.1, 2.4, 5.6, 1.2));

    printf("avg(1.0, 2.0, 3.0, 4.0, 5.0) = %.1f\n",
           average(5, 1.0, 2.0, 3.0, 4.0, 5.0));

    return 0;
}
```

## 5. v 开头的变体函数

`v` 开头的函数接受 `va_list` 而非可变参数，用于在包装函数中使用：

```c
#include <stdio.h>
#include <stdarg.h>

int vprintf(const char *format, va_list ap);
int vfprintf(FILE *stream, const char *format, va_list ap);
int vsprintf(char *str, const char *format, va_list ap);
int vsnprintf(char *str, size_t size, const char *format, va_list ap);  // C99
int vscanf(const char *format, va_list ap);  // C99
int vfscanf(FILE *stream, const char *format, va_list ap);  // C99
int vsscanf(const char *str, const char *format, va_list ap);  // C99
```

```c
#include <stdio.h>
#include <stdarg.h>

// 包装vfprintf的示例
void error_printf(const char *fmt, ...) {
    fprintf(stderr, "[ERROR] ");

    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
}

// 写入文件的日志
void log_to_file(FILE *fp, const char *fmt, ...) {
    va_list args;
    va_start(args, fmt);
    vfprintf(fp, fmt, args);
    va_end(args);
}

int main(void) {
    error_printf("文件 '%s' 不存在 (错误码: %d)\n",
                 "test.txt", 2);

    return 0;
}
```

## 6. 重要注意事项

> **要点一**：`va_start` 的第二个参数必须是最后一个固定参数（不能是数组类型）。

> **要点二**：`va_arg` 的类型必须与实际传递的参数类型完全匹配。对于小于 `int` 的整数类型，使用 `int`；对于 `float`，使用 `double`（默认参数提升）。

> **要点三**：必须在函数返回前调用 `va_end`。

> **要点四**：没有方法可以确定可变参数的数量和类型，必须通过某种方式（固定参数、格式串、结束标记）来确定。

> **要点五**：C99的 `va_copy` 用于需要多次遍历参数列表的情况。

> **要点六**：可变参数函数不是类型安全的，编译器无法检查参数类型是否正确。

> **要点七**：`char` 和 `short` 类型的参数会被提升为 `int`，`float` 会被提升为 `double`。

> **要点八**：在64位系统上，`va_list` 的内部结构可能比较复杂，不要直接操作它的成员。
