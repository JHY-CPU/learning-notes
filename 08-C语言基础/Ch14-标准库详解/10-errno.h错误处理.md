# errno.h - 错误处理

## 1. 概述

`<errno.h>` 定义了用于错误报告的宏和全局变量 `errno`。当标准库函数执行出错时，会设置 `errno` 为特定的错误码来指示错误类型。

## 2. errno 变量

### 2.1 基本概念

```c
#include <errno.h>

// errno 是一个可修改的左值（C99起通常实现为宏）
// 每个线程有独立的errno值（C11起）
// 函数成功时errno的值未定义，因此应在调用后立即检查
```

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main(void) {
    // 错误的使用方式：不先检查返回值就看errno
    FILE *fp = fopen("test.txt", "r");
    // errno可能在fopen之前就有值

    // 正确的使用方式：先检查函数返回值
    fp = fopen("不存在的文件.txt", "r");
    if (fp == NULL) {
        printf("errno = %d\n", errno);
        printf("错误信息: %s\n", strerror(errno));
    }

    // 重置errno
    errno = 0;

    // 检查数学函数错误
    double result = sqrt(-1.0);
    if (errno != 0) {
        printf("sqrt错误: %s\n", strerror(errno));
    }

    return 0;
}
```

## 3. 标准错误码

### 3.1 C标准定义的错误码

| 宏 | 值（通常） | 含义 |
|----|-----------|------|
| `EDOM` | 33 | 数学函数参数超出定义域 |
| `ERANGE` | 34 | 结果超出范围 |
| `EILSEQ` | 84 | 非法多字节序列（C95） |

### 3.2 POSIX常见错误码

| 宏 | 含义 |
|----|------|
| `EPERM` | 操作不允许 |
| `ENOENT` | 文件或目录不存在 |
| `EIO` | I/O错误 |
| `EACCES` | 权限不足 |
| `EEXIST` | 文件已存在 |
| `ENOSPC` | 磁盘空间不足 |
| `ENOMEM` | 内存不足 |
| `EINVAL` | 无效参数 |
| `EINTR` | 系统调用被中断 |
| `EPIPE` | 管道破裂 |

```c
#include <stdio.h>
#include <errno.h>
#include <math.h>
#include <string.h>
#include <stdlib.h>

// 打印标准错误码信息
void print_standard_errors(void) {
    printf("标准C错误码:\n");

    errno = 0;
    sqrt(-1.0);
    if (errno == EDOM) {
        printf("  EDOM (%d): 域错误 - %s\n", EDOM, strerror(EDOM));
    }

    errno = 0;
    pow(10, 1000);
    if (errno == ERANGE) {
        printf("  ERANGE (%d): 范围错误 - %s\n", ERANGE, strerror(ERANGE));
    }
}

int main(void) {
    print_standard_errors();

    // 常见的错误处理模式
    printf("\n常见错误场景:\n");

    // 文件操作错误
    FILE *fp = fopen("/root/protected.txt", "r");
    if (fp == NULL) {
        printf("  fopen: errno=%d, %s\n", errno, strerror(errno));
    }

    // 动态内存分配
    errno = 0;
    void *huge = malloc(SIZE_MAX);
    if (huge == NULL) {
        printf("  malloc: errno=%d, %s\n", errno, strerror(errno));
    }

    return 0;
}
```

## 4. 错误信息输出函数

```c
#include <string.h>
#include <stdio.h>

char *strerror(int errnum);   // 返回错误码对应的字符串
void perror(const char *s);   // 输出用户消息 + 系统错误信息
```

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

int main(void) {
    // strerror - 获取错误信息字符串
    printf("错误码0: %s\n", strerror(0));
    printf("错误码%d: %s\n", EDOM, strerror(EDOM));
    printf("错误码%d: %s\n", ERANGE, strerror(ERANGE));

    // perror - 自动输出错误信息
    FILE *fp = fopen("missing.txt", "r");
    if (fp == NULL) {
        perror("打开文件失败");
        // 输出类似: 打开文件失败: No such file or directory
    }

    // perror的工作原理:
    // 1. 输出参数字符串
    // 2. 输出 ": "
    // 3. 输出 strerror(errno) 的结果
    // 4. 输出换行符

    return 0;
}
```

## 5. 错误处理模式

### 5.1 基本错误处理

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

// 模式1：立即处理
int read_config(const char *filename) {
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        fprintf(stderr, "无法打开配置文件 '%s': %s\n",
                filename, strerror(errno));
        return -1;
    }

    // 处理文件...
    fclose(fp);
    return 0;
}

// 模式2：返回错误码
typedef enum {
    SUCCESS = 0,
    ERR_FILE_NOT_FOUND,
    ERR_PERMISSION,
    ERR_NO_MEMORY,
    ERR_INVALID_PARAM
} ErrorCode;

ErrorCode open_database(const char *path, FILE **out) {
    if (path == NULL || out == NULL) {
        return ERR_INVALID_PARAM;
    }

    *out = fopen(path, "r+b");
    if (*out == NULL) {
        switch (errno) {
            case ENOENT:
                return ERR_FILE_NOT_FOUND;
            case EACCES:
                return ERR_PERMISSION;
            case ENOMEM:
                return ERR_NO_MEMORY;
            default:
                return ERR_FILE_NOT_FOUND;
        }
    }
    return SUCCESS;
}

// 模式3：带错误消息的处理
typedef struct {
    int code;
    char message[256];
} Error;

Error make_error(int code, const char *fmt, ...) {
    Error err = {0};
    err.code = code;
    va_list args;
    va_start(args, fmt);
    vsnprintf(err.message, sizeof(err.message), fmt, args);
    va_end(args);
    return err;
}

int main(void) {
    // 测试错误处理
    FILE *db;
    ErrorCode result = open_database("nonexistent.db", &db);

    switch (result) {
        case SUCCESS:
            printf("数据库打开成功\n");
            fclose(db);
            break;
        case ERR_FILE_NOT_FOUND:
            fprintf(stderr, "错误: 数据库文件不存在\n");
            break;
        case ERR_PERMISSION:
            fprintf(stderr, "错误: 没有访问权限\n");
            break;
        case ERR_NO_MEMORY:
            fprintf(stderr, "错误: 内存不足\n");
            break;
        default:
            fprintf(stderr, "未知错误\n");
    }

    return 0;
}
```

### 5.2 线程安全的错误处理

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>

// C11保证errno是线程安全的
// 每个线程有独立的errno副本

#ifdef _WIN32
#include <windows.h>
// Windows使用GetLastError()
void windows_error_example(void) {
    HANDLE h = CreateFile("nonexistent.txt", GENERIC_READ,
                          0, NULL, OPEN_EXISTING,
                          FILE_ATTRIBUTE_NORMAL, NULL);
    if (h == INVALID_HANDLE_VALUE) {
        DWORD err = GetLastError();
        char *msg = NULL;
        FormatMessageA(FORMAT_MESSAGE_ALLOCATE_BUFFER |
                       FORMAT_MESSAGE_FROM_SYSTEM,
                       NULL, err, 0, (LPSTR)&msg, 0, NULL);
        fprintf(stderr, "Windows错误 %lu: %s\n", err, msg);
        LocalFree(msg);
    }
}
#endif

int main(void) {
    // errno是线程局部的，不需要加锁
    errno = 0;
    FILE *fp = fopen("test.txt", "r");
    if (fp == NULL) {
        // 在多线程环境中安全使用
        int local_errno = errno;  // 立即保存
        printf("错误: %d - %s\n", local_errno, strerror(local_errno));
    }

    return 0;
}
```

## 6. 重要注意事项

> **要点一**：只有在函数返回值指示出错时，`errno` 才有意义。成功时不保证 `errno` 为0。

> **要点二**：`errno` 在成功时不被清零，因此应在调用函数前设置 `errno = 0` 来确保准确检测错误。

> **要点三**：`strerror` 返回的字符串不应被修改（可能指向静态内存）。

> **要点四**：`perror` 输出到 `stderr`，适合在终端显示错误。

> **要点五**：C11起，`errno` 是线程安全的，每个线程有独立副本。

> **要点六**：标准C只定义了 `EDOM`、`ERANGE` 和 `EILSEQ` 三个错误码，其余是POSIX扩展。

> **要点七**：不要依赖 `errno` 的具体数值，应使用宏名。

> **要点八**：某些函数（如 `math.h` 中的）可能同时设置 `errno` 和返回特殊值，应同时检查。
