# signal.h - 信号处理

## 1. 概述

`<signal.h>` 提供了信号处理机制。信号是操作系统通知程序发生了某个事件的方式，程序可以注册信号处理函数来响应特定信号。

## 2. 标准信号

| 信号 | 含义 | 说明 |
|------|------|------|
| `SIGABRT` | 异常终止 | 由 `abort()` 产生 |
| `SIGFPE` | 浮点异常 | 除以零、溢出等 |
| `SIGILL` | 非法指令 | 无效的机器指令 |
| `SIGINT` | 中断 | 用户按 Ctrl+C |
| `SIGSEGV` | 段错误 | 非法内存访问 |
| `SIGTERM` | 终止请求 | 由 `kill` 命令发送 |

## 3. 核心函数

```c
#include <signal.h>

void (*signal(int sig, void (*handler)(int)))(int);
int raise(int sig);
```

## 4. 基本用法

### 4.1 注册信号处理函数

```c
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

// 信号处理函数
void signal_handler(int sig) {
    switch (sig) {
        case SIGINT:
            printf("\n收到中断信号 (Ctrl+C)\n");
            printf("优雅退出...\n");
            exit(0);
            break;
        case SIGSEGV:
            printf("段错误！发生了非法内存访问\n");
            exit(1);
            break;
        case SIGFPE:
            printf("浮点异常！可能是除以零\n");
            exit(1);
            break;
        default:
            printf("收到信号: %d\n", sig);
    }
}

int main(void) {
    // 注册信号处理函数
    signal(SIGINT, signal_handler);
    signal(SIGSEGV, signal_handler);
    signal(SIGFPE, signal_handler);

    printf("程序运行中，按 Ctrl+C 测试信号处理...\n");
    printf("PID: %d\n", getpid());

    // 主循环
    int count = 0;
    while (1) {
        printf("\r计数: %d", count++);
        fflush(stdout);

        // 模拟工作
        for (volatile long i = 0; i < 10000000L; i++);
    }

    return 0;
}
```

### 4.2 raise - 发送信号

```c
#include <stdio.h>
#include <signal.h>

void my_handler(int sig) {
    printf("处理信号: %d\n", sig);
}

int main(void) {
    signal(SIGTERM, my_handler);

    printf("发送 SIGTERM 信号给自己...\n");
    raise(SIGTERM);  // 向当前进程发送信号

    printf("信号处理完成，继续执行\n");

    return 0;
}
```

### 4.3 忽略和默认处理

```c
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>

int main(void) {
    // 忽略信号
    signal(SIGINT, SIG_IGN);   // 忽略 Ctrl+C
    printf("SIGINT 已被忽略，Ctrl+C 无效\n");

    // 恢复默认处理
    signal(SIGINT, SIG_DFL);   // 恢复默认行为
    printf("SIGINT 已恢复默认处理\n");

    return 0;
}
```

## 5. 实用示例

### 5.1 安全的清理机制

```c
#include <stdio.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>

static volatile sig_atomic_t shutdown_requested = 0;
static FILE *log_file = NULL;

void cleanup_handler(int sig) {
    shutdown_requested = 1;
    // 注意：信号处理函数中只能调用异步信号安全的函数
    // fprintf 和 fclose 不是异步信号安全的，这里简化处理
}

int main(void) {
    // 注册清理处理函数
    signal(SIGINT, cleanup_handler);
    signal(SIGTERM, cleanup_handler);

    log_file = fopen("app.log", "w");
    if (log_file == NULL) {
        perror("无法打开日志文件");
        return 1;
    }

    printf("程序运行中，按 Ctrl+C 退出...\n");

    int counter = 0;
    while (!shutdown_requested) {
        fprintf(log_file, "计数: %d\n", counter++);
        fflush(log_file);

        // 模拟工作
        for (volatile long i = 0; i < 50000000L; i++);
    }

    // 安全关闭
    printf("\n收到关闭信号，正在清理...\n");
    if (log_file != NULL) {
        fclose(log_file);
        printf("日志文件已关闭\n");
    }

    printf("程序正常退出\n");
    return 0;
}
```

### 5.2 超时机制

```c
#include <stdio.h>
#include <signal.h>
#include <setjmp.h>

static jmp_buf timeout_env;

void timeout_handler(int sig) {
    longjmp(timeout_env, 1);
}

// 带超时的输入
int timed_input(char *buffer, int size, int seconds) {
    // 设置超时信号
    signal(SIGALRM, timeout_handler);
    alarm(seconds);

    if (setjmp(timeout_env) == 0) {
        printf("请在 %d 秒内输入: ", seconds);
        if (fgets(buffer, size, stdin) != NULL) {
            alarm(0);  // 取消定时器
            return 1;
        }
    }

    // 超时
    printf("\n输入超时!\n");
    return 0;
}

int main(void) {
    char input[100];
    timed_input(input, sizeof(input), 5);
    return 0;
}
```

## 6. 重要注意事项

> **要点一**：信号处理函数中只能调用异步信号安全（async-signal-safe）的函数。标准POSIX规定了约110个这样的函数。

> **要点二**：`sig_atomic_t` 类型保证在信号处理函数中读写是原子的。

> **要点三**：`signal` 函数的行为在不同系统上可能不同，推荐使用 `sigaction`（POSIX）。

> **要点四**：不能捕获 `SIGKILL` 和 `SIGSTOP` 信号。

> **要点五**：信号处理函数执行期间，对应的信号会被阻塞（自动屏蔽）。

> **要点六**：如果信号处理函数中调用了不可重入函数（如 `malloc`、`printf`），可能导致死锁或数据损坏。

> **要点七**：`SIG_DFL` 和 `SIG_IGN` 是特殊的信号处理函数指针，分别表示默认处理和忽略。

> **要点八**：在信号处理函数中设置 `volatile sig_atomic_t` 标志，在主循环中检查是最安全的做法。
