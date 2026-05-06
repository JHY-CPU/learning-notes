# Linux 系统编程概述

## 1. 什么是系统编程

系统编程是指编写直接与操作系统内核交互的程序，使用**系统调用**（System Call）来完成文件操作、进程管理、内存管理、网络通信等底层功能。与应用层编程不同，系统编程更贴近操作系统，需要理解内核的工作机制。

### 系统编程的典型应用场景

- 操作系统工具（shell、文件管理器）
- 服务器程序（Web服务器、数据库服务器）
- 嵌入式系统开发
- 设备驱动程序
- 高性能计算

## 2. 系统调用 vs 库函数

### 系统调用

系统调用是用户程序请求内核服务的接口，是从用户态切换到内核态的唯一途径。

```c
#include <sys/syscall.h>
#include <unistd.h>

// 直接使用系统调用
ssize_t ret = syscall(SYS_write, 1, "Hello\n", 6);
```

### 常见系统调用分类

| 类别 | 系统调用示例 |
|------|-------------|
| 文件IO | open, read, write, close, lseek |
| 进程管理 | fork, exec, wait, exit |
| 内存管理 | mmap, brk, sbrk |
| 信号处理 | signal, kill, sigaction |
| 进程间通信 | pipe, shmget, semget, msgget |
| 网络通信 | socket, bind, listen, accept, connect |
| 时间管理 | gettimeofday, clock_gettime, nanosleep |

### 库函数

库函数是建立在系统调用之上的封装，提供了更方便的接口和额外功能。

```c
#include <stdio.h>

// fopen 是库函数，内部调用了 open 系统调用
FILE *fp = fopen("test.txt", "r");
// fgetc 是库函数，内部使用了带缓冲的 read
int ch = fgetc(fp);
fclose(fp);
```

### 两者的区别

```c
/*
 * 系统调用 vs 库函数 对比
 *
 * 系统调用:
 *   - 直接进入内核态
 *   - 开销较大（用户态到内核态的切换）
 *   - 不带缓冲
 *   - POSIX 标准定义
 *
 * 库函数:
 *   - 可能在用户态完成（如字符串操作）
 *   - 可能带缓冲（如 stdio）
 *   - 可移植性更好
 *   - 可以调用多个系统调用
 */

// 系统调用方式：无缓冲直接写入
#include <unistd.h>
#include <fcntl.h>
#include <string.h>

void syscall_example(void)
{
    int fd = open("output.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd == -1) {
        perror("open");
        return;
    }

    const char *msg = "Hello, Linux System Programming!\n";
    ssize_t nwritten = write(fd, msg, strlen(msg));
    if (nwritten == -1) {
        perror("write");
    }

    close(fd);
}

// 库函数方式：带缓冲写入
#include <stdio.h>

void libc_example(void)
{
    FILE *fp = fopen("output.txt", "w");
    if (fp == NULL) {
        perror("fopen");
        return;
    }

    fprintf(fp, "Hello, Linux System Programming!\n");

    fclose(fp);  // 缓冲区在这里被刷新
}
```

## 3. POSIX 标准

POSIX（Portable Operating System Interface）是一系列 IEEE 标准，定义了操作系统应提供的接口。

```c
// 检查 POSIX 版本
#include <unistd.h>
#include <stdio.h>

int main(void)
{
    // _POSIX_VERSION 定义了支持的 POSIX 版本
    // 值格式：YYYYMMLL（年-月-级别）
    printf("POSIX Version: %ld\n", (long)_POSIX_VERSION);

    // 检查特定功能是否支持
    #ifdef _POSIX_THREADS
        printf("POSIX Threads: Supported\n");
    #else
        printf("POSIX Threads: Not Supported\n");
    #endif

    #ifdef _POSIX_REALTIME_SIGNALS
        printf("Realtime Signals: Supported\n");
    #endif

    #ifdef _POSIX_MAPPED_FILES
        printf("Memory Mapped Files: Supported\n");
    #endif

    // sysconf 可在运行时查询系统配置
    long pagesize = sysconf(_SC_PAGESIZE);
    long nprocs = sysconf(_SC_NPROCESSORS_ONLN);
    long open_max = sysconf(_SC_OPEN_MAX);

    printf("Page Size: %ld bytes\n", pagesize);
    printf("Online Processors: %ld\n", nprocs);
    printf("Max Open Files: %ld\n", open_max);

    return 0;
}
```

### POSIX 标准覆盖范围

- POSIX.1：核心系统调用和库函数
- POSIX.1b：实时扩展（调度、时钟、信号、消息队列、信号量、共享内存）
- POSIX.1c：线程（pthread）
- POSIX.2：Shell 和工具

## 4. 开发环境搭建

### 编译器和工具链

```bash
# 安装 GCC 和开发工具（Debian/Ubuntu）
sudo apt-get install build-essential

# 安装 man 手册
sudo apt-get install manpages-dev manpages-posix-dev

# 查看系统调用手册
man 2 open
# 查看库函数手册
man 3 printf
# 查看 POSIX 手册
man 7 posix
```

### 编译选项

```bash
# 基本编译
gcc -o myprogram myprogram.c

# 启用所有警告和 POSIX 标准
gcc -Wall -Wextra -std=c11 -D_POSIX_C_SOURCE=200809L -o myprogram myprogram.c

# 链接数学库和线程库
gcc -Wall -std=c11 -D_GNU_SOURCE -o myprogram myprogram.c -lpthread -lm

# 编译时开启调试信息
gcc -g -O0 -Wall -o myprogram myprogram.c
```

### 错误处理机制

系统编程中几乎所有系统调用都可能失败，必须正确处理错误。

```c
#include <stdio.h>
#include <errno.h>
#include <string.h>
#include <stdlib.h>

/*
 * 标准错误处理模式:
 * 系统调用失败时通常返回 -1，并设置 errno 全局变量
 */

void error_handling_example(void)
{
    // 方式1：使用 perror（推荐）
    FILE *fp = fopen("/nonexistent/file.txt", "r");
    if (fp == NULL) {
        perror("fopen");  // 输出: fopen: No such file or directory
        // 注意：popen 等函数也会设置 errno
    }

    // 方式2：使用 strerror
    int fd = open("/root/secret.txt", O_RDONLY);
    if (fd == -1) {
        fprintf(stderr, "open failed: %s (errno=%d)\n",
                strerror(errno), errno);
    }

    // 方式3：自定义错误处理函数
    // 见下方 err_exit 定义
}

// 自定义错误退出函数
void err_exit(const char *msg)
{
    perror(msg);
    exit(EXIT_FAILURE);
}

// 带格式的错误退出
void err_exit_fmt(const char *fmt, ...)
{
    va_list args;
    va_start(args, fmt);
    vfprintf(stderr, fmt, args);
    va_end(args);
    fprintf(stderr, ": %s\n", strerror(errno));
    exit(EXIT_FAILURE);
}
```

### 常见 errno 值

```c
#include <errno.h>

/*
 * 常见 errno 值含义:
 *
 * EACCES     - 权限不足
 * ENOENT     - 文件或目录不存在
 * EEXIST     - 文件已存在
 * EAGAIN     - 资源暂时不可用（非阻塞IO）
 * EINTR      - 系统调用被信号中断
 * EINVAL     - 无效参数
 * ENOMEM     - 内存不足
 * EPIPE      - 管道破裂（写端已关闭）
 * ECONNRESET - 连接被重置
 * ETIMEDOUT  - 操作超时
 */
```

## 5. 文件描述符

文件描述符是一个非负整数，内核用来标识打开的文件、管道、socket 等。

```c
#include <unistd.h>
#include <fcntl.h>
#include <stdio.h>

/*
 * 标准文件描述符:
 *   0 - STDIN_FILENO  (标准输入)
 *   1 - STDOUT_FILENO (标准输出)
 *   2 - STDERR_FILENO (标准错误)
 */

void fd_demo(void)
{
    // 查看进程最大打开文件数
    long max_fd = sysconf(_SC_OPEN_MAX);
    printf("Max open files: %ld\n", max_fd);

    // 打开文件，获取文件描述符
    int fd = open("test.txt", O_RDONLY);
    if (fd == -1) {
        perror("open");
        return;
    }
    printf("File descriptor: %d\n", fd);  // 通常是 3（0,1,2 已被占用）

    // dup 复制文件描述符
    int fd2 = dup(fd);
    printf("Duplicated fd: %d\n", fd2);

    close(fd);
    close(fd2);
}
```

## 6. 未缓冲 IO vs 缓冲 IO

```c
#include <unistd.h>
#include <stdio.h>
#include <fcntl.h>

/*
 * 未缓冲IO（系统调用）:
 *   open, close, read, write, lseek, fcntl, ioctl
 *   直接操作文件描述符，没有用户态缓冲
 *
 * 缓冲IO（标准IO库）:
 *   fopen, fclose, fread, fwrite, fgets, fputs
 *   操作 FILE* 指针，有用户态缓冲
 *
 * 缓冲模式:
 *   全缓冲 - 缓冲区满时刷新（普通文件）
 *   行缓冲 - 遇到换行符时刷新（终端）
 *   无缓冲 - 立即输出（stderr）
 */

void buffer_demo(void)
{
    // 未缓冲IO：直接调用 write
    int fd = open("unbuf.txt", O_WRONLY | O_CREAT | O_TRUNC, 0644);
    write(fd, "Line 1\n", 7);   // 立即写入内核
    write(fd, "Line 2\n", 7);
    close(fd);

    // 缓冲IO：经过 FILE 缓冲区
    FILE *fp = fopen("buf.txt", "w");
    fprintf(fp, "Line 1\n");     // 写入用户态缓冲区
    fprintf(fp, "Line 2\n");
    fflush(fp);                   // 强制刷新缓冲区
    fclose(fp);                   // 关闭时自动刷新
}
```

## 7. 综合示例：简易文件复制程序

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <fcntl.h>
#include <errno.h>
#include <string.h>

#define BUFFER_SIZE 4096

/*
 * 使用系统调用实现文件复制
 * 演示：open, read, write, close 的基本用法
 */
int copy_file(const char *src, const char *dst)
{
    int fd_src, fd_dst;
    char buffer[BUFFER_SIZE];
    ssize_t nread, nwritten;

    // 打开源文件（只读）
    fd_src = open(src, O_RDONLY);
    if (fd_src == -1) {
        fprintf(stderr, "Cannot open source '%s': %s\n",
                src, strerror(errno));
        return -1;
    }

    // 创建目标文件（写入，若存在则截断）
    fd_dst = open(dst, O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd_dst == -1) {
        fprintf(stderr, "Cannot create destination '%s': %s\n",
                dst, strerror(errno));
        close(fd_src);
        return -1;
    }

    // 循环读写
    while ((nread = read(fd_src, buffer, BUFFER_SIZE)) > 0) {
        char *ptr = buffer;
        ssize_t remaining = nread;

        // 处理 write 可能未写完的情况
        while (remaining > 0) {
            nwritten = write(fd_dst, ptr, remaining);
            if (nwritten == -1) {
                if (errno == EINTR) {
                    continue;  // 被信号中断，重试
                }
                perror("write");
                close(fd_src);
                close(fd_dst);
                return -1;
            }
            remaining -= nwritten;
            ptr += nwritten;
        }
    }

    if (nread == -1) {
        perror("read");
        close(fd_src);
        close(fd_dst);
        return -1;
    }

    close(fd_src);
    close(fd_dst);
    return 0;
}

int main(int argc, char *argv[])
{
    if (argc != 3) {
        fprintf(stderr, "Usage: %s <source> <destination>\n", argv[0]);
        return EXIT_FAILURE;
    }

    if (copy_file(argv[1], argv[2]) == 0) {
        printf("File copied successfully: %s -> %s\n", argv[1], argv[2]);
    } else {
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
```

## 8. 小结

- 系统编程直接与内核交互，通过系统调用完成底层操作
- POSIX 标准保证了不同 Unix 系统之间的可移植性
- 必须始终检查系统调用的返回值并正确处理错误
- 文件描述符是理解后续所有 IO 操作的基础
- 理解缓冲 IO 与未缓冲 IO 的区别对于正确编写程序至关重要
