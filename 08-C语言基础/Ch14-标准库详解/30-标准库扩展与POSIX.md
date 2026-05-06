# 标准库扩展与 POSIX

## 1. 概述

C标准库之外，还存在许多重要的扩展，其中最广泛使用的是POSIX（Portable Operating System Interface）标准。本章介绍这些扩展的主要内容以及编写可移植代码的注意事项。

## 2. POSIX 标准

### 2.1 POSIX 简介

POSIX 是IEEE制定的一系列操作系统接口标准，定义了UNIX-like系统应提供的API。它扩展了C标准库，添加了许多实用的系统级函数。

### 2.2 POSIX 主要头文件

```c
// 文件和目录操作
#include <dirent.h>        // 目录操作
#include <fcntl.h>         // 文件控制
#include <sys/stat.h>      // 文件状态
#include <unistd.h>        // UNIX标准函数

// 进程管理
#include <sys/types.h>     // 基本系统数据类型
#include <sys/wait.h>      // 进程等待
#include <signal.h>        // 信号处理（扩展）

// 线程（POSIX线程）
#include <pthread.h>       // POSIX线程

// 网络
#include <sys/socket.h>    // 套接字
#include <netinet/in.h>    // 网络地址
#include <arpa/inet.h>     // 网络工具
#include <netdb.h>         // 网络数据库

// 定时器
#include <sys/time.h>      // 时间（扩展）
#include <time.h>          // 时间（扩展）

// 正则表达式
#include <regex.h>         // 正则表达式
```

## 3. POSIX 文件操作扩展

```c
#include <stdio.h>
#include <sys/stat.h>
#include <unistd.h>
#include <dirent.h>

// 文件状态
void file_info(const char *path) {
    struct stat st;
    if (stat(path, &st) == 0) {
        printf("文件: %s\n", path);
        printf("  大小: %ld 字节\n", (long)st.st_size);
        printf("  权限: %o\n", st.st_mode & 0777);
        printf("  链接数: %ld\n", (long)st.st_nlink);

        if (S_ISREG(st.st_mode))  printf("  类型: 普通文件\n");
        if (S_ISDIR(st.st_mode))  printf("  类型: 目录\n");
        if (S_ISLNK(st.st_mode))  printf("  类型: 符号链接\n");
    }
}

// 列出目录内容
void list_directory(const char *path) {
    DIR *dir = opendir(path);
    if (dir == NULL) {
        perror("无法打开目录");
        return;
    }

    struct dirent *entry;
    while ((entry = readdir(dir)) != NULL) {
        printf("  %s\n", entry->d_name);
    }

    closedir(dir);
}

int main(void) {
    // 获取当前工作目录
    char cwd[1024];
    if (getcwd(cwd, sizeof(cwd)) != NULL) {
        printf("当前目录: %s\n", cwd);
    }

    // 获取文件信息
    file_info(__FILE__);

    // 列出目录
    printf("\n目录内容:\n");
    list_directory(".");

    return 0;
}
```

## 4. POSIX 线程（pthreads）

```c
#include <stdio.h>
#include <pthread.h>

#define NUM_THREADS 4

int shared_counter = 0;
pthread_mutex_t mutex = PTHREAD_MUTEX_INITIALIZER;

void *worker(void *arg) {
    int id = *(int *)arg;

    for (int i = 0; i < 10000; i++) {
        pthread_mutex_lock(&mutex);
        shared_counter++;
        pthread_mutex_unlock(&mutex);
    }

    printf("线程 %d 完成\n", id);
    return NULL;
}

int main(void) {
    pthread_t threads[NUM_THREADS];
    int ids[NUM_THREADS];

    for (int i = 0; i < NUM_THREADS; i++) {
        ids[i] = i;
        pthread_create(&threads[i], NULL, worker, &ids[i]);
    }

    for (int i = 0; i < NUM_THREADS; i++) {
        pthread_join(threads[i], NULL);
    }

    printf("最终计数: %d\n", shared_counter);

    pthread_mutex_destroy(&mutex);
    return 0;
}
```

## 5. POSIX 进程管理

```c
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>
#include <sys/wait.h>

int main(void) {
    pid_t pid = fork();

    if (pid < 0) {
        perror("fork失败");
        return 1;
    } else if (pid == 0) {
        // 子进程
        printf("子进程 PID: %d\n", getpid());
        printf("父进程 PID: %d\n", getppid());

        // 执行命令
        execlp("echo", "echo", "Hello from child", NULL);
        perror("exec失败");
        exit(1);
    } else {
        // 父进程
        printf("父进程: 创建了子进程 %d\n", pid);

        int status;
        waitpid(pid, &status, 0);

        if (WIFEXITED(status)) {
            printf("子进程退出码: %d\n", WEXITSTATUS(status));
        }
    }

    return 0;
}
```

## 6. POSIX 时间函数扩展

```c
#include <stdio.h>
#include <time.h>
#include <sys/time.h>

int main(void) {
    // 高精度时间
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    printf("单调时钟: %ld.%09ld 秒\n", ts.tv_sec, ts.tv_nsec);

    clock_gettime(CLOCK_REALTIME, &ts);
    printf("实时时钟: %ld.%09ld 秒\n", ts.tv_sec, ts.tv_nsec);

    // 计时
    struct timespec start, end;
    clock_gettime(CLOCK_MONOTONIC, &start);

    // 执行操作
    volatile long sum = 0;
    for (long i = 0; i < 10000000L; i++) sum += i;

    clock_gettime(CLOCK_MONOTONIC, &end);

    double elapsed = (end.tv_sec - start.tv_sec) +
                     (end.tv_nsec - start.tv_nsec) / 1e9;
    printf("耗时: %.6f 秒\n", elapsed);

    return 0;
}
```

## 7. POSIX 字符串扩展

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    // POSIX扩展的字符串函数
    // 注意：这些不是标准C函数

    // strdup - 复制字符串（自动分配内存）
    // char *strdup(const char *s);
    // char *s = strdup("Hello");
    // free(s);

    // strcasecmp - 不区分大小写比较
    // int result = strcasecmp("Hello", "hello");

    // strtok_r - 线程安全的字符串分割
    // char *strtok_r(char *str, const char *delim, char **saveptr);

    // strsep - 字符串分割（替代strtok）
    // char *strsep(char **stringp, const char *delim);

    printf("POSIX字符串函数示例:\n");
    printf("注意：需要在支持POSIX的系统上编译\n");

    return 0;
}
```

## 8. 平台特定的扩展

### 8.1 Windows 特有

```c
#ifdef _WIN32
#include <windows.h>
#include <io.h>
#include <direct.h>

// Windows特定函数
void windows_specific(void) {
    // 获取系统目录
    char path[MAX_PATH];
    GetSystemDirectoryA(path, MAX_PATH);
    printf("系统目录: %s\n", path);

    // 创建目录
    _mkdir("test_dir");

    // 文件查找
    WIN32_FIND_DATAA findData;
    HANDLE hFind = FindFirstFileA("*.*", &findData);
    if (hFind != INVALID_HANDLE_VALUE) {
        do {
            printf("文件: %s\n", findData.cFileName);
        } while (FindNextFileA(hFind, &findData));
        FindClose(hFind);
    }
}
#endif
```

### 8.2 Linux 特有

```c
#ifdef __linux__
#include <sys/sysinfo.h>
#include <sys/utsname.h>

void linux_specific(void) {
    // 系统信息
    struct utsname sys_info;
    uname(&sys_info);
    printf("系统: %s %s\n", sys_info.sysname, sys_info.release);

    // 内存信息
    struct sysinfo si;
    sysinfo(&si);
    printf("总内存: %lu MB\n", si.totalram / 1024 / 1024);
    printf("可用内存: %lu MB\n", si.freeram / 1024 / 1024);
}
#endif
```

### 8.3 macOS 特有

```c
#ifdef __APPLE__
#include <mach/mach.h>
#include <sys/sysctl.h>

void macos_specific(void) {
    // 获取CPU信息
    char cpu_brand[256];
    size_t size = sizeof(cpu_brand);
    sysctlbyname("machdep.cpu.brand_string",
                 cpu_brand, &size, NULL, 0);
    printf("CPU: %s\n", cpu_brand);
}
#endif
```

## 9. 编写可移植代码

### 9.1 特性检测

```c
#include <stdio.h>

// 使用标准特性检测宏
int main(void) {
    // C标准版本
    #ifdef __STDC__
    printf("支持ANSI C\n");
    #endif

    #ifdef __STDC_VERSION__
    printf("C标准版本: %ld\n", __STDC_VERSION__);
    // 199901L = C99
    // 201112L = C11
    // 201710L = C17
    #endif

    // 编译器检测
    #ifdef __GNUC__
    printf("GCC版本: %d.%d.%d\n",
           __GNUC__, __GNUC_MINOR__, __GNUC_PATCHLEVEL__);
    #endif

    #ifdef _MSC_VER
    printf("MSVC版本: %d\n", _MSC_VER);
    #endif

    // 操作系统检测
    #ifdef _WIN32
    printf("平台: Windows\n");
    #elif defined(__linux__)
    printf("平台: Linux\n");
    #elif defined(__APPLE__)
    printf("平台: macOS\n");
    #endif

    // POSIX检测
    #ifdef _POSIX_VERSION
    printf("POSIX版本: %ld\n", _POSIX_VERSION);
    #endif

    return 0;
}
```

### 9.2 条件编译模式

```c
#include <stdio.h>

// 统一接口模式
#ifdef _WIN32
    #include <windows.h>
    #define SLEEP_MS(ms) Sleep(ms)
    #define MYPATH_SEP '\\'
#else
    #include <unistd.h>
    #define SLEEP_MS(ms) usleep((ms) * 1000)
    #define MYPATH_SEP '/'
#endif

// 抽象接口
typedef struct {
#ifdef _WIN32
    HANDLE handle;
#else
    int fd;
#endif
} FileHandle;

// 跨平台的路径构建
void build_path(char *dest, const char *dir, const char *file) {
    sprintf(dest, "%s%c%s", dir, MYPATH_SEP, file);
}

int main(void) {
    char path[256];
    build_path(path, "/home/user", "file.txt");
    printf("路径: %s\n", path);

    SLEEP_MS(100);  // 休眠100毫秒
    printf("休眠完成\n");

    return 0;
}
```

## 10. 重要注意事项

> **要点一**：POSIX函数在Windows上不可用，需要使用兼容层（如Cygwin、MSYS2）或Windows原生API。

> **要点二**：编写可移植代码时，始终提供标准C的回退实现。

> **要点三**：`_POSIX_C_SOURCE` 和 `_XOPEN_SOURCE` 宏用于启用POSIX扩展。

> **要点四**：`pthread` 库在Linux/Mac上需要链接 `-lpthread`。

> **要点五**：文件描述符（POSIX）和文件句柄（Windows）是不同平台的文件操作抽象。

> **要点六**：线程安全函数（以 `_r` 结尾，如 `strtok_r`）是POSIX扩展，不是标准C的一部分。

> **要点七**：`getopt` 函数用于命令行参数解析，是POSIX标准但不是C标准。

> **要点八**：使用 `#ifdef` / `#elif` / `#else` 来处理平台差异，保持代码整洁。
