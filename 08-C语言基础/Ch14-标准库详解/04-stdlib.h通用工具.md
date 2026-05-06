# stdlib.h - 通用工具函数

## 1. 概述

`<stdlib.h>` 提供了内存管理、数值转换、随机数生成、排序搜索和程序控制等通用工具函数。这是C标准库中功能最丰富的头文件之一。

## 2. 内存管理函数

### 2.1 动态内存分配

```c
#include <stdlib.h>

void *malloc(size_t size);                    // 分配内存（未初始化）
void *calloc(size_t nmemb, size_t size);      // 分配并清零内存
void *realloc(void *ptr, size_t size);        // 调整内存大小
void free(void *ptr);                         // 释放内存
```

### 2.2 内存管理示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(void) {
    // malloc - 分配未初始化的内存
    int *arr1 = (int *)malloc(5 * sizeof(int));
    if (arr1 == NULL) {
        fprintf(stderr, "malloc 分配失败\n");
        return 1;
    }
    // malloc不初始化，内容未定义
    for (int i = 0; i < 5; i++) {
        arr1[i] = i * 10;
    }

    // calloc - 分配并清零的内存
    int *arr2 = (int *)calloc(5, sizeof(int));
    if (arr2 == NULL) {
        fprintf(stderr, "calloc 分配失败\n");
        free(arr1);
        return 1;
    }
    // calloc初始化为0
    for (int i = 0; i < 5; i++) {
        printf("arr2[%d] = %d\n", i, arr2[i]);  // 全为0
    }

    // realloc - 调整内存大小
    int *arr3 = (int *)realloc(arr1, 10 * sizeof(int));
    if (arr3 == NULL) {
        fprintf(stderr, "realloc 失败\n");
        free(arr1);  // arr1仍然有效
        free(arr2);
        return 1;
    }
    // 注意：realloc可能移动内存，所以使用返回值
    arr1 = NULL;  // arr3现在管理这块内存

    // 填充新增的元素
    for (int i = 5; i < 10; i++) {
        arr3[i] = i * 10;
    }

    // 显示所有元素
    printf("realloc后的数组: ");
    for (int i = 0; i < 10; i++) {
        printf("%d ", arr3[i]);
    }
    printf("\n");

    // 释放内存
    free(arr3);
    free(arr2);

    // 悬空指针防护
    arr3 = NULL;
    arr2 = NULL;

    return 0;
}
```

### 2.3 aligned_alloc（C11）

```c
#include <stdlib.h>

// C11: 分配对齐的内存
void *aligned_alloc(size_t alignment, size_t size);

int main(void) {
    // 分配64字节对齐的256字节内存
    // size 必须是 alignment 的整数倍
    void *ptr = aligned_alloc(64, 256);
    if (ptr != NULL) {
        printf("对齐分配的地址: %p\n", ptr);
        printf("地址是否64字节对齐: %s\n",
               ((uintptr_t)ptr % 64 == 0) ? "是" : "否");
        free(ptr);
    }
    return 0;
}
```

## 3. 数值转换函数

### 3.1 字符串转数值

```c
#include <stdlib.h>

int atoi(const char *nptr);            // 转为int
long atol(const char *nptr);           // 转为long
long long atoll(const char *nptr);     // 转为long long (C99)
double atof(const char *nptr);         // 转为double

// 更安全的版本（带错误检测）
long strtol(const char *nptr, char **endptr, int base);
long long strtoll(const char *nptr, char **endptr, int base);
unsigned long strtoul(const char *nptr, char **endptr, int base);
unsigned long long strtoull(const char *nptr, char **endptr, int base);
double strtod(const char *nptr, char **endptr);
float strtof(const char *nptr, char **endptr);        // C99
long double strtold(const char *nptr, char **endptr);  // C99
```

### 3.2 转换示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>

int main(void) {
    // atoi 系列（简单但无错误检测）
    int num1 = atoi("12345");
    double num2 = atof("3.14159");
    printf("atoi: %d, atof: %f\n", num1, num2);

    // 错误情况：atoi不报错
    int bad = atoi("xyz");  // 返回0，无法区分错误
    printf("atoi(\"xyz\") = %d\n", bad);

    // strtol - 推荐使用，带错误检测
    char *endptr;
    errno = 0;
    long val = strtol("12345xyz", &endptr, 10);
    if (errno != 0) {
        perror("strtol");
    } else if (*endptr != '\0') {
        printf("解析到: %ld, 剩余: \"%s\"\n", val, endptr);
    }

    // 不同进制转换
    long hex_val = strtol("FF", NULL, 16);     // 255
    long oct_val = strtol("77", NULL, 8);      // 63
    long bin_val = strtol("1010", NULL, 2);    // 10
    printf("十六进制FF = %ld, 八进制77 = %ld, 二进制1010 = %ld\n",
           hex_val, oct_val, bin_val);

    // 自动检测进制（base=0）
    long auto1 = strtol("0xFF", NULL, 0);   // 检测为十六进制
    long auto2 = strtol("077", NULL, 0);    // 检测为八进制
    long auto3 = strtol("100", NULL, 0);    // 检测为十进制
    printf("自动检测: %ld, %ld, %ld\n", auto1, auto2, auto3);

    return 0;
}
```

## 4. 随机数函数

```c
#include <stdlib.h>

int rand(void);                // 生成伪随机数
void srand(unsigned int seed); // 设置随机种子
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(void) {
    // 使用当前时间作为种子（只调用一次）
    srand((unsigned int)time(NULL));

    // 生成0到RAND_MAX之间的随机数
    printf("RAND_MAX = %d\n", RAND_MAX);

    // 生成指定范围的随机数
    // 公式: rand() % (max - min + 1) + min
    for (int i = 0; i < 5; i++) {
        int dice = rand() % 6 + 1;          // 1-6
        int range = rand() % 91 + 10;       // 10-100
        printf("骰子: %d, 范围[10-100]: %d\n", dice, range);
    }

    // 生成0到1之间的浮点随机数
    for (int i = 0; i < 5; i++) {
        double ratio = (double)rand() / RAND_MAX;
        printf("随机浮点数: %f\n", ratio);
    }

    return 0;
}
```

## 5. 程序控制函数

### 5.1 程序终止

```c
#include <stdlib.h>

void exit(int status);          // 正常终止程序
void _Exit(int status);         // 快速终止（C99）
void abort(void);               // 异常终止
int atexit(void (*func)(void)); // 注册退出处理函数
```

### 5.2 程序终止示例

```c
#include <stdio.h>
#include <stdlib.h>

// 退出处理函数
void cleanup1(void) {
    printf("清理函数1: 释放资源\n");
}

void cleanup2(void) {
    printf("清理函数2: 关闭日志\n");
}

int main(void) {
    // 注册退出处理函数（后注册的先执行）
    atexit(cleanup1);
    atexit(cleanup2);

    printf("程序开始执行\n");

    // exit: 正常终止，刷新缓冲区，调用atexit注册的函数
    // 参数: EXIT_SUCCESS(0) 或 EXIT_FAILURE(非0)
    exit(EXIT_SUCCESS);

    // 以下代码不会执行
    printf("这行不会被执行\n");
    return 0;
}
// 输出:
// 程序开始执行
// 清理函数2: 关闭日志
// 清理函数1: 释放资源
```

### 5.3 系统命令

```c
#include <stdlib.h>

int system(const char *command);

int main(void) {
    // 执行系统命令
    int ret = system("echo Hello from shell");
    printf("命令返回值: %d\n", ret);

    // 检查命令是否可用
    if (system(NULL)) {
        printf("命令解释器可用\n");
    }

    return 0;
}
```

## 6. 排序与搜索

```c
#include <stdlib.h>

void qsort(void *base, size_t nmemb, size_t size,
           int (*compar)(const void *, const void *));

void *bsearch(const void *key, const void *base, size_t nmemb, size_t size,
              int (*compar)(const void *, const void *));
```

### 排序与搜索示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 整数比较函数
int compare_int(const void *a, const void *b) {
    return (*(int *)a - *(int *)b);
}

// 浮点数比较函数
int compare_double(const void *a, const void *b) {
    double diff = *(double *)a - *(double *)b;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

// 字符串比较函数
int compare_string(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

int main(void) {
    // 排序整数数组
    int nums[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(nums) / sizeof(nums[0]);

    qsort(nums, n, sizeof(int), compare_int);

    printf("排序后的整数: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", nums[i]);
    }
    printf("\n");

    // 二分查找
    int key = 25;
    int *result = bsearch(&key, nums, n, sizeof(int), compare_int);
    if (result != NULL) {
        printf("找到 %d, 位置: %ld\n", *result, result - nums);
    } else {
        printf("未找到 %d\n", key);
    }

    // 排序字符串数组
    const char *fruits[] = {"banana", "apple", "cherry", "date"};
    int fn = sizeof(fruits) / sizeof(fruits[0]);

    qsort(fruits, fn, sizeof(char *), compare_string);

    printf("排序后的水果: ");
    for (int i = 0; i < fn; i++) {
        printf("%s ", fruits[i]);
    }
    printf("\n");

    return 0;
}
```

## 7. 环境变量

```c
#include <stdlib.h>

char *getenv(const char *name);  // 获取环境变量
```

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 获取常见环境变量
    char *path = getenv("PATH");
    if (path != NULL) {
        printf("PATH = %s\n", path);
    }

    char *home = getenv("HOME");
    if (home != NULL) {
        printf("HOME = %s\n", home);
    }

    char *user = getenv("USER");
    if (user != NULL) {
        printf("USER = %s\n", user);
    }

    return 0;
}
```

## 8. 整数算术函数

```c
#include <stdlib.h>

int abs(int j);                 // 绝对值
long labs(long j);              // long绝对值
long long llabs(long long j);   // long long绝对值 (C99)

div_t div(int numer, int denom);       // 除法同时得商和余数
ldiv_t ldiv(long numer, long denom);
lldiv_t lldiv(long long numer, long long denom);  // C99
```

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 绝对值
    printf("abs(-42) = %d\n", abs(-42));
    printf("labs(-123456L) = %ld\n", labs(-123456L));

    // div - 同时获取商和余数（比分别用 / 和 % 更高效）
    div_t result = div(17, 5);
    printf("17 / 5 = 商%d 余%d\n", result.quot, result.rem);

    ldiv_t lresult = ldiv(100000L, 3L);
    printf("100000 / 3 = 商%ld 余%ld\n", lresult.quot, lresult.rem);

    return 0;
}
```

## 9. 多字节/宽字符函数

```c
#include <stdlib.h>

int mblen(const char *s, size_t n);           // 多字节字符长度
int mbtowc(wchar_t *pwc, const char *s, size_t n);  // 多字节转宽字符
int wctomb(char *s, wchar_t wchar);           // 宽字符转多字节
size_t mbstowcs(wchar_t *dest, const char *src, size_t n);  // 字符串转换
size_t wcstombs(char *dest, const wchar_t *src, size_t n);  // 字符串转换
```

## 10. 重要注意事项

> **要点一**：`malloc` 返回的指针类型是 `void*`，可隐式转换为任何指针类型（C语言中），但C++需要显式转换。

> **要点二**：`realloc` 可能移动内存块，返回的地址可能与原地址不同，务必使用返回值。

> **要点三**：`free` 后应将指针置为 `NULL`，防止悬空指针。

> **要点四**：`atoi` 无法检测错误，推荐使用 `strtol` 系列函数。

> **要点五**：`rand()` 生成的是伪随机数，不适合密码学用途。

> **要点六**：`srand` 只需在程序开始时调用一次，重复调用反而会降低随机性。

> **要点七**：`exit()` 会刷新所有打开的输出流并关闭所有打开的文件，而 `_Exit()` 不会。

> **要点八**：`system()` 函数存在安全风险（命令注入），生产环境中慎用。

> **要点九**：`qsort` 的比较函数必须遵循严格的规范：相等时返回0，前者大于后者返回正数。
