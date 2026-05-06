# time.h - 时间与日期

## 1. 概述

`<time.h>` 提供了时间获取、转换和格式化的功能。C语言的时间分为两种表示方式：日历时间（calendar time）和处理器时间（processor time）。

## 2. 时间类型

```c
#include <time.h>

time_t          // 日历时间类型（通常是从1970-01-01 00:00:00 UTC的秒数）
clock_t         // 处理器时间类型
struct tm       // 分解时间结构体
size_t          // 无符号整数类型

// struct tm 的成员:
struct tm {
    int tm_sec;    // 秒 [0, 60]（60用于闰秒）
    int tm_min;    // 分 [0, 59]
    int tm_hour;   // 时 [0, 23]
    int tm_mday;   // 日 [1, 31]
    int tm_mon;    // 月 [0, 11]（0=一月）
    int tm_year;   // 年（从1900开始）
    int tm_wday;   // 星期 [0, 6]（0=周日）
    int tm_yday;   // 年中天数 [0, 365]
    int tm_isdst;  // 夏令时标志
};
```

## 3. 获取时间

```c
#include <time.h>

time_t time(time_t *tp);          // 获取当前日历时间
clock_t clock(void);              // 获取处理器时间
```

```c
#include <stdio.h>
#include <time.h>

int main(void) {
    // time - 获取当前时间
    time_t now = time(NULL);
    printf("当前时间戳: %lld\n", (long long)now);

    // 也可以传入指针
    time_t t;
    time(&t);
    printf("当前时间戳: %lld\n", (long long)t);

    // clock - 获取CPU时间
    clock_t start = clock();

    // 执行一些操作
    volatile long sum = 0;
    for (long i = 0; i < 10000000L; i++) {
        sum += i;
    }

    clock_t end = clock();
    double cpu_time = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU时间: %f 秒\n", cpu_time);
    printf("CLOCKS_PER_SEC = %ld\n", CLOCKS_PER_SEC);

    return 0;
}
```

## 4. 时间转换

```c
#include <time.h>

struct tm *gmtime(const time_t *tp);       // UTC时间
struct tm *localtime(const time_t *tp);    // 本地时间
time_t mktime(struct tm *tp);              // tm转time_t
double difftime(time_t time1, time_t time0);  // 计算时间差
```

```c
#include <stdio.h>
#include <time.h>

int main(void) {
    time_t now = time(NULL);

    // 转换为本地时间
    struct tm *local = localtime(&now);
    printf("本地时间:\n");
    printf("  %d年%02d月%02d日\n",
           local->tm_year + 1900, local->tm_mon + 1, local->tm_mday);
    printf("  %02d:%02d:%02d\n",
           local->tm_hour, local->tm_min, local->tm_sec);
    printf("  星期%d (0=周日)\n", local->tm_wday);
    printf("  今年第%d天\n", local->tm_yday);
    printf("  夏令时: %s\n", local->tm_isdst > 0 ? "是" : "否");

    // 转换为UTC时间
    struct tm *utc = gmtime(&now);
    printf("\nUTC时间:\n");
    printf("  %d年%02d月%02d日 %02d:%02d:%02d\n",
           utc->tm_year + 1900, utc->tm_mon + 1, utc->tm_mday,
           utc->tm_hour, utc->tm_min, utc->tm_sec);

    // mktime - 从tm构建time_t
    struct tm birthday = {0};
    birthday.tm_year = 2000 - 1900;  // 2000年
    birthday.tm_mon = 5;             // 6月（0-based）
    birthday.tm_mday = 15;           // 15日
    birthday.tm_hour = 10;
    birthday.tm_min = 30;

    time_t bday_time = mktime(&birthday);
    printf("\n生日时间戳: %lld\n", (long long)bday_time);

    // 计算时间差
    time_t start = time(NULL);
    // ... 执行操作 ...
    time_t end = time(NULL);
    double elapsed = difftime(end, start);
    printf("经过时间: %.0f 秒\n", elapsed);

    return 0;
}
```

## 5. 时间格式化

```c
#include <time.h>

size_t strftime(char *s, size_t max, const char *format, const struct tm *tp);
```

### strftime 格式说明符

| 说明符 | 含义 | 示例 |
|--------|------|------|
| `%Y` | 四位年份 | 2023 |
| `%y` | 两位年份 | 23 |
| `%m` | 月份 (01-12) | 06 |
| `%d` | 日期 (01-31) | 15 |
| `%H` | 小时 24制 (00-23) | 14 |
| `%I` | 小时 12制 (01-12) | 02 |
| `%M` | 分钟 (00-59) | 30 |
| `%S` | 秒 (00-60) | 45 |
| `%A` | 星期全名 | Thursday |
| `%a` | 星期缩写 | Thu |
| `%B` | 月份全名 | June |
| `%b` | 月份缩写 | Jun |
| `%p` | AM/PM | PM |
| `%Z` | 时区名 | CST |
| `%j` | 年中天数 (001-366) | 166 |
| `%U` | 年中周数 (周日起始) | 24 |
| `%W` | 年中周数 (周一起始) | 24 |
| `%c` | 本地日期时间 | Thu Jun 15 14:30:45 2023 |
| `%x` | 本地日期 | 06/15/23 |
| `%X` | 本地时间 | 14:30:45 |
| `%%` | 百分号 | % |

```c
#include <stdio.h>
#include <time.h>

int main(void) {
    time_t now = time(NULL);
    struct tm *local = localtime(&now);
    char buffer[256];

    // 常用格式
    strftime(buffer, sizeof(buffer), "%Y-%m-%d %H:%M:%S", local);
    printf("标准格式: %s\n", buffer);

    // 中文格式
    strftime(buffer, sizeof(buffer), "%Y年%m月%d日 %H时%M分%S秒", local);
    printf("中文格式: %s\n", buffer);

    // 带星期
    strftime(buffer, sizeof(buffer), "%A, %B %d, %Y", local);
    printf("英文格式: %s\n", buffer);

    // ISO 8601格式
    strftime(buffer, sizeof(buffer), "%Y-%m-%dT%H:%M:%S%z", local);
    printf("ISO 8601: %s\n", buffer);

    // 12小时制
    strftime(buffer, sizeof(buffer), "%I:%M:%S %p", local);
    printf("12小时制: %s\n", buffer);

    // 自定义格式
    strftime(buffer, sizeof(buffer),
             "今天是%Y年的第%j天, 第%W周", local);
    printf("详细信息: %s\n", buffer);

    return 0;
}
```

## 6. 时间解析

```c
#include <stdio.h>
#include <time.h>
#include <string.h>

int main(void) {
    // strptime (POSIX，非标准C)
    // C标准中没有strptime，但许多系统支持
    // 手动解析示例:
    struct tm parsed = {0};
    char date_str[] = "2023-12-25 15:30:00";

    // 使用sscanf解析
    sscanf(date_str, "%d-%d-%d %d:%d:%d",
           &parsed.tm_year, &parsed.tm_mon, &parsed.tm_mday,
           &parsed.tm_hour, &parsed.tm_min, &parsed.tm_sec);
    parsed.tm_year -= 1900;
    parsed.tm_mon -= 1;

    time_t t = mktime(&parsed);
    char output[100];
    strftime(output, sizeof(output), "%Y年%m月%d日 %H:%M:%S", &parsed);
    printf("解析结果: %s (时间戳: %lld)\n", output, (long long)t);

    return 0;
}
```

## 7. 性能测量

```c
#include <stdio.h>
#include <time.h>

// 使用clock测量CPU时间
void measure_cpu_time(void) {
    clock_t start = clock();

    // CPU密集型操作
    double sum = 0;
    for (long i = 0; i < 100000000L; i++) {
        sum += 1.0 / (i + 1);
    }

    clock_t end = clock();
    double cpu_seconds = (double)(end - start) / CLOCKS_PER_SEC;
    printf("CPU时间: %.3f 秒\n", cpu_seconds);
    printf("部分和: %f\n", sum);
}

// 使用time测量实际时间
void measure_wall_time(void) {
    time_t start = time(NULL);

    // 模拟耗时操作
    for (volatile int i = 0; i < 1000000; i++);

    time_t end = time(NULL);
    printf("实际经过: %lld 秒\n", (long long)(end - start));
}

// 使用clock_gettime获取高精度时间（POSIX）
#ifdef _POSIX_C_SOURCE
#include <time.h>
void measure_high_precision(void) {
    struct timespec ts_start, ts_end;
    clock_gettime(CLOCK_MONOTONIC, &ts_start);

    // 操作...

    clock_gettime(CLOCK_MONOTONIC, &ts_end);
    double elapsed = (ts_end.tv_sec - ts_start.tv_sec) +
                     (ts_end.tv_nsec - ts_start.tv_nsec) / 1e9;
    printf("高精度时间: %.9f 秒\n", elapsed);
}
#endif

int main(void) {
    printf("=== CPU时间测量 ===\n");
    measure_cpu_time();

    printf("\n=== 实际时间测量 ===\n");
    measure_wall_time();

    return 0;
}
```

## 8. 定时器示例

```c
#include <stdio.h>
#include <time.h>
#include <stdlib.h>

int main(void) {
    // 简单的倒计时器
    printf("程序将在3秒后执行操作...\n");

    time_t start = time(NULL);
    time_t target = start + 3;

    while (time(NULL) < target) {
        // 忙等待（不推荐用于实际应用）
    }

    printf("时间到！\n");

    // 打印当前时间（每秒更新，共5次）
    printf("\n实时时钟:\n");
    for (int i = 0; i < 5; i++) {
        time_t now = time(NULL);
        struct tm *t = localtime(&now);
        char buf[64];
        strftime(buf, sizeof(buf), "%H:%M:%S", t);
        printf("\r当前时间: %s", buf);
        fflush(stdout);

        // 等待1秒
        time_t target = time(NULL) + 1;
        while (time(NULL) < target);
    }
    printf("\n");

    return 0;
}
```

## 9. 重要注意事项

> **要点一**：`localtime` 和 `gmtime` 返回指向静态内存的指针，多次调用会覆盖之前的值。可使用 `localtime_r`（POSIX）或复制结果。

> **要点二**：`tm_mon` 范围是 0-11（0=一月），`tm_year` 是从1900开始的年数。

> **要点三**：`clock()` 返回的是处理器时间，不是实际经过的时间。

> **要点四**：`time_t` 的具体类型和范围由实现定义，通常是自1970年1月1日以来的秒数。

> **要点五**：`mktime` 会自动规范化超出范围的 `tm` 成员值。

> **要点六**：`strftime` 缓冲区大小要足够大，否则返回0且内容未定义。

> **要点七**：`difftime` 返回 `double` 类型，即使 `time_t` 是整数类型。

> **要点八**：`CLOCK_PER_SEC` 通常等于1000000（微秒），但应使用此常量而非硬编码。
