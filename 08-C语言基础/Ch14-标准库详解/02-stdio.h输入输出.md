# stdio.h - 标准输入输出

## 1. 概述

`<stdio.h>` 是C标准库中最常用的头文件之一，提供了格式化输入输出、字符输入输出和字符串输入输出的功能。所有I/O操作都围绕"流"（stream）的概念进行。

## 2. 标准流

C语言定义了三个标准流，在程序启动时自动打开：

| 流名称 | 类型 | 说明 |
|--------|------|------|
| `stdin` | 标准输入 | 默认连接键盘 |
| `stdout` | 标准输出 | 默认连接终端 |
| `stderr` | 标准错误 | 默认连接终端，无缓冲 |

```c
#include <stdio.h>

int main(void) {
    // 三个标准流无需手动打开
    fprintf(stdout, "这是标准输出\n");
    fprintf(stderr, "这是标准错误输出\n");

    char buf[100];
    fprintf(stdout, "请输入: ");
    fgets(buf, sizeof(buf), stdin);
    fprintf(stdout, "你输入了: %s", buf);

    return 0;
}
```

## 3. 格式化输出函数

### 3.1 printf 系列

```c
#include <stdio.h>

int printf(const char *format, ...);                          // 输出到stdout
int fprintf(FILE *stream, const char *format, ...);           // 输出到指定流
int sprintf(char *str, const char *format, ...);              // 输出到字符串（不安全）
int snprintf(char *str, size_t size, const char *format, ...);// 输出到字符串（安全）
```

#### 基本用法

```c
#include <stdio.h>

int main(void) {
    // printf - 输出到标准输出
    printf("Hello, %s! 年龄: %d\n", "张三", 25);

    // fprintf - 输出到文件
    FILE *fp = fopen("output.txt", "w");
    if (fp != NULL) {
        fprintf(fp, "写入文件: %f\n", 3.14159);
        fclose(fp);
    }

    // snprintf - 安全地输出到字符串
    char buffer[50];
    int written = snprintf(buffer, sizeof(buffer), "值: %d", 42);
    printf("buffer = \"%s\", 写入了 %d 个字符\n", buffer, written);

    // sprintf - 不推荐使用，存在缓冲区溢出风险
    char unsafe[10];
    // sprintf(unsafe, "这是一个很长的字符串"); // 危险！

    return 0;
}
```

### 3.2 格式说明符

| 说明符 | 类型 | 说明 |
|--------|------|------|
| `%d` / `%i` | int | 有符号十进制整数 |
| `%u` | unsigned int | 无符号十进制整数 |
| `%o` | unsigned int | 八进制整数 |
| `%x` / `%X` | unsigned int | 十六进制整数 |
| `%f` | double | 十进制浮点数 |
| `%e` / `%E` | double | 科学计数法 |
| `%g` / `%G` | double | 自动选择%f或%e |
| `%c` | int | 单个字符 |
| `%s` | char* | 字符串 |
| `%p` | void* | 指针地址 |
| `%n` | int* | 已输出字符数 |
| `%%` | - | 百分号本身 |

#### 格式控制

```c
#include <stdio.h>

int main(void) {
    // 宽度与对齐
    printf("|%10d|\n", 42);       // |        42| 右对齐
    printf("|%-10d|\n", 42);      // |42        | 左对齐
    printf("|%010d|\n", 42);      // |0000000042| 零填充

    // 精度控制
    printf("%.2f\n", 3.14159);    // 3.14
    printf("%10.2f\n", 3.14159);  //       3.14
    printf("%.5s\n", "HelloWorld"); // Hello

    // 前缀标记
    printf("%+d\n", 42);          // +42
    printf("%+d\n", -42);         // -42
    printf("% d\n", 42);          //  42 (正数前加空格)
    printf("%#x\n", 255);         // 0xff (带前缀)
    printf("%#o\n", 255);         // 0377

    // 长度修饰符
    printf("%ld\n", 123456789L);  // long
    printf("%lld\n", 123456789LL);// long long
    printf("%hd\n", (short)42);   // short
    printf("%zu\n", sizeof(int)); // size_t

    return 0;
}
```

## 4. 格式化输入函数

### 4.1 scanf 系列

```c
#include <stdio.h>

int scanf(const char *format, ...);                          // 从stdin读取
int fscanf(FILE *stream, const char *format, ...);           // 从文件读取
int sscanf(const char *str, const char *format, ...);        // 从字符串读取
```

#### 基本用法

```c
#include <stdio.h>

int main(void) {
    int age;
    char name[50];
    float score;

    // scanf - 从标准输入读取
    printf("请输入姓名、年龄、分数: ");
    scanf("%49s %d %f", name, &age, &score);
    printf("姓名: %s, 年龄: %d, 分数: %.1f\n", name, age, score);

    // sscanf - 从字符串解析
    char data[] = "2023-12-25";
    int year, month, day;
    sscanf(data, "%d-%d-%d", &year, &month, &day);
    printf("日期: %d年%d月%d日\n", year, month, day);

    // fscanf - 从文件读取
    FILE *fp = fopen("data.txt", "r");
    if (fp != NULL) {
        int value;
        while (fscanf(fp, "%d", &value) == 1) {
            printf("读取: %d\n", value);
        }
        fclose(fp);
    }

    return 0;
}
```

## 5. 字符输入输出

```c
#include <stdio.h>

int main(void) {
    // 单字符输出
    putchar('A');           // 输出到stdout
    fputc('B', stdout);     // 输出到指定流

    // 单字符输入
    printf("\n请输入一个字符: ");
    int ch = getchar();     // 从stdin读取
    printf("你输入了: %c (ASCII: %d)\n", ch, ch);

    // ungetc - 将字符推回输入流
    ungetc(ch, stdin);
    int ch2 = getchar();
    printf("推回后再次读取: %c\n", ch2);

    return 0;
}
```

## 6. 字符串输入输出

```c
#include <stdio.h>

int main(void) {
    // puts - 输出字符串到stdout，自动添加换行
    puts("Hello, World!");

    // fputs - 输出字符串到指定流，不自动添加换行
    fputs("Hello", stdout);
    fputs(" World\n", stdout);

    // fgets - 安全地读取字符串
    char buffer[100];
    printf("请输入一行文本: ");
    if (fgets(buffer, sizeof(buffer), stdin) != NULL) {
        // 移除末尾的换行符
        size_t len = strlen(buffer);
        if (len > 0 && buffer[len - 1] == '\n') {
            buffer[len - 1] = '\0';
        }
        printf("你输入了: %s\n", buffer);
    }

    // 注意：gets() 已在C11中被移除，不要使用！

    return 0;
}
```

## 7. 缓冲控制

```c
#include <stdio.h>

int main(void) {
    // 设置缓冲模式
    char buf[BUFSIZ];

    // 行缓冲 - 遇到换行时刷新
    setvbuf(stdout, buf, _IOLBF, sizeof(buf));

    printf("这行会行缓冲");
    fflush(stdout);  // 手动刷新缓冲区

    // 无缓冲
    setvbuf(stderr, NULL, _IONBF, 0);

    // 全缓冲
    FILE *fp = fopen("test.txt", "w");
    setvbuf(fp, NULL, _IOFBF, 4096);

    fclose(fp);
    return 0;
}
```

## 8. 临时文件

```c
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    // 创建临时文件
    FILE *tmp = tmpfile();
    if (tmp == NULL) {
        perror("创建临时文件失败");
        return 1;
    }

    fprintf(tmp, "临时数据: %d\n", 42);
    rewind(tmp);  // 回到文件开头

    char buffer[100];
    if (fgets(buffer, sizeof(buffer), tmp) != NULL) {
        printf("从临时文件读取: %s", buffer);
    }

    fclose(tmp);  // 关闭时自动删除

    // 创建临时文件名
    char name[L_tmpnam];
    tmpnam(name);
    printf("临时文件名: %s\n", name);

    return 0;
}
```

## 9. 文件定位

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("test.txt", "w+");
    if (fp == NULL) return 1;

    fprintf(fp, "Hello, World!");
    fflush(fp);

    // 获取当前位置
    long pos = ftell(fp);
    printf("当前位置: %ld\n", pos);

    // 回到文件开头
    rewind(fp);

    // 定位到指定位置
    fseek(fp, 7, SEEK_SET);  // 从开头偏移7字节

    char buffer[20];
    fgets(buffer, sizeof(buffer), fp);
    printf("从位置7读取: %s\n", buffer);  // "World!"

    // 使用fgetpos/fsetpos处理大文件
    fpos_t position;
    fgetpos(fp, &position);
    fsetpos(fp, &position);

    fclose(fp);
    return 0;
}
```

## 10. 错误处理

```c
#include <stdio.h>
#include <errno.h>

int main(void) {
    FILE *fp = fopen("不存在的文件.txt", "r");
    if (fp == NULL) {
        perror("fopen失败");  // 打印错误信息
        printf("错误码: %d\n", ferror(stdin));
        return 1;
    }

    // 检查文件错误状态
    if (ferror(fp)) {
        printf("文件操作出错\n");
        clearerr(fp);  // 清除错误标志
    }

    // 检查文件结束
    if (feof(fp)) {
        printf("已到达文件末尾\n");
    }

    fclose(fp);
    return 0;
}
```

## 11. 重要注意事项

> **要点一**：始终使用 `snprintf` 代替 `sprintf`，避免缓冲区溢出。

> **要点二**：使用 `fgets` 代替已废弃的 `gets`，`gets` 在C11中已被移除。

> **要点三**：`scanf` 读取字符串时务必指定最大宽度（如 `%49s`），防止缓冲区溢出。

> **要点四**：`scanf` 的返回值是成功读取的项目数，应当检查以确认输入有效。

> **要点五**：`stdout` 通常是行缓冲的，在关键输出后使用 `fflush(stdout)` 确保数据写出。

> **要点六**：`printf` 返回值是输出的字符数（不含终止符），出错时返回负值。

> **要点七**：使用 `%n` 格式说明符可以获取已输出的字符数，但要注意安全风险。

> **要点八**：`fopen` 失败时返回 `NULL`，应当始终检查返回值。
