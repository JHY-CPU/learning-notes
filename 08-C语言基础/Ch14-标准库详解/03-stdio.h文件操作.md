# stdio.h - 文件操作详解

## 1. 概述

C语言的文件操作通过 `<stdio.h>` 中的函数实现，采用"流"（stream）的抽象概念。文件操作分为文本模式和二进制模式两种，通过 `FILE` 结构体指针进行操作。

## 2. 文件打开与关闭

### 2.1 fopen - 打开文件

```c
#include <stdio.h>

FILE *fopen(const char *filename, const char *mode);
```

#### 打开模式

| 模式 | 说明 | 文件存在 | 文件不存在 |
|------|------|----------|------------|
| `"r"` | 只读（文本） | 打开 | 错误 |
| `"w"` | 只写（文本） | 清空 | 创建 |
| `"a"` | 追加（文本） | 打开末尾 | 创建 |
| `"r+"` | 读写（文本） | 打开 | 错误 |
| `"w+"` | 读写（文本） | 清空 | 创建 |
| `"a+"` | 读/追加（文本） | 打开末尾 | 创建 |
| `"rb"` | 只读（二进制） | 打开 | 错误 |
| `"wb"` | 只写（二进制） | 清空 | 创建 |
| `"ab"` | 追加（二进制） | 打开末尾 | 创建 |
| `"r+b"` / `"rb+"` | 读写（二进制） | 打开 | 错误 |
| `"w+b"` / `"wb+"` | 读写（二进制） | 清空 | 创建 |
| `"a+b"` / `"ab+"` | 读/追加（二进制） | 打开末尾 | 创建 |

### 2.2 基本文件操作示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <errno.h>
#include <string.h>

int main(void) {
    FILE *fp;

    // 1. 创建并写入文件
    fp = fopen("example.txt", "w");
    if (fp == NULL) {
        fprintf(stderr, "无法创建文件: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }
    fprintf(fp, "第一行内容\n");
    fprintf(fp, "第二行内容\n");
    fprintf(fp, "第三行: 数值 = %d\n", 42);
    fclose(fp);

    // 2. 读取文件
    fp = fopen("example.txt", "r");
    if (fp == NULL) {
        fprintf(stderr, "无法打开文件: %s\n", strerror(errno));
        return EXIT_FAILURE;
    }

    char line[256];
    int line_num = 0;
    while (fgets(line, sizeof(line), fp) != NULL) {
        line_num++;
        // 去除末尾换行符
        size_t len = strlen(line);
        if (len > 0 && line[len - 1] == '\n') {
            line[len - 1] = '\0';
        }
        printf("第%d行: %s\n", line_num, line);
    }

    // 检查是否因为到达文件末尾而结束
    if (feof(fp)) {
        printf("已到达文件末尾，共读取 %d 行\n", line_num);
    } else if (ferror(fp)) {
        perror("读取文件出错");
    }

    fclose(fp);
    return EXIT_SUCCESS;
}
```

## 3. 字符级文件操作

```c
#include <stdio.h>

int fgetc(FILE *stream);              // 读取一个字符
int ungetc(int c, FILE *stream);      // 将字符推回流
int fputc(int c, FILE *stream);       // 写入一个字符
```

### 字符操作示例

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    FILE *src = fopen("source.txt", "r");
    FILE *dst = fopen("destination.txt", "w");

    if (src == NULL || dst == NULL) {
        perror("打开文件失败");
        return 1;
    }

    int ch;
    int char_count = 0;
    int word_count = 0;
    int in_word = 0;

    // 逐字符复制并统计
    while ((ch = fgetc(src)) != EOF) {
        fputc(ch, dst);
        char_count++;

        if (isspace(ch)) {
            in_word = 0;
        } else if (!in_word) {
            in_word = 1;
            word_count++;
        }
    }

    printf("复制完成: %d 个字符, %d 个单词\n", char_count, word_count);

    fclose(src);
    fclose(dst);
    return 0;
}
```

## 4. 行级文件操作

```c
#include <stdio.h>

char *fgets(char *s, int size, FILE *stream);   // 读取一行
int fputs(const char *s, FILE *stream);          // 写入字符串
```

### 行操作示例

```c
#include <stdio.h>
#include <string.h>
#include <stdlib.h>

#define MAX_LINE 1024

int main(void) {
    FILE *fp = fopen("lines.txt", "w+");
    if (fp == NULL) return 1;

    // 写入多行
    const char *lines[] = {
        "C语言编程",
        "标准库详解",
        "文件操作",
        NULL
    };

    for (int i = 0; lines[i] != NULL; i++) {
        fputs(lines[i], fp);
        fputc('\n', fp);
    }

    // 回到开头读取
    rewind(fp);

    char buffer[MAX_LINE];
    printf("文件内容:\n");
    while (fgets(buffer, MAX_LINE, fp) != NULL) {
        // 去除换行符
        buffer[strcspn(buffer, "\n")] = '\0';
        printf("  > %s\n", buffer);
    }

    fclose(fp);
    return 0;
}
```

## 5. 块读写操作

```c
#include <stdio.h>

size_t fread(void *ptr, size_t size, size_t nmemb, FILE *stream);
size_t fwrite(const void *ptr, size_t size, size_t nmemb, FILE *stream);
```

### 二进制读写示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 定义一个结构体
typedef struct {
    int id;
    char name[32];
    float score;
} Student;

int main(void) {
    // 准备数据
    Student students[] = {
        {1, "张三", 95.5f},
        {2, "李四", 88.0f},
        {3, "王五", 92.3f},
    };
    int count = sizeof(students) / sizeof(students[0]);

    // 写入二进制文件
    FILE *fp = fopen("students.dat", "wb");
    if (fp == NULL) {
        perror("无法创建文件");
        return 1;
    }

    // 先写入记录数
    fwrite(&count, sizeof(int), 1, fp);
    // 写入所有学生数据
    fwrite(students, sizeof(Student), count, fp);
    fclose(fp);
    printf("已写入 %d 条学生记录\n", count);

    // 读取二进制文件
    fp = fopen("students.dat", "rb");
    if (fp == NULL) {
        perror("无法打开文件");
        return 1;
    }

    int read_count;
    fread(&read_count, sizeof(int), 1, fp);

    Student *read_students = malloc(read_count * sizeof(Student));
    if (read_students == NULL) {
        fclose(fp);
        return 1;
    }

    fread(read_students, sizeof(Student), read_count, fp);
    fclose(fp);

    // 显示读取的数据
    printf("\n读取的学生记录:\n");
    for (int i = 0; i < read_count; i++) {
        printf("  ID: %d, 姓名: %s, 分数: %.1f\n",
               read_students[i].id,
               read_students[i].name,
               read_students[i].score);
    }

    free(read_students);
    return 0;
}
```

## 6. 文件定位

```c
#include <stdio.h>

int fseek(FILE *stream, long offset, int whence);
long ftell(FILE *stream);
void rewind(FILE *stream);
int fgetpos(FILE *stream, fpos_t *pos);
int fsetpos(FILE *stream, const fpos_t *pos);
```

### 定位参数

| whence 值 | 含义 |
|-----------|------|
| `SEEK_SET` | 从文件开头 |
| `SEEK_CUR` | 从当前位置 |
| `SEEK_END` | 从文件末尾 |

### 文件定位示例

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("data.bin", "w+b");
    if (fp == NULL) return 1;

    // 写入一些数据
    int data[] = {10, 20, 30, 40, 50};
    fwrite(data, sizeof(int), 5, fp);

    // 获取文件大小
    fseek(fp, 0, SEEK_END);
    long size = ftell(fp);
    printf("文件大小: %ld 字节\n", size);

    // 随机访问第3个元素（索引2）
    fseek(fp, 2 * sizeof(int), SEEK_SET);
    int value;
    fread(&value, sizeof(int), 1, fp);
    printf("第3个元素: %d\n", value);  // 30

    // 修改第3个元素
    fseek(fp, 2 * sizeof(int), SEEK_SET);
    int new_value = 999;
    fwrite(&new_value, sizeof(int), 1, fp);

    // 验证修改
    rewind(fp);
    for (int i = 0; i < 5; i++) {
        fread(&value, sizeof(int), 1, fp);
        printf("data[%d] = %d\n", i, value);
    }

    fclose(fp);
    return 0;
}
```

## 7. 文件删除与重命名

```c
#include <stdio.h>

int remove(const char *filename);   // 删除文件
int rename(const char *old, const char *new);  // 重命名文件
```

```c
#include <stdio.h>

int main(void) {
    // 重命名文件
    if (rename("old_name.txt", "new_name.txt") != 0) {
        perror("重命名失败");
    }

    // 删除文件
    if (remove("temp.txt") != 0) {
        perror("删除失败");
    }

    return 0;
}
```

## 8. 临时文件

```c
#include <stdio.h>

FILE *tmpfile(void);                          // 创建临时文件
char *tmpnam(char *s);                        // 生成临时文件名
```

```c
#include <stdio.h>

int main(void) {
    // tmpfile 创建的临时文件在关闭时自动删除
    FILE *tmp = tmpfile();
    if (tmp == NULL) {
        perror("创建临时文件失败");
        return 1;
    }

    fprintf(tmp, "临时数据\n");
    rewind(tmp);

    char buf[100];
    fgets(buf, sizeof(buf), tmp);
    printf("读取: %s", buf);

    fclose(tmp);  // 文件自动删除

    return 0;
}
```

## 9. 文件状态检测

```c
#include <stdio.h>

int feof(FILE *stream);    // 检查是否到达文件末尾
int ferror(FILE *stream);  // 检查是否发生错误
void clearerr(FILE *stream); // 清除错误和EOF标志
int fflush(FILE *stream);  // 刷新缓冲区
```

```c
#include <stdio.h>

int main(void) {
    FILE *fp = fopen("test.txt", "r");
    if (fp == NULL) return 1;

    // 读取到EOF
    char buf[100];
    while (fgets(buf, sizeof(buf), fp) != NULL) {
        // 正常处理
    }

    // 确定结束原因
    if (feof(fp)) {
        printf("正常到达文件末尾\n");
    }
    if (ferror(fp)) {
        printf("读取过程中发生错误\n");
        clearerr(fp);  // 清除错误状态
    }

    fclose(fp);
    return 0;
}
```

## 10. 文件操作最佳实践

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>

// 安全的文件复制函数
int safe_copy(const char *src_path, const char *dst_path) {
    FILE *src = fopen(src_path, "rb");
    if (src == NULL) {
        fprintf(stderr, "无法打开源文件 '%s': %s\n",
                src_path, strerror(errno));
        return -1;
    }

    FILE *dst = fopen(dst_path, "wb");
    if (dst == NULL) {
        fprintf(stderr, "无法创建目标文件 '%s': %s\n",
                dst_path, strerror(errno));
        fclose(src);
        return -1;
    }

    char buffer[4096];
    size_t bytes;
    while ((bytes = fread(buffer, 1, sizeof(buffer), src)) > 0) {
        if (fwrite(buffer, 1, bytes, dst) != bytes) {
            fprintf(stderr, "写入失败: %s\n", strerror(errno));
            fclose(src);
            fclose(dst);
            return -1;
        }
    }

    if (ferror(src)) {
        fprintf(stderr, "读取失败: %s\n", strerror(errno));
        fclose(src);
        fclose(dst);
        return -1;
    }

    fclose(src);
    fclose(dst);
    return 0;
}

int main(void) {
    if (safe_copy("source.dat", "backup.dat") == 0) {
        printf("文件复制成功\n");
    }
    return 0;
}
```

## 11. 重要注意事项

> **要点一**：二进制模式和文本模式在Windows上有区别（换行符转换），在Linux上无区别。

> **要点二**：`fread` 和 `fwrite` 返回实际读写成功的元素个数，应与请求的个数比较。

> **要点三**：`fseek` 对二进制文件支持 `SEEK_END`，对文本文件行为未定义。

> **要点四**：使用 `fopen` 后务必检查返回值是否为 `NULL`。

> **要点五**：`fflush` 用于输出流时刷新缓冲区，用于输入流的行为是未定义的。

> **要点六**：操作完成后必须调用 `fclose`，否则可能丢失数据。

> **要点七**：不要同时以文本模式和二进制模式打开同一文件。
