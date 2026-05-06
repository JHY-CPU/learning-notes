# string.h - 字符串与内存操作

## 1. 概述

`<string.h>` 提供了字符串操作和内存操作函数。C语言中的字符串是以 `\0` 结尾的字符数组，所有字符串函数都依赖于这个约定。

## 2. 字符串长度

```c
#include <string.h>

size_t strlen(const char *s);
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *str = "Hello, World!";
    size_t len = strlen(str);
    printf("\"%s\" 的长度: %zu\n", str, len);  // 13

    // strlen 不包含终止符 '\0'
    char buf[50] = "测试";
    printf("长度: %zu\n", strlen(buf));  // 2 (中文每个字符通常占3字节UTF-8)

    // 注意：strlen 需要遍历整个字符串，O(n) 时间复杂度
    // 对于很长的字符串，频繁调用可能影响性能

    return 0;
}
```

## 3. 字符串复制

```c
#include <string.h>

char *strcpy(char *dest, const char *src);        // 复制字符串
char *strncpy(char *dest, const char *src, size_t n);  // 复制最多n个字符
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char dest[50];

    // strcpy - 完整复制（不安全，不检查目标大小）
    strcpy(dest, "Hello, World!");
    printf("strcpy: %s\n", dest);

    // strncpy - 安全复制
    // 注意：如果src长度>=n，dest不会以'\0'结尾！
    char small[5];
    strncpy(small, "Hello, World!", sizeof(small) - 1);
    small[sizeof(small) - 1] = '\0';  // 手动添加终止符
    printf("strncpy: %s\n", small);

    // 推荐的安全复制方式
    char safe[20];
    size_t n = sizeof(safe) - 1;
    strncpy(safe, "Hello, World!", n);
    safe[n] = '\0';
    printf("安全复制: %s\n", safe);

    return 0;
}
```

## 4. 字符串拼接

```c
#include <string.h>

char *strcat(char *dest, const char *src);        // 拼接字符串
char *strncat(char *dest, const char *src, size_t n);  // 拼接最多n个字符
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char greeting[100] = "你好";

    // strcat - 在dest末尾追加src
    strcat(greeting, ", ");
    strcat(greeting, "世界!");
    printf("拼接结果: %s\n", greeting);

    // strncat - 限制拼接长度
    char path[50] = "/home/user";
    strncat(path, "/documents/file.txt", sizeof(path) - strlen(path) - 1);
    printf("路径: %s\n", path);

    // strncat 总是添加 '\0'，比 strncpy 更安全

    return 0;
}
```

## 5. 字符串比较

```c
#include <string.h>

int strcmp(const char *s1, const char *s2);           // 比较字符串
int strncmp(const char *s1, const char *s2, size_t n);  // 比较最多n个字符
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    // strcmp 返回值:
    // < 0: s1 < s2
    // = 0: s1 == s2
    // > 0: s1 > s2

    printf("strcmp结果:\n");
    printf("  \"abc\" vs \"abc\": %d\n", strcmp("abc", "abc"));     // 0
    printf("  \"abc\" vs \"abd\": %d\n", strcmp("abc", "abd"));     // < 0
    printf("  \"abd\" vs \"abc\": %d\n", strcmp("abd", "abc"));     // > 0
    printf("  \"abc\" vs \"abcd\": %d\n", strcmp("abc", "abcd"));   // < 0

    // 实际应用：字符串排序
    const char *names[] = {"Charlie", "Alice", "Bob", "David"};
    int n = 4;

    // 简单冒泡排序
    for (int i = 0; i < n - 1; i++) {
        for (int j = 0; j < n - i - 1; j++) {
            if (strcmp(names[j], names[j + 1]) > 0) {
                const char *temp = names[j];
                names[j] = names[j + 1];
                names[j + 1] = temp;
            }
        }
    }

    printf("排序后: ");
    for (int i = 0; i < n; i++) {
        printf("%s ", names[i]);
    }
    printf("\n");

    // strncmp - 前缀比较
    const char *url = "https://example.com";
    if (strncmp(url, "https://", 8) == 0) {
        printf("这是一个HTTPS链接\n");
    }

    return 0;
}
```

## 6. 字符串搜索

```c
#include <string.h>

char *strchr(const char *s, int c);            // 查找字符首次出现
char *strrchr(const char *s, int c);           // 查找字符最后出现
size_t strcspn(const char *s, const char *reject);  // 查找首个reject中的字符
size_t strspn(const char *s, const char *accept);   // 查找首个不在accept中的字符
char *strpbrk(const char *s, const char *accept);   // 查找accept中任意字符
char *strstr(const char *haystack, const char *needle);  // 查找子串
char *strtok(char *str, const char *delim);    // 字符串分割
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    const char *text = "Hello, World! Hello, C!";

    // strchr - 查找字符首次出现
    char *pos = strchr(text, 'W');
    if (pos != NULL) {
        printf("首次出现'W'的位置: %ld, 子串: %s\n",
               pos - text, pos);
    }

    // strrchr - 查找字符最后出现
    pos = strrchr(text, 'l');
    if (pos != NULL) {
        printf("最后出现'l'的位置: %ld\n", pos - text);
    }

    // strstr - 查找子串
    pos = strstr(text, "World");
    if (pos != NULL) {
        printf("找到\"World\"在位置: %ld\n", pos - text);
    }

    // strpbrk - 查找字符集合中任意字符
    pos = strpbrk(text, "!?");
    if (pos != NULL) {
        printf("首个标点符号: '%c'\n", *pos);
    }

    // strcspn - 返回不在reject中的前缀长度
    const char *data = "12345abcde";
    size_t len = strcspn(data, "abcde");
    printf("数字前缀长度: %zu, 前缀: %.*s\n", len, (int)len, data);

    return 0;
}
```

### strtok 字符串分割

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char csv[] = "张三,25,北京,程序员";

    // strtok 会修改原字符串，用 '\0' 替换分隔符
    char *token = strtok(csv, ",");
    int field = 0;

    printf("CSV字段解析:\n");
    while (token != NULL) {
        printf("  字段%d: %s\n", field++, token);
        // 后续调用传入NULL
        token = strtok(NULL, ",");
    }

    // 注意：strtok 不是线程安全的
    // 线程安全版本: strtok_r (POSIX)

    // 多分隔符
    char cmd[] = "ls  -la  /home/user";
    token = strtok(cmd, " \t");  // 空格和制表符
    while (token != NULL) {
        printf("  \"%s\"\n", token);
        token = strtok(NULL, " \t");
    }

    return 0;
}
```

## 7. 内存操作函数

```c
#include <string.h>

void *memcpy(void *dest, const void *src, size_t n);     // 复制内存
void *memmove(void *dest, const void *src, size_t n);    // 移动内存（可重叠）
void *memset(void *s, int c, size_t n);                  // 设置内存
int memcmp(const void *s1, const void *s2, size_t n);    // 比较内存
void *memchr(const void *s, int c, size_t n);            // 在内存中查找
```

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    // memcpy - 复制内存（不处理重叠区域）
    int src[] = {1, 2, 3, 4, 5};
    int dst[5];
    memcpy(dst, src, sizeof(src));
    printf("memcpy结果: ");
    for (int i = 0; i < 5; i++) printf("%d ", dst[i]);
    printf("\n");

    // memmove - 安全的内存复制（可处理重叠区域）
    int arr[] = {1, 2, 3, 4, 5};
    memmove(arr + 1, arr, 4 * sizeof(int));  // 向右移动1位
    printf("memmove结果: ");
    for (int i = 0; i < 5; i++) printf("%d ", arr[i]);
    printf("\n");

    // memset - 设置内存（按字节）
    char buffer[100];
    memset(buffer, 0, sizeof(buffer));  // 清零
    memset(buffer, 'A', 10);            // 前10字节设为'A'
    buffer[10] = '\0';
    printf("memset结果: %s\n", buffer);

    // 常见用法：清零结构体
    struct { int x; int y; } point;
    memset(&point, 0, sizeof(point));

    // memcmp - 比较内存
    int a[] = {1, 2, 3};
    int b[] = {1, 2, 4};
    int cmp = memcmp(a, b, sizeof(a));
    printf("memcmp: %s\n", cmp < 0 ? "a < b" : cmp > 0 ? "a > b" : "a == b");

    // memchr - 在内存中查找字节
    char data[] = "Hello, World!";
    char *found = memchr(data, 'W', sizeof(data));
    if (found != NULL) {
        printf("找到'W'在位置: %ld\n", found - data);
    }

    return 0;
}
```

### memcpy vs memmove

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    // 当源和目标内存区域重叠时：
    char str[] = "Hello, World!";

    // memcpy: 行为未定义（如果区域重叠）
    // memmove: 保证正确处理重叠

    // 安全的做法：始终使用 memmove 处理可能重叠的内存
    memmove(str, str + 7, 5);  // 将"World"移到开头
    str[5] = '\0';
    printf("memmove结果: %s\n", str);  // "World"

    return 0;
}
```

## 8. 错误信息

```c
#include <string.h>

char *strerror(int errnum);  // 将错误码转换为可读字符串
```

```c
#include <stdio.h>
#include <string.h>
#include <errno.h>

int main(void) {
    // 查看各种错误码的含义
    int errors[] = {0, 1, 2, 13, 22, 36};
    const char *names[] = {"成功", "EPERM", "ENOENT", "EACCES", "EINVAL", "ENAMETOOLONG"};

    for (int i = 0; i < 6; i++) {
        printf("错误码 %d (%s): %s\n",
               errors[i], names[i], strerror(errors[i]));
    }

    // 实际使用场景
    FILE *fp = fopen("不存在.txt", "r");
    if (fp == NULL) {
        printf("错误: %s\n", strerror(errno));
        // 或使用 perror("fopen");
    }

    return 0;
}
```

## 9. 字符串操作的安全版本（C11 Annex K）

```c
// 以下函数属于C11可选附件K，并非所有编译器支持
errno_t strcpy_s(char *dest, rsize_t destsz, const char *src);
errno_t strncpy_s(char *dest, rsize_t destsz, const char *src, rsize_t count);
errno_t strcat_s(char *dest, rsize_t destsz, const char *src);
errno_t strncat_s(char *dest, rsize_t destsz, const char *src, rsize_t count);
```

## 10. 重要注意事项

> **要点一**：C语言字符串以 `\0` 结尾，忘记终止符会导致未定义行为。

> **要点二**：`strncpy` 不保证添加 `\0`，当源字符串长度 >= n 时。

> **要点三**：`strtok` 会修改原字符串，且不是线程安全的。

> **要点四**：`memcpy` 不处理重叠内存，`memmove` 可以处理。

> **要点五**：`memset` 按字节设置，用来清零浮点数组可能有问题（虽然通常可以工作）。

> **要点六**：字符串函数不检查目标缓冲区大小，程序员需确保目标有足够空间。

> **要点七**：`strcmp` 的返回值具体值因实现而异，只应检查正负和零。

> **要点八**：对字符串常量使用字符串修改函数会导致未定义行为。
