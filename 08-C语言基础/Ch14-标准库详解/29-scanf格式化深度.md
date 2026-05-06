# scanf 格式化深度解析

## 1. 概述

`scanf` 系列函数用于格式化输入。其格式字符串语法与 `printf` 类似但有重要区别。本章深入解析 `scanf` 的所有格式化选项和行为细节。

## 2. 通用语法

```
%[*][width][length]specifier
```

- `*`：赋值抑制（读取但不存储）
- `width`：最大读取字符数
- `length`：长度修饰符
- `specifier`：格式说明符

## 3. 格式说明符

| 说明符 | 类型 | 说明 |
|--------|------|------|
| `%d` | int* | 十进制整数 |
| `%i` | int* | 自动进制整数 |
| `%u` | unsigned int* | 无符号十进制 |
| `%o` | unsigned int* | 八进制整数 |
| `%x`, `%X` | unsigned int* | 十六进制整数 |
| `%f` | float* | 浮点数 |
| `%e`, `%E`, `%g`, `%G` | float* | 浮点数 |
| `%lf` | double* | 双精度浮点 |
| `%Lf` | long double* | 扩展精度浮点 |
| `%c` | char* | 字符（不跳过空白） |
| `%s` | char* | 字符串（跳过空白） |
| `%[` | char* | 字符集扫描 |
| `%p` | void** | 指针 |
| `%n` | int* | 已读取字符数 |
| `%%` | - | 匹配百分号 |

## 4. 基本用法

```c
#include <stdio.h>

int main(void) {
    int age;
    char name[50];
    float score;

    printf("请输入: 姓名 年龄 分数\n");

    // scanf返回成功读取的项目数
    int items = scanf("%49s %d %f", name, &age, &score);

    printf("成功读取 %d 项\n", items);
    printf("姓名: %s, 年龄: %d, 分数: %.1f\n", name, age, score);

    return 0;
}
```

## 5. 空白处理

```c
#include <stdio.h>

int main(void) {
    // %c 不跳过空白字符
    char ch;
    printf("输入一个字符: ");
    scanf(" %c", &ch);  // 空格前缀跳过空白
    printf("字符: '%c'\n", ch);

    // %s 跳过前导空白，遇到空白停止
    char word[50];
    scanf("%49s", word);  // 只读取一个单词
    printf("单词: %s\n", word);

    // 格式串中的空白匹配任意数量的空白
    int a, b;
    scanf("%d %d", &a, &b);  // 两个%d之间的空格匹配任意空白
    printf("a=%d, b=%d\n", a, b);

    // 格式串中的非空白字符匹配具体字符
    int hour, minute;
    scanf("%d:%d", &hour, &minute);  // 匹配如 "12:30"
    printf("时间: %d:%d\n", hour, minute);

    return 0;
}
```

## 6. 赋值抑制符 *

```c
#include <stdio.h>

int main(void) {
    // * 跳过输入项，不存储
    int x, z;

    // 输入 "1 2 3"，跳过中间的2
    scanf("%d %*d %d", &x, &z);
    printf("x=%d, z=%d (跳过了中间值)\n", x, z);

    // 跳过日期中的年和月，只取日
    int day;
    scanf("%*d-%*d-%d", &day);
    printf("日: %d\n", day);

    // 跳过行首的空白
    char line[100];
    scanf("%*[ \t]%99[^\n]", line);
    printf("去除前导空白: \"%s\"\n", line);

    return 0;
}
```

## 7. 宽度限制

```c
#include <stdio.h>

int main(void) {
    // 限制读取的最大字符数
    char buf[10];

    // %9s 最多读取9个字符（留1个给'\0'）
    scanf("%9s", buf);
    printf("读取: %s\n", buf);

    // 限制数字的位数
    int num;
    scanf("%3d", &num);  // 最多读取3位数字
    printf("数字: %d\n", num);

    return 0;
}
```

## 8. 长度修饰符

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    // h: short
    short s;
    scanf("%hd", &s);

    // hh: signed char / unsigned char
    unsigned char uc;
    scanf("%hhu", &uc);

    // l: long（整数）或 double（浮点）
    long l;
    scanf("%ld", &l);

    double d;
    scanf("%lf", &d);

    // ll: long long
    long long ll;
    scanf("%lld", &ll);

    // j: intmax_t
    intmax_t imax;
    scanf("%jd", &imax);

    // z: size_t
    size_t sz;
    scanf("%zu", &sz);

    // t: ptrdiff_t
    ptrdiff_t pd;
    scanf("%td", &pd);

    // L: long double
    long double ld;
    scanf("%Lf", &ld);

    return 0;
}
```

## 9. 字符集扫描 %[]

```c
#include <stdio.h>
#include <string.h>

int main(void) {
    char buf[100];

    // [abc] 只匹配指定字符
    scanf("%[abc]", buf);
    printf("匹配abc: %s\n", buf);

    // [^abc] 匹配不在集合中的字符
    scanf("%[^abc]", buf);
    printf("匹配非abc: %s\n", buf);

    // 读取整行（包括空格）
    // %[^\n] 匹配除换行外的所有字符
    scanf(" %[^\n]", buf);
    printf("整行: %s\n", buf);

    // 读取到逗号停止
    scanf(" %[^,]", buf);
    printf("到逗号: %s\n", buf);

    // 数字字符集
    scanf("%[0-9]", buf);
    printf("数字: %s\n", buf);

    // 字母字符集
    scanf("%[a-zA-Z]", buf);
    printf("字母: %s\n", buf);

    // 读取引号内的字符串
    // 输入: "Hello World"
    scanf("\"%[^\"]\"", buf);
    printf("引号内: %s\n", buf);

    return 0;
}
```

### 解析CSV示例

```c
#include <stdio.h>

int main(void) {
    // 解析 "name,age,city" 格式
    char name[50], city[50];
    int age;

    char input[] = "张三,25,北京";
    sscanf(input, "%49[^,],%d,%49[^,\n]", name, &age, city);
    printf("姓名: %s, 年龄: %d, 城市: %s\n", name, age, city);

    // 解析日志行
    char log[] = "[2023-12-25 10:30:45] ERROR: Something went wrong";
    char date[20], time[10], level[10], message[100];
    sscanf(log, "[%19[^ ] %9[^]]] %9[^:]: %99[^\n]",
           date, time, level, message);
    printf("日期: %s\n时间: %s\n级别: %s\n消息: %s\n",
           date, time, level, message);

    return 0;
}
```

## 10. %i 与 %d 的区别

```c
#include <stdio.h>

int main(void) {
    int a, b, c;

    // %d 总是十进制
    // %i 根据前缀自动判断进制
    //   0开头 -> 八进制
    //   0x开头 -> 十六进制
    //   其他 -> 十进制

    sscanf("10", "%i", &a);   // 十进制 10
    sscanf("010", "%i", &b);  // 八进制 8
    sscanf("0x10", "%i", &c); // 十六进制 16

    printf("%%i: %d, %d, %d\n", a, b, c);  // 10, 8, 16

    sscanf("10", "%d", &a);   // 十进制 10
    sscanf("010", "%d", &b);  // 十进制 10（不识别八进制）
    sscanf("0x10", "%d", &c); // 0（遇到x停止）

    printf("%%d: %d, %d, %d\n", a, b, c);  // 10, 10, 0

    return 0;
}
```

## 11. %n - 已读取字符数

```c
#include <stdio.h>

int main(void) {
    int chars_read;

    // 记录已读取的字符数
    int val;
    sscanf("  42  ", "%d%n", &val, &chars_read);
    printf("读取了 %d, 消耗了 %d 个字符\n", val, chars_read);

    // 用于部分解析
    char remaining[100];
    int offset;
    sscanf("123abc", "%3d%n", &val, &offset);
    sscanf("123abc" + offset, "%s", remaining);
    printf("数字: %d, 剩余: %s\n", val, remaining);

    return 0;
}
```

## 12. 安全输入

```c
#include <stdio.h>
#include <string.h>

// 安全的行读取
int safe_read_line(char *buf, size_t size) {
    if (fgets(buf, size, stdin) == NULL) {
        return -1;
    }
    // 移除换行符
    size_t len = strlen(buf);
    if (len > 0 && buf[len - 1] == '\n') {
        buf[len - 1] = '\0';
    } else {
        // 清除剩余输入
        int ch;
        while ((ch = getchar()) != '\n' && ch != EOF);
    }
    return 0;
}

// 安全的整数读取
int safe_read_int(int *out) {
    char buf[32];
    if (safe_read_line(buf, sizeof(buf)) != 0) {
        return -1;
    }
    char *end;
    long val = strtol(buf, &end, 10);
    if (*end != '\0' || end == buf) {
        return -1;  // 不是有效整数
    }
    *out = (int)val;
    return 0;
}

int main(void) {
    int num;
    printf("请输入一个整数: ");
    if (safe_read_int(&num) == 0) {
        printf("你输入了: %d\n", num);
    } else {
        printf("输入无效\n");
    }

    return 0;
}
```

## 13. 重要注意事项

> **要点一**：`scanf` 的返回值是成功读取的项目数，应当检查以确认输入有效。

> **要点二**：`%c` 和 `%[` 不会跳过前导空白，其他说明符会。

> **要点三**：读取字符串时务必指定最大宽度（如 `%49s`），防止缓冲区溢出。

> **要点四**：`%d` 需要 `int*` 参数，`%f` 需要 `float*` 参数（不是 `double*`），`%lf` 才是 `double*`。

> **要点五**：`scanf` 在遇到不匹配的输入时会停止，剩余输入留在缓冲区。

> **要点六**：格式串中的普通字符（非空白）必须与输入完全匹配。

> **要点七**：`%[^\n]` 可以读取包含空格的整行，但不会读取换行符本身。

> **要点八**：在实际应用中，使用 `fgets` + `sscanf` 或 `strtol` 系列函数通常比直接使用 `scanf` 更安全可靠。
