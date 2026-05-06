# ctype.h - 字符分类与转换

## 1. 概述

`<ctype.h>` 提供了字符分类和字符转换函数。这些函数接受一个 `int` 参数（表示 `unsigned char` 值或 `EOF`），返回非零值表示真，0表示假。

## 2. 字符分类函数

### 2.1 函数一览

| 函数 | 功能 | 匹配字符 |
|------|------|----------|
| `isalnum` | 字母或数字 | `[a-zA-Z0-9]` |
| `isalpha` | 字母 | `[a-zA-Z]` |
| `isblank` | 空白字符（C99） | 空格、`\t` |
| `iscntrl` | 控制字符 | ASCII 0-31, 127 |
| `isdigit` | 十进制数字 | `[0-9]` |
| `isgraph` | 可显示字符（除空格） | ASCII 33-126 |
| `islower` | 小写字母 | `[a-z]` |
| `isprint` | 可显示字符（含空格） | ASCII 32-126 |
| `ispunct` | 标点符号 | 非字母数字的可显示字符 |
| `isspace` | 空白字符 | 空格, `\t`, `\n`, `\v`, `\f`, `\r` |
| `isupper` | 大写字母 | `[A-Z]` |
| `isxdigit` | 十六进制数字 | `[0-9a-fA-F]` |

### 2.2 基本用法

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    char test_chars[] = {'A', 'z', '5', ' ', '\n', '@', '\t', 'G'};
    int n = sizeof(test_chars) / sizeof(test_chars[0]);

    for (int i = 0; i < n; i++) {
        char c = test_chars[i];
        printf("字符 '%c' (ASCII %3d):\n", c, c);
        printf("  isalpha:  %s\n", isalpha(c)  ? "是" : "否");
        printf("  isdigit:  %s\n", isdigit(c)  ? "是" : "否");
        printf("  isalnum:  %s\n", isalnum(c)  ? "是" : "否");
        printf("  isspace:  %s\n", isspace(c)  ? "是" : "否");
        printf("  isupper:  %s\n", isupper(c)  ? "是" : "否");
        printf("  islower:  %s\n", islower(c)  ? "是" : "否");
        printf("  isxdigit: %s\n", isxdigit(c) ? "是" : "否");
        printf("  ispunct:  %s\n", ispunct(c)  ? "是" : "否");
        printf("  iscntrl:  %s\n", iscntrl(c)  ? "是" : "否");
        printf("  isprint:  %s\n", isprint(c)  ? "是" : "否");
        printf("  isgraph:  %s\n", isgraph(c)  ? "是" : "否");
        printf("\n");
    }

    return 0;
}
```

### 2.3 实用示例

```c
#include <stdio.h>
#include <ctype.h>
#include <string.h>

// 统计字符串中各类字符的数量
void count_characters(const char *str) {
    int letters = 0, digits = 0, spaces = 0, puncts = 0, others = 0;

    for (int i = 0; str[i] != '\0'; i++) {
        if (isalpha(str[i]))       letters++;
        else if (isdigit(str[i]))  digits++;
        else if (isspace(str[i]))  spaces++;
        else if (ispunct(str[i]))  puncts++;
        else                       others++;
    }

    printf("文本: \"%s\"\n", str);
    printf("  字母: %d, 数字: %d, 空白: %d, 标点: %d, 其他: %d\n",
           letters, digits, spaces, puncts, others);
}

// 验证密码强度
int check_password(const char *pwd) {
    int has_upper = 0, has_lower = 0, has_digit = 0, has_punct = 0;
    size_t len = strlen(pwd);

    if (len < 8) return 0;  // 太短

    for (size_t i = 0; i < len; i++) {
        if (isupper(pwd[i]))  has_upper = 1;
        if (islower(pwd[i]))  has_lower = 1;
        if (isdigit(pwd[i]))  has_digit = 1;
        if (ispunct(pwd[i]))  has_punct = 1;
    }

    // 需要至少3种字符类型
    return (has_upper + has_lower + has_digit + has_punct) >= 3;
}

// 简单的词法分析：提取标识符和数字
void tokenize(const char *code) {
    printf("词法分析: \"%s\"\n", code);
    int i = 0;
    while (code[i] != '\0') {
        if (isspace(code[i])) {
            i++;
            continue;
        }

        if (isalpha(code[i]) || code[i] == '_') {
            // 标识符
            printf("  标识符: ");
            while (isalnum(code[i]) || code[i] == '_') {
                putchar(code[i++]);
            }
            printf("\n");
        } else if (isdigit(code[i])) {
            // 数字
            printf("  数字: ");
            while (isdigit(code[i]) || code[i] == '.') {
                putchar(code[i++]);
            }
            printf("\n");
        } else {
            // 其他符号
            printf("  符号: %c\n", code[i++]);
        }
    }
}

int main(void) {
    count_characters("Hello, World! 123");
    count_characters("C语言编程 test@2023");

    printf("\n密码检查:\n");
    printf("  \"abc123\": %s\n", check_password("abc123") ? "合格" : "不合格");
    printf("  \"Abc123!@\": %s\n", check_password("Abc123!@") ? "合格" : "不合格");

    printf("\n");
    tokenize("int x = 42; float y = 3.14;");

    return 0;
}
```

## 3. 字符转换函数

```c
#include <ctype.h>

int tolower(int c);  // 转为小写
int toupper(int c);  // 转为大写
```

```c
#include <stdio.h>
#include <ctype.h>
#include <string.h>

// 安全的大小写转换（只转换字母，其他字符不变）
void to_upper_string(char *str) {
    for (int i = 0; str[i] != '\0'; i++) {
        str[i] = toupper((unsigned char)str[i]);
    }
}

void to_lower_string(char *str) {
    for (int i = 0; str[i] != '\0'; i++) {
        str[i] = tolower((unsigned char)str[i]);
    }
}

// 不区分大小写的字符串比较
int strcasecmp_custom(const char *s1, const char *s2) {
    while (*s1 && *s2) {
        int c1 = tolower((unsigned char)*s1);
        int c2 = tolower((unsigned char)*s2);
        if (c1 != c2) return c1 - c2;
        s1++;
        s2++;
    }
    return (unsigned char)*s1 - (unsigned char)*s2;
}

int main(void) {
    char text[] = "Hello, World! 123";

    printf("原始: %s\n", text);
    to_upper_string(text);
    printf("大写: %s\n", text);
    to_lower_string(text);
    printf("小写: %s\n", text);

    // 不区分大小写比较
    printf("\n不区分大小写比较:\n");
    printf("  \"Hello\" vs \"hello\": %d\n",
           strcasecmp_custom("Hello", "hello"));  // 0
    printf("  \"abc\" vs \"ABD\": %d\n",
           strcasecmp_custom("abc", "ABD"));      // < 0

    // 单字符转换
    printf("\n单字符转换:\n");
    printf("  toupper('a') = '%c'\n", toupper('a'));  // 'A'
    printf("  tolower('Z') = '%c'\n", tolower('Z'));  // 'z'
    printf("  toupper('5') = '%c'\n", toupper('5'));  // '5'（非字母不变）

    return 0;
}
```

## 4. ASCII码表速查

```
ASCII码表（可打印字符部分）:
32-47:   空格 !"#$%&'()*+,-./
48-57:   0-9 (数字)
58-64:   :;<=>?@
65-90:   A-Z (大写字母)
91-96:   [\]^_`
97-122:  a-z (小写字母)
123-126: {|}~

控制字符: 0-31, 127
```

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    // 打印ASCII分类表
    printf("ASCII字符分类:\n");
    printf("字符 数值 iscntrl isspace isdigit isupper islower isalpha isalnum isprint isgraph ispunct isxdigit\n");
    printf("---- ---- ------- ------- ------- ------- ------- ------- ------- ------- ------- ------- --------\n");

    for (int c = 0; c < 128; c++) {
        if (isprint(c)) {
            printf("  %c   %3d    %d       %d       %d       %d       %d       %d       %d       %d       %d       %d       %d\n",
                   c, c,
                   iscntrl(c), isspace(c), isdigit(c),
                   isupper(c), islower(c), isalpha(c), isalnum(c),
                   isprint(c), isgraph(c), ispunct(c), isxdigit(c));
        }
    }

    return 0;
}
```

## 5. 重要注意事项

> **要点一**：ctype函数的参数类型是 `int`，必须是 `unsigned char` 值或 `EOF`。传入负值（除了EOF）会导致未定义行为。安全做法是先强制转换：`isalpha((unsigned char)c)`。

> **要点二**：这些函数的行为受当前 locale 影响。在默认的 "C" locale 下只处理ASCII字符。

> **要点三**：`toupper` 和 `tolower` 对非字母字符返回原字符，不会报错。

> **要点四**：`isblank` 是C99新增的，只匹配空格和制表符，而 `isspace` 还包括换行符等。

> **要点五**：`isprint` 包含空格，`isgraph` 不包含空格。

> **要点六**：ctype函数通常通过查表实现，比手写判断更快。

> **要点七**：不要假设字符编码一定是ASCII，虽然在大多数现代系统上确实如此。使用 `unsigned char` 强制转换确保安全。
