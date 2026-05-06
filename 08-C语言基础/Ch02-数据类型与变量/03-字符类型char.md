# 03 - 字符类型 char

## 一、char 类型基础

`char` 是 C 语言中用于存储**单个字符**的基本类型，本质上是一个占用 **1 字节**的整数。

```c
char ch = 'A';       // 存储字符 'A'
char num = 65;       // 同样存储 'A'，因为 ASCII 中 'A' == 65
```

`char` 虽然名为"字符类型"，但它在内存中存储的是字符对应的**编码值**（通常是 ASCII 码），因此 `char` 本质上是一个整型。

---

## 二、ASCII 码表

ASCII（American Standard Code for Information Interchange）使用 7 位编码，共定义了 128 个字符。C 语言中 `char` 的默认编码方式就是 ASCII。

### 2.1 常见 ASCII 字符分类

| 范围 | 类别 | 示例 |
|------|------|------|
| 0 ~ 31 | 控制字符 | `\0`(0), `\n`(10), `\t`(9) |
| 32 | 空格 | `' '` |
| 48 ~ 57 | 数字字符 | `'0'`(48) ~ `'9'`(57) |
| 65 ~ 90 | 大写字母 | `'A'`(65) ~ `'Z'`(90) |
| 97 ~ 122 | 小写字母 | `'a'`(97) ~ `'z'`(122) |

### 2.2 常用技巧

```c
// 大小写转换：相差 32
char upper = 'A';
char lower = upper + 32;     // 'a'

// 或使用位运算
char lower2 = upper | 0x20;  // 'a'

// 数字字符转整数
char digit = '7';
int value = digit - '0';     // 7

// 整数转数字字符
int num = 3;
char ch = num + '0';         // '3'

// 判断字符类型
if (ch >= 'A' && ch <= 'Z') {
    printf("大写字母\n");
}
```

---

## 三、signed char 与 unsigned char

`char` 是否有符号取决于编译器和平台的实现。C 标准允许 `char` 等同于 `signed char` 或 `unsigned char`。

### 3.1 三种 char 的区别

| 类型 | 范围 | 说明 |
|------|------|------|
| `char` | 实现定义 | 由编译器决定是有符号还是无符号 |
| `signed char` | -128 ~ 127 | 明确有符号 |
| `unsigned char` | 0 ~ 255 | 明确无符号 |

```c
// 检查当前平台 char 的符号性
#include <stdio.h>

int main(void) {
    char ch = -1;
    if (ch < 0) {
        printf("char 在本平台是有符号的\n");
    } else {
        printf("char 在本平台是无符号的\n");
    }
    // 或使用宏
    // #include <limits.h>
    // if (CHAR_MIN < 0) { /* signed */ }
    return 0;
}
```

### 3.2 使用建议

- 处理**文本数据**时：使用 `char`
- 需要**算术运算**且有负数：使用 `signed char`
- 处理**原始字节数据**：使用 `unsigned char`

---

## 四、转义字符

转义字符用于表示那些无法直接输入或有特殊含义的字符：

| 转义序列 | 含义 | ASCII 值 |
|----------|------|----------|
| `\0` | 空字符（NUL） | 0 |
| `\a` | 响铃（BEL） | 7 |
| `\b` | 退格（BS） | 8 |
| `\t` | 水平制表符（TAB） | 9 |
| `\n` | 换行（LF） | 10 |
| `\v` | 垂直制表符（VT） | 11 |
| `\f` | 换页（FF） | 12 |
| `\r` | 回车（CR） | 13 |
| `\\` | 反斜杠 | 92 |
| `\'` | 单引号 | 39 |
| `\"` | 双引号 | 34 |
| `\?` | 问号 | 63 |
| `\xhh` | 十六进制表示 | 对应值 |
| `\ooo` | 八进制表示（1-3 位） | 对应值 |

```c
printf("Hello\tWorld\n");        // Hello   World（中间有制表符）
printf("Path: C:\\Users\\file\n"); // Path: C:\Users\file
printf("Say \"Hi\"\n");           // Say "Hi"
printf("\x48\x65\x6C\x6C\x6F\n"); // Hello（十六进制表示）
printf("\110\145\154\154\157\n"); // Hello（八进制表示）
```

---

## 五、字符的输入输出

### 5.1 使用 printf 和 scanf

```c
#include <stdio.h>

int main(void) {
    char ch;

    printf("请输入一个字符: ");
    scanf("%c", &ch);

    printf("字符: %c\n", ch);
    printf("ASCII 码: %d\n", ch);

    return 0;
}
```

### 5.2 使用 getchar 和 putchar

```c
#include <stdio.h>

int main(void) {
    int c;  // 注意：getchar 返回 int 类型！

    printf("输入字符（按 Ctrl+D/Ctrl+Z 结束）:\n");
    while ((c = getchar()) != EOF) {
        putchar(c);
    }

    return 0;
}
```

> **重要**：`getchar()` 返回 `int` 而非 `char`，因为它需要返回 EOF（通常是 -1）来表示输入结束。如果用 `char` 接收，将无法正确检测 EOF。

---

## 六、ctype.h 中的字符处理函数

`<ctype.h>` 提供了丰富的字符判断和转换函数：

```c
#include <ctype.h>

// 判断函数（返回非零表示真）
isalpha(c);   // 是否为字母
isdigit(c);   // 是否为数字
isalnum(c);   // 是否为字母或数字
isspace(c);   // 是否为空白字符（空格、\t、\n 等）
isupper(c);   // 是否为大写字母
islower(c);   // 是否为小写字母
ispunct(c);   // 是否为标点符号
isprint(c);   // 是否为可打印字符

// 转换函数
toupper(c);   // 转为大写
tolower(c);   // 转为小写
```

```c
#include <stdio.h>
#include <ctype.h>

int main(void) {
    char str[] = "Hello, World! 123";

    int letters = 0, digits = 0, spaces = 0;
    for (int i = 0; str[i] != '\0'; i++) {
        if (isalpha(str[i])) letters++;
        else if (isdigit(str[i])) digits++;
        else if (isspace(str[i])) spaces++;
    }

    printf("字母: %d, 数字: %d, 空白: %d\n", letters, digits, spaces);
    return 0;
}
```

---

## 七、'\0' 的特殊地位

`'\0'`（空字符，ASCII 值为 0）在 C 语言中有特殊的意义——它是**字符串的结束标志**：

```c
char str1[] = {'H', 'i', '\0'};  // 合法的字符串
char str2[] = "Hi";               // 自动包含 '\0'

printf("str1 长度: %zu\n", sizeof(str1));  // 3
printf("str2 长度: %zu\n", sizeof(str2));  // 3（含 '\0'）
```

> 注意区分 `'\0'`（空字符，值为 0）和 `'0'`（数字零，值为 48），二者完全不同。

---

## 八、要点总结

1. `char` 占 1 字节，存储的是字符的**编码值**（通常为 ASCII）
2. `char` 的符号性由**实现定义**，可使用 `CHAR_MIN` 宏检测
3. 处理字节数据时使用 `unsigned char`，避免符号扩展问题
4. 掌握常用转义字符，特别是 `\0`、`\n`、`\t`
5. `getchar()` 返回 `int` 类型，不能用 `char` 来检测 EOF
6. 使用 `<ctype.h>` 中的函数进行字符判断和转换
7. 大小写转换可通过加减 32 或位运算实现
8. `'\0'` 是字符串的终止标志，与 `'0'`（ASCII 48）完全不同
