# 22 - Unicode 与 UTF-8 处理

## 一、Unicode 基础

### 1.1 什么是 Unicode

Unicode 是一个国际标准，为世界上几乎所有语言的每个字符分配唯一的数字编号（码点 Code Point）。

```
Unicode 码点范围:
- 基本多文种平面 (BMP): U+0000 ~ U+FFFF
- 补充平面: U+10000 ~ U+10FFFF

示例:
'A'      = U+0041
'中'     = U+4E2D
''     = U+1F600 (笑脸emoji)
```

### 1.2 Unicode 编码方式

```
UTF-8:   变长 1-4 字节，兼容 ASCII，互联网最常用
UTF-16:  变长 2 或 4 字节，Windows 内部使用
UTF-32:  定长 4 字节，简单但浪费空间
```

## 二、UTF-8 编码原理

### 2.1 编码规则

```
码点范围               UTF-8字节数  编码格式
U+0000 ~ U+007F       1字节        0xxxxxxx
U+0080 ~ U+07FF       2字节        110xxxxx 10xxxxxx
U+0800 ~ U+FFFF       3字节        1110xxxx 10xxxxxx 10xxxxxx
U+10000 ~ U+10FFFF    4字节        11110xxx 10xxxxxx 10xxxxxx 10xxxxxx
```

### 2.2 编码示例

```c
/*
  'A' = U+0041 = 0100 0001 (1字节)
  UTF-8: 01000001 = 0x41 = 'A'

  '中' = U+4E2D = 0100 1110 0010 1101 (3字节)
  UTF-8: 11100100 10111000 10101101
        = 0xE4 0xB8 0xAD

  '' = U+1F600 (4字节)
  UTF-8: 11110000 10011111 10011000 10000000
        = 0xF0 0x9F 0x98 0x80
*/

#include <stdio.h>

int main() {
    char *str = "A中";
    unsigned char *p = (unsigned char *)str;

    while (*p) {
        printf("字节: 0x%02X\n", *p);
        p++;
    }
    // A: 0x41
    // 中: 0xE4, 0xB8, 0xAD

    return 0;
}
```

## 三、UTF-8 基本操作

### 3.1 计算 UTF-8 字符数

```c
#include <stdio.h>

// 判断是否是 UTF-8 续字节
static inline int is_utf8_continuation(unsigned char c) {
    return (c & 0xC0) == 0x80;  // 10xxxxxx
}

// 获取一个 UTF-8 字符的字节数
int utf8_char_len(unsigned char c) {
    if ((c & 0x80) == 0x00) return 1;      // 0xxxxxxx
    if ((c & 0xE0) == 0xC0) return 2;      // 110xxxxx
    if ((c & 0xF0) == 0xE0) return 3;      // 1110xxxx
    if ((c & 0xF8) == 0xF0) return 4;      // 11110xxx
    return -1;  // 无效的 UTF-8 起始字节
}

// 计算 UTF-8 字符串的字符数
size_t utf8_strlen(const char *s) {
    size_t count = 0;
    const unsigned char *p = (const unsigned char *)s;

    while (*p) {
        if (!is_utf8_continuation(*p)) {
            count++;
        }
        p++;
    }
    return count;
}

int main() {
    const char *str = "Hello你好世界";
    printf("字节数: %zu\n", strlen(str));       // 15
    printf("字符数: %zu\n", utf8_strlen(str));   // 8 (5个ASCII + 3个中文)
    return 0;
}
```

### 3.2 UTF-8 字符串遍历

```c
#include <stdio.h>
#include <string.h>

// 遍历 UTF-8 字符串的每个字符
void utf8_foreach(const char *s) {
    const unsigned char *p = (const unsigned char *)s;
    int char_num = 0;

    while (*p) {
        int len = utf8_char_len(*p);
        if (len <= 0) {
            printf("无效 UTF-8 字节: 0x%02X\n", *p);
            p++;
            continue;
        }

        printf("字符 #%d: ", ++char_num);
        for (int i = 0; i < len; i++) {
            printf("%02X ", p[i]);
        }
        printf("(%.*)\n", len, (const char *)p);

        p += len;
    }
}
```

### 3.3 UTF-8 随机访问

```c
#include <stdio.h>
#include <stdlib.h>

// 获取第 n 个 UTF-8 字符的位置
const char *utf8_at(const char *s, size_t n) {
    const unsigned char *p = (const unsigned char *)s;
    size_t count = 0;

    while (*p) {
        if (count == n) return (const char *)p;
        if (!is_utf8_continuation(*p)) {
            count++;
        }
        p++;
    }
    return NULL;
}

// 获取指定位置的字符（复制到 buf）
int utf8_get_char(const char *s, size_t index, char *buf, size_t buf_size) {
    const char *pos = utf8_at(s, index);
    if (!pos) return -1;

    int len = utf8_char_len(*(unsigned char *)pos);
    if ((size_t)len >= buf_size) return -1;

    memcpy(buf, pos, len);
    buf[len] = '\0';
    return len;
}
```

## 四、UTF-8 与宽字符互转

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>

#ifdef _WIN32
#include <windows.h>
#endif

// 跨平台 UTF-8 转宽字符
wchar_t *utf8_to_wide(const char *utf8) {
#ifdef _WIN32
    int needed = MultiByteToWideChar(CP_UTF8, 0, utf8, -1, NULL, 0);
    wchar_t *wide = (wchar_t *)malloc(needed * sizeof(wchar_t));
    MultiByteToWideChar(CP_UTF8, 0, utf8, -1, wide, needed);
    return wide;
#else
    setlocale(LC_ALL, "en_US.UTF-8");
    size_t len = mbstowcs(NULL, utf8, 0);
    if (len == (size_t)-1) return NULL;
    wchar_t *wide = (wchar_t *)malloc((len + 1) * sizeof(wchar_t));
    mbstowcs(wide, utf8, len + 1);
    return wide;
#endif
}

// 跨平台 宽字符 转 UTF-8
char *wide_to_utf8(const wchar_t *wide) {
#ifdef _WIN32
    int needed = WideCharToMultiByte(CP_UTF8, 0, wide, -1,
                                      NULL, 0, NULL, NULL);
    char *utf8 = (char *)malloc(needed);
    WideCharToMultiByte(CP_UTF8, 0, wide, -1,
                         utf8, needed, NULL, NULL);
    return utf8;
#else
    setlocale(LC_ALL, "en_US.UTF-8");
    size_t len = wcstombs(NULL, wide, 0);
    if (len == (size_t)-1) return NULL;
    char *utf8 = (char *)malloc(len + 1);
    wcstombs(utf8, wide, len + 1);
    return utf8;
#endif
}
```

## 五、UTF-8 验证

```c
#include <stdio.h>

// 验证 UTF-8 字符串是否合法
int utf8_validate(const unsigned char *s) {
    while (*s) {
        int len = utf8_char_len(*s);
        if (len <= 0) return 0;  // 无效起始字节

        // 检查续字节
        for (int i = 1; i < len; i++) {
            if (s[i] == '\0' || !is_utf8_continuation(s[i])) {
                return 0;
            }
        }
        s += len;
    }
    return 1;
}
```

## 六、重要注意事项

> **关键要点：**
> 1. **UTF-8 中一个"字符"可能占 1-4 字节**：不能用 `char` 索引随机访问
> 2. **`strlen` 返回字节数不是字符数**
> 3. **UTF-8 续字节的特征**：最高两位是 `10`
> 4. **C 标准库对 UTF-8 支持有限**：需要自己实现或使用第三方库（如 ICU）
> 5. **设置正确的 locale**：使用 `setlocale(LC_ALL, "")` 或指定编码
> 6. **处理文本时优先使用 UTF-8**：已成为互联网事实标准
