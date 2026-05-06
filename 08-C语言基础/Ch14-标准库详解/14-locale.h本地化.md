# locale.h - 本地化

## 1. 概述

`<locale.h>` 提供了国际化和本地化支持，允许程序根据不同地区的习惯调整输出格式、字符分类、货币表示等行为。

## 2. locale 的概念

Locale（区域设置）定义了程序的本地化行为，包括：
- 数字、日期、货币的格式
- 字符分类和排序规则
- 字符编码

## 3. 核心函数与常量

```c
#include <locale.h>

char *setlocale(int category, const char *locale);
struct lconv *localeconv(void);
```

### 3.1 本地化类别

| 类别宏 | 影响范围 |
|--------|----------|
| `LC_ALL` | 所有类别 |
| `LC_COLLATE` | 字符串比较（strcoll, strxfrm） |
| `LC_CTYPE` | 字符分类（ctype函数） |
| `LC_MONETARY` | 货币格式化 |
| `LC_NUMERIC` | 数字格式化 |
| `LC_TIME` | 日期时间格式化（strftime） |

```c
#include <stdio.h>
#include <locale.h>

int main(void) {
    // 获取当前locale
    char *current = setlocale(LC_ALL, NULL);
    printf("当前locale: %s\n", current);  // 通常是 "C"

    // 设置为系统默认locale
    setlocale(LC_ALL, "");

    // 设置为中文locale
    #ifdef _WIN32
    setlocale(LC_ALL, "Chinese_China.936");  // Windows
    #else
    setlocale(LC_ALL, "zh_CN.UTF-8");       // Linux/Mac
    #endif

    // 仅修改数字格式
    setlocale(LC_NUMERIC, "C");

    return 0;
}
```

## 4. 数字与货币格式化

```c
#include <stdio.h>
#include <locale.h>

int main(void) {
    setlocale(LC_ALL, "");

    struct lconv *lc = localeconv();

    printf("=== 数字格式 ===\n");
    printf("小数点: \"%s\"\n", lc->decimal_point);
    printf("千位分隔符: \"%s\"\n", lc->thousands_sep);
    printf("分组: %d\n", lc->grouping[0]);

    printf("\n=== 货币格式 ===\n");
    printf("国际货币符号: \"%s\"\n", lc->int_curr_symbol);
    printf("本地货币符号: \"%s\"\n", lc->currency_symbol);
    printf("小数点: \"%s\"\n", lc->mon_decimal_point);
    printf("千位分隔符: \"%s\"\n", lc->mon_thousands_sep);
    printf("正号: \"%s\"\n", lc->positive_sign);
    printf("负号: \"%s\"\n", lc->negative_sign);
    printf("int_frac_digits: %d\n", lc->int_frac_digits);
    printf("frac_digits: %d\n", lc->frac_digits);

    // 不同locale的差异
    printf("\n=== Locale对比 ===\n");

    // C locale
    setlocale(LC_NUMERIC, "C");
    printf("C locale: 小数点 = \"%s\"\n",
           localeconv()->decimal_point);

    // 设置为德语locale（使用逗号作为小数点）
    #ifdef _WIN32
    setlocale(LC_NUMERIC, "German_Germany.1252");
    #else
    setlocale(LC_NUMERIC, "de_DE.UTF-8");
    #endif
    printf("德语 locale: 小数点 = \"%s\"\n",
           localeconv()->decimal_point);

    return 0;
}
```

## 5. 字符排序

```c
#include <stdio.h>
#include <locale.h>
#include <string.h>

int main(void) {
    const char *words[] = {"apple", "Banana", "cherry", "Apple"};
    int n = 4;

    // C locale排序（基于ASCII值）
    printf("C locale排序:\n");
    setlocale(LC_COLLATE, "C");
    for (int i = 0; i < n - 1; i++) {
        for (int j = i + 1; j < n; j++) {
            if (strcmp(words[i], words[j]) > 0) {
                const char *temp = words[i];
                words[i] = words[j];
                words[j] = temp;
            }
        }
    }
    for (int i = 0; i < n; i++) {
        printf("  %s\n", words[i]);
    }

    // 注意：strcoll使用当前locale进行比较
    // 在某些locale下，大小写排序规则不同

    return 0;
}
```

## 6. 多字节字符与宽字符

```c
#include <stdio.h>
#include <locale.h>
#include <wchar.h>
#include <string.h>

int main(void) {
    // 设置中文locale以正确处理中文字符
    setlocale(LC_ALL, "");

    // 多字节字符串转宽字符
    const char *mb_str = "你好世界";
    wchar_t wc_buf[50];
    size_t len = mbstowcs(wc_buf, mb_str, 49);

    if (len != (size_t)-1) {
        wc_buf[len] = L'\0';
        printf("多字节字符串: %s\n", mb_str);
        printf("宽字符长度: %zu 个字符\n", len);

        // 使用宽字符输出
        wprintf(L"宽字符输出: %ls\n", wc_buf);
    }

    // 字符级别的处理
    printf("\n逐字节分析:\n");
    for (int i = 0; mb_str[i] != '\0'; i++) {
        printf("  byte[%d] = 0x%02X\n", i,
               (unsigned char)mb_str[i]);
    }

    return 0;
}
```

## 7. 常见 Locale 名称

```
Windows locale名称:
  "Chinese_China.936"     - 简体中文 (GBK)
  "Chinese_China.65001"   - 简体中文 (UTF-8)
  "English_United States.1252" - 美国英语
  "C"                     - 默认C locale
  ""                      - 系统默认

Linux/Mac locale名称:
  "zh_CN.UTF-8"           - 简体中文
  "zh_TW.UTF-8"           - 繁体中文
  "en_US.UTF-8"           - 美国英语
  "ja_JP.UTF-8"           - 日语
  "C" 或 "POSIX"          - 默认C locale
  ""                      - 系统默认
```

## 8. 重要注意事项

> **要点一**：`setlocale(LC_ALL, "")` 使用环境变量（如 `LANG`、`LC_ALL`）设置locale。

> **要点二**：`localeconv()` 返回指向静态数据的指针，不应修改其内容。

> **要点三**：默认的 "C" locale 使用ASCII排序规则和点号作为小数点。

> **要点四**：在处理中文字符时，需要正确设置locale，否则多字节字符处理函数可能不工作。

> **要点五**：`setlocale` 是全局设置，影响整个程序，在多线程环境中需要小心。

> **要点六**：`grouping` 字段描述千位分组规则，不同语言可能不同（如印度使用不同的分组）。

> **要点七**：ctype函数的行为受 `LC_CTYPE` 影响，不同locale下 `isalpha` 可能识别不同的字符。

> **要点八**：Windows和Linux的locale名称格式不同，编写跨平台代码时需要条件编译。
