# 24 - 正则表达式 (regex.h)

## 一、POSIX 正则表达式简介

C 标准库不包含正则表达式，但 POSIX 标准提供了 `<regex.h>`。

### 1.1 基本概念

```c
#include <regex.h>
#include <stdio.h>

/*
  两种风格:
  - BRE (Basic Regular Expressions): 基本正则
  - ERE (Extended Regular Expressions): 扩展正则

  BRE vs ERE 区别:
  功能        BRE        ERE
  分组        \(...\)    (...)
  选择        \|         |
  量词+       \+         +
  量词?       \?         ?
  量词{}      \{n,m\}    {n,m}

  编译标志:
  REG_EXTENDED  - 使用 ERE
  REG_ICASE     - 忽略大小写
  REG_NEWLINE   - 多行模式
  REG_NOSUB     - 不报告匹配位置
*/
```

## 二、基本使用流程

### 2.1 编译、匹配、释放

```c
#include <stdio.h>
#include <regex.h>
#include <string.h>

int main() {
    regex_t regex;
    int ret;

    // 1. 编译正则表达式
    ret = regcomp(&regex, "^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\\.[a-zA-Z]{2,}$",
                  REG_EXTENDED);
    if (ret != 0) {
        char errbuf[100];
        regerror(ret, &regex, errbuf, sizeof(errbuf));
        printf("正则编译错误: %s\n", errbuf);
        return 1;
    }

    // 2. 执行匹配
    const char *emails[] = {
        "user@example.com",       // 匹配
        "test.user+tag@mail.org", // 匹配
        "invalid@",               // 不匹配
        "no-at-sign.com",         // 不匹配
    };

    for (int i = 0; i < 4; i++) {
        ret = regexec(&regex, emails[i], 0, NULL, 0);
        if (ret == 0) {
            printf("'%s' -> 有效邮箱\n", emails[i]);
        } else {
            printf("'%s' -> 无效邮箱\n", emails[i]);
        }
    }

    // 3. 释放正则表达式
    regfree(&regex);

    return 0;
}
```

## 三、捕获组（匹配子串）

### 3.1 使用 regmatch_t 获取匹配位置

```c
#include <stdio.h>
#include <regex.h>
#include <string.h>

int main() {
    regex_t regex;
    regmatch_t matches[4];  // matches[0]=完整匹配, matches[1-3]=捕获组

    // 编译带捕获组的正则
    // 捕获组用 (...)
    regcomp(&regex,
            "([0-9]{4})-([0-9]{2})-([0-9]{2})",
            REG_EXTENDED);

    const char *text = "日期: 2024-01-15, 另一个: 2023-12-25";

    const char *search = text;
    while (regexec(&regex, search, 4, matches, 0) == 0) {
        // matches[0]: 完整匹配
        // matches[1]: 第一个捕获组（年）
        // matches[2]: 第二个捕获组（月）
        // matches[3]: 第三个捕获组（日）

        if (matches[0].rm_so != -1) {
            int len = matches[0].rm_eo - matches[0].rm_so;
            printf("完整匹配: '%.*s'\n", len, search + matches[0].rm_so);
        }

        // 提取捕获组
        for (int i = 1; i < 4; i++) {
            if (matches[i].rm_so != -1) {
                int len = matches[i].rm_eo - matches[i].rm_so;
                printf("  组%d: '%.*s'\n", i, len,
                       search + matches[i].rm_so);
            }
        }

        // 移动到匹配位置之后继续搜索
        search += matches[0].rm_eo;
    }

    regfree(&regex);
    return 0;
}
```

## 四、常用正则模式

### 4.1 常用正则表达式

```c
// 整数（含负数）
regcomp(&re, "^-?[0-9]+$", REG_EXTENDED);

// 浮点数
regcomp(&re, "^-?[0-9]+\\.?[0-9]*$", REG_EXTENDED);

// IPv4 地址
regcomp(&re,
    "^([0-9]{1,3}\\.){3}[0-9]{1,3}$",
    REG_EXTENDED);

// 日期 YYYY-MM-DD
regcomp(&re,
    "^[0-9]{4}-(0[1-9]|1[0-2])-(0[1-9]|[12][0-9]|3[01])$",
    REG_EXTENDED);

// URL
regcomp(&re,
    "^https?://[a-zA-Z0-9.-]+(/[^ ]*)?$",
    REG_EXTENDED | REG_ICASE);

// 中国手机号
regcomp(&re, "^1[3-9][0-9]{9}$", REG_EXTENDED);

// 中文字符（UTF-8）
regcomp(&re,
    "[\xE4-\xE9][\x80-\xBF]{2}",  // 简单匹配3字节UTF-8中文
    REG_EXTENDED);
```

## 五、替换功能（手动实现）

POSIX 正则不直接支持替换，需要自己实现：

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <regex.h>

// 简单的正则替换
char *regex_replace(const char *pattern, const char *text,
                    const char *replacement) {
    regex_t regex;
    regmatch_t match;

    if (regcomp(&regex, pattern, REG_EXTENDED) != 0) {
        return NULL;
    }

    size_t result_cap = strlen(text) * 2 + 1;
    char *result = (char *)malloc(result_cap);
    size_t result_len = 0;

    const char *search = text;

    while (regexec(&regex, search, 1, &match, 0) == 0) {
        // 复制匹配前的部分
        size_t before = (size_t)match.rm_so;
        if (result_len + before >= result_cap) {
            result_cap *= 2;
            result = (char *)realloc(result, result_cap);
        }
        memcpy(result + result_len, search, before);
        result_len += before;

        // 添加替换文本
        size_t rep_len = strlen(replacement);
        if (result_len + rep_len >= result_cap) {
            result_cap = result_len + rep_len + 1;
            result = (char *)realloc(result, result_cap);
        }
        memcpy(result + result_len, replacement, rep_len);
        result_len += rep_len;

        // 移动到匹配之后
        search += match.rm_eo;
    }

    // 复制剩余部分
    size_t remain = strlen(search);
    if (result_len + remain >= result_cap) {
        result = (char *)realloc(result, result_len + remain + 1);
    }
    memcpy(result + result_len, search, remain);
    result_len += remain;
    result[result_len] = '\0';

    regfree(&regex);
    return result;
}

int main() {
    char *result = regex_replace(
        "[0-9]+",
        "今天是2024年1月15日，温度是-5度",
        "X"
    );
    if (result) {
        printf("%s\n", result);  // 今天是XXXX年X月XX日，温度是-X度
        free(result);
    }
    return 0;
}
```

## 六、错误处理

```c
void check_regex_error(int errcode, regex_t *regex) {
    if (errcode != 0) {
        char errbuf[256];
        regerror(errcode, regex, errbuf, sizeof(errbuf));
        fprintf(stderr, "正则表达式错误: %s\n", errbuf);
    }
}
```

## 七、重要注意事项

> **关键要点：**
> 1. **POSIX 正则功能有限**：不如 Perl/Python 正则强大
> 2. **必须 `regfree` 释放资源**：避免内存泄漏
> 3. **使用 `REG_EXTENDED` 获取 ERE 语法**：更接近现代正则表达式
> 4. **`regmatch_t` 的 `rm_so = -1` 表示未匹配**
> 5. **跨平台可考虑 PCRE2 库**：功能更强大
> 6. **简单字符串操作可能更快**：正则有编译和执行开销
