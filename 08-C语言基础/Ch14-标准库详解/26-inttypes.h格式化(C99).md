# inttypes.h - 整数格式化（C99）

## 1. 概述

`<inttypes.h>`（C99引入）提供了 `stdint.h` 中固定宽度整数类型的格式化宏和转换函数，解决了不同平台上格式化 `int32_t`、`uint64_t` 等类型的可移植性问题。

## 2. 格式化宏

### 2.1 printf 格式宏

```c
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

int main(void) {
    int8_t   v8  = 127;
    int16_t  v16 = 32767;
    int32_t  v32 = 2147483647;
    int64_t  v64 = 9223372036854775807LL;
    uint8_t  u8  = 255;
    uint16_t u16 = 65535;
    uint32_t u32 = 4294967295U;
    uint64_t u64 = 18446744073709551615ULL;

    // 十进制格式
    printf("PRId8:   %" PRId8 "\n", v8);
    printf("PRId16:  %" PRId16 "\n", v16);
    printf("PRId32:  %" PRId32 "\n", v32);
    printf("PRId64:  %" PRId64 "\n", v64);

    printf("PRIu8:   %" PRIu8 "\n", u8);
    printf("PRIu16:  %" PRIu16 "\n", u16);
    printf("PRIu32:  %" PRIu32 "\n", u32);
    printf("PRIu64:  %" PRIu64 "\n", u64);

    // 十六进制格式
    printf("PRIx32:  0x%" PRIx32 "\n", u32);
    printf("PRIX32:  0x%" PRIX32 "\n", u32);

    // 八进制格式
    printf("PRIo32:  0%" PRIo32 "\n", u32);

    return 0;
}
```

### 2.2 scanf 格式宏

```c
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

int main(void) {
    int32_t val32;
    int64_t val64;
    uint32_t hex_val;

    // 使用SCN宏从字符串解析
    const char *str1 = "12345";
    sscanf(str1, "%" SCNd32, &val32);
    printf("解析 int32_t: %d\n", val32);

    const char *str2 = "999999999999";
    sscanf(str2, "%" SCNd64, &val64);
    printf("解析 int64_t: %lld\n", (long long)val64);

    const char *str3 = "FF";
    sscanf(str3, "%" SCNx32, &hex_val);
    printf("解析 hex: %u\n", hex_val);

    return 0;
}
```

### 2.3 完整的格式化宏列表

```c
#include <stdio.h>
#include <inttypes.h>

int main(void) {
    // printf 系列宏
    // 有符号十进制
    printf("PRId8=%s PRId16=%s PRId32=%s PRId64=%s\n",
           PRId8, PRId16, PRId32, PRId64);
    printf("PRIdLEAST8=%s PRIdLEAST16=%s PRIdLEAST32=%s PRIdLEAST64=%s\n",
           PRIdLEAST8, PRIdLEAST16, PRIdLEAST32, PRIdLEAST64);
    printf("PRIdFAST8=%s PRIdFAST16=%s PRIdFAST32=%s PRIdFAST64=%s\n",
           PRIdFAST8, PRIdFAST16, PRIdFAST32, PRIdFAST64);
    printf("PRIdMAX=%s PRIdPTR=%s\n", PRIdMAX, PRIdPTR);

    // 无符号十进制
    printf("PRIu32=%s PRIu64=%s\n", PRIu32, PRIu64);

    // 无符号八进制
    printf("PRIo32=%s PRIo64=%s\n", PRIo32, PRIo64);

    // 无符号十六进制（小写）
    printf("PRIx32=%s PRIx64=%s\n", PRIx32, PRIx64);

    // 无符号十六进制（大写）
    printf("PRIX32=%s PRIX64=%s\n", PRIX32, PRIX64);

    // scanf 系列宏（类似结构）
    printf("SCNd32=%s SCNd64=%s\n", SCNd32, SCNd64);
    printf("SCNu32=%s SCNu64=%s\n", SCNu32, SCNu64);

    return 0;
}
```

## 3. 整数转换函数

```c
#include <inttypes.h>

intmax_t strtoimax(const char *nptr, char **endptr, int base);
uintmax_t strtoumax(const char *nptr, char **endptr, int base);
```

```c
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>
#include <errno.h>

int main(void) {
    char *endptr;

    // strtoimax - 字符串转intmax_t
    errno = 0;
    intmax_t val1 = strtoimax("123456789012345", &endptr, 10);
    if (errno != 0) {
        perror("strtoimax");
    } else {
        printf("strtoimax: %jd\n", val1);
        if (*endptr != '\0') {
            printf("剩余: %s\n", endptr);
        }
    }

    // 不同进制
    intmax_t hex_val = strtoimax("7FFFFFFF", NULL, 16);
    printf("十六进制 7FFFFFFF = %jd\n", hex_val);

    intmax_t bin_val = strtoimax("1010", NULL, 2);
    printf("二进制 1010 = %jd\n", bin_val);

    // strtoumax - 字符串转uintmax_t
    uintmax_t uval = strtoumax("18446744073709551615", NULL, 10);
    printf("strtoumax: %ju\n", uval);

    return 0;
}
```

## 4. imaxdiv 函数

```c
#include <inttypes.h>

imaxdiv_t imaxdiv(intmax_t numer, intmax_t denom);
// 同时返回商和余数，类似div()
```

```c
#include <stdio.h>
#include <inttypes.h>

int main(void) {
    intmax_t a = 100;
    intmax_t b = 7;

    imaxdiv_t result = imaxdiv(a, b);
    printf("%jd / %jd = 商%jd 余%jd\n",
           a, b, result.quot, result.rem);

    // 负数的情况
    imaxdiv_t neg = imaxdiv(-100, 7);
    printf("-100 / 7 = 商%jd 余%jd\n", neg.quot, neg.rem);

    return 0;
}
```

## 5. 可移植的格式化技巧

```c
#include <stdio.h>
#include <inttypes.h>
#include <stdint.h>

// 不使用inttypes.h的替代方式
void portable_format(void) {
    int32_t val32 = -12345;
    uint64_t val64 = 1234567890123456789ULL;

    // 方式1: 使用PRId64宏（推荐）
    printf("方式1: %" PRId32 ", %" PRIu64 "\n", val32, val64);

    // 方式2: 强制转换为long long
    printf("方式2: %lld, %llu\n",
           (long long)val32, (unsigned long long)val64);

    // 方式3: 使用imaxx
    printf("方式3: %jd, %ju\n",
           (intmax_t)val32, (uintmax_t)val64);
}

int main(void) {
    portable_format();
    return 0;
}
```

## 6. 重要注意事项

> **要点一**：PRId64 等宏展开为字符串字面量，与格式字符串拼接使用：`"%" PRId64 "\n"`。

> **要点二**：在某些平台上，`long` 是32位（Windows），`PRId64` 可能展开为 `"lld"` 而非 `"ld"`。

> **要点三**：`strtoimax` 和 `strtoumax` 的行为与 `strtol` 系列类似。

> **要点四**：`imaxdiv` 的行为与 `div` 类似，`quot * denom + rem == numer`。

> **要点五**：`<inttypes.h>` 自动包含 `<stdint.h>`，不需要重复包含。

> **要点六**：宏名中的 "PRI" 表示 printf，"SCN" 表示 scanf。

> **要点七**：宏名中的 "d" 表示有符号十进制，"u" 无符号十进制，"x/X" 十六进制，"o" 八进制。

> **要点八**：如果不确定目标平台，使用这些宏是最安全的选择。
