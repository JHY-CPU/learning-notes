# stdint.h - 固定宽度整数类型（C99）

## 1. 概述

`<stdint.h>`（C99引入）提供了具有精确宽度的整数类型，解决了不同平台上 `int`、`long` 等类型宽度不一致的可移植性问题。

## 2. 精确宽度类型

```c
#include <stdint.h>

// 精确宽度类型
int8_t    // 恰好8位有符号
uint8_t   // 恰好8位无符号
int16_t   // 恰好16位有符号
uint16_t  // 恰好16位无符号
int32_t   // 恰好32位有符号
uint32_t  // 恰好32位无符号
int64_t   // 恰好64位有符号
uint64_t  // 恰好64位无符号
```

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    // 精确宽度整数
    int8_t   a = 127;           // -128 ~ 127
    uint8_t  b = 255;           // 0 ~ 255
    int16_t  c = 32767;         // -32768 ~ 32767
    uint16_t d = 65535;         // 0 ~ 65535
    int32_t  e = 2147483647;    // 约±21亿
    uint32_t f = 4294967295u;   // 约42亿
    int64_t  g = 9223372036854775807LL;
    uint64_t h = 18446744073709551615ULL;

    printf("int8_t:   %d\n", a);
    printf("uint8_t:  %u\n", b);
    printf("int16_t:  %d\n", c);
    printf("uint16_t: %u\n", d);
    printf("int32_t:  %d\n", e);
    printf("uint32_t: %u\n", f);
    printf("int64_t:  %lld\n", (long long)g);
    printf("uint64_t: %llu\n", (unsigned long long)h);

    // 确认大小
    printf("\n类型大小:\n");
    printf("sizeof(int8_t)   = %zu\n", sizeof(int8_t));
    printf("sizeof(int16_t)  = %zu\n", sizeof(int16_t));
    printf("sizeof(int32_t)  = %zu\n", sizeof(int32_t));
    printf("sizeof(int64_t)  = %zu\n", sizeof(int64_t));

    return 0;
}
```

## 3. 最小宽度类型

```c
#include <stdint.h>

// 最小宽度类型（保证至少N位，可能更大）
int_least8_t    // 至少8位
uint_least8_t
int_least16_t   // 至少16位
uint_least16_t
int_least32_t   // 至少32位
uint_least32_t
int_least64_t   // 至少64位
uint_least64_t
```

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    // 最小宽度类型
    int_least8_t  i8  = 100;
    int_least16_t i16 = 30000;
    int_least32_t i32 = 1000000000;
    int_least64_t i64 = 9000000000000000000LL;

    printf("int_least8_t:  %d (大小: %zu)\n", i8, sizeof(i8));
    printf("int_least16_t: %d (大小: %zu)\n", i16, sizeof(i16));
    printf("int_least32_t: %d (大小: %zu)\n", i32, sizeof(i32));
    printf("int_least64_t: %lld (大小: %zu)\n",
           (long long)i64, sizeof(i64));

    return 0;
}
```

## 4. 最快宽度类型

```c
#include <stdint.h>

// 最快的至少N位类型（可能比N位大以提高速度）
int_fast8_t     // 最快的至少8位
uint_fast8_t
int_fast16_t    // 最快的至少16位
uint_fast16_t
int_fast32_t    // 最快的至少32位
uint_fast32_t
int_fast64_t    // 最快的至少64位
uint_fast64_t
```

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    // 在大多数64位平台上：
    // int_fast8_t 实际是 int 或更大
    // int_fast16_t 实际是 int 或更大
    // int_fast32_t 实际是 int 或更大
    // int_fast64_t 实际是 long long

    printf("最快宽度类型的实际大小:\n");
    printf("int_fast8_t:  %zu\n", sizeof(int_fast8_t));
    printf("int_fast16_t: %zu\n", sizeof(int_fast16_t));
    printf("int_fast32_t: %zu\n", sizeof(int_fast32_t));
    printf("int_fast64_t: %zu\n", sizeof(int_fast64_t));

    // int_fast8_t 在许多平台上实际上和int一样大
    // 因为CPU操作int大小的数据最快

    return 0;
}
```

## 5. 指针宽度整数

```c
#include <stdint.h>

intptr_t    // 可以存储指针的有符号整数
uintptr_t   // 可以存储指针的无符号整数
```

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    int value = 42;
    int *ptr = &value;

    // intptr_t 和 uintptr_t 可以安全存储指针
    intptr_t  iptr = (intptr_t)ptr;
    uintptr_t uptr = (uintptr_t)ptr;

    printf("指针: %p\n", (void*)ptr);
    printf("intptr_t:  %ld (大小: %zu)\n", (long)iptr, sizeof(iptr));
    printf("uintptr_t: %lu (大小: %zu)\n",
           (unsigned long)uptr, sizeof(uptr));

    // 转换回指针
    int *restored = (int*)iptr;
    printf("恢复的值: %d\n", *restored);

    // 指针运算
    printf("指针是否对齐到4字节: %s\n",
           (uptr % 4 == 0) ? "是" : "否");

    return 0;
}
```

## 6. 最大宽度类型

```c
#include <stdint.h>

intmax_t     // 最大的有符号整数类型
uintmax_t    // 最大的无符号整数类型
```

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    intmax_t  imax = INTMAX_MAX;
    uintmax_t umax = UINTMAX_MAX;

    printf("intmax_t 最大值:  %jd\n", (intmax_t)imax);
    printf("uintmax_t 最大值: %ju\n", (uintmax_t)umax);
    printf("intmax_t 大小:    %zu\n", sizeof(intmax_t));
    printf("uintmax_t 大小:   %zu\n", sizeof(uintmax_t));

    // INTMAX_C / UINTMAX_C 宏
    // 创建 intmax_t / uintmax_t 类型的常量
    intmax_t val = INTMAX_C(1234567890123456789);
    printf("大常量: %jd\n", val);

    return 0;
}
```

## 7. 限制宏

```c
#include <stdio.h>
#include <stdint.h>

int main(void) {
    // 精确宽度类型的限制
    printf("INT8_MIN  = %d,   INT8_MAX  = %d\n", INT8_MIN, INT8_MAX);
    printf("UINT8_MAX = %u\n", UINT8_MAX);
    printf("INT16_MIN = %d,  INT16_MAX = %d\n", INT16_MIN, INT16_MAX);
    printf("INT32_MIN = %d, INT32_MAX = %d\n", INT32_MIN, INT32_MAX);
    printf("INT64_MIN = %lld, INT64_MAX = %lld\n",
           (long long)INT64_MIN, (long long)INT64_MAX);

    // 最小宽度类型的限制
    printf("INT_LEAST8_MIN  = %d\n", INT_LEAST8_MIN);
    printf("INT_LEAST32_MAX = %d\n", INT_LEAST32_MAX);

    // 最快宽度类型的限制
    printf("INT_FAST8_MIN  = %d\n", INT_FAST8_MIN);
    printf("INT_FAST32_MAX = %d\n", INT_FAST32_MAX);

    // 指针宽度类型
    printf("INTPTR_MIN  = %ld\n", (long)INTPTR_MIN);
    printf("INTPTR_MAX  = %ld\n", (long)INTPTR_MAX);
    printf("UINTPTR_MAX = %lu\n", (unsigned long)UINTPTR_MAX);

    // 最大宽度类型
    printf("INTMAX_MIN  = %jd\n", (intmax_t)INTMAX_MIN);
    printf("INTMAX_MAX  = %jd\n", (intmax_t)INTMAX_MAX);

    // SIZE_MAX (来自stdint.h或stddef.h)
    printf("SIZE_MAX    = %zu\n", SIZE_MAX);

    return 0;
}
```

## 8. 格式化宏

```c
#include <stdio.h>
#include <stdint.h>
#include <inttypes.h>

int main(void) {
    // inttypes.h提供了格式化宏
    int32_t  val32 = -12345;
    uint64_t val64 = 1234567890123456789ULL;

    // printf 格式化宏
    printf("int32_t:  %" PRId32 "\n", val32);
    printf("uint64_t: %" PRIu64 "\n", val64);
    printf("hex64:    %" PRIx64 "\n", val64);

    // scanf 格式化宏
    // int32_t input;
    // scanf("%" SCNd32, &input);

    // 可移植的替代方式
    printf("可移植方式: %lld, %llu\n",
           (long long)val32, (unsigned long long)val64);

    return 0;
}
```

## 9. 重要注意事项

> **要点一**：精确宽度类型（如 `int32_t`）在某些平台上可能不存在（如果硬件不支持）。需要它们时应该使用条件编译。

> **要点二**：`int_fast8_t` 在许多平台上实际是 `int`（32位或64位），因为CPU操作自然字长的数据最快。

> **要点三**：`intptr_t` 和 `uintptr_t` 是可选类型，但几乎所有现代平台都支持。

> **要点四**：格式化 `int64_t` 需要 `%lld`（转为 `long long`）或使用 `inttypes.h` 的 `PRId64` 宏。

> **要点五**：`SIZE_MAX` 定义在 `stdint.h` 和 `stddef.h` 中都可用。

> **要点六**：`intmax_t` 通常是 `long long`，但可能更大（取决于平台）。

> **要点七**：这些类型对编写可移植的底层代码（网络协议、文件格式、加密算法等）非常有用。

> **要点八**：`INTMAX_C` 和 `UINTMAX_C` 宏可以创建对应类型的字面量常量。
