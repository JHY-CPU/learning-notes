# stddef.h - 常用定义

## 1. 概述

`<stddef.h>` 定义了若干常用的类型、宏和常量。这些定义被许多其他标准头文件使用，是C标准库的基础组件。

## 2. 核心定义

### 2.1 类型定义

```c
#include <stddef.h>

size_t      // 无符号整数类型，sizeof运算符的返回类型
ptrdiff_t   // 有符号整数类型，两个指针相减的结果类型
wchar_t     // 宽字符类型
```

### 2.2 宏定义

```c
#include <stddef.h>

NULL        // 空指针常量
offsetof(type, member)  // 结构体成员的偏移量
```

## 3. size_t 详解

```c
#include <stdio.h>
#include <stddef.h>
#include <string.h>

int main(void) {
    // size_t 是 sizeof 的返回类型
    size_t int_size = sizeof(int);
    size_t double_size = sizeof(double);
    size_t ptr_size = sizeof(void*);

    printf("sizeof(int)    = %zu\n", int_size);
    printf("sizeof(double) = %zu\n", double_size);
    printf("sizeof(void*)  = %zu\n", ptr_size);

    // size_t 的范围
    printf("SIZE_MAX = %zu\n", SIZE_MAX);  // size_t的最大值

    // 使用 size_t 作为循环变量（推荐）
    char str[] = "Hello, World!";
    size_t len = strlen(str);
    for (size_t i = 0; i < len; i++) {
        putchar(str[i]);
    }
    printf("\n");

    // 数组大小
    int arr[] = {1, 2, 3, 4, 5};
    size_t arr_size = sizeof(arr) / sizeof(arr[0]);
    printf("数组元素个数: %zu\n", arr_size);

    return 0;
}
```

### size_t 的格式说明符

```c
#include <stdio.h>
#include <stddef.h>

int main(void) {
    size_t val = 12345;

    // C99: 使用 %zu 格式化size_t
    printf("size_t值: %zu\n", val);

    // 可移植的替代方式
    printf("使用 %%lu: %lu\n", (unsigned long)val);

    return 0;
}
```

## 4. ptrdiff_t 详解

```c
#include <stdio.h>
#include <stddef.h>

int main(void) {
    int arr[] = {10, 20, 30, 40, 50};
    int *p1 = &arr[0];
    int *p2 = &arr[4];

    // 指针相减得到ptrdiff_t
    ptrdiff_t diff = p2 - p1;
    printf("指针差: %td 个元素\n", diff);  // 4

    // ptrdiff_t 是有符号的
    ptrdiff_t reverse = p1 - p2;
    printf("反向差: %td\n", reverse);  // -4

    // 注意：指针减法只在同一数组内有效
    // 否则是未定义行为

    // 使用 ptrdiff_t 进行数组下标计算
    int *mid = arr + 2;
    ptrdiff_t index = mid - arr;
    printf("中间元素索引: %td, 值: %d\n", index, *mid);

    return 0;
}
```

## 5. NULL 宏

```c
#include <stdio.h>
#include <stddef.h>

int main(void) {
    // NULL 是空指针常量
    int *ptr = NULL;

    printf("NULL的值: %p\n", (void*)ptr);

    // NULL 可能的定义方式:
    // #define NULL 0
    // #define NULL 0L
    // #define NULL (void*)0
    // #define NULL __nullptr  (C++11)

    // 检查指针是否为空
    if (ptr == NULL) {
        printf("指针为空\n");
    }

    // C语言中 NULL 和 0 可以互换（但推荐用NULL）
    ptr = 0;    // 合法但不推荐
    ptr = NULL; // 推荐

    return 0;
}
```

## 6. offsetof 宏

```c
#include <stdio.h>
#include <stddef.h>

struct Person {
    char name[32];
    int age;
    double salary;
    char address[64];
};

int main(void) {
    // offsetof 获取结构体成员相对于结构体开头的偏移量
    printf("Person结构体成员偏移量:\n");
    printf("  name:    %zu\n", offsetof(struct Person, name));
    printf("  age:     %zu\n", offsetof(struct Person, age));
    printf("  salary:  %zu\n", offsetof(struct Person, salary));
    printf("  address: %zu\n", offsetof(struct Person, address));
    printf("  sizeof:  %zu\n", sizeof(struct Person));

    // 可以看到内存对齐的影响
    // age 可能不在 name 的第32字节处

    // 使用 offsetof 进行序列化
    struct Person p = {"张三", 25, 50000.0, "北京市"};
    unsigned char *raw = (unsigned char*)&p;

    printf("\nage成员的原始字节:\n");
    size_t age_offset = offsetof(struct Person, age);
    unsigned char *age_bytes = raw + age_offset;
    for (size_t i = 0; i < sizeof(int); i++) {
        printf("  [%zu] = 0x%02X\n", i, age_bytes[i]);
    }

    return 0;
}
```

### offsetof 的底层实现原理

```c
// offsetof 的一种实现方式
// （实际实现可能依赖编译器内置）
#define MY_OFFSETOF(type, member) \
    ((size_t)&(((type *)0)->member))

#include <stdio.h>
#include <stddef.h>

struct Example {
    char a;
    int b;
    short c;
};

int main(void) {
    printf("标准 offsetof(a): %zu\n", offsetof(struct Example, a));
    printf("标准 offsetof(b): %zu\n", offsetof(struct Example, b));
    printf("标准 offsetof(c): %zu\n", offsetof(struct Example, c));

    printf("自定义 offsetof(a): %zu\n", MY_OFFSETOF(struct Example, a));
    printf("自定义 offsetof(b): %zu\n", MY_OFFSETOF(struct Example, b));
    printf("自定义 offsetof(c): %zu\n", MY_OFFSETOF(struct Example, c));

    return 0;
}
```

## 7. 重要注意事项

> **要点一**：`size_t` 是无符号类型，用于表示大小和索引。使用 `%zu` 格式化。

> **要点二**：`ptrdiff_t` 是有符号类型，用于指针运算。使用 `%td` 格式化。

> **要点三**：在32位系统上 `size_t` 通常是32位，64位系统上是64位。

> **要点四**：`NULL` 的实际定义可能是 `0`、`0L` 或 `(void*)0`，但语义相同。

> **要点五**：`offsetof` 只能用于结构体（struct），不能用于联合体（union）的非首个成员。

> **要点六**：使用 `size_t` 作为数组索引和循环变量可以避免符号相关的警告。

> **要点七**：`ptrdiff_t` 在两个指针差距超过其范围时，结果是未定义的。

> **要点八**：`stddef.h` 中的定义也被其他头文件（如 `stdio.h`、`stdlib.h`）包含，通常不需要单独包含。
