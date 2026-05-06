# stdalign.h - 内存对齐（C11）

## 1. 概述

`<stdalign.h>`（C11引入）提供了 `alignof` 和 `alignas` 关键字的宏定义，用于查询和控制内存对齐。

## 2. 核心宏

```c
#include <stdalign.h>

#define alignas _Alignas     // 指定对齐要求
#define alignof _Alignof     // 查询对齐要求

// C11还定义了:
// __alignas_is_defined  - alignas可用时为1
// __alignof_is_defined  - alignof可用时为1
```

## 3. alignof - 查询对齐要求

```c
#include <stdio.h>
#include <stdalign.h>

int main(void) {
    // 查询基本类型的对齐要求
    printf("char 的对齐:      %zu\n", alignof(char));      // 1
    printf("short 的对齐:     %zu\n", alignof(short));     // 2
    printf("int 的对齐:       %zu\n", alignof(int));       // 4
    printf("long 的对齐:      %zu\n", alignof(long));      // 4或8
    printf("long long 的对齐: %zu\n", alignof(long long)); // 8
    printf("float 的对齐:     %zu\n", alignof(float));     // 4
    printf("double 的对齐:    %zu\n", alignof(double));    // 8
    printf("void* 的对齐:     %zu\n", alignof(void*));     // 4或8

    // 查询结构体的对齐
    struct Data {
        char c;
        int i;
        double d;
    };
    printf("struct Data 的对齐: %zu\n", alignof(struct Data));

    return 0;
}
```

## 4. alignas - 指定对齐

### 4.1 基本用法

```c
#include <stdio.h>
#include <stdalign.h>

int main(void) {
    // 默认对齐的变量
    int normal_var;
    printf("normal_var 地址: %p, 对齐: %zu\n",
           (void*)&normal_var,
           (size_t)((uintptr_t)&normal_var % alignof(int)));

    // 指定16字节对齐
    alignas(16) int aligned_var;
    printf("aligned_var 地址: %p\n", (void*)&aligned_var);
    printf("aligned_var 是否16字节对齐: %s\n",
           ((uintptr_t)&aligned_var % 16 == 0) ? "是" : "否");

    // 指定64字节对齐（常用于缓存行优化）
    alignas(64) char cache_line[256];
    printf("cache_line 地址: %p\n", (void*)cache_line);

    return 0;
}
```

### 4.2 结构体对齐

```c
#include <stdio.h>
#include <stdalign.h>
#include <stdatomic.h>

// 结构体成员对齐
struct Packed {
    alignas(8) char c;      // char 但8字节对齐
    alignas(16) int i;      // int 但16字节对齐
    double d;
};

// 整个结构体对齐
struct alignas(32) CacheAligned {
    int data[8];
};

// 使用另一个类型的对齐要求
struct SameAsDouble {
    alignas(double) char bytes[8];  // 与double相同的对齐
};

int main(void) {
    printf("struct Packed 大小: %zu\n", sizeof(struct Packed));
    printf("struct Packed 对齐: %zu\n", alignof(struct Packed));

    printf("struct CacheAligned 大小: %zu\n", sizeof(struct CacheAligned));
    printf("struct CacheAligned 对齐: %zu\n", alignof(struct CacheAligned));

    // 验证对齐
    struct CacheAligned ca;
    printf("ca 地址是否32字节对齐: %s\n",
           ((uintptr_t)&ca % 32 == 0) ? "是" : "否");

    return 0;
}
```

## 5. SIMD与性能优化

```c
#include <stdio.h>
#include <stdalign.h>

// SIMD操作通常要求16字节对齐（SSE）或32字节对齐（AVX）
int main(void) {
    // 16字节对齐用于SSE
    alignas(16) float sse_data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    printf("SSE数据地址: %p (16字节对齐: %s)\n",
           (void*)sse_data,
           ((uintptr_t)sse_data % 16 == 0) ? "是" : "否");

    // 32字节对齐用于AVX
    alignas(32) double avx_data[4] = {1.0, 2.0, 3.0, 4.0};
    printf("AVX数据地址: %p (32字节对齐: %s)\n",
           (void*)avx_data,
           ((uintptr_t)avx_data % 32 == 0) ? "是" : "否");

    // 缓存行对齐（通常64字节）
    alignas(64) int hot_data[16];
    printf("热数据地址: %p (64字节对齐: %s)\n",
           (void*)hot_data,
           ((uintptr_t)hot_data % 64 == 0) ? "是" : "否");

    return 0;
}
```

## 6. alignas与sizeof的交互

```c
#include <stdio.h>
#include <stdalign.h>

int main(void) {
    // alignas 可能增加结构体大小
    struct A {
        char c;
        int i;
    };

    struct B {
        alignas(16) char c;  // c现在是16字节对齐
        int i;
    };

    printf("sizeof(struct A) = %zu\n", sizeof(struct A));  // 8
    printf("sizeof(struct B) = %zu\n", sizeof(struct B));  // 32 (可能)
    printf("alignof(struct A) = %zu\n", alignof(struct A));
    printf("alignof(struct B) = %zu\n", alignof(struct B));

    // alignas 可以指定比自然对齐更宽松的对齐
    // 但不能指定更严格的对齐（标准要求）

    return 0;
}
```

## 7. 重要注意事项

> **要点一**：`alignas` 的参数必须是2的幂次，且不能超过 `alignof(max_align_t)`（除非实现支持）。

> **要点二**：`alignof` 只能用于完整的对象类型，不能用于函数类型或不完整类型。

> **要点三**：过大的对齐要求可能导致内存浪费和性能下降。

> **要点四**：`alignas` 可以用于变量声明和结构体成员。

> **要点五**：C11的 `_Alignas` 和 `_Alignof` 是关键字，`<stdalign.h>` 提供了宏别名。

> **要点六**：`max_align_t`（来自 `<stddef.h>`）是具有最大基本对齐的类型。

> **要点七**：`malloc` 返回的指针保证对齐到 `max_align_t`。

> **要点八**：对齐要求必须是编译时常量表达式。
