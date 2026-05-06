# 13 - #pragma 指令

## #pragma 概述

`#pragma` 是一个特殊的预处理指令，用于向编译器提供**额外的控制信息**。与标准预处理指令不同，`#pragma` 的具体内容由编译器定义，不同编译器的 pragma 支持不同。

## 基本语法

```c
#pragma 指令名 [参数...]
```

如果编译器不认识某个 pragma，它会**忽略**该指令（不会产生错误）。

## 标准相关的 pragma

### #pragma once

```c
// 防止头文件重复包含（非标准但广泛支持）
#pragma once

// 等价于：
#ifndef MYHEADER_H
#define MYHEADER_H
// ...
#endif
```

### #pragma STDC（C99 标准定义）

```c
// 浮点环境访问
#pragma STDC FP_CONTRACT ON   // 允许浮点收缩（默认）
#pragma STDC FP_CONTRACT OFF  // 禁止浮点收缩
#pragma STDC FP_CONTRACT DEFAULT

// 舍入模式
#pragma STDC FENV_ROUND FE_TONEAREST   // 四舍五入
#pragma STDC FENV_ROUND FE_UPWARD      // 向上取整
#pragma STDC FENV_ROUND FE_DOWNWARD    // 向下取整
#pragma STDC FENV_ROUND FE_TOWARDZERO  // 向零取整

// 异常标志
#pragma STDC FENV_ACCESS ON   // 允许访问浮点环境
#pragma STDC FENV_ACCESS OFF  // 不访问（默认）
```

## GCC 特定 pragma

### #pragma GCC diagnostic — 诊断控制

```c
// 关闭特定警告
#pragma GCC diagnostic push              // 保存当前诊断状态
#pragma GCC diagnostic ignored "-Wunused-variable"
    int unused_var;  // 不会产生警告
#pragma GCC diagnostic pop               // 恢复诊断状态

// 忽略多个警告
#pragma GCC diagnostic ignored "-Wunused-function"
#pragma GCC diagnostic ignored "-Wshadow"

// 将警告视为错误
#pragma GCC diagnostic error "-Wimplicit-function-declaration"

// 恢复默认
#pragma GCC diagnostic warning "-Wall"
```

实际使用示例：

```c
// 忽略第三方代码的警告
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic ignored "-Wextra"
#include "third_party/legacy_code.h"
#pragma GCC diagnostic pop
```

### #pragma GCC optimize — 优化控制

```c
// 对特定函数应用优化
#pragma GCC optimize("O3")
void hot_function(void) {
    // 高度优化的代码
}
#pragma GCC optimize("O0")
void debug_function(void) {
    // 不优化的代码
}

// 对整个文件应用优化
#pragma GCC optimize("Ofast")
```

### #pragma GCC target — 目标架构

```c
// 为特定函数启用特定 CPU 指令集
#pragma GCC target("avx2")
void process_avx2(float *data, int n) {
    // 可以使用 AVX2 指令
}

#pragma GCC target("sse4.1")
void process_sse(float *data, int n) {
    // 可以使用 SSE4.1 指令
}
```

## MSVC 特定 pragma

### #pragma warning

```c
// 禁用特定警告
#pragma warning(disable: 4996)    // 禁用警告 4996
#pragma warning(disable: 4100 4201)  // 禁用多个警告

// 改变警告级别
#pragma warning(once: 4700)       // 只报告一次
#pragma warning(error: 4700)      // 将警告视为错误

// 保存和恢复
#pragma warning(push)
#pragma warning(disable: 4996)
    // ...
#pragma warning(pop)
```

### #pragma comment

```c
// 链接器注释
#pragma comment(lib, "ws2_32.lib")   // 链接 Windows Socket 库
#pragma comment(lib, "user32.lib")   // 链接用户界面库

// 编译器注释
#pragma comment(compiler, "优化选项: /O2")

// 链接器选项
#pragma comment(linker, "/STACK:1048576")  // 设置栈大小
```

## #pragma pack — 结构体对齐

```c
// 设置结构体成员的对齐方式
#pragma pack(push, 1)  // 保存当前对齐，设置为 1 字节对齐
struct PackedStruct {
    char  a;    // 1 字节
    int   b;    // 4 字节（但因为 pack(1)，紧跟 a）
    char  c;    // 1 字节
};
#pragma pack(pop)       // 恢复之前的对齐设置

// sizeof(struct PackedStruct) == 6（而不是通常的 12）
```

使用场景：

```c
// 网络协议头
#pragma pack(push, 1)
struct PacketHeader {
    uint16_t type;
    uint32_t length;
    uint16_t checksum;
};
#pragma pack(pop)

// 二进制文件格式
#pragma pack(push, 1)
struct BMPFileHeader {
    uint16_t signature;    // "BM"
    uint32_t file_size;
    uint16_t reserved1;
    uint16_t reserved2;
    uint32_t data_offset;
};
#pragma pack(pop)
```

## _Pragma 运算符（C99）

`_Pragma` 是 C99 引入的运算符，允许在宏中使用 pragma。

### 基本用法

```c
// #pragma 的等价写法
#pragma message("Hello")    // 不能在宏中使用

_Pragma("message(\"Hello\")")  // 可以在宏中使用！
```

### 在宏中使用

```c
// 定义一个禁用警告的宏
#define DISABLE_WARNING(warn) _Pragma(#warn)
#define PUSH_DIAGNOSTIC _Pragma("GCC diagnostic push")
#define POP_DIAGNOSTIC _Pragma("GCC diagnostic pop")
#define IGNORE_UNUSED _Pragma("GCC diagnostic ignored \"-Wunused-variable\""

// 使用
PUSH_DIAGNOSTIC
IGNORE_UNUSED
    int x;  // 不会警告
POP_DIAGNOSTIC
```

### 条件编译中使用

```c
#ifdef DEBUG
    _Pragma("GCC optimize(\"O0\")")
#else
    _Pragma("GCC optimize(\"O3\")")
#endif
```

## 自定义 pragma（编译器支持）

某些编译器允许注册自定义 pragma：

```c
// GCC 4.3+ 的 pragma 钩子
// （这是编译器插件功能，不是标准 C）
```

## 常用 pragma 汇总

```c
// 头文件守卫
#pragma once

// 结构体对齐
#pragma pack(push, 1)
#pragma pack(pop)

// GCC 诊断控制
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wall"
#pragma GCC diagnostic pop

// GCC 优化
#pragma GCC optimize("O2")

// MSVC 警告
#pragma warning(push)
#pragma warning(disable: 4996)
#pragma warning(pop)

// MSVC 链接
#pragma comment(lib, "library_name")
```

## 重要注意事项

> **关键点总结**：
> 1. `#pragma` 的具体行为由**编译器决定**，不保证跨编译器兼容
> 2. 编译器不认识的 pragma 会被**静默忽略**（这是标准行为）
> 3. `_Pragma` 运算符允许在宏中使用 pragma，弥补了 `#pragma` 的限制
> 4. `#pragma pack` 影响结构体布局，跨平台通信时常用
> 5. `#pragma once` 非标准但广泛支持，比传统头文件守卫更简洁
> 6. 使用 `#pragma GCC diagnostic` 可以精确控制警告，避免全局禁用
