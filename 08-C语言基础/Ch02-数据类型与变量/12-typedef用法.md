# 12 - typedef 用法

## 一、typedef 基本概念

`typedef` 用于为已有的类型创建**别名**（alias），使代码更简洁、更易读。

```c
typedef unsigned long ulong;      // ulong 是 unsigned long 的别名
typedef unsigned int uint;        // uint 是 unsigned int 的别名

ulong a = 100UL;   // 等价于 unsigned long a = 100UL;
uint b = 42;       // 等价于 unsigned int b = 42;
```

> **注意**：`typedef` 不创建新类型，只是给已有类型起一个新名字。

---

## 二、typedef 基本用法

### 2.1 基本类型别名

```c
typedef int               Integer;
typedef unsigned int      UInt;
typedef unsigned char     Byte;
typedef unsigned short    Word;
typedef unsigned long     DWord;  // Double Word
typedef double            Real;
typedef char              Bool;   // C99 之前
```

### 2.2 结构体别名

```c
// 不用 typedef
struct Point {
    int x, y;
};
struct Point p1;   // 必须写 struct

// 用 typedef
typedef struct {
    int x, y;
} Point;
Point p1;          // 直接使用 Point

// 或者
typedef struct Point {
    int x, y;
} Point_t;
```

### 2.3 联合体和枚举别名

```c
typedef union {
    int i;
    float f;
    char c;
} Data;

typedef enum {
    JAN, FEB, MAR, APR, MAY, JUN,
    JUL, AUG, SEP, OCT, NOV, DEC
} Month;
```

---

## 三、typedef 与指针

### 3.1 简化指针类型

```c
typedef int *IntPtr;
typedef char *String;
typedef void (*Callback)(void);   // 函数指针别名

IntPtr p;           // 等价于 int *p;
String name;        // 等价于 char *name;
```

### 3.2 常见陷阱

```c
typedef int *IntPtr;
const IntPtr p;     // 等价于 int * const p（常量指针）
                    // 不是 const int *p！
```

```c
// 陷阱分析
typedef int *IntPtr;
IntPtr a, b;        // a 和 b 都是指针！

// 对比宏定义
#define IntPtr2 int *
IntPtr2 a, b;       // a 是指针，b 是 int！（宏只是文本替换）
```

---

## 四、typedef 与函数指针

`typedef` 在函数指针场景中特别有用：

```c
// 不用 typedef
int (*compare)(const void *, const void *);

// 用 typedef
typedef int (*CompareFunc)(const void *, const void *);

// 使用
CompareFunc cmp = my_compare;
qsort(arr, n, sizeof(int), cmp);
```

```c
// 更多例子
typedef void (*SignalHandler)(int);
typedef int (*MathFunc)(int, int);

int add(int a, int b) { return a + b; }
int mul(int a, int b) { return a * b; }

MathFunc op = add;
printf("%d\n", op(3, 4));  // 7
```

---

## 五、typedef 与数组

```c
typedef int IntArray10[10];    // 10 个 int 的数组类型
typedef double Matrix3[3][3];  // 3x3 矩阵类型

IntArray10 arr;         // 等价于 int arr[10];
Matrix3 identity = {    // 等价于 double identity[3][3] = {...}
    {1, 0, 0},
    {0, 1, 0},
    {0, 0, 1}
};
```

---

## 六、typedef vs #define

| 特性 | typedef | #define |
|------|---------|---------|
| 处理阶段 | 编译 | 预处理 |
| 作用域 | 遵循块作用域 | 从定义到文件末尾 |
| 指针声明 | 正确处理多个变量 | 可能出错 |
| 复杂类型 | 支持 | 受限 |
| 可调试 | 可见 | 被替换 |

```c
// typedef 正确处理指针
typedef int *IntPtr;
IntPtr a, b;      // a, b 都是 int*

// #define 可能出错
#define IntPtr int *
IntPtr a, b;      // a 是 int*, b 是 int!
```

---

## 七、实际项目中的常见用法

### 7.1 标准库中的 typedef

```c
// <stddef.h>
typedef unsigned long size_t;
typedef signed long ptrdiff_t;

// <stdint.h>
typedef signed char int8_t;
typedef unsigned char uint8_t;
typedef signed int int32_t;
typedef unsigned int uint32_t;

// <stdio.h>
typedef struct _IO_FILE FILE;

// <time.h>
typedef long time_t;
```

### 7.2 可移植性

```c
// 根据平台定义统一类型
typedef long long Int64;
typedef unsigned long long UInt64;

// 或使用 stdint.h
#include <stdint.h>
int32_t x;      // 确保在所有平台上都是 32 位
uint64_t y;     // 确保在所有平台上都是 64 位无符号
```

### 7.3 链表节点

```c
typedef struct Node {
    int data;
    struct Node *next;
} Node;

// 使用
Node *head = NULL;
Node *new_node = malloc(sizeof(Node));
new_node->data = 42;
new_node->next = NULL;
```

---

## 八、typedef 的最佳实践

1. **命名约定**：类型别名首字母大写或加 `_t` 后缀（注意 `_t` 是 POSIX 保留的）
2. **不要过度使用**：简单的 `int`、`char` 不需要别名
3. **提高可读性**：在复杂类型（函数指针、数组）上使用 typedef
4. **放在头文件中**：如果多个文件需要使用

```c
// 推荐：放在头文件中
// types.h
#ifndef TYPES_H
#define TYPES_H

typedef struct {
    double x, y;
} Vec2D;

typedef struct {
    int width, height;
} Size;

typedef void (*EventCallback)(int event_type, void *data);

#endif
```

---

## 九、要点总结

1. `typedef` 为已有类型创建别名，不创建新类型
2. 常用于简化结构体、联合体、枚举和函数指针的声明
3. `typedef` 遵循作用域规则，`#define` 不遵循
4. `typedef` 正确处理多个指针变量声明，`#define` 可能出错
5. 函数指针配合 `typedef` 是最重要的应用场景之一
6. 标准库大量使用 `typedef`（`size_t`、`FILE`、`uint32_t` 等）
7. 避免对简单类型过度使用 typedef
8. `_t` 后缀是 POSIX 保留的，自定义类型应避免使用
