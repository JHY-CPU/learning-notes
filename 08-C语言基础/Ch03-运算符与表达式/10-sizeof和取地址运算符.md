# 10 - sizeof 和取地址运算符

## 一、sizeof 运算符

### 1.1 概述

`sizeof` 是一个编译时运算符，用于获取类型或变量所占用的**字节数**。它的返回值类型是 `size_t`（无符号整数类型，定义在 `<stddef.h>` 中）。

### 1.2 基本用法

```c
#include <stdio.h>

int main(void) {
    // sizeof(类型) —— 获取类型的大小
    printf("sizeof(char)      = %zu\n", sizeof(char));       // 1
    printf("sizeof(short)     = %zu\n", sizeof(short));      // 通常2
    printf("sizeof(int)       = %zu\n", sizeof(int));        // 通常4
    printf("sizeof(long)      = %zu\n", sizeof(long));       // 4或8
    printf("sizeof(long long) = %zu\n", sizeof(long long));  // 通常8
    printf("sizeof(float)     = %zu\n", sizeof(float));      // 通常4
    printf("sizeof(double)    = %zu\n", sizeof(double));     // 通常8

    // sizeof(变量) —— 获取变量的大小（括号可省略）
    int x = 42;
    printf("sizeof x   = %zu\n", sizeof x);    // 4（省略括号）
    printf("sizeof(x)  = %zu\n", sizeof(x));   // 4（带括号）

    return 0;
}
```

> **注意**：使用 `%zu` 格式化符输出 `size_t` 类型。

### 1.3 sizeof 的特殊性质

```c
// 1. sizeof 是编译时运算符，不执行表达式
int i = 0;
size_t s = sizeof(i++);  // i++ 不会被执行！
printf("i = %d\n", i);   // i仍然是0

// 2. 数组的 sizeof
int arr[10];
printf("sizeof(arr) = %zu\n", sizeof(arr));       // 40（10 * 4）
printf("数组元素个数 = %zu\n", sizeof(arr) / sizeof(arr[0]));  // 10

// 3. 指针的 sizeof（注意区分指针和数组）
int *ptr = arr;
printf("sizeof(ptr) = %zu\n", sizeof(ptr));  // 4或8（指针大小，不是数组大小）

// 4. sizeof 是运算符，不是函数
// 可以直接用于表达式
printf("%zu\n", sizeof 1 + 2);  // sizeof(1) + 2 = 4 + 2 = 6（不是sizeof(1+2)）
```

### 1.4 字符串和结构体的 sizeof

```c
// 字符串字面量
printf("sizeof(\"hello\") = %zu\n", sizeof("hello"));  // 6（包含'\0'）

// 字符数组 vs 字符指针
char str1[] = "hello";
char *str2 = "hello";
printf("sizeof(str1) = %zu\n", sizeof(str1));  // 6（数组包含6个char）
printf("sizeof(str2) = %zu\n", sizeof(str2));  // 4或8（指针的大小）

// 结构体（可能包含填充字节）
struct Data {
    char c;     // 1字节
    int i;      // 4字节
    char d;     // 1字节
};
printf("sizeof(struct Data) = %zu\n", sizeof(struct Data));  // 可能是12（有对齐填充）
```

## 二、取地址运算符 `&`

### 2.1 概述

`&` 运算符用于获取变量的**内存地址**，返回一个指向该变量类型的指针。

```c
#include <stdio.h>

int main(void) {
    int x = 42;
    int *p = &x;   // p存储了x的地址

    printf("x 的值: %d\n", x);      // 42
    printf("x 的地址: %p\n", (void *)&x);  // 某个内存地址
    printf("p 的值: %p\n", (void *)p);     // 与上面相同
    printf("p 的地址: %p\n", (void *)&p);  // p本身的地址（另一个地址）

    return 0;
}
```

### 2.2 & 的限制

- 不能对**字面量**取地址：`&42` 是非法的。
- 不能对**寄存器变量**取地址：`register int x; &x;` 是非法的。
- 不能对**表达式结果**取地址（除非是数组下标或解引用）。
- 可以对**数组元素**和**结构体成员**取地址。

```c
int arr[5];
int *p1 = &arr[3];   // 合法：对数组元素取地址

struct Point { int x; int y; };
struct Point pt;
int *p2 = &pt.x;     // 合法：对结构体成员取地址

int *p3 = &*arr;     // 合法：等价于 &arr[0]
int *p4 = &arr[0];   // 合法
```

## 三、解引用运算符 `*`

### 3.1 概述

`*` 运算符用于**通过指针访问**其所指向的变量的值（与 `&` 互为逆操作）。

```c
int x = 42;
int *p = &x;

printf("*p = %d\n", *p);  // 42（通过p访问x的值）

*p = 100;   // 通过指针修改x的值
printf("x = %d\n", x);   // 100
```

### 3.2 & 和 * 的关系

```c
int x = 42;

// & 和 * 互为逆运算
printf("%d\n", *(&x));   // 42（先取地址，再解引用，回到x）

// 但 &(*p) 不一定等于 p（如果 p 指向无效地址）
int *p = &x;
printf("%p\n", (void *)&(*p));  // 等于 p 的值
```

## 四、优先级

```
()  []  ->  .  后缀++  后缀--   >   前缀++  前缀--  + - ! ~ (type) * & sizeof   >   * / %
```

- `sizeof`、`&`（取地址）、`*`（解引用）都是**单目运算符**，优先级相同。
- 它们的优先级**高于**双目算术运算符。
- 结合性：**右结合**。

```c
int arr[10];
int *p = arr;

// 优先级示例
sizeof *p;       // sizeof(*p)：获取p指向类型的大小
sizeof(int) * 2; // (sizeof(int)) * 2：int大小乘以2
&*p;             // &(*p)：p指向的变量的地址（等于p）
*&x;             // &(*(&x))：等于 &x，再解引用，再取地址（等于x的地址）
*p++;            // *(p++)：后置++，先取*p，然后p移动
```

## 五、关键要点

1. `sizeof` 是编译时运算符，**不执行其操作数中的表达式**（如 `sizeof(i++)` 中 `i++` 不执行）。
2. `sizeof` 返回 `size_t` 类型，使用 `%zu` 格式化输出。
3. `sizeof(数组)` 得到整个数组的字节数，`sizeof(指针)` 得到指针本身的大小。
4. `sizeof` 可以省略括号（当操作数是变量名时），但建议始终加上以提高可读性。
5. `&` 不能用于字面量和寄存器变量。
6. `*` 和 `&` 互为逆操作。
7. `&`、`*`、`sizeof` 优先级相同且右结合，高于算术运算符。
8. 函数参数中的数组会**退化为指针**，此时 `sizeof` 返回指针大小而非数组大小。
