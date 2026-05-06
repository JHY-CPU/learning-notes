# 空指针与 NULL

## 一、什么是空指针

空指针（Null Pointer）是指**不指向任何有效对象或函数**的指针。在 C 语言中，空指针常量用 `NULL` 表示，其值为 `0` 或 `(void*)0`。

```c
#include <stdio.h>
#include <stdlib.h>

int main() {
    int *p = NULL;
    printf("NULL 的值: %p\n", (void*)p);  // 0x0 或 (nil)
    printf("p == NULL: %s\n", p == NULL ? "true" : "false");
    printf("p == 0: %s\n", p == 0 ? "true" : "false");
    return 0;
}
```

## 二、NULL 的定义

### 在标准头文件中

```c
// <stddef.h> / <stdlib.h> / <stdio.h> 中
#define NULL ((void*)0)
// 或
#define NULL 0
```

- C 标准中 NULL 的具体定义由实现决定
- 通常是 `((void*)0)` 或字面量 `0`
- 在 C++ 中通常定义为 `0`（不是 `(void*)0`）

## 三、空指针的用途

### 1. 表示指针未初始化

```c
int *p = NULL;  // 明确表示 p 未指向任何对象
```

### 2. 使用前的安全检查

```c
void safe_print(int *p) {
    if (p == NULL) {
        printf("指针为空\n");
        return;
    }
    printf("值: %d\n", *p);
}
```

### 3. 作为链表/树的终止标记

```c
struct Node {
    int data;
    struct Node *next;
};

// 链表末尾的 next 指向 NULL
struct Node *head = NULL;  // 空链表
```

### 4. 函数返回错误状态

```c
int* find_value(int *arr, int n, int target) {
    for (int i = 0; i < n; i++) {
        if (arr[i] == target) {
            return &arr[i];
        }
    }
    return NULL;  // 未找到
}

// 调用者检查返回值
int *result = find_value(arr, 5, 42);
if (result != NULL) {
    printf("找到: %d\n", *result);
} else {
    printf("未找到\n");
}
```

### 5. 函数的可选参数

```c
void process(int *data, int *result) {
    if (data == NULL) {
        // 使用默认值
        return;
    }
    // 处理 data
    if (result != NULL) {
        *result = *data * 2;
    }
}
```

## 四、空指针解引用

### 立即崩溃

```c
int *p = NULL;
// *p = 10;  // 段错误（Segmentation Fault）或访问违规
```

### 为什么解引用空指针会崩溃

- 操作系统将地址 0 标记为不可访问的内存区域
- 尝试访问地址 0 会触发硬件异常
- 操作系统捕获异常并终止程序

### 不保证一定崩溃

```c
// 在嵌入式系统中，地址 0 可能是有效地址
// 解引用 NULL 的行为是实现定义的
// 不要依赖"解引用 NULL 会崩溃"来检测错误
```

## 五、NULL 与 0 的关系

```c
// 在指针上下文中，0 等价于 NULL
int *p1 = NULL;
int *p2 = 0;
int *p3 = '\0';

// p1、p2、p3 都是空指针

// 但数值上 NULL 不一定等于 0（虽然通常都是）
// NULL 的位模式可能不是全 0（理论上）
```

### 比较中的注意事项

```c
int *p = NULL;

if (p == 0)      // OK，0 是空指针常量
if (p == NULL)   // OK，推荐写法
if (!p)          // OK，非 NULL 为真
if (p)           // OK，等价于 p != NULL
```

## 六、C++ 中的 nullptr（仅供参考）

C++11 引入了 `nullptr` 来解决 `NULL` 在重载中的歧义：

```c++
// C++ 中 NULL 通常是 0，导致以下歧义
void foo(int);
void foo(char*);
foo(NULL);  // 调用哪个？

// C++11 解决方案
foo(nullptr);  // 明确调用 char* 版本
```

> **注意**：`nullptr` 是 C++ 特性，C 语言中不可使用。

## 七、空指针的常见问题

### 1. 忘记检查

```c
int *p = malloc(sizeof(int));
// 如果 malloc 失败，p 为 NULL
*p = 10;  // 如果 p 为 NULL，崩溃

// 正确
if (p != NULL) {
    *p = 10;
}
```

### 2. 释放后未置空

```c
int *p = malloc(sizeof(int));
free(p);
// p 不是 NULL，但也不再有效
// 必须手动置空
p = NULL;
```

### 3. 函数返回 NULL 未检查

```c
FILE *fp = fopen("file.txt", "r");
// 如果文件不存在，fp 为 NULL
// 必须检查
if (fp == NULL) {
    perror("打开文件失败");
    return 1;
}
```

## 八、关键要点总结

> **核心概念**
> - NULL 是空指针常量，值为 0 或 (void*)0
> - 空指针不指向任何有效对象
> - 解引用空指针是未定义行为（通常导致崩溃）

> **使用规范**
> - 未初始化的指针设为 NULL
> - 使用前检查指针是否为 NULL
> - 函数返回 NULL 表示失败或未找到
> - 释放后置 NULL

> **注意**
> - NULL 在 C 和 C++ 中的定义有细微差异
> - C 语言没有 nullptr（这是 C++ 特性）
