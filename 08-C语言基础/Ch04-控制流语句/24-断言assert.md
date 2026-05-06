# 断言 assert

## 1. assert 概述

`assert` 是C语言标准库 `<assert.h>` 提供的一个宏，用于在程序运行时检测条件是否为真。如果条件为假，assert 会终止程序并输出诊断信息。

assert 主要用于**调试阶段**检测程序中的逻辑错误，而不是处理运行时的用户输入错误。

## 2. 基本语法

```c
#include <assert.h>

assert(表达式);
```

- 如果表达式为**真（非零）**：程序继续执行，assert 不做任何事
- 如果表达式为**假（零）**：程序终止，输出错误信息并调用 `abort()`

## 3. 基本使用

```c
#include <stdio.h>
#include <assert.h>

int divide(int a, int b) {
    assert(b != 0);    // 断言除数不为零
    return a / b;
}

int main(void) {
    printf("10 / 3 = %d\n", divide(10, 3));  // 正常
    printf("10 / 0 = %d\n", divide(10, 0));  // 触发assert！
    return 0;
}
```

输出（断言失败时）：
```
10 / 3 = 3
Assertion failed: b != 0, file program.c, line 5
Aborted (core dumped)
```

## 4. assert 的输出信息

当断言失败时，assert 输出：
- 被断言的表达式
- 文件名
- 行号
- 然后调用 `abort()` 终止程序

```c
assert(x > 0);
// 失败时输出:
// Assertion failed: x > 0, file test.c, line 15
```

## 5. 典型使用场景

### 5.1 函数参数验证

```c
#include <assert.h>
#include <string.h>

void string_copy(char *dest, const char *src, size_t dest_size) {
    assert(dest != NULL);       // 目标不能是空指针
    assert(src != NULL);        // 源不能是空指针
    assert(dest_size > 0);      // 目标大小必须有效

    size_t i;
    for (i = 0; i < dest_size - 1 && src[i] != '\0'; i++) {
        dest[i] = src[i];
    }
    dest[i] = '\0';
}
```

### 5.2 前置条件检查

```c
#include <assert.h>

// 要求数组已排序
int binary_search(int arr[], int n, int target) {
    assert(arr != NULL);
    assert(n > 0);

    // 可选：断言数组是有序的（调试时验证）
    #ifdef DEBUG
    for (int i = 1; i < n; i++) {
        assert(arr[i] >= arr[i-1]);
    }
    #endif

    int left = 0, right = n - 1;
    while (left <= right) {
        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}
```

### 5.3 后置条件检查

```c
#include <assert.h>

int find_max(int arr[], int n) {
    assert(n > 0);

    int max = arr[0];
    for (int i = 1; i < n; i++) {
        if (arr[i] > max) {
            max = arr[i];
        }
    }

    // 后置条件：max 一定是数组中的某个元素
    #ifdef DEBUG
    int found = 0;
    for (int i = 0; i < n; i++) {
        if (arr[i] == max) {
            found = 1;
            break;
        }
    }
    assert(found);
    #endif

    return max;
}
```

### 5.4 不可到达的代码

```c
#include <assert.h>

enum Color { RED, GREEN, BLUE };

const char *color_name(enum Color c) {
    switch (c) {
        case RED:   return "红色";
        case GREEN: return "绿色";
        case BLUE:  return "蓝色";
        default:
            assert(0 && "不可到达的代码：未知的颜色值");
            return "未知";
    }
}
```

`assert(0 && "message")` 是一个惯用法：表达式恒为假，触发断言失败，引号内的字符串作为提示信息的一部分出现在断言输出中。

## 6. 禁用 assert —— NDEBUG

在发布版本中，通常不需要 assert 检查（它们会降低性能）。通过定义 `NDEBUG` 宏可以禁用所有 assert：

```c
#define NDEBUG         // 在 #include <assert.h> 之前定义
#include <assert.h>

int main(void) {
    assert(0);         // 不会触发，因为 NDEBUG 已定义
    printf("assert 被跳过了\n");
    return 0;
}
```

### 编译时控制

```bash
# 调试版本：assert 有效
gcc -g program.c -o program_debug

# 发布版本：assert 被禁用
gcc -O2 -DNDEBUG program.c -o program_release
```

### NDEBUG 的效果

```c
// 未定义 NDEBUG 时
#define assert(e) ((e) ? (void)0 : _assert_fail(#e, __FILE__, __LINE__))

// 定义 NDEBUG 后
#define assert(e) ((void)0)    // 完全不产生代码
```

## 7. assert vs 运行时检查

| 特性 | assert | 运行时检查 |
|------|--------|-----------|
| 目的 | 检测程序逻辑错误 | 处理合法的异常情况 |
| 时机 | 调试阶段 | 始终运行 |
| 失败行为 | 终止程序 | 返回错误码/异常 |
| 性能 | 发布版零开销 | 始终有开销 |
| 可禁用 | 是（NDEBUG） | 否 |

### 何时用 assert，何时用运行时检查

```c
// 用 assert：程序员的错误（调用方式不对）
void set_value(int *ptr, int value) {
    assert(ptr != NULL);    // 传 NULL 是调用者的bug
    *ptr = value;
}

// 用运行时检查：用户/环境的错误
FILE *open_file(const char *name) {
    FILE *fp = fopen(name, "r");
    if (fp == NULL) {
        fprintf(stderr, "无法打开文件: %s\n", name);
        return NULL;        // 文件不存在是正常情况
    }
    return fp;
}
```

**原则**：
- assert：检测"不应该发生"的事情（程序bug）
- 运行时检查：处理"可能发生"的事情（用户错误、资源不足）

## 8. assert 的最佳实践

### 8.1 断言中不要有副作用

```c
// 错误：assert 中的表达式有副作用
assert(printf("调试信息\n"));   // 发布版中printf被禁用！

// 正确
printf("调试信息\n");
assert(condition);

// 错误：assert 中有状态修改
assert(++count < MAX);    // count的自增在发布版中消失了！

// 正确
count++;
assert(count < MAX);
```

### 8.2 使用有意义的断言消息

```c
// 不好
assert(ptr);

// 更好
assert(ptr != NULL && "指针不能为空");

// 不好
assert(x > 0);

// 更好
assert(x > 0 && "x必须为正数，用于计算平方根");
```

### 8.3 断言验证不变量

```c
#include <assert.h>

typedef struct {
    int *data;
    int size;
    int capacity;
} Vector;

void vector_push(Vector *v, int value) {
    assert(v != NULL);
    assert(v->data != NULL);
    assert(v->size >= 0);
    assert(v->capacity > 0);
    assert(v->size <= v->capacity);   // 不变量

    if (v->size == v->capacity) {
        // 扩容...
    }

    v->data[v->size++] = value;

    // 验证操作后的不变量
    assert(v->size <= v->capacity);
}
```

## 9. 自定义 assert 宏

```c
#include <stdio.h>
#include <stdlib.h>

// 自定义 assert，提供更多信息
#ifdef DEBUG
    #define MY_ASSERT(cond, msg) do { \
        if (!(cond)) { \
            fprintf(stderr, "断言失败: %s\n", msg); \
            fprintf(stderr, "  条件: %s\n", #cond); \
            fprintf(stderr, "  文件: %s\n", __FILE__); \
            fprintf(stderr, "  行号: %d\n", __LINE__); \
            fprintf(stderr, "  函数: %s\n", __func__); \
            abort(); \
        } \
    } while(0)
#else
    #define MY_ASSERT(cond, msg) ((void)0)
#endif

int main(void) {
    int x = -1;
    MY_ASSERT(x > 0, "x必须为正数");
    return 0;
}
```

## 10. 静态断言（C11）

C11 引入了 `_Static_assert`（或 `static_assert`），在**编译时**检查条件：

```c
#include <assert.h>

_Static_assert(sizeof(int) >= 4, "int 至少需要4字节");
static_assert(sizeof(long) >= sizeof(int), "long 不能比 int 小");

int main(void) {
    // 如果上面的断言失败，编译会报错
    return 0;
}
```

静态断言在编译时求值，如果不满足，编译器会报错。适合验证数据结构大小、类型假设等。

## 11. 要点总结

1. `assert(condition)` 检查条件，失败时终止程序并输出诊断信息
2. assert 主要用于**调试阶段**，检测程序的逻辑错误
3. 定义 `NDEBUG` 宏后 assert 被禁用，发布版本应该禁用
4. assert 中不应该有副作用（自增、函数调用等）
5. assert vs 运行时检查：前者检测"不应该发生"的事，后者处理"可能发生"的事
6. `_Static_assert`（C11）提供编译时断言

## 12. 练习题

1. 为一个动态数组的增删改查操作添加 assert 验证
2. 编写一个自定义 assert 宏，在失败时输出更多上下文信息
3. 思考：哪些情况应该用 assert，哪些应该用运行时错误处理？
