# assert.h - 断言与调试

## 1. 概述

`<assert.h>` 提供了运行时断言机制，用于在调试阶段检查程序的假设条件是否成立。断言在开发和测试阶段非常有用，可以帮助快速定位逻辑错误。

## 2. assert 宏

### 2.1 基本用法

```c
#include <assert.h>

void assert(int expression);
// 如果 expression 为假（0），程序终止并输出错误信息
// 如果 expression 为真（非0），不做任何操作
```

```c
#include <stdio.h>
#include <assert.h>

int divide(int a, int b) {
    assert(b != 0);  // 断言除数不为零
    return a / b;
}

int *create_array(int size) {
    assert(size > 0);          // 断言大小为正
    assert(size <= 1000000);   // 断言大小合理

    int *arr = malloc(size * sizeof(int));
    assert(arr != NULL);       // 断言分配成功

    return arr;
}

double sqrt_custom(double x) {
    assert(x >= 0.0);  // 断言非负数
    return sqrt(x);
}

int main(void) {
    // 正常情况
    printf("10 / 3 = %d\n", divide(10, 3));

    // 以下会触发断言失败
    // printf("10 / 0 = %d\n", divide(10, 0));

    int *arr = create_array(100);
    free(arr);

    printf("sqrt(4) = %f\n", sqrt_custom(4.0));

    return 0;
}
```

### 2.2 断言失败输出

当断言失败时，输出格式如下：

```
assertion "expression" failed: file "filename", line line_number, function "function_name"
```

```c
#include <stdio.h>
#include <assert.h>

void process(int *data, int len) {
    assert(data != NULL);
    assert(len > 0);
    assert(len <= 1000);

    for (int i = 0; i < len; i++) {
        assert(data[i] >= 0);  // 断言数据非负
        printf("%d ", data[i]);
    }
    printf("\n");
}

int main(void) {
    int data[] = {1, 2, 3, 4, 5};
    process(data, 5);

    // 触发断言
    // int bad_data[] = {1, -2, 3};
    // process(bad_data, 3);
    // 输出: assert "data[i] >= 0" failed: file "test.c", line 10

    return 0;
}
```

## 3. NDEBUG 宏

### 3.1 禁用断言

定义 `NDEBUG` 宏后，`assert` 宏会被预处理器移除，不产生任何代码。

```c
// 方式1：在包含头文件前定义
#define NDEBUG
#include <assert.h>

// 方式2：编译时定义
// gcc -DNDEBUG program.c

// 方式3：在Makefile中
// CFLAGS += -DNDEBUG
```

### 3.2 Release 模式的断言处理

```c
#include <stdio.h>

// 在包含assert.h之前定义NDEBUG来禁用断言
#ifdef NDEBUG
  #define DEBUG_MODE 0
#else
  #define DEBUG_MODE 1
#endif

#include <assert.h>

int main(void) {
    printf("调试模式: %s\n", DEBUG_MODE ? "开启" : "关闭");

    // 在NDEBUG模式下，这行会被完全移除
    assert(1 == 2);  // Release模式下不存在

    printf("程序继续运行\n");
    return 0;
}
```

### 3.3 典型的编译配置

```makefile
# Makefile 示例
CC = gcc
CFLAGS_DEBUG = -g -Wall -Wextra -O0
CFLAGS_RELEASE = -O2 -DNDEBUG

debug:
	$(CC) $(CFLAGS_DEBUG) -o program program.c

release:
	$(CC) $(CFLAGS_RELEASE) -o program program.c
```

## 4. 静态断言 - _Static_assert（C11）

### 4.1 概述

`_Static_assert` 在编译时检查条件，不满足时编译失败。这对于检查编译时常量非常有用。

```c
#include <assert.h>  // C11: assert.h 中也提供了 static_assert 宏

// C11语法
_Static_assert(constant_expression, "error message");

// C11通过assert.h提供的宏
static_assert(constant_expression, "error message");
```

### 4.2 使用示例

```c
#include <stdio.h>
#include <assert.h>
#include <stdint.h>

// 编译时检查类型大小
_Static_assert(sizeof(int) >= 4, "int必须至少4字节");
_Static_assert(sizeof(long long) == 8, "long long必须是8字节");
_Static_assert(sizeof(void*) == 8, "仅支持64位系统");

// 检查结构体大小
typedef struct {
    int x;
    int y;
    int z;
} Point3D;

_Static_assert(sizeof(Point3D) == 12, "Point3D应该是12字节");

// 检查枚举值
enum Color { RED = 0, GREEN = 1, BLUE = 2 };
_Static_assert(BLUE == 2, "BLUE应该是2");

int main(void) {
    // 函数内的静态断言
    _Static_assert(sizeof(double) == 8, "double必须是8字节");

    printf("所有编译时断言通过\n");
    printf("int大小: %zu\n", sizeof(int));
    printf("long long大小: %zu\n", sizeof(long long));
    printf("指针大小: %zu\n", sizeof(void*));
    printf("Point3D大小: %zu\n", sizeof(Point3D));

    return 0;
}
```

## 5. 自定义断言宏

### 5.1 带更多信息的断言

```c
#include <stdio.h>
#include <stdlib.h>

// 自定义断言：提供更多信息
#ifdef NDEBUG
    #define MY_ASSERT(expr) ((void)0)
#else
    #define MY_ASSERT(expr) \
        do { \
            if (!(expr)) { \
                fprintf(stderr, \
                    "断言失败: %s\n" \
                    "  文件: %s\n" \
                    "  行号: %d\n" \
                    "  函数: %s\n" \
                    "  表达式: %s\n", \
                    #expr, __FILE__, __LINE__, \
                    __func__, #expr); \
                abort(); \
            } \
        } while(0)
#endif

// 带自定义消息的断言
#define ASSERT_MSG(expr, msg) \
    do { \
        if (!(expr)) { \
            fprintf(stderr, "断言失败 [%s]: %s (文件 %s, 行 %d)\n", \
                    #expr, msg, __FILE__, __LINE__); \
            abort(); \
        } \
    } while(0)

// 参数检查宏
#define CHECK_NOT_NULL(ptr) \
    ASSERT_MSG((ptr) != NULL, "指针不能为NULL")

#define CHECK_RANGE(val, min, max) \
    ASSERT_MSG((val) >= (min) && (val) <= (max), \
               "值超出有效范围")

int main(void) {
    int x = 42;
    MY_ASSERT(x > 0);
    MY_ASSERT(x < 100);

    int *ptr = &x;
    CHECK_NOT_NULL(ptr);
    CHECK_RANGE(x, 0, 100);

    printf("所有断言通过\n");

    return 0;
}
```

## 6. 断言的最佳实践

### 6.1 适用场景

```c
#include <assert.h>
#include <stdlib.h>

// 好的用法：检查函数前置条件
int array_get(int *arr, int len, int index) {
    assert(arr != NULL);       // 前置条件
    assert(index >= 0);        // 前置条件
    assert(index < len);       // 前置条件
    return arr[index];
}

// 好的用法：检查内部不变量
int binary_search(int *arr, int len, int target) {
    assert(arr != NULL);
    // 假设arr已排序（不变量）

    int left = 0, right = len - 1;
    while (left <= right) {
        assert(left >= 0 && right < len);  // 内部不变量

        int mid = left + (right - left) / 2;
        if (arr[mid] == target) return mid;
        if (arr[mid] < target) left = mid + 1;
        else right = mid - 1;
    }
    return -1;
}

// 好的用法：检查后置条件
int factorial(int n) {
    assert(n >= 0);

    int result = 1;
    for (int i = 2; i <= n; i++) {
        result *= i;
        assert(result > 0);  // 检查是否溢出
    }

    assert(result >= 1);  // 后置条件
    return result;
}
```

### 6.2 不应该使用断言的情况

```c
#include <assert.h>
#include <stdio.h>
#include <errno.h>

// 错误：不要用断言处理用户输入
// 用户输入不是编程错误
void bad_example(const char *filename) {
    // 错误做法
    // FILE *fp = fopen(filename, "r");
    // assert(fp != NULL);  // 用户可能输入不存在的文件名

    // 正确做法：使用错误处理
    FILE *fp = fopen(filename, "r");
    if (fp == NULL) {
        perror("打开文件失败");
        return;
    }
    fclose(fp);
}

// 错误：不要在断言中有副作用
void bad_side_effect(int x) {
    // 错误：Release模式下assert被移除，++x不会执行
    // assert(++x > 0);

    // 正确：副作用放在断言外面
    x++;
    assert(x > 0);
}

// 错误：不要用断言检查不应该发生的运行时错误
void bad_runtime_check(void) {
    // 内存分配失败是运行时错误，不是程序bug
    // int *p = malloc(1000000000);
    // assert(p != NULL);

    // 正确做法
    int *p = malloc(1000000000);
    if (p == NULL) {
        fprintf(stderr, "内存分配失败\n");
        exit(EXIT_FAILURE);
    }
    free(p);
}
```

## 7. 重要注意事项

> **要点一**：断言用于捕捉编程错误（bugs），不应用于处理正常的运行时错误。

> **要点二**：定义 `NDEBUG` 后断言被移除，因此不要在断言表达式中放置有副作用的操作。

> **要点三**：`_Static_assert`（C11）在编译时检查，失败时产生编译错误，不影响运行时性能。

> **要点四**：断言失败时调用 `abort()` 终止程序，不会执行清理操作。

> **要点五**：在开发阶段使用断言，在Release版本中禁用（定义 `NDEBUG`）。

> **要点六**：断言检查的应该是"不应该发生"的情况，而不是用户输入错误等正常错误。

> **要点七**：好的断言文档化了程序的假设条件，是代码自文档化的一部分。

> **要点八**：`static_assert` 在 C11 中需要包含 `<assert.h>` 或 `<stdalign.h>` 才能使用宏形式。
