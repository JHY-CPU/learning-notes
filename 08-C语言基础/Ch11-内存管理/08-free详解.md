# free详解

## 一、函数原型

```c
#include <stdlib.h>
void free(void *ptr);
```

`free`用于释放先前通过`malloc`、`calloc`或`realloc`分配的内存，将其归还给堆管理器。

### 1.1 参数说明

| 参数 | 说明 |
|------|------|
| `ptr` | 指向先前分配的内存块的指针，如果为NULL则函数不做任何操作 |

## 二、free的基本用法

### 2.1 标准释放流程

```c
#include <stdio.h>
#include <stdlib.h>

void basic_free_example(void) {
    // 分配
    int *data = (int *)malloc(sizeof(int) * 100);
    if (data == NULL) {
        fprintf(stderr, "分配失败\n");
        return;
    }

    // 使用
    for (int i = 0; i < 100; i++) {
        data[i] = i;
    }
    printf("data[99] = %d\n", data[99]);

    // 释放
    free(data);

    // 置NULL防止悬空指针
    data = NULL;

    // free(NULL)是安全的，不会有任何效果
    free(data);  // 安全，什么都不做
}

int main(void) {
    basic_free_example();
    return 0;
}
```

### 2.2 free(NULL)的安全性

```c
#include <stdio.h>
#include <stdlib.h>

void free_null_safety(void) {
    int *ptr = NULL;

    // free(NULL) 是完全安全的，C标准保证不做任何操作
    free(ptr);  // 合法，安全

    // 常见的防御性编程模式
    // 无需检查 ptr != NULL
    // free 内部会自动处理 NULL

    // 条件释放
    int *maybe_alloc = NULL;
    int should_allocate = 0;

    if (should_allocate) {
        maybe_alloc = malloc(100);
    }

    // 无论是否分配过，都可以安全free
    free(maybe_alloc);
}

int main(void) {
    free_null_safety();
    return 0;
}
```

## 三、free后置NULL模式

### 3.1 为什么要置NULL

```c
#include <stdio.h>
#include <stdlib.h>

void why_set_null_after_free(void) {
    int *ptr = malloc(sizeof(int) * 10);
    *ptr = 42;

    free(ptr);
    // 此时ptr是悬空指针(dangling pointer)
    // *ptr;   // 未定义行为！
    // ptr[5]; // 未定义行为！

    ptr = NULL;  // 关键步骤！

    // 现在即使误用，也会导致明显的段错误
    // 而不是难以调试的随机行为
    // *ptr;  // 段错误（立即崩溃，容易发现）
}

int main(void) {
    why_set_null_after_free();
    return 0;
}
```

### 3.2 封装安全释放宏/函数

```c
#include <stdio.h>
#include <stdlib.h>

// 安全释放宏
#define SAFE_FREE(ptr) do { \
    free(ptr);              \
    (ptr) = NULL;           \
} while(0)

// 安全释放函数（需要传入指针的指针）
void safe_free(void **ptr) {
    if (ptr != NULL && *ptr != NULL) {
        free(*ptr);
        *ptr = NULL;
    }
}

void safe_free_examples(void) {
    int *a = malloc(sizeof(int) * 100);
    char *b = malloc(256);
    double *c = malloc(sizeof(double) * 50);

    // 使用宏释放
    SAFE_FREE(a);
    // a 现在是 NULL

    // 使用函数释放
    safe_free((void **)&b);
    // b 现在是 NULL

    // 再次释放是安全的
    SAFE_FREE(a);     // free(NULL) 安全
    safe_free((void **)&b);  // free(NULL) 安全
    SAFE_FREE(c);
}

int main(void) {
    safe_free_examples();
    return 0;
}
```

## 四、free的错误用法

### 4.1 常见错误

```c
#include <stdio.h>
#include <stdlib.h>

void free_mistakes(void) {
    int *p1 = malloc(sizeof(int) * 10);
    int *p2 = p1;

    // 错误1：double free - 重复释放
    // free(p1);
    // free(p2);  // p2指向同一块内存！double free!

    // 正确：只释放一次
    free(p1);
    p1 = NULL;
    p2 = NULL;

    // 错误2：free栈上的变量
    // int stack_var = 42;
    // free(&stack_var);  // 未定义行为！

    // 错误3：free静态变量
    // static int static_var = 0;
    // free(&static_var);  // 未定义行为！

    // 错误4：free部分分配的内存
    int *arr = malloc(sizeof(int) * 100);
    // free(arr + 50);  // 必须free原始指针！
    free(arr);

    // 错误5：free后使用
    int *ptr = malloc(sizeof(int));
    *ptr = 100;
    free(ptr);
    // printf("%d\n", *ptr);  // use-after-free 未定义行为!
    ptr = NULL;
}

int main(void) {
    free_mistakes();
    return 0;
}
```

### 4.2 free与malloc配对

```c
#include <stdio.h>
#include <stdlib.h>

void free_pairing_demo(void) {
    // 规则：每malloc一次，必须free一次

    // 情况1：简单的1对1
    int *a = malloc(sizeof(int));
    free(a);  // 正确

    // 情况2：realloc后的指针
    int *b = malloc(sizeof(int) * 10);
    b = realloc(b, sizeof(int) * 20);
    free(b);  // 只free一次（realloc已处理旧的）

    // 情况3：多级指针
    int **matrix = malloc(sizeof(int *) * 5);
    for (int i = 0; i < 5; i++) {
        matrix[i] = malloc(sizeof(int) * 5);
    }
    // 先释放内层
    for (int i = 0; i < 5; i++) {
        free(matrix[i]);
    }
    // 再释放外层
    free(matrix);
}

int main(void) {
    free_pairing_demo();
    return 0;
}
```

## 五、释放后的内存状态

```c
#include <stdio.h>
#include <stdlib.h>

void after_free_state(void) {
    int *ptr = malloc(sizeof(int));
    *ptr = 42;

    printf("释放前: *ptr = %d, 地址 = %p\n", *ptr, (void *)ptr);

    free(ptr);
    // 释放后：
    // 1. ptr指针本身的值不变（仍指向原地址）
    // 2. 该地址的内存已归还系统
    // 3. 该内存内容不确定（可能被覆盖、可能保留原值）

    printf("释放后: 地址 = %p (值不确定)\n", (void *)ptr);
    // printf("%d\n", *ptr);  // 未定义行为！

    ptr = NULL;
    printf("置NULL后: ptr = %p\n", (void *)ptr);
}

int main(void) {
    after_free_state();
    return 0;
}
```

## 六、关键要点

> **free使用要点**
> 1. free(NULL)是安全的，C标准明确保证
> 2. free后立即将指针置NULL
> 3. 只能free通过malloc/calloc/realloc返回的指针
> 4. 同一内存只能free一次（避免double free）
> 5. 必须用原始分配时的指针来free（不能偏移）
> 6. free不改变指针本身的值，只是释放指向的内存
> 7. 使用SAFE_FREE宏可以自动置NULL，减少错误
