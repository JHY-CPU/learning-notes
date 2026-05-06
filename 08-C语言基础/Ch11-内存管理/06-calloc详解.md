# calloc详解

## 一、函数原型

```c
#include <stdlib.h>
void *calloc(size_t nmemb, size_t size);
```

`calloc`（contiguous allocation）分配一块连续内存，用于存储`nmemb`个大小为`size`的元素，并将所有字节**初始化为零**。

### 1.1 参数与返回值

| 参数 | 类型 | 说明 |
|------|------|------|
| `nmemb` | `size_t` | 元素个数 |
| `size` | `size_t` | 每个元素的大小(字节) |

| 返回值 | 说明 |
|--------|------|
| 成功 | 返回指向已分配并清零内存的指针 |
| 失败 | 返回`NULL` |

## 二、calloc vs malloc

### 2.1 核心区别

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void compare_malloc_calloc(void) {
    size_t count = 10;

    // malloc: 分配但不初始化
    int *arr_malloc = (int *)malloc(sizeof(int) * count);

    // calloc: 分配并清零
    int *arr_calloc = (int *)calloc(count, sizeof(int));

    printf("malloc 分配后的内容(未初始化):\n");
    for (size_t i = 0; i < count; i++) {
        printf("  arr_malloc[%zu] = %d\n", i, arr_malloc[i]);
        // 未定义的垃圾值
    }

    printf("\ncalloc 分配后的内容(已清零):\n");
    for (size_t i = 0; i < count; i++) {
        printf("  arr_calloc[%zu] = %d\n", i, arr_calloc[i]);
        // 全部为0
    }

    // 等价写法：malloc + memset
    int *arr_manual = (int *)malloc(sizeof(int) * count);
    if (arr_manual) {
        memset(arr_manual, 0, sizeof(int) * count);
    }

    free(arr_malloc);
    free(arr_calloc);
    free(arr_manual);
}

int main(void) {
    compare_malloc_calloc();
    return 0;
}
```

### 2.2 功能对比表

| 特性 | malloc | calloc |
|------|--------|--------|
| 初始化 | 不初始化(垃圾值) | 全部清零 |
| 参数 | 总字节数 | 元素个数 + 元素大小 |
| 溢出检查 | 无自动检查 | 可能检查乘法溢出 |
| 性能 | 稍快(不初始化) | 稍慢(需清零) |
| 返回值 | void* | void* |

## 三、calloc的典型用法

### 3.1 数组初始化

```c
#include <stdio.h>
#include <stdlib.h>

void array_initialization(void) {
    int count = 100;

    // 分配整型数组，所有元素自动初始化为0
    int *scores = (int *)calloc(count, sizeof(int));
    if (scores == NULL) {
        fprintf(stderr, "分配失败\n");
        return;
    }

    // 无需手动初始化，直接使用
    for (int i = 0; i < count; i++) {
        printf("scores[%d] = %d\n", i, scores[i]);  // 全是0
    }

    free(scores);
}

int main(void) {
    array_initialization();
    return 0;
}
```

### 3.2 字符串缓冲区

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

void string_buffer(void) {
    size_t buf_size = 256;

    // calloc 自动包含 '\0' 终止
    char *buffer = (char *)calloc(buf_size, sizeof(char));
    if (buffer == NULL) {
        fprintf(stderr, "缓冲区分配失败\n");
        return;
    }

    // buffer 已经全部是 '\0'，安全地使用
    strcpy(buffer, "Hello, calloc!");
    printf("内容: %s\n", buffer);

    // 检查剩余部分是否为0
    printf("buffer[15] = %d (应该是0)\n", buffer[15]);

    free(buffer);
}

int main(void) {
    string_buffer();
    return 0;
}
```

### 3.3 结构体数组

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int x;
    int y;
    int z;
} Point3D;

void struct_array(void) {
    int n = 5;

    // 分配结构体数组，所有成员自动初始化为0
    Point3D *points = (Point3D *)calloc(n, sizeof(Point3D));
    if (points == NULL) {
        fprintf(stderr, "分配失败\n");
        return;
    }

    // 所有成员都是0
    for (int i = 0; i < n; i++) {
        printf("point[%d] = (%d, %d, %d)\n",
               i, points[i].x, points[i].y, points[i].z);
    }

    free(points);
}

int main(void) {
    struct_array();
    return 0;
}
```

## 四、calloc的溢出保护

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>

void overflow_protection(void) {
    // calloc 可能检测到乘法溢出
    // 如果 nmemb * size 超过 SIZE_MAX，calloc可能返回NULL

    // 这种情况会溢出
    size_t huge = SIZE_MAX / 2;
    int *ptr = (int *)calloc(huge, sizeof(int));
    if (ptr == NULL) {
        printf("calloc 检测到溢出，返回NULL\n");
    } else {
        free(ptr);
    }

    // 正常情况
    int *safe = (int *)calloc(100, sizeof(int));
    if (safe != NULL) {
        printf("正常分配成功\n");
        free(safe);
    }
}

int main(void) {
    overflow_protection();
    return 0;
}
```

## 五、calloc的内部实现思路

```c
// calloc 的概念实现
void *conceptual_calloc(size_t nmemb, size_t size) {
    // 检查乘法溢出
    size_t total = nmemb * size;
    if (nmemb != 0 && total / nmemb != size) {
        return NULL;  // 溢出
    }

    // 分配内存
    void *ptr = malloc(total);
    if (ptr == NULL) {
        return NULL;
    }

    // 清零
    memset(ptr, 0, total);

    return ptr;
}
```

## 六、何时使用calloc

```c
#include <stdio.h>
#include <stdlib.h>

// 适合使用calloc的场景
void when_to_use_calloc(void) {
    // 场景1：需要零初始化的数组
    int *counters = calloc(100, sizeof(*counters));

    // 场景2：位图/标志数组
    unsigned char *bitmap = calloc(1024, sizeof(*bitmap));

    // 场景3：安全敏感的数据（避免信息泄漏）
    char *secret_buf = calloc(256, sizeof(*secret_buf));

    // 场景4：需要初始值为0的统计数组
    double *sums = calloc(10, sizeof(*sums));

    free(counters);
    free(bitmap);
    free(secret_buf);
    free(sums);
}

// 适合使用malloc的场景
void when_to_use_malloc(void) {
    // 场景1：马上会覆盖全部数据
    int *data = malloc(100 * sizeof(*data));
    for (int i = 0; i < 100; i++) {
        data[i] = compute_value(i);  // 全部覆盖
    }

    // 场景2：性能敏感且不需要初始化
    // malloc跳过清零步骤，稍微更快

    free(data);
}

int main(void) {
    when_to_use_calloc();
    when_to_use_malloc();
    return 0;
}
```

## 七、关键要点

> **calloc使用要点**
> 1. calloc将分配的内存全部初始化为零
> 2. 参数是(元素个数, 元素大小)，自动计算总字节数
> 3. 对于需要零初始化的场景，calloc比malloc+memset更简洁
> 4. calloc可能检测乘法溢出，更安全
> 5. calloc比malloc稍慢（多了一个清零操作）
> 6. 同样需要检查返回值，同样需要free释放
> 7. 对于浮点数，零初始化得到的是0.0（IEEE 754标准）
