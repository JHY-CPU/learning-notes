# realloc详解

## 一、函数原型

```c
#include <stdlib.h>
void *realloc(void *ptr, size_t size);
```

`realloc`用于调整已分配内存块的大小，可以扩大或缩小。

### 1.1 参数与返回值

| 参数 | 类型 | 说明 |
|------|------|------|
| `ptr` | `void*` | 原内存块指针（malloc/calloc/realloc返回的，或NULL） |
| `size` | `size_t` | 新的大小(字节)，0表示释放 |

| 返回值 | 说明 |
|--------|------|
| 成功(扩大/缩小) | 返回指向新内存块的指针 |
| size=0 | 可能返回NULL或非NULL（实现定义） |
| ptr=NULL | 等效于 malloc(size) |
| 失败 | 返回NULL，原内存块不变 |

## 二、realloc的三种使用模式

### 2.1 原地扩展

如果原内存块后方有足够空闲空间，realloc直接在原地扩展：

```c
#include <stdio.h>
#include <stdlib.h>

void in_place_expansion(void) {
    int *arr = malloc(sizeof(int) * 10);
    for (int i = 0; i < 10; i++) arr[i] = i;

    printf("原地址: %p\n", (void *)arr);

    // 尝试扩展到20个元素
    int *new_arr = realloc(arr, sizeof(int) * 20);

    if (new_arr != NULL) {
        arr = new_arr;
        printf("新地址: %p\n", (void *)arr);

        // 如果地址相同，说明是原地扩展
        // 如果不同，realloc已经拷贝了数据
        // 原有数据(arr[0]-arr[9])保留完好
        for (int i = 0; i < 10; i++) {
            printf("arr[%d] = %d\n", i, arr[i]);
        }

        // 可以使用新扩展的空间
        for (int i = 10; i < 20; i++) {
            arr[i] = i * 10;
        }
    }

    free(arr);
}

int main(void) {
    in_place_expansion();
    return 0;
}
```

### 2.2 新分配并拷贝

如果原地无法扩展，realloc会：
1. 分配新的更大的内存块
2. 将原数据拷贝到新块
3. 释放原内存块
4. 返回新块指针

```c
#include <stdio.h>
#include <stdlib.h>

void new_allocation_copy(void) {
    int *arr = malloc(sizeof(int) * 10);
    for (int i = 0; i < 10; i++) arr[i] = i * 10;

    int *old_addr = arr;

    // realloc可能需要移动到新位置
    int *new_arr = realloc(arr, sizeof(int) * 1000000);
    if (new_arr == NULL) {
        fprintf(stderr, "扩展失败\n");
        free(arr);  // 原内存仍然有效
        return;
    }

    arr = new_arr;

    printf("原地址: %p\n", (void *)old_addr);
    printf("新地址: %p\n", (void *)arr);
    printf("地址是否变化: %s\n", (old_addr != arr) ? "是(已拷贝)" : "否(原地扩展)");

    // 数据完整性验证
    for (int i = 0; i < 10; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);  // 应该是 0, 10, 20...90
    }

    free(arr);
}

int main(void) {
    new_allocation_copy();
    return 0;
}
```

### 2.3 缩小内存块

```c
#include <stdio.h>
#include <stdlib.h>

void shrink_memory(void) {
    int *arr = malloc(sizeof(int) * 1000);
    for (int i = 0; i < 1000; i++) arr[i] = i;

    printf("原大小: 1000个int, 地址: %p\n", (void *)arr);

    // 缩小到100个int
    int *smaller = realloc(arr, sizeof(int) * 100);
    if (smaller != NULL) {
        arr = smaller;
        printf("缩小后地址: %p\n", (void *)arr);

        // 前100个数据保留
        for (int i = 0; i < 10; i++) {
            printf("arr[%d] = %d\n", i, arr[i]);
        }
        // 后900个数据可能被释放（不可访问）
    }

    free(arr);
}

int main(void) {
    shrink_memory();
    return 0;
}
```

## 三、realloc的正确使用模式

### 3.1 安全的realloc模式

```c
#include <stdio.h>
#include <stdlib.h>

void safe_realloc_pattern(void) {
    int *arr = malloc(sizeof(*arr) * 10);
    if (arr == NULL) return;

    for (int i = 0; i < 10; i++) arr[i] = i;

    // 正确模式：使用临时指针
    int *temp = realloc(arr, sizeof(*arr) * 20);
    if (temp == NULL) {
        // realloc失败，原内存arr仍然有效
        fprintf(stderr, "扩展失败，保留原数组\n");
        free(arr);
        return;
    }
    arr = temp;  // 成功才更新原指针

    // 使用扩展后的数组
    for (int i = 10; i < 20; i++) {
        arr[i] = i * 2;
    }

    free(arr);
    arr = NULL;
}

int main(void) {
    safe_realloc_pattern();
    return 0;
}
```

### 3.2 错误的realloc模式

```c
#include <stdio.h>
#include <stdlib.h>

void wrong_realloc_pattern(void) {
    int *arr = malloc(sizeof(int) * 10);

    // 错误！如果realloc失败，arr指向的内存已被释放且无法找回
    // arr = realloc(arr, sizeof(int) * 20);
    // if (arr == NULL) {
    //     // 此时原内存已丢失！内存泄漏！
    //     return;
    // }

    // 正确做法总是使用临时变量
    int *temp = realloc(arr, sizeof(int) * 20);
    if (temp != NULL) {
        arr = temp;
    }

    free(arr);
}

int main(void) {
    wrong_realloc_pattern();
    return 0;
}
```

## 四、动态数组实现

### 4.1 自动增长的数组

```c
#include <stdio.h>
#include <stdlib.h>

typedef struct {
    int *data;
    size_t size;      // 当前元素数
    size_t capacity;  // 总容量
} IntVector;

// 初始化
IntVector *vector_create(size_t initial_capacity) {
    IntVector *vec = malloc(sizeof(*vec));
    if (!vec) return NULL;

    vec->data = malloc(sizeof(*vec->data) * initial_capacity);
    if (!vec->data) {
        free(vec);
        return NULL;
    }

    vec->size = 0;
    vec->capacity = initial_capacity;
    return vec;
}

// 添加元素（自动扩容）
int vector_push(IntVector *vec, int value) {
    if (vec->size >= vec->capacity) {
        // 容量翻倍策略
        size_t new_cap = vec->capacity * 2;
        int *temp = realloc(vec->data, sizeof(*vec->data) * new_cap);
        if (temp == NULL) return -1;

        vec->data = temp;
        vec->capacity = new_cap;
        printf("  扩容到 %zu\n", new_cap);
    }

    vec->data[vec->size++] = value;
    return 0;
}

// 释放
void vector_free(IntVector *vec) {
    if (vec) {
        free(vec->data);
        free(vec);
    }
}

int main(void) {
    IntVector *vec = vector_create(4);
    if (!vec) {
        fprintf(stderr, "创建失败\n");
        return 1;
    }

    printf("添加元素（容量自动增长）:\n");
    for (int i = 0; i < 20; i++) {
        if (vector_push(vec, i * 10) != 0) {
            fprintf(stderr, "添加失败\n");
            break;
        }
    }

    printf("\n数组内容 (%zu/%zu):\n", vec->size, vec->capacity);
    for (size_t i = 0; i < vec->size; i++) {
        printf("  vec[%zu] = %d\n", i, vec->data[i]);
    }

    vector_free(vec);
    return 0;
}
```

## 五、realloc(NULL) 和 realloc(ptr, 0)

```c
#include <stdio.h>
#include <stdlib.h>

void realloc_edge_cases(void) {
    // realloc(NULL, size) 等价于 malloc(size)
    int *p1 = realloc(NULL, sizeof(int) * 10);
    if (p1) {
        printf("realloc(NULL, 40) 成功\n");
        p1[0] = 42;
        printf("p1[0] = %d\n", p1[0]);
    }

    // realloc(ptr, 0) 行为实现定义
    // 一些实现等价于 free(ptr) 并返回 NULL
    // 另一些可能返回非NULL但不能使用
    // 为了可移植性，不要依赖此行为
    // 建议直接使用 free() 来释放内存

    free(p1);
}

int main(void) {
    realloc_edge_cases();
    return 0;
}
```

## 六、性能考虑

```c
#include <stdio.h>
#include <stdlib.h>

// 增长策略对比
void growth_strategies(void) {
    // 策略1：每次增加固定大小
    // 缺点：频繁realloc，每次可能需要拷贝所有数据
    // O(n)次realloc, O(n^2)总拷贝次数

    // 策略2：每次翻倍 (推荐)
    // 优点：摊销O(1)时间复杂度
    // O(log n)次realloc, O(n)总拷贝次数

    // 策略3：黄金比例增长 (~1.618倍)
    // 一些库使用此策略，平衡内存使用和性能

    printf("推荐：使用翻倍增长策略\n");
}

int main(void) {
    growth_strategies();
    return 0;
}
```

## 七、关键要点

> **realloc使用要点**
> 1. **总是使用临时指针** - `temp = realloc(ptr, size)`，检查temp后再赋值
> 2. realloc失败时原内存块保持不变（不会被释放）
> 3. 数据在扩展/缩小时自动保留
> 4. realloc(ptr, 0)行为实现定义，建议用free代替
> 5. realloc(NULL, size)等价于malloc(size)
> 6. 扩容策略推荐翻倍增长，摊销O(1)复杂度
> 7. 减小尺寸时，超出新大小的数据不可访问
