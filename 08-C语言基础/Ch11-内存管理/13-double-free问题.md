# Double-Free问题

## 一、什么是Double Free

Double Free（重复释放）是指对同一块已分配的内存调用两次或多次`free`。这会破坏堆管理器的内部数据结构，导致未定义行为。

```
Double Free 的过程:

  ptr = malloc(100);      ptr ──→ [有效内存块]
  free(ptr);              ptr ──→ [已释放，堆管理器标记为空闲]
  free(ptr);              重复释放！堆结构损坏！

  后果：
  - 堆管理结构被破坏
  - 后续 malloc/free 可能崩溃
  - 可能被利用进行安全攻击
```

## 二、常见的Double Free场景

### 2.1 直接重复释放

```c
#include <stdio.h>
#include <stdlib.h>

void direct_double_free(void) {
    int *ptr = malloc(sizeof(int));
    if (ptr == NULL) return;

    *ptr = 42;
    printf("值: %d\n", *ptr);

    free(ptr);
    // free(ptr);  // 错误！double free!

    ptr = NULL;  // 置NULL后，再次free是安全的
    free(ptr);   // free(NULL) 是安全的

    printf("安全释放完成\n");
}

int main(void) {
    direct_double_free();
    return 0;
}
```

### 2.2 多个指针指向同一内存

```c
#include <stdio.h>
#include <stdlib.h>

void multiple_pointers_double_free(void) {
    int *p1 = malloc(sizeof(int));
    if (p1 == NULL) return;

    *p1 = 100;

    int *p2 = p1;  // p2和p1指向同一块内存
    int *p3 = p1;  // p3也指向同一块

    printf("p1=%p, p2=%p, p3=%p\n", (void *)p1, (void *)p2, (void *)p3);

    free(p1);
    p1 = NULL;

    // free(p2);  // 错误！double free!
    // free(p3);  // 错误！double free!
    p2 = NULL;
    p3 = NULL;

    printf("正确：只释放一次\n");
}

int main(void) {
    multiple_pointers_double_free();
    return 0;
}
```

### 2.3 控制流导致的Double Free

```c
#include <stdio.h>
#include <stdlib.h>

// 场景：多个代码路径可能导致重复释放
void control_flow_double_free(int error_condition) {
    int *data = malloc(sizeof(int) * 100);
    if (data == NULL) return;

    int is_freed = 0;  // 跟踪释放状态

    if (error_condition) {
        free(data);
        is_freed = 1;
        // 可能直接返回
    }

    // 后续操作...
    if (!is_freed) {
        // 只在未释放时使用
        data[0] = 42;
    }

    // 错误：统一释放，不检查是否已释放
    // free(data);  // 如果error_condition为真，double free!

    // 正确做法
    if (!is_freed) {
        free(data);
    }
    data = NULL;
}

int main(void) {
    control_flow_double_free(0);
    control_flow_double_free(1);
    return 0;
}
```

### 2.4 共享资源的Double Free

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    char *buffer;
    int ref_count;
} SharedResource;

SharedResource *create_resource(const char *content) {
    SharedResource *res = malloc(sizeof(*res));
    if (!res) return NULL;

    res->buffer = malloc(256);
    if (!res->buffer) { free(res); return NULL; }

    strcpy(res->buffer, content);
    res->ref_count = 1;
    return res;
}

void add_reference(SharedResource *res) {
    if (res) res->ref_count++;
}

void release_reference(SharedResource **res) {
    if (res == NULL || *res == NULL) return;

    (*res)->ref_count--;
    if ((*res)->ref_count == 0) {
        free((*res)->buffer);
        free(*res);
        *res = NULL;
        printf("资源已完全释放\n");
    } else {
        printf("引用计数减为 %d\n", (*res)->ref_count);
    }
}

int main(void) {
    SharedResource *res = create_resource("共享数据");
    printf("引用计数: %d\n", res->ref_count);

    add_reference(res);  // 第二个引用
    printf("引用计数: %d\n", res->ref_count);

    release_reference(&res);  // ref_count = 1
    // res不为NULL，数据仍在
    if (res) printf("数据: %s\n", res->buffer);

    release_reference(&res);  // ref_count = 0，释放
    // res已为NULL

    release_reference(&res);  // free(NULL)安全，什么都不做

    return 0;
}
```

## 三、Double Free的安全影响

```c
#include <stdio.h>
#include <stdlib.h>

void security_implications(void) {
    printf("Double Free的安全影响:\n\n");

    printf("1. 堆管理结构破坏\n");
    printf("   - malloc/free使用的链表或树结构被损坏\n");
    printf("   - 后续内存操作可能写入任意地址\n\n");

    printf("2. Use-After-Free攻击\n");
    printf("   - 释放的内存被重新分配\n");
    printf("   - 攻击者可以操控数据内容\n\n");

    printf("3. 类型混淆\n");
    printf("   - 同一地址被分配给不同类型的数据\n");
    printf("   - 导致数据解释错误\n\n");

    printf("4. CVE漏洞\n");
    printf("   - Double Free是许多安全漏洞的根源\n");
    printf("   - 可能导致远程代码执行\n");
}

int main(void) {
    security_implications();
    return 0;
}
```

## 四、避免Double Free的方法

### 4.1 使用SAFE_FREE宏

```c
#include <stdio.h>
#include <stdlib.h>

#define SAFE_FREE(p) do { free(p); (p) = NULL; } while(0)

void use_safe_free(void) {
    int *a = malloc(sizeof(int) * 10);
    int *b = a;  // 共享指针

    SAFE_FREE(a);  // free + NULL
    // a = NULL, b仍指向旧地址

    SAFE_FREE(b);  // free(NULL) 安全！
    // a = NULL, b = NULL

    printf("安全释放完成\n");
}

int main(void) {
    use_safe_free();
    return 0;
}
```

### 4.2 跟踪释放状态

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    void *ptr;
    int is_freed;
} TrackedPointer;

void tracked_init(TrackedPointer *tp, void *ptr) {
    tp->ptr = ptr;
    tp->is_freed = 0;
}

void tracked_free(TrackedPointer *tp) {
    if (tp->is_freed) {
        fprintf(stderr, "警告：尝试重复释放!\n");
        return;
    }
    if (tp->ptr) {
        free(tp->ptr);
        tp->ptr = NULL;
        tp->is_freed = 1;
    }
}

int main(void) {
    TrackedPointer tp;
    tracked_init(&tp, malloc(256));

    tracked_free(&tp);    // 正常释放
    tracked_free(&tp);    // 被拦截，打印警告

    return 0;
}
```

## 五、关键要点

> **Double Free要点**
> 1. 同一块内存只能free一次
> 2. 多个指针指向同一内存时，free后所有指针都应置NULL
> 3. 使用引用计数管理共享资源
> 4. free后立即置NULL是防止double free的最有效方法
> 5. AddressSanitizer可以检测double free
> 6. Double Free可能被利用进行安全攻击
> 7. 复杂控制流中使用is_freed标志跟踪释放状态
