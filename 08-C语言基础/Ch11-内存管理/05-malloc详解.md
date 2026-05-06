# malloc详解

## 一、函数原型与基本用法

```c
#include <stdlib.h>
void *malloc(size_t size);
```

`malloc`是C语言中最基本的动态内存分配函数，它在堆上分配指定大小的连续内存块。

### 1.1 参数说明

| 参数 | 类型 | 说明 |
|------|------|------|
| `size` | `size_t` | 要分配的字节数 |

### 1.2 返回值

| 返回值 | 说明 |
|--------|------|
| 成功 | 返回指向已分配内存的`void*`指针 |
| 失败 | 返回`NULL`（内存不足时） |

## 二、基本使用模式

### 2.1 分配单个变量

```c
#include <stdio.h>
#include <stdlib.h>

void allocate_single_variable(void) {
    // 分配一个int大小的内存
    int *ptr = (int *)malloc(sizeof(int));

    // 必须检查返回值
    if (ptr == NULL) {
        fprintf(stderr, "malloc分配失败\n");
        return;
    }

    // 使用分配的内存
    *ptr = 42;
    printf("值: %d\n", *ptr);

    // 释放内存
    free(ptr);
    ptr = NULL;
}

int main(void) {
    allocate_single_variable();
    return 0;
}
```

### 2.2 分配数组

```c
#include <stdio.h>
#include <stdlib.h>

void allocate_array(void) {
    int count = 100;

    // 分配100个int大小的内存
    int *arr = (int *)malloc(sizeof(int) * count);
    if (arr == NULL) {
        fprintf(stderr, "malloc分配失败\n");
        return;
    }

    // 注意：malloc不初始化内存，内容是未定义的
    // 需要手动初始化
    for (int i = 0; i < count; i++) {
        arr[i] = i * i;
    }

    // 打印部分结果
    for (int i = 0; i < 10; i++) {
        printf("arr[%d] = %d\n", i, arr[i]);
    }

    free(arr);
    arr = NULL;
}

int main(void) {
    allocate_array();
    return 0;
}
```

### 2.3 分配结构体

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[50];
    double score;
} Student;

void allocate_struct(void) {
    // 分配单个结构体
    Student *stu = (Student *)malloc(sizeof(Student));
    if (stu == NULL) {
        fprintf(stderr, "分配学生结构体失败\n");
        return;
    }

    // 使用 -> 运算符访问成员
    stu->id = 1001;
    strcpy(stu->name, "张三");
    stu->score = 95.5;

    printf("学号: %d\n", stu->id);
    printf("姓名: %s\n", stu->name);
    printf("成绩: %.1f\n", stu->score);

    free(stu);
    stu = NULL;

    // 分配结构体数组
    int n = 5;
    Student *students = (Student *)malloc(sizeof(Student) * n);
    if (students == NULL) {
        fprintf(stderr, "分配学生数组失败\n");
        return;
    }

    for (int i = 0; i < n; i++) {
        students[i].id = 1001 + i;
        sprintf(students[i].name, "学生%d", i + 1);
        students[i].score = 60.0 + i * 10;
    }

    for (int i = 0; i < n; i++) {
        printf("[%d] %s - %.1f\n",
               students[i].id, students[i].name, students[i].score);
    }

    free(students);
    students = NULL;
}

int main(void) {
    allocate_struct();
    return 0;
}
```

## 三、正确使用malloc的最佳实践

### 3.1 安全的分配模式

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 推荐模式：使用 sizeof(*ptr) 而非 sizeof(Type)
void safe_allocation_pattern(void) {
    // 好的做法：sizeof(*arr) 自动适配类型
    int *arr = malloc(sizeof(*arr) * 100);
    // 即使 arr 的类型改变，这里也不需要修改

    if (arr == NULL) {
        perror("malloc");
        return;
    }

    // 使用 memset 清零初始化
    memset(arr, 0, sizeof(*arr) * 100);

    free(arr);
    arr = NULL;
}

// 推荐：封装安全的分配函数
void *safe_malloc(size_t size) {
    void *ptr = malloc(size);
    if (ptr == NULL) {
        fprintf(stderr, "内存分配失败: 请求 %zu 字节\n", size);
        exit(EXIT_FAILURE);  // 或返回NULL让调用者处理
    }
    return ptr;
}

// 使用示例
void use_safe_malloc(void) {
    int *data = safe_malloc(sizeof(*data) * 1000);
    // 如果分配失败，程序已退出，不需要检查NULL

    for (int i = 0; i < 1000; i++) {
        data[i] = i;
    }

    free(data);
    data = NULL;
}

int main(void) {
    safe_allocation_pattern();
    use_safe_malloc();
    return 0;
}
```

### 3.2 处理分配失败

```c
#include <stdio.h>
#include <stdlib.h>

// 策略1：检查并优雅降级
int strategy_check_and_degrade(void) {
    int *large_buf = malloc(1024 * 1024 * 100);  // 尝试100MB
    if (large_buf == NULL) {
        // 尝试更小的缓冲区
        fprintf(stderr, "大缓冲区分配失败，尝试小缓冲区\n");
        large_buf = malloc(1024 * 1024);  // 尝试1MB
        if (large_buf == NULL) {
            fprintf(stderr, "小缓冲区也失败\n");
            return -1;
        }
    }

    // 使用缓冲区...
    free(large_buf);
    return 0;
}

// 策略2：重试机制
void *retry_malloc(size_t size, int max_retries) {
    void *ptr = NULL;
    for (int i = 0; i < max_retries; i++) {
        ptr = malloc(size);
        if (ptr != NULL) return ptr;

        fprintf(stderr, "第 %d 次分配失败，重试...\n", i + 1);
        // 可选：等待一段时间让系统释放内存
    }
    return NULL;
}

// 策略3：分配失败的清理
int allocate_with_cleanup(void) {
    int *a = NULL, *b = NULL, *c = NULL;

    a = malloc(sizeof(*a) * 100);
    if (a == NULL) goto cleanup;

    b = malloc(sizeof(*b) * 200);
    if (b == NULL) goto cleanup;

    c = malloc(sizeof(*c) * 300);
    if (c == NULL) goto cleanup;

    // 使用 a, b, c ...
    printf("全部分配成功\n");

cleanup:
    // 统一清理，free(NULL)是安全的
    free(a);
    free(b);
    free(c);
    return (c != NULL) ? 0 : -1;
}

int main(void) {
    strategy_check_and_degrade();
    allocate_with_cleanup();
    return 0;
}
```

## 四、malloc的内部行为

### 4.1 内存对齐

```c
#include <stdio.h>
#include <stdlib.h>
#include <stdalign.h>

void demonstrate_alignment(void) {
    // malloc返回的指针满足最大对齐要求
    char *p1 = malloc(1);
    int  *p2 = malloc(sizeof(int));
    double *p3 = malloc(sizeof(double));

    printf("p1地址: %p (mod 8 = %zu)\n", (void *)p1, (size_t)p1 % 8);
    printf("p2地址: %p (mod 8 = %zu)\n", (void *)p2, (size_t)p2 % 8);
    printf("p3地址: %p (mod 8 = %zu)\n", (void *)p3, (size_t)p3 % 8);
    // 通常都是8字节对齐的

    free(p1);
    free(p2);
    free(p3);
}

int main(void) {
    demonstrate_alignment();
    return 0;
}
```

### 4.2 实际分配大小

```c
#include <stdio.h>
#include <stdlib.h>

void actual_allocation_size(void) {
    // malloc可能分配比请求更大的内存（用于头部信息和对齐）
    // 但我们只能使用请求的大小

    size_t requested = 3;
    char *ptr = malloc(requested);

    if (ptr) {
        // 只能安全使用 requested 字节
        ptr[0] = 'A';
        ptr[1] = 'B';
        ptr[2] = 'C';
        // ptr[3] = 'D';  // 越界！即使实际分配了更多

        printf("请求: %zu 字节\n", requested);
        printf("地址: %p\n", (void *)ptr);

        free(ptr);
    }
}

int main(void) {
    actual_allocation_size();
    return 0;
}
```

## 五、malloc与类型安全

```c
#include <stdio.h>
#include <stdlib.h>

// C++中需要强制转换，C中不需要（但加上更清晰）
void type_casting_examples(void) {
    // C语言中 malloc 返回 void*，可以自动转换
    int *arr1 = malloc(sizeof(int) * 10);        // C合法
    int *arr2 = (int *)malloc(sizeof(int) * 10); // 更清晰的写法

    // 推荐使用 sizeof(*ptr) 模式
    double *dbl = malloc(sizeof(*dbl) * 5);
    // 等价于 double *dbl = malloc(sizeof(double) * 5);

    // 结构体
    typedef struct { int x; int y; } Point;
    Point *pts = malloc(sizeof(*pts) * 10);

    free(arr1);
    free(arr2);
    free(dbl);
    free(pts);
}

int main(void) {
    type_casting_examples();
    return 0;
}
```

## 六、malloc(0) 的行为

```c
#include <stdio.h>
#include <stdlib.h>

void malloc_zero_behavior(void) {
    // malloc(0) 是合法的，但行为由实现定义
    // 可能返回NULL，也可能返回非NULL指针
    // 不能解引用返回的指针
    char *ptr = malloc(0);

    if (ptr == NULL) {
        printf("malloc(0) 返回 NULL\n");
    } else {
        printf("malloc(0) 返回非NULL: %p\n", (void *)ptr);
        // free(ptr);  // 如果非NULL，仍然需要free
    }

    // 但这种写法没有实际意义
    free(ptr);  // 安全：free(NULL) 是允许的
}

int main(void) {
    malloc_zero_behavior();
    return 0;
}
```

## 七、关键要点

> **malloc使用要点**
> 1. 总是检查malloc返回值是否为NULL
> 2. 使用`sizeof(*ptr)`模式避免类型不匹配
> 3. malloc不初始化内存，内容是未定义的
> 4. 分配失败时需要有合适的错误处理策略
> 5. malloc(0)是合法的但行为实现定义
> 6. 返回的指针满足最大对齐要求
> 7. 只能访问请求大小范围内的内存
> 8. 配合free使用，防止内存泄漏
