# qsort 与 bsearch 详解

## 1. 概述

`qsort` 和 `bsearch` 是 `<stdlib.h>` 中提供的通用排序和查找函数。它们通过函数指针实现类型无关的操作，是C标准库中最实用的算法函数。

## 2. qsort - 快速排序

### 2.1 函数原型

```c
#include <stdlib.h>

void qsort(void *base, size_t nmemb, size_t size,
           int (*compar)(const void *, const void *));
```

| 参数 | 说明 |
|------|------|
| `base` | 数组首地址 |
| `nmemb` | 元素个数 |
| `size` | 每个元素的大小（字节） |
| `compar` | 比较函数指针 |

### 2.2 比较函数规范

```c
int compar(const void *a, const void *b);
```

- 返回值 < 0：`a` 排在 `b` 前面
- 返回值 = 0：`a` 和 `b` 相等
- 返回值 > 0：`a` 排在 `b` 后面

```c
// 整数升序比较
int compare_int_asc(const void *a, const void *b) {
    return (*(const int *)a - *(const int *)b);
}

// 整数降序比较
int compare_int_desc(const void *a, const void *b) {
    return (*(const int *)b - *(const int *)a);
}

// 安全的整数比较（避免溢出）
int compare_int_safe(const void *a, const void *b) {
    int va = *(const int *)a;
    int vb = *(const int *)b;
    if (va < vb) return -1;
    if (va > vb) return 1;
    return 0;
}

// 浮点数比较
int compare_double(const void *a, const void *b) {
    double diff = *(const double *)a - *(const double *)b;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}
```

## 3. qsort 使用示例

### 3.1 基本类型排序

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int cmp_int(const void *a, const void *b) {
    return (*(const int *)a - *(const int *)b);
}

int cmp_double(const void *a, const void *b) {
    double diff = *(const double *)a - *(const double *)b;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

int main(void) {
    // 整数数组排序
    int nums[] = {64, 34, 25, 12, 22, 11, 90};
    int n = sizeof(nums) / sizeof(nums[0]);

    qsort(nums, n, sizeof(int), cmp_int);

    printf("排序后的整数: ");
    for (int i = 0; i < n; i++) {
        printf("%d ", nums[i]);
    }
    printf("\n");

    // 浮点数数组排序
    double vals[] = {3.14, 1.41, 2.72, 0.58, 1.73};
    int m = sizeof(vals) / sizeof(vals[0]);

    qsort(vals, m, sizeof(double), cmp_double);

    printf("排序后的浮点数: ");
    for (int i = 0; i < m; i++) {
        printf("%.2f ", vals[i]);
    }
    printf("\n");

    return 0;
}
```

### 3.2 字符串排序

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// 字符串数组排序（指针数组）
int cmp_string(const void *a, const void *b) {
    return strcmp(*(const char **)a, *(const char **)b);
}

// 按字符串长度排序
int cmp_string_len(const void *a, const void *b) {
    size_t la = strlen(*(const char **)a);
    size_t lb = strlen(*(const char **)b);
    if (la < lb) return -1;
    if (la > lb) return 1;
    return 0;
}

// 不区分大小写的字符串比较
int cmp_string_nocase(const void *a, const void *b) {
    const char *sa = *(const char **)a;
    const char *sb = *(const char **)b;
    #ifdef _WIN32
    return _stricmp(sa, sb);
    #else
    return strcasecmp(sa, sb);
    #endif
}

int main(void) {
    // 字符串指针数组
    const char *fruits[] = {"banana", "Apple", "cherry", "date", "Elderberry"};
    int n = 5;

    printf("原始顺序:\n");
    for (int i = 0; i < n; i++) printf("  %s\n", fruits[i]);

    // 字典序排序
    qsort(fruits, n, sizeof(char *), cmp_string);
    printf("\n字典序排序:\n");
    for (int i = 0; i < n; i++) printf("  %s\n", fruits[i]);

    // 不区分大小写排序
    qsort(fruits, n, sizeof(char *), cmp_string_nocase);
    printf("\n不区分大小写排序:\n");
    for (int i = 0; i < n; i++) printf("  %s\n", fruits[i]);

    // 按长度排序
    qsort(fruits, n, sizeof(char *), cmp_string_len);
    printf("\n按长度排序:\n");
    for (int i = 0; i < n; i++) printf("  %s\n", fruits[i]);

    return 0;
}
```

### 3.3 结构体排序

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[32];
    double score;
} Student;

// 按分数排序（降序）
int cmp_student_score(const void *a, const void *b) {
    double diff = ((const Student *)b)->score - ((const Student *)a)->score;
    if (diff > 0) return 1;
    if (diff < 0) return -1;
    return 0;
}

// 按姓名排序
int cmp_student_name(const void *a, const void *b) {
    return strcmp(((const Student *)a)->name,
                  ((const Student *)b)->name);
}

// 多级排序：先按分数降序，再按姓名升序
int cmp_student_multi(const void *a, const void *b) {
    const Student *sa = (const Student *)a;
    const Student *sb = (const Student *)b;

    // 先比较分数
    double score_diff = sb->score - sa->score;
    if (score_diff > 0) return 1;
    if (score_diff < 0) return -1;

    // 分数相同，按姓名排序
    return strcmp(sa->name, sb->name);
}

int main(void) {
    Student students[] = {
        {1, "张三", 95.5},
        {2, "李四", 88.0},
        {3, "王五", 92.3},
        {4, "赵六", 95.5},
        {5, "钱七", 88.0},
    };
    int n = 5;

    // 按分数排序
    qsort(students, n, sizeof(Student), cmp_student_score);
    printf("按分数排序:\n");
    for (int i = 0; i < n; i++) {
        printf("  %s: %.1f\n", students[i].name, students[i].score);
    }

    // 多级排序
    qsort(students, n, sizeof(Student), cmp_student_multi);
    printf("\n多级排序(分数+姓名):\n");
    for (int i = 0; i < n; i++) {
        printf("  %s: %.1f\n", students[i].name, students[i].score);
    }

    return 0;
}
```

## 4. bsearch - 二分查找

### 4.1 函数原型

```c
#include <stdlib.h>

void *bsearch(const void *key, const void *base, size_t nmemb, size_t size,
              int (*compar)(const void *, const void *));
```

| 参数 | 说明 |
|------|------|
| `key` | 查找的目标值指针 |
| `base` | 已排序数组的首地址 |
| `nmemb` | 元素个数 |
| `size` | 每个元素的大小 |
| `compar` | 比较函数 |

**返回值**：找到则返回指向该元素的指针，否则返回 `NULL`。

### 4.2 使用示例

```c
#include <stdio.h>
#include <stdlib.h>

int cmp_int(const void *a, const void *b) {
    return (*(const int *)a - *(const int *)b);
}

int main(void) {
    int arr[] = {2, 5, 8, 12, 16, 23, 38, 56, 72, 91};
    int n = sizeof(arr) / sizeof(arr[0]);

    // 二分查找
    int key = 23;
    int *result = bsearch(&key, arr, n, sizeof(int), cmp_int);

    if (result != NULL) {
        printf("找到 %d, 位置: %ld\n", *result, result - arr);
    } else {
        printf("未找到 %d\n", key);
    }

    // 查找不存在的元素
    key = 99;
    result = bsearch(&key, arr, n, sizeof(int), cmp_int);
    printf("查找 %d: %s\n", key, result ? "找到" : "未找到");

    return 0;
}
```

### 4.3 结构体查找

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

typedef struct {
    int id;
    char name[32];
} Record;

int cmp_record_id(const void *a, const void *b) {
    return ((const Record *)a)->id - ((const Record *)b)->id;
}

int main(void) {
    Record records[] = {
        {1001, "Alice"},
        {1002, "Bob"},
        {1003, "Charlie"},
        {1004, "David"},
        {1005, "Eve"},
    };
    int n = 5;

    // 按ID查找
    Record key = {1003, ""};
    Record *found = bsearch(&key, records, n, sizeof(Record), cmp_record_id);

    if (found != NULL) {
        printf("找到: ID=%d, Name=%s\n", found->id, found->name);
    }

    // 查找不存在的
    key.id = 9999;
    found = bsearch(&key, records, n, sizeof(Record), cmp_record_id);
    printf("查找ID 9999: %s\n", found ? "找到" : "未找到");

    return 0;
}
```

## 5. 重要注意事项

> **要点一**：`bsearch` 要求数组必须已排序，通常先用 `qsort` 排序。

> **要点二**：`qsort` 和 `bsearch` 使用相同的比较函数。

> **要点三**：比较函数中对指针的强制类型转换必须与实际数据类型匹配。

> **要点四**：整数比较时要小心溢出。两个 `int` 相减的结果可能超出 `int` 范围，建议使用安全比较。

> **要点五**：`qsort` 是不稳定的排序（相等元素的相对顺序可能改变）。

> **要点六**：`bsearch` 只返回第一个匹配的元素（如果数组中有重复元素）。

> **要点七**：`qsort` 和 `bsearch` 的时间复杂度分别为 O(n log n) 和 O(log n)。

> **要点八**：比较函数必须满足严格弱序关系：自反性、反对称性、传递性。
