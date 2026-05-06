# 23 - 限定符 restrict

## 一、restrict 基本概念

`restrict` 是 C99 引入的**指针限定符**，它向编译器承诺：**该指针是访问所指向对象的唯一方式**（在该指针的生命周期内）。

```c
void func(int * restrict ptr) {
    // 承诺：ptr 是访问该内存的唯一途径
}
```

`restrict` 不改变程序的行为，但它给编译器提供了**优化**的机会。

---

## 二、为什么需要 restrict

### 2.1 指针别名问题

当两个指针可能指向同一块内存（称为**别名**）时，编译器无法做某些优化：

```c
void add_arrays(int *a, int *b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}
```

编译器不确定 `a` 和 `b` 是否指向同一块内存，因此：
- 不能将 `a[i]` 和 `b[i]` 并行加载
- 不能重新排序读写操作

### 2.2 使用 restrict 后

```c
void add_arrays(int * restrict a, int * restrict b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}
```

编译器知道 `a` 和 `b` 不会指向重叠的内存，可以：
- 并行加载数据
- 重新排序指令
- 使用 SIMD 指令优化

---

## 三、restrict 的语义

### 3.1 承诺的含义

```c
int x = 10;
int * restrict p = &x;

// 承诺：在 p 的作用域内，只有通过 p 才能访问 x
// 如果有其他指针也访问 x，则违反承诺，行为未定义
```

### 3.2 违反 restrict 的后果

```c
int data[10] = {0};

void func(int * restrict a, int * restrict b) {
    *a = 10;
    *b = 20;
    // 编译器假设 a 和 b 指向不同的对象
    // 可能假设 *a 仍然是 10
    printf("%d\n", *a);  // 可能输出 10（编译器优化后的结果）
}

// 如果这样调用：
func(&data[0], &data[0]);  // 违反 restrict！行为未定义
```

---

## 四、标准库中的 restrict

C 标准库中大量使用了 `restrict`：

### 4.1 memcpy

```c
void *memcpy(void * restrict dest, const void * restrict src, size_t n);
// dest 和 src 不能重叠！
```

### 4.2 memcmp

```c
int memcmp(const void * restrict s1, const void * restrict s2, size_t n);
```

### 4.3 printf

```c
int printf(const char * restrict format, ...);
```

### 4.4 对比 memcpy 和 memmove

```c
// memcpy：使用 restrict，假设不重叠（更快）
void *memcpy(void * restrict dest, const void * restrict src, size_t n);

// memmove：不使用 restrict，允许重叠（更安全）
void *memmove(void *dest, const void *src, size_t n);
```

```c
#include <string.h>

int arr[] = {1, 2, 3, 4, 5};

// 安全：源和目标不重叠
memcpy(&arr[2], &arr[0], 2 * sizeof(int));  // {1, 2, 1, 2, 5}

// 源和目标重叠时，应该使用 memmove
memmove(&arr[1], &arr[0], 4 * sizeof(int));  // 安全
// 如果用 memcpy，结果未定义
```

---

## 五、restrict 的常见用法

### 5.1 矩阵乘法优化

```c
// 无 restrict：编译器保守生成代码
void multiply(double *a, double *b, double *c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i*n+j] += a[i*n+k] * b[k*n+j];
}

// 有 restrict：编译器可以更积极地优化
void multiply(double * restrict a, double * restrict b,
              double * restrict c, int n) {
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++)
            for (int k = 0; k < n; k++)
                c[i*n+j] += a[i*n+k] * b[k*n+j];
}
```

### 5.2 信号处理

```c
void fir_filter(const float * restrict input,
                const float * restrict coefficients,
                float * restrict output,
                int n, int taps) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < taps; j++) {
            if (i - j >= 0)
                sum += input[i - j] * coefficients[j];
        }
        output[i] = sum;
    }
}
```

---

## 六、restrict 的限制和注意事项

### 6.1 只对指针有效

```c
restrict int x;          // 错误！restrict 只能修饰指针
int * restrict p;        // 正确
```

### 6.2 不改变程序语义

如果程序在没有 `restrict` 的情况下是正确的，加上 `restrict` 后行为不变。`restrict` 只是允许编译器产生更快的代码。

### 6.3 程序员的责任

`restrict` 是程序员对编译器的**承诺**，编译器不会检查承诺是否被遵守。如果违反，会产生未定义行为。

---

## 七、restrict 与其他限定符

```c
// restrict 与 const
const int * restrict p1;       // 指向常量的 restrict 指针
int * const restrict p2;       // restrict 的常量指针
const int * const restrict p3; // 两者都是

// restrict 与 volatile
volatile int * restrict p4;    // 指向 volatile 的 restrict 指针
```

---

## 八、restrict 在函数参数中的使用

```c
// 标准库风格
void vector_add(int n,
                const double * restrict a,
                const double * restrict b,
                double * restrict result) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
}

// 调用者确保三个数组不重叠
double x[100], y[100], z[100];
vector_add(100, x, y, z);    // 正确
// vector_add(100, x, x, x); // 违反 restrict，未定义行为
```

---

## 九、要点总结

1. `restrict` 是 C99 引入的指针限定符，向编译器承诺没有别名
2. `restrict` 不改变程序语义，只允许编译器进行更积极的优化
3. 标准库函数（如 `memcpy`、`printf`）使用 `restrict` 约束参数
4. `memcpy` 要求源和目标不重叠，`memmove` 则允许重叠
5. 违反 `restrict` 承诺会导致**未定义行为**
6. `restrict` 只能修饰指针，不能修饰普通变量
7. 编译器不会验证 `restrict` 承诺是否被遵守
8. 在性能关键的数值计算代码中使用 `restrict` 可以显著提升性能
