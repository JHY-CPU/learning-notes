# restrict 指针（C99）

## 一、什么是 restrict

`restrict` 是 C99 标准引入的指针限定符。它向编译器承诺：**该指针是访问其所指向内存的唯一方式**（在该指针的生命周期内）。

```c
void func(int *restrict ptr) {
    // 承诺：ptr 是访问这块内存的唯一指针
    // 编译器可以基于此做更激进的优化
}
```

## 二、restrict 的作用

### 编译器优化

没有 `restrict` 时，编译器必须假设不同指针可能指向同一内存（别名），因此不能随意重排或合并内存访问。

```c
// 没有 restrict：编译器必须保守
void add(int *a, int *b, int *result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
    // 编译器不确定 result 是否与 a 或 b 重叠
    // 每次循环都必须实际读写内存
}
```

```c
// 使用 restrict：编译器可以优化
void add(int *restrict a, int *restrict b, int *restrict result, int n) {
    for (int i = 0; i < n; i++) {
        result[i] = a[i] + b[i];
    }
    // 编译器知道 a、b、result 不重叠
    // 可以使用向量化、循环展开等优化
}
```

## 三、使用示例

### 示例一：memcpy vs memmove

```c
// memcpy 使用 restrict（要求不重叠）
void* my_memcpy(void *restrict dest, const void *restrict src, size_t n) {
    char *d = dest;
    const char *s = src;
    while (n--) {
        *d++ = *s++;
    }
    return dest;
}

// memmove 不使用 restrict（允许重叠）
void* my_memmove(void *dest, const void *src, size_t n) {
    char *d = dest;
    const char *s = src;
    if (d < s) {
        while (n--) *d++ = *s++;
    } else {
        d += n; s += n;
        while (n--) *--d = *--s;
    }
    return dest;
}
```

### 示例二：数组运算

```c
void scale_add(float *restrict out,
               const float *restrict a,
               const float *restrict b,
               float scale, int n) {
    for (int i = 0; i < n; i++) {
        out[i] = a[i] + scale * b[i];
    }
}
```

### 示例三：restrict 与 const 结合

```c
// 输入数组只读且不与其他指针重叠
void compute(const float *restrict src,
             float *restrict dst, int n) {
    for (int i = 0; i < n; i++) {
        dst[i] = src[i] * 2.0f;
    }
}
```

## 四、restrict 的限制与后果

### 违反 restrict 契约

```c
void process(int *restrict a, int *restrict b, int n) {
    for (int i = 0; i < n; i++) {
        a[i] += b[i];
    }
}

int main() {
    int arr[10] = {0};
    // 违反契约：a 和 b 指向同一内存
    process(arr, arr, 10);  // 未定义行为！
    return 0;
}
```

当程序员承诺指针不重叠但实际重叠时，编译器可能生成错误代码。

## 五、restrict 的应用场景

### 数值计算库

```c
// 矩阵乘法的内核函数
void matmul_kernel(const double *restrict A,
                   const double *restrict B,
                   double *restrict C,
                   int M, int N, int K) {
    for (int i = 0; i < M; i++) {
        for (int j = 0; j < N; j++) {
            double sum = 0.0;
            for (int k = 0; k < K; k++) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
}
```

### 信号处理

```c
void fir_filter(const float *restrict input,
                const float *restrict coeffs,
                float *restrict output,
                int n, int taps) {
    for (int i = 0; i < n; i++) {
        float sum = 0.0f;
        for (int j = 0; j < taps; j++) {
            if (i - j >= 0) {
                sum += input[i - j] * coeffs[j];
            }
        }
        output[i] = sum;
    }
}
```

## 六、标准库中的 restrict

C99 标准库中许多函数使用 `restrict`：

```c
void *memcpy(void *restrict dest, const void *restrict src, size_t n);
char *strcpy(char *restrict dest, const char *restrict src);
int printf(const char *restrict format, ...);
int scanf(const char *restrict format, ...);
int sprintf(char *restrict s, const char *restrict format, ...);
```

## 七、restrict 与 volatile

`restrict` 和 `volatile` 可以同时使用：

```c
// 硬件寄存器：既限制别名，又防止编译器优化掉
volatile int *restrict reg = (volatile int*)0x40021000;
```

## 八、什么时候使用 restrict

| 场景 | 是否使用 restrict |
|------|------------------|
| 指针可能指向同一内存 | 不使用 |
| 性能关键的数值计算 | 考虑使用 |
| 明确知道指针不重叠 | 可以使用 |
| 通用库函数（安全优先） | 谨慎使用 |

## 九、关键要点总结

> **核心概念**
> - restrict 向编译器承诺：指针是访问内存的唯一途径
> - 允许编译器进行更激进的优化
> - 违反承诺导致未定义行为

> **使用原则**
> - 只在确信不重叠时使用
> - 数值计算、信号处理等场景适合使用
> - 通用代码中谨慎使用
> - 与 const 结合使用效果更好
