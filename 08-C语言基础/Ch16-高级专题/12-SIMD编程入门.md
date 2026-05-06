# 12 - SIMD编程入门

## 概述

SIMD（Single Instruction, Multiple Data）是一种数据级并行技术，一条指令同时处理多个数据元素。现代CPU通过SSE/AVX指令集支持SIMD，可显著提升计算密集型任务的性能。

---

## 1. SSE/AVX寄存器

| 指令集 | 寄存器宽度 | 可同时处理 |
|--------|-----------|-----------|
| SSE | 128位 | 4个float / 2个double |
| AVX | 256位 | 8个float / 4个double |
| AVX-512 | 512位 | 16个float / 8个double |

---

## 2. 使用GCC内建向量类型

```c
#include <stdio.h>
#include <string.h>

// GCC向量化扩展
typedef float v4sf __attribute__((vector_size(16)));   // 4个float
typedef float v8sf __attribute__((vector_size(32)));   // 8个float (AVX)
typedef int v4si __attribute__((vector_size(16)));     // 4个int

// 向量加法
void vector_add_gcc(float *a, float *b, float *result, int n) {
    for (int i = 0; i < n; i += 4) {
        v4sf va = *(v4sf *)&a[i];
        v4sf vb = *(v4sf *)&b[i];
        v4sf vr = va + vb;  // 编译器自动生成SIMD指令
        *(v4sf *)&result[i] = vr;
    }
}
```

---

## 3. SSE内在函数（Intrinsics）

### 需要头文件

```c
#include <xmmintrin.h>  // SSE
#include <emmintrin.h>  // SSE2
#include <pmmintrin.h>  // SSE3
#include <smmintrin.h>  // SSE4.1
#include <immintrin.h>  // AVX (包含所有)
```

### 基本操作

```c
#include <immintrin.h>
#include <stdio.h>

void sse_demo(void) {
    // 从数组加载
    float data[4] = {1.0f, 2.0f, 3.0f, 4.0f};
    __m128 vec = _mm_load_ps(data);  // 加载4个float

    // 从标量设置
    __m128 vec2 = _mm_set_ps(4.0f, 3.0f, 2.0f, 1.0f);  // 注意顺序！
    __m128 vec3 = _mm_set1_ps(2.0f);  // 所有元素设为2.0

    // 算术运算
    __m128 sum = _mm_add_ps(vec, vec2);   // 加法
    __m128 mul = _mm_mul_ps(vec, vec3);   // 乘法
    __m128 sub = _mm_sub_ps(vec, vec2);   // 减法
    __m128 div = _mm_div_ps(vec, vec3);   // 除法

    // 存回数组
    float result[4];
    _mm_store_ps(result, sum);
    printf("结果: %.1f %.1f %.1f %.1f\n",
           result[0], result[1], result[2], result[3]);

    // 比较
    __m128 cmp = _mm_cmplt_ps(vec, vec2);  // vec < vec2
    int mask = _mm_movemask_ps(cmp);       // 提取比较结果

    // 水平操作
    __m128 hadd = _mm_hadd_ps(vec, vec);   // 水平加法
}
```

---

## 4. AVX内在函数

```c
#include <immintrin.h>

void avx_demo(void) {
    // AVX: 256位，8个float
    float data[8] = {1,2,3,4,5,6,7,8};
    __m256 vec = _mm256_load_ps(data);

    __m256 vec2 = _mm256_set1_ps(2.0f);
    __m256 result = _mm256_add_ps(vec, vec2);

    float output[8];
    _mm256_store_ps(output, result);

    for (int i = 0; i < 8; i++)
        printf("%.1f ", output[i]);
    printf("\n");
}
```

---

## 5. SIMD向量求和示例

```c
#include <immintrin.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// 标量版本
float sum_scalar(float *arr, int n) {
    float sum = 0;
    for (int i = 0; i < n; i++)
        sum += arr[i];
    return sum;
}

// SSE版本
float sum_sse(float *arr, int n) {
    __m128 sum_vec = _mm_setzero_ps();
    int i;
    for (i = 0; i <= n - 4; i += 4) {
        __m128 vec = _mm_loadu_ps(&arr[i]);  // 未对齐加载
        sum_vec = _mm_add_ps(sum_vec, vec);
    }
    // 水平求和
    float result[4];
    _mm_store_ps(result, sum_vec);
    float sum = result[0] + result[1] + result[2] + result[3];
    // 处理剩余元素
    for (; i < n; i++)
        sum += arr[i];
    return sum;
}

// AVX版本
float sum_avx(float *arr, int n) {
    __m256 sum_vec = _mm256_setzero_ps();
    int i;
    for (i = 0; i <= n - 8; i += 8) {
        __m256 vec = _mm256_loadu_ps(&arr[i]);
        sum_vec = _mm256_add_ps(sum_vec, vec);
    }
    // 水平求和
    float result[8] __attribute__((aligned(32)));
    _mm256_store_ps(result, sum_vec);
    float sum = 0;
    for (int j = 0; j < 8; j++) sum += result[j];
    for (; i < n; i++) sum += arr[i];
    return sum;
}

// 性能测试
void benchmark(void) {
    const int N = 10000000;
    float *arr = (float *)aligned_alloc(32, N * sizeof(float));
    for (int i = 0; i < N; i++)
        arr[i] = (float)(rand() % 100) / 100.0f;

    clock_t start;

    start = clock();
    float s1 = sum_scalar(arr, N);
    printf("标量: %.2f, 耗时: %.3fms\n", s1,
           (double)(clock() - start) / CLOCKS_PER_SEC * 1000);

    start = clock();
    float s2 = sum_sse(arr, N);
    printf("SSE:  %.2f, 耗时: %.3fms\n", s2,
           (double)(clock() - start) / CLOCKS_PER_SEC * 1000);

    start = clock();
    float s3 = sum_avx(arr, N);
    printf("AVX:  %.2f, 耗时: %.3fms\n", s3,
           (double)(clock() - start) / CLOCKS_PER_SEC * 1000);

    free(arr);
}
```

---

## 6. 矩阵乘法SIMD优化

```c
// 4x4矩阵乘法（SSE优化）
void mat4x4_mul_sse(float A[4][4], float B[4][4], float C[4][4]) {
    for (int i = 0; i < 4; i++) {
        __m128 row = _mm_load_ps(A[i]);
        for (int j = 0; j < 4; j++) {
            __m128 col = _mm_set_ps(B[3][j], B[2][j], B[1][j], B[0][j]);
            __m128 prod = _mm_mul_ps(row, col);
            // 水平求和
            float result[4] __attribute__((aligned(16)));
            _mm_store_ps(result, prod);
            C[i][j] = result[0] + result[1] + result[2] + result[3];
        }
    }
}
```

---

## 7. 编译选项

```bash
# SSE支持（大多数现代CPU默认启用）
gcc -msse -msse2 -msse3 -mssse3 -msse4.1 -msse4.2 -o app app.c

# AVX支持
gcc -mavx -mavx2 -o app app.c

# 自动向量化
gcc -O2 -ftree-vectorize -mavx2 -o app app.c

# 查看向量化报告
gcc -O2 -ftree-vectorize -fopt-info-vec -mavx2 app.c

# 检查CPU支持的指令集
cat /proc/cpuinfo | grep flags
```

---

## 要点总结

1. SIMD适合数据密集的并行计算（向量运算、图像处理等）
2. SSE处理128位（4个float），AVX处理256位（8个float）
3. 使用`_mm_load_ps`要求数据16字节对齐，`_mm_loadu_ps`不要求
4. 编译器`-O2 -ftree-vectorize`可自动向量化简单循环
5. 手动SIMD优化通常比自动向量化效果更好
6. AVX-512需要更新的CPU支持
