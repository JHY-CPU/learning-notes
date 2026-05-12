# OpenMP 高级特性

## 1. 任务并行（Task Parallelism）

OpenMP 3.0 引入的 `task` 指令支持动态、非规则的任务并行，适合递归算法和不规则计算。

### 基本任务模型

```c
#include <stdio.h>
#include <omp.h>

int main() {
    #pragma omp parallel
    {
        #pragma omp single
        {
            // single 确保只有一个线程创建任务
            for (int i = 0; i < 8; i++) {
                #pragma omp task
                {
                    // 每个任务由任意空闲线程执行
                    int tid = omp_get_thread_num();
                    printf("任务%d 由线程%d 执行\n", i, tid);
                }
            }
        }
    }
    return 0;
}
```

### 递归任务——并行斐波那契

```c
#include <stdio.h>
#include <omp.h>

long fib_serial(int n) {
    if (n < 2) return n;
    return fib_serial(n - 1) + fib_serial(n - 2);
}

long fib_parallel(int n) {
    if (n < 20)  // 小问题直接串行，避免任务开销过大
        return fib_serial(n);

    long x, y;
    #pragma omp task shared(x)
    x = fib_parallel(n - 1);

    #pragma omp task shared(y)
    y = fib_parallel(n - 2);

    #pragma omp taskwait  // 等待两个子任务完成
    return x + y;
}

int main() {
    int n = 40;
    double start, end;

    start = omp_get_wtime();
    long result_s = fib_serial(n);
    end = omp_get_wtime();
    printf("串行 fib(%d) = %ld, 耗时 %.3f秒\n", n, result_s, end - start);

    start = omp_get_wtime();
    long result_p;
    #pragma omp parallel
    {
        #pragma omp single
        result_p = fib_parallel(n);
    }
    end = omp_get_wtime();
    printf("并行 fib(%d) = %ld, 耗时 %.3f秒\n", n, result_p, end - start);

    return 0;
}
```

### taskgroup——等待一组任务

```c
#pragma omp taskgroup
{
    #pragma omp task
    process_subtree(left_child);

    #pragma omp task
    process_subtree(right_child);

    // taskgroup 隐式等待组内所有任务完成
}
// 此处两个子树都已处理完毕
```

## 2. SIMD 指令

OpenMP 4.0 引入 SIMD 指令，显式告诉编译器进行向量化。

```c
#include <stdio.h>
#include <omp.h>

#define N 1000000

// simd: 提示编译器对循环进行向量化
void vector_add(double *a, double *b, double *c, int n) {
    #pragma omp simd
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// parallel for + simd 联合使用
void vector_mul_add(double *a, double *b, double *c, double factor, int n) {
    #pragma omp parallel for simd
    for (int i = 0; i < n; i++) {
        c[i] = a[i] * factor + b[i];
    }
}

// simd reduction
double simd_dot_product(double *a, double *b, int n) {
    double sum = 0.0;
    #pragma omp simd reduction(+:sum)
    for (int i = 0; i < n; i++) {
        sum += a[i] * b[i];
    }
    return sum;
}

// simdlen: 指定向量宽度
void vector_add_512(double *a, double *b, double *c, int n) {
    #pragma omp simd simdlen(8)
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];  // 每次处理8个double（AVX-512）
    }
}

int main() {
    double *a = (double *)malloc(N * sizeof(double));
    double *b = (double *)malloc(N * sizeof(double));
    double *c = (double *)malloc(N * sizeof(double));

    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
        b[i] = (N - i) * 1.0;
    }

    double start = omp_get_wtime();
    vector_add(a, b, c, N);
    printf("SIMD 向量加法: %.4f秒\n", omp_get_wtime() - start);

    double dot = simd_dot_product(a, b, N);
    printf("SIMD 点积: %.2f\n", dot);

    free(a); free(b); free(c);
    return 0;
}
```

## 3. 线程亲和性（Thread Affinity）

线程亲和性控制 OpenMP 线程与 CPU 核心的绑定关系，对 NUMA 架构尤为重要。

```c
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>

int main() {
    // 环境变量控制亲和性（运行时设置，非代码内）
    // OMP_PROC_BIND=close  —— 线程绑定到相邻核心（适合NUMA）
    // OMP_PROC_BIND=spread  —— 线程分散绑定（最大化带宽）
    // OMP_PROC_BIND=master  —— 线程绑定到主线程所在的核心

    omp_set_num_threads(8);

    #pragma omp parallel
    {
        int tid = omp_get_thread_num();

        // 查询当前线程绑定的 place
        int place = omp_get_place_num();
        int start = omp_get_place_num_procs(place);
        int *ids = (int *)malloc(start * sizeof(int));
        omp_get_place_proc_ids(place, ids);

        printf("线程%d 绑定到place %d, 处理器IDs: ", tid, place);
        for (int i = 0; i < start; i++) printf("%d ", ids[i]);
        printf("\n");
        free(ids);
    }

    return 0;
}
```

运行时设置：

```bash
OMP_PROC_BIND=close OMP_PLACES=cores ./my_program
OMP_PROC_BIND=spread OMP_PLACES=threads ./my_program
```

## 4. 嵌套并行（Nested Parallelism）

嵌套并行允许在 parallel 区内部再创建 parallel 区。

```c
#include <stdio.h>
#include <omp.h>

int main() {
    // 启用嵌套并行
    omp_set_nested(1);
    // 或使用环境变量 OMP_NESTED=true

    // 也可以设置最大活跃并行级别
    omp_set_max_active_levels(2);

    printf("最大嵌套级别: %d\n", omp_get_max_active_levels());

    #pragma omp parallel num_threads(2)
    {
        int outer_tid = omp_get_thread_num();
        printf("外层线程%d 开始\n", outer_tid);

        // 嵌套并行
        #pragma omp parallel num_threads(4)
        {
            int inner_tid = omp_get_thread_num();
            printf("  外层%d -> 内层线程%d\n", outer_tid, inner_tid);
        }

        printf("外层线程%d 结束\n", outer_tid);
    }

    return 0;
}
```

**输出示例**（共 2 x 4 = 8 个线程组合）：

```
外层线程0 开始
  外层0 -> 内层线程0
  外层0 -> 内层线程1
  外层0 -> 内层线程2
  外层0 -> 内层线程3
外层线程0 结束
外层线程1 开始
  外层1 -> 内层线程0
  ...
```

**注意事项**：
- 嵌套并行可能导致线程爆炸（总线程数 = 外层线程数 x 内层线程数）
- 大多数场景下 MPI + OpenMP 混合模型比嵌套 OpenMP 更常用
- 嵌套并行在 NUMA 系统上需要特别注意内存分配的亲和性

## 5. 取消（Cancellation）

OpenMP 4.0 支持取消操作，提前终止并行区域。

```c
#include <stdio.h>
#include <omp.h>

int main() {
    // 必须设置环境变量 OMP_CANCELLATION=true
    int found = 0;

    #pragma omp parallel
    {
        #pragma omp for
        for (int i = 0; i < 1000000; i++) {
            if (found) {
                #pragma omp cancel for  // 取消 for 循环
            }
            if (/* 找到目标 */) {
                found = 1;
                #pragma omp cancel for
            }
        }
    }

    return 0;
}
```
