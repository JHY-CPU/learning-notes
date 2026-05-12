# OpenMP 编程

OpenMP（Open Multi-Processing）是基于共享内存的并行编程模型，通过编译器指令（pragma）和运行时库实现多线程并行。它是 POSIX Threads（Pthreads）的高层抽象，让开发者无需手动管理线程的创建、同步和销毁。

## 0. 底层原理

### Fork-Join 模型

OpenMP 采用 **fork-join** 并行模型：

```
主线程（Master Thread）
    |
    | 串行代码
    |
    |--- #pragma omp parallel --- fork
    |        |
    |   线程0   线程1   线程2   线程3
    |    |        |       |       |
    |   [计算]  [计算]  [计算]  [计算]   ← 并行区域
    |    |        |       |       |
    |   --- barrier / end parallel --- join
    |
    | 串行代码
    |
    V
```

1. 程序开始时只有一个主线程（master thread）
2. 遇到 `#pragma omp parallel` 时，主线程**fork**出多个子线程
3. 子线程并行执行并行区域的代码
4. 并行区域结束时，所有子线程同步（隐式 barrier），然后**join**回主线程
5. 程序继续串行执行

### 线程创建的底层实现

OpenMP 运行时库（如 libgomp、libomp）维护一个**线程池**：

- 首次遇到 `#pragma omp parallel` 时，创建 N 个线程（N = `OMP_NUM_THREADS` 或 `omp_get_num_procs()`）
- 线程创建后**不销毁**，而是进入休眠等待状态
- 后续的并行区域复用已有线程，只需唤醒和分配任务
- 这种线程池机制避免了反复创建/销毁线程的开销

在 Linux 上，OpenMP 线程实际上就是 POSIX 线程（pthreads），OpenMP 运行时库在其上层提供了更高级的抽象。

### 内存模型

OpenMP 使用**弱一致性内存模型**（Weak Consistency）：

```
线程0的视图:  x = 1   x = 2   #pragma omp barrier   读 x → 2
线程1的视图:  x = 10  x = 20  #pragma omp barrier   读 x → 可能是 2
                                   ↑
                         barrier 保证之前的所有写入对所有线程可见
```

- 在没有同步点（barrier、critical、atomic 等）时，一个线程对共享变量的修改可能不会立即被其他线程看到
- 这是因为 CPU 缓存和编译器优化可能导致变量值暂存在寄存器或本地缓存中
- OpenMP 的同步原语（barrier、flush）强制将缓存中的修改写回主存

## 1. 基本结构

OpenMP 程序以 pragma 指令为核心，编译器根据指令自动创建和管理线程。

```c
#include <stdio.h>
#include <omp.h>

int main() {
    // 设置线程数（也可以用环境变量 OMP_NUM_THREADS）
    omp_set_num_threads(4);

    // parallel 指令：以下代码块由多个线程并行执行
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("线程 %d / %d: Hello OpenMP!\n", tid, nthreads);
    }

    return 0;
}
```

**编译与运行**：

```bash
gcc -fopenmp -o hello_omp hello_omp.c
./hello_omp
# 输出（顺序不确定）：
# 线程 0 / 4: Hello OpenMP!
# 线程 2 / 4: Hello OpenMP!
# 线程 1 / 4: Hello OpenMP!
# 线程 3 / 4: Hello OpenMP!

# 通过环境变量控制线程数
OMP_NUM_THREADS=8 ./hello_omp
```

**Makefile**：

```makefile
CC = gcc
CFLAGS = -fopenmp -Wall -O2 -lm

TARGETS = hello_omp parallel_for pi_compute

all: $(TARGETS)

hello_omp: hello_omp.c
	$(CC) $(CFLAGS) -o $@ $<

parallel_for: parallel_for.c
	$(CC) $(CFLAGS) -o $@ $<

pi_compute: pi_compute.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
```

### 关键运行时函数

```c
omp_set_num_threads(n);    // 设置后续并行区域使用的线程数
int n = omp_get_num_threads();  // 获取当前并行区域的线程数
int tid = omp_get_thread_num(); // 获取当前线程的编号（0 到 n-1）
int max = omp_get_max_threads(); // 获取最大可用线程数
int procs = omp_get_num_procs(); // 获取系统 CPU 核心数
double t = omp_get_wtime();      // 获取高精度墙钟时间
double r = omp_get_wtick();      // 获取 wtime 的精度（秒）
```

## 2. parallel for 循环并行

`#pragma omp parallel for` 将紧接的 for 循环迭代分配给各线程执行。

```c
#include <stdio.h>
#include <omp.h>

int main() {
    int N = 1000000;
    double a[N], b[N], c[N];

    // 初始化
    for (int i = 0; i < N; i++) {
        a[i] = i * 1.0;
        b[i] = (N - i) * 1.0;
    }

    double start = omp_get_wtime();

    // parallel for 自动将迭代分配给线程
    #pragma omp parallel for
    for (int i = 0; i < N; i++) {
        c[i] = a[i] + b[i];
    }

    double end = omp_get_wtime();
    printf("向量加法完成，耗时 %.6f 秒\n", end - start);
    printf("验证: c[0]=%.1f, c[N-1]=%.1f\n", c[0], c[N - 1]);

    return 0;
}
```

### parallel for 的展开方式

```c
// 用户写的代码：
#pragma omp parallel for
for (int i = 0; i < 100; i++) {
    work(i);
}

// 编译器等效展开为（伪代码）：
int i;
#pragma omp parallel private(i)
{
    int tid = omp_get_thread_num();
    int nthreads = omp_get_num_threads();
    int chunk = 100 / nthreads;
    int start = tid * chunk;
    int end = (tid == nthreads - 1) ? 100 : start + chunk;
    for (i = start; i < end; i++) {
        work(i);
    }
}
```

### 循环限制

OpenMP 要求 for 循环满足以下条件：
- 循环变量必须是整数类型
- 循环条件必须是 `<`、`<=`、`>`、`>=` 形式
- 增量必须是循环不变的（每次迭代增加固定值）
- 循环次数必须在进入并行区域时确定

```c
// 合法
#pragma omp parallel for
for (int i = 0; i < N; i++) { ... }

#pragma omp parallel for
for (int i = N - 1; i >= 0; i--) { ... }

#pragma omp parallel for
for (int i = 0; i < N; i += 2) { ... }

// 非法
#pragma omp parallel for
for (int i = 0; i < N; i += variable_stride) { ... }  // 增量不是常量

#pragma omp parallel for
for (int i = 0; condition(i); i++) { ... }  // 条件形式不对
```

## 3. Reduction 归约

当多个线程需要累加到同一个变量时，使用 `reduction` 子句避免数据竞争。

```c
#include <stdio.h>
#include <omp.h>
#include <math.h>

int main() {
    int N = 10000000;
    double sum = 0.0;

    double start = omp_get_wtime();

    // reduction(+:sum) 表示每个线程维护一个 sum 的私有副本
    // 最后自动将所有副本用 + 合并
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        double x = (i + 0.5) / N;
        sum += 4.0 / (1.0 + x * x);
    }
    double pi = sum / N;

    double end = omp_get_wtime();
    printf("PI ≈ %.15f (耗时 %.4f 秒)\n", pi, end - start);

    return 0;
}
```

### 底层实现原理

```
用户代码:  reduction(+:sum)

编译器展开:
  double sum = 0;                          // 主线程的初始值
  #pragma omp parallel
  {
      double sum_private = 0;              // 每个线程的私有副本（初始值为操作的单位元）
      #pragma omp for
      for (int i = 0; i < N; i++) {
          sum_private += work(i);          // 各线程只修改自己的副本
      }
      #pragma omp atomic (或 critical)     // 同步合并
      sum += sum_private;                  // 所有副本合并回共享变量
  }
```

### 支持的操作符

| 操作符 | 初始值 | 说明 |
|--------|--------|------|
| `+` | 0 | 求和 |
| `-` | 0 | 求差（注意：实际合并行为是求和） |
| `*` | 1 | 求积 |
| `&` | ~0（全 1） | 按位与 |
| `\|` | 0 | 按位或 |
| `^` | 0 | 按位异或 |
| `&&` | 1 | 逻辑与 |
| `\|\|` | 0 | 逻辑或 |
| `max` | 最小可能值 | 求最大值 |
| `min` | 最大可能值 | 求最小值 |

### 多变量归约

```c
double sum = 0, product = 1;
int count = 0;

#pragma omp parallel for reduction(+:sum,count) reduction(*:product)
for (int i = 0; i < N; i++) {
    sum += data[i];
    product *= (1.0 + data[i] * 0.01);
    count++;
}
printf("sum=%.2f, product=%.2f, count=%d\n", sum, product, count);
```

## 4. 调度策略（Schedule）

`schedule` 子句控制迭代如何分配给线程。

### static（静态调度）

```c
// 将迭代均匀分成连续块，每个线程预先分配固定块
// 块大小 = N / num_threads
#pragma omp parallel for schedule(static)
for (int i = 0; i < N; i++) {
    // 每个线程处理连续的 N/num_threads 个迭代
}

// 指定块大小
#pragma omp parallel for schedule(static, 64)
for (int i = 0; i < N; i++) {
    // 每个线程每次取 64 个连续迭代，直到分配完
}
```

### dynamic（动态调度）

```c
// 每个线程完成一块后从队列中取下一块
// 适合迭代耗时不均匀的情况
#pragma omp parallel for schedule(dynamic, 100)
for (int i = 0; i < N; i++) {
    // 每次取 100 个迭代，完成后再取
}
```

### guided（引导调度）

```c
// 块大小从大到小递减（初始块大，后续逐渐变小）
// 平衡了初始分配效率和负载均衡
#pragma omp parallel for schedule(guided, 50)
for (int i = 0; i < N; i++) {
    // 初始块 = 剩余迭代数 / 线程数，逐渐减小到最小 50
}
```

### auto（自动调度）

```c
// 让运行时或编译器决定最优调度策略
#pragma omp parallel for schedule(auto)
for (int i = 0; i < N; i++) { ... }
```

### 调度策略选择指南

| 场景 | 推荐策略 | 原因 |
|------|---------|------|
| 均匀工作量 | static | 无调度开销，缓存友好 |
| 不均匀工作量 | dynamic | 自动负载均衡 |
| 渐变工作量 | guided | 平衡开销和均衡 |
| 不确定 | auto | 运行时自行选择 |

### 性能对比示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>

int main() {
    int N = 1000;
    int *work = (int *)malloc(N * sizeof(int));

    // 模拟不均匀的工作量（前半部分轻，后半部分重）
    for (int i = 0; i < N; i++)
        work[i] = (i < N / 2) ? rand() % 100 : rand() % 100000;

    omp_set_num_threads(4);

    // static: 可能负载不均
    double start = omp_get_wtime();
    double sum1 = 0;
    #pragma omp parallel for schedule(static) reduction(+:sum1)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < work[i]; j++) sum1 += 1.0;
    }
    printf("static:  %.4f 秒 (sum=%.0f)\n", omp_get_wtime() - start, sum1);

    // dynamic: 自动负载均衡
    start = omp_get_wtime();
    double sum2 = 0;
    #pragma omp parallel for schedule(dynamic, 10) reduction(+:sum2)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < work[i]; j++) sum2 += 1.0;
    }
    printf("dynamic: %.4f 秒 (sum=%.0f)\n", omp_get_wtime() - start, sum2);

    // guided
    start = omp_get_wtime();
    double sum3 = 0;
    #pragma omp parallel for schedule(guided, 10) reduction(+:sum3)
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < work[i]; j++) sum3 += 1.0;
    }
    printf("guided:  %.4f 秒 (sum=%.0f)\n", omp_get_wtime() - start, sum3);

    free(work);
    return 0;
}
```

## 5. critical 与 atomic

### critical 临界区

`critical` 确保同一时刻只有一个线程执行临界区代码。

```c
#include <stdio.h>
#include <omp.h>

int main() {
    int counter = 0;

    #pragma omp parallel for
    for (int i = 0; i < 100000; i++) {
        // critical 确保只有一个线程修改 counter
        #pragma omp critical
        {
            counter++;
        }
    }
    printf("counter = %d (期望 100000)\n", counter);
    return 0;
}
```

#### 底层实现

`critical` 在底层使用互斥锁（mutex）实现：

```
#pragma omp critical
{
    // 伪代码
    pthread_mutex_lock(&critical_section_mutex);
    counter++;                       // 临界区代码
    pthread_mutex_unlock(&critical_section_mutex);
}
```

因此 `critical` 的开销较大（约 100-500 纳秒每次），在热循环中应避免使用。

### atomic 原子操作

`atomic` 适用于简单操作，比 `critical` 开销更小。

```c
int sum = 0;

#pragma omp parallel for
for (int i = 0; i < 100000; i++) {
    // atomic 只适用于单条简单赋值语句
    #pragma omp atomic
    sum += i;
}
printf("sum = %d\n", sum);
```

#### atomic 的多种形式

```c
#pragma omp atomic
x++;              // read-modify-write

#pragma omp atomic
x += expr;        // update

#pragma omp atomic
x = x op expr;    // 另一种 update 写法

#pragma omp atomic capture
{ old = x; x++; } // 原子读取旧值并修改

#pragma omp atomic read
y = x;            // 原子读取

#pragma omp atomic write
x = expr;         // 原子写入
```

#### 底层实现

`atomic` 在底层直接使用 CPU 的原子指令：

```c
// #pragma omp atomic  x += val
// 底层等价于：
__atomic_add_fetch(&x, val, __ATOMIC_SEQ_CST);  // GCC 内建函数
// 或
_InterlockedAdd(&x, val);  // Windows 上的原子操作
```

原子操作的开销远小于互斥锁（约 10-50 纳秒），但只支持有限的操作类型。

### critical vs atomic 对比

| 特性 | critical | atomic |
|------|----------|--------|
| 适用范围 | 任意代码块 | 单条简单语句 |
| 开销 | 较大（~100-500 ns） | 较小（~10-50 ns） |
| 实现 | 互斥锁 | CPU 原子指令 |
| 命名 critical | 支持 | 不支持 |
| 嵌套 | 可以嵌套不同名的 critical | 不适用 |

### 命名 critical

```c
// 不同名称的 critical 互不阻塞
#pragma omp critical(region_a)
{
    update_resource_a();
}

#pragma omp critical(region_b)
{
    update_resource_b();  // 可以与 region_a 同时执行
}
```

## 6. 变量作用域

```c
int shared_var = 0;     // 默认在 parallel 区是 shared
int private_var;         // 需要显式声明为 private

#pragma omp parallel shared(shared_var) private(private_var)
{
    private_var = omp_get_thread_num();  // 每个线程有自己的副本
    // shared_var 所有线程共享，访问需小心竞争
}
```

### 子句详解

| 子句 | 说明 |
|------|------|
| `shared(x)` | x 在所有线程间共享（默认行为） |
| `private(x)` | 每个线程有自己的 x 副本，初始值未定义 |
| `firstprivate(x)` | 每个线程有自己的 x 副本，初始值来自主线程的值 |
| `lastprivate(x)` | 循环结束时，最后一个迭代的值赋给主线程变量 |
| `default(none)` | 要求显式指定所有变量的作用域（推荐） |

```c
// firstprivate 示例
int start_val = 100;
#pragma omp parallel firstprivate(start_val)
{
    start_val += omp_get_thread_num();
    // 线程0: start_val = 100, 线程1: start_val = 101, ...
}
// 主线程的 start_val 仍为 100

// lastprivate 示例
int result;
#pragma omp parallel for lastprivate(result)
for (int i = 0; i < N; i++) {
    result = compute(i);
}
// result 包含最后一次迭代 (i=N-1) 的值

// default(none) 强制显式声明
double sum = 0;
#pragma omp parallel for default(none) shared(sum, N) private(i)
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    sum += work(i);
}
```

### 常见错误：数据竞争

```c
// 错误！多个线程同时修改 sum，产生数据竞争
double sum = 0;
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    sum += data[i];  // DATA RACE! 结果不确定
}

// 修复方法1: reduction
#pragma omp parallel for reduction(+:sum)
for (int i = 0; i < N; i++) {
    sum += data[i];
}

// 修复方法2: atomic
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    #pragma omp atomic
    sum += data[i];
}
```

### 使用 ThreadSanitizer 检测数据竞争

```bash
# 编译时启用 ThreadSanitizer
gcc -fopenmp -fsanitize=thread -g -o program program.c

# 运行程序，ThreadSanitizer 会报告竞争
./program

# 输出示例：
# ==================
# WARNING: ThreadSanitizer: data race (pid=12345)
#   Write of size 8 at 0x7ffd... by thread T1:
#     #0 main._omp_fn.0 program.c:15
#   Previous write of size 8 at 0x7ffd... by thread T2:
#     #0 main._omp_fn.0 program.c:15
# ==================
```

## 7. 同步与 barrier

```c
#include <stdio.h>
#include <omp.h>

void do_work_part1() { /* ... */ }
void do_work_part2() { /* ... */ }

int main() {
    #pragma omp parallel
    {
        // 各线程执行自己的工作
        do_work_part1();

        // barrier: 所有线程在此等待，直到全部到达
        #pragma omp barrier

        // 所有线程都完成了 part1 后才执行 part2
        do_work_part2();
    }

    // single: 只有一个线程执行（通常是第一个到达的）
    #pragma omp parallel
    {
        #pragma omp single
        printf("只有一个线程执行了这条语句\n");
    }

    // master: 只有主线程（tid=0）执行（OpenMP 3.1+ 建议用 omp masked）
    #pragma omp parallel
    {
        #pragma omp master
        printf("只有主线程（tid=0）执行\n");
    }

    // sections: 不同线程执行不同代码块
    #pragma omp parallel sections
    {
        #pragma omp section
        {
            printf("线程 %d: 执行任务 A\n", omp_get_thread_num());
            task_A();
        }
        #pragma omp section
        {
            printf("线程 %d: 执行任务 B\n", omp_get_thread_num());
            task_B();
        }
        #pragma omp section
        {
            printf("线程 %d: 执行任务 C\n", omp_get_thread_num());
            task_C();
        }
    }

    return 0;
}
```

### 隐式 barrier

```c
// parallel for 结束时有隐式 barrier
#pragma omp parallel for
for (int i = 0; i < N; i++) {
    work(i);
}
// ← 所有线程在此同步

// 可以用 nowait 消除隐式 barrier
#pragma omp parallel
{
    #pragma omp for nowait
    for (int i = 0; i < N; i++) {
        work(i);
    }
    // ← 无 barrier，线程完成 for 后立即继续

    #pragma omp for
    for (int j = 0; j < M; j++) {
        work2(j);
    }
}
```

## 8. 嵌套并行与任务模型

### 嵌套并行

```c
#pragma omp parallel num_threads(2)
{
    #pragma omp parallel num_threads(2)
    {
        // 如果启用嵌套并行，将创建 2*2=4 个线程
        printf("外层线程 %d, 内层线程 %d\n",
               omp_get_ancestor_thread_num(1),
               omp_get_thread_num());
    }
}

// 启用嵌套并行
omp_set_nested(1);  // 或设置 OMP_NESTED=true
```

### Task 模型（OpenMP 3.0+）

Task 提供了比 for 循环更灵活的并行化方式，适合递归和不规则并行：

```c
#include <stdio.h>
#include <omp.h>

// 并行计算斐波那契数列
long fib(int n) {
    if (n < 2) return n;

    long x, y;

    #pragma omp task shared(x)
    x = fib(n - 1);

    #pragma omp task shared(y)
    y = fib(n - 2);

    #pragma omp taskwait  // 等待两个子任务完成

    return x + y;
}

int main() {
    int n = 30;
    long result;

    #pragma omp parallel
    {
        #pragma omp single
        {
            result = fib(n);
        }
    }

    printf("fib(%d) = %ld\n", n, result);
    return 0;
}
```

### Task 的应用场景

```c
// 1. 处理不规则数据结构（链表、树）
#pragma omp parallel
{
    #pragma omp single
    {
        Node *curr = head;
        while (curr) {
            #pragma omp task firstprivate(curr)
            process_node(curr);
            curr = curr->next;
        }
    }
}

// 2. 任务依赖图（OpenMP 4.0+）
int a, b, c;
#pragma omp parallel
#pragma omp single
{
    #pragma omp task depend(out: a)
    a = compute_a();

    #pragma omp task depend(out: b)
    b = compute_b();

    #pragma omp task depend(in: a, b) depend(out: c)
    c = combine(a, b);  // 等 a 和 b 都完成后执行
}
```

## 9. OpenMP 性能分析与调优

### 缓存亲和性（False Sharing）

```c
// 错误：可能发生 false sharing
// 相邻的数组元素可能在同一缓存行中，不同线程修改同一缓存行导致缓存失效
int counters[4];  // 4 个 int 在同一缓存行中（64 字节）

#pragma omp parallel for
for (int i = 0; i < 4; i++) {
    for (int j = 0; j < 1000000; j++)
        counters[i]++;  // 每个线程修改不同元素，但同一缓存行
}

// 修复：对齐到缓存行边界
_Alignas(64) int counters[4];  // C11 对齐
// 或使用 padding
struct { int val; char pad[60]; } counters[4];
```

### 缓存行伪共享原理

```
缓存行大小: 64 字节

场景1（false sharing）：
  缓存行: [counters[0] | counters[1] | counters[2] | counters[3]]
  线程0 修改 counters[0] → 整个缓存行标记为脏
  线程1 修改 counters[1] → 需要先使线程0的缓存行失效
  结果：缓存行在核之间反复弹跳（cache line bouncing）

场景2（修复后）：
  缓存行0: [counters[0] | padding ...]
  缓存行1: [counters[1] | padding ...]
  每个线程修改独立的缓存行 → 无干扰
```

### 比较不同线程数的加速比

```c
#include <stdio.h>
#include <stdlib.h>
#include <omp.h>
#include <math.h>

double compute(int N) {
    double sum = 0;
    #pragma omp parallel for reduction(+:sum)
    for (int i = 0; i < N; i++) {
        double x = (i + 0.5) / N;
        sum += 4.0 / (1.0 + x * x);
    }
    return sum / N;
}

int main() {
    int N = 100000000;
    double seq_time, par_time;

    // 串行基准
    omp_set_num_threads(1);
    double start = omp_get_wtime();
    double pi = compute(N);
    seq_time = omp_get_wtime() - start;
    printf("串行: %.4f 秒 (PI=%.15f)\n", seq_time, pi);

    // 不同线程数
    int thread_counts[] = {1, 2, 4, 8, 16};
    for (int t = 0; t < 5; t++) {
        omp_set_num_threads(thread_counts[t]);
        start = omp_get_wtime();
        compute(N);
        par_time = omp_get_wtime() - start;
        printf("线程数 %2d: %.4f 秒, 加速比 %.2fx\n",
               thread_counts[t], par_time, seq_time / par_time);
    }
    return 0;
}
```

典型输出（8 核 CPU）：

```
串行: 0.5823 秒 (PI=3.141592653589793)
线程数  1: 0.5841 秒, 加速比 1.00x
线程数  2: 0.2934 秒, 加速比 1.99x
线程数  4: 0.1482 秒, 加速比 3.93x
线程数  8: 0.0765 秒, 加速比 7.61x
线程数 16: 0.0789 秒, 加速比 7.38x  ← 超线程收益递减
```

## 10. 完整实战：OpenMP 并行图像处理

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <omp.h>

/*
 * 并行图像高斯模糊
 * 使用 3x3 高斯核对灰度图像进行卷积
 *
 * 高斯核 (归一化):
 *   1/16 * [1 2 1]
 *          [2 4 2]
 *          [1 2 1]
 */
void gaussian_blur(const unsigned char *input, unsigned char *output,
                   int width, int height) {
    // 高斯核权重
    const int kernel[3][3] = {
        {1, 2, 1},
        {2, 4, 2},
        {1, 2, 1}
    };
    const int kernel_sum = 16;

    // 对图像每个像素进行卷积（并行化外层循环）
    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sum = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = input[(y + ky) * width + (x + kx)];
                    sum += pixel * kernel[ky + 1][kx + 1];
                }
            }
            output[y * width + x] = (unsigned char)(sum / kernel_sum);
        }
    }
}

// 边缘检测（Sobel 算子）
void sobel_edge(const unsigned char *input, unsigned char *output,
                int width, int height) {
    const int gx[3][3] = {{-1, 0, 1}, {-2, 0, 2}, {-1, 0, 1}};
    const int gy[3][3] = {{-1, -2, -1}, {0, 0, 0}, {1, 2, 1}};

    #pragma omp parallel for schedule(dynamic, 16)
    for (int y = 1; y < height - 1; y++) {
        for (int x = 1; x < width - 1; x++) {
            int sx = 0, sy = 0;
            for (int ky = -1; ky <= 1; ky++) {
                for (int kx = -1; kx <= 1; kx++) {
                    int pixel = input[(y + ky) * width + (x + kx)];
                    sx += pixel * gx[ky + 1][kx + 1];
                    sy += pixel * gy[ky + 1][kx + 1];
                }
            }
            int mag = (int)sqrt((double)(sx * sx + sy * sy));
            output[y * width + x] = (unsigned char)(mag > 255 ? 255 : mag);
        }
    }
}

int main() {
    int width = 4096, height = 4096;

    // 生成测试图像
    unsigned char *image = (unsigned char *)malloc(width * height);
    unsigned char *blurred = (unsigned char *)calloc(width * height, 1);
    unsigned char *edges = (unsigned char *)calloc(width * height, 1);

    srand(42);
    for (int i = 0; i < width * height; i++)
        image[i] = (unsigned char)(rand() % 256);

    // 测试不同线程数
    int threads[] = {1, 2, 4, 8};
    for (int t = 0; t < 4; t++) {
        omp_set_num_threads(threads[t]);

        double start = omp_get_wtime();
        gaussian_blur(image, blurred, width, height);
        double blur_time = omp_get_wtime() - start;

        start = omp_get_wtime();
        sobel_edge(blurred, edges, width, height);
        double edge_time = omp_get_wtime() - start;

        printf("线程数 %d: 模糊 %.4f 秒, 边缘检测 %.4f 秒, 总计 %.4f 秒\n",
               threads[t], blur_time, edge_time, blur_time + edge_time);
    }

    free(image); free(blurred); free(edges);
    return 0;
}
```

## 11. 真实应用场景

### 11.1 OpenMP 在图像处理中的应用

OpenCV、FFmpeg 等广泛使用 OpenMP 加速图像处理：

- **滤波操作**：高斯模糊、中值滤波等，每个像素独立计算，天然并行
- **色彩空间转换**：RGB 到 YUV 的转换，每个像素可并行处理
- **直方图统计**：使用 `reduction` 或 `atomic` 统计像素值分布
- **视频编解码**：帧级并行（不同帧由不同线程解码）或宏块级并行

### 11.2 OpenMP 在数值计算中的应用

- **矩阵乘法**：外层循环并行化，每个线程计算结果矩阵的一部分行
- **LU 分解**：外层循环并行化，但需要注意数据依赖
- **快速傅里叶变换（FFT）**：大规模 FFT 的级间并行
- **蒙特卡洛模拟**：每个线程独立生成随机样本，最后汇总

### 11.3 OpenMP 在科学计算中的应用

- **分子动力学**：短程力的计算（如 Lennard-Jones 势）对每个原子对独立
- **有限元分析**：刚度矩阵的组装，每个单元的贡献可并行计算
- **粒子模拟**：粒子间力的计算和位置更新可并行化

## 12. MPI vs OpenMP vs Pthreads 对比

| 特性 | MPI | OpenMP | Pthreads |
|------|-----|--------|----------|
| **内存模型** | 分布式内存 | 共享内存 | 共享内存 |
| **并行粒度** | 进程级 | 线程级 | 线程级 |
| **通信方式** | 显式消息传递 | 隐式（共享变量） | 显式同步原语 |
| **编程难度** | 较高 | 较低 | 高 |
| **可扩展性** | 极好（万级以上节点） | 受限于单节点核心数 | 受限于单节点核心数 |
| **适用场景** | 集群、超算 | 单节点多核 | 需要精细控制线程 |
| **数据分布** | 手动管理 | 自动（共享地址空间） | 手动管理 |
| **编译器支持** | 需要 MPI 库 | GCC/Clang/ICC 内建 | 需要 pthread 库 |
| **典型用例** | 气候模拟、CFD | 图像处理、矩阵运算 | 服务器、嵌入式 |

**选择建议**：
- 超过单节点规模（跨节点） → 用 MPI
- 单节点多核加速（共享内存） → 用 OpenMP（更简单）
- 需要精细线程控制（锁、条件变量） → 用 Pthreads
- 超算上的混合编程 → MPI（跨节点）+ OpenMP（节点内），这是目前最流行的 HPC 编程范式

### 混合编程示例

```c
// MPI + OpenMP 混合编程
// MPI 负责跨节点通信，OpenMP 负责节点内多线程并行
#include <mpi.h>
#include <omp.h>
#include <stdio.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 每个 MPI 进程内部使用 OpenMP 多线程
    #pragma omp parallel
    {
        int tid = omp_get_thread_num();
        int nthreads = omp_get_num_threads();
        printf("MPI rank %d, OpenMP thread %d / %d\n",
               rank, tid, nthreads);
    }

    // MPI 通信（每个进程的主线程参与）
    double local_val = (double)rank;
    double global_sum;
    MPI_Reduce(&local_val, &global_sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("全局 sum = %.1f\n", global_sum);

    MPI_Finalize();
    return 0;
}
```

```bash
# 编译
mpicc -fopenmp -o hybrid hybrid.c -lm

# 运行：4 个 MPI 进程，每个进程 8 个 OpenMP 线程
export OMP_NUM_THREADS=8
mpirun -np 4 ./hybrid
# 总共使用 4 * 8 = 32 个线程
```
