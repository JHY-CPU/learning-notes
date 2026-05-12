# MPI 集合通信

集合通信（Collective Communication）涉及通信子内所有进程，相比点对点通信效率更高，语义更清晰。MPI 实现可以针对集合通信进行全局优化（如使用树形算法、环形算法），而非简单的点对点组合。

## 0. 底层实现原理

集合通信之所以比手动编写的点对点通信更快，是因为 MPI 实现可以使用高效的并行算法：

### 广播的树形算法

朴素广播（根进程逐个发送）的时间复杂度为 O(P)，而树形广播为 O(log P)：

```
朴素方式（O(P)）：
根 → 进程1
根 → 进程2
根 → 进程3
...

树形方式（O(log P)）：
第1轮：根 → 进程1
第2轮：根 → 进程2,  进程1 → 进程3
第3轮：根 → 进程4,  进程1 → 进程5,  进程2 → 进程6,  进程3 → 进程7
...
```

### 归约的双二叉树算法

MPI 实现（如 MPICH、Open MPI）通常使用**双二叉树**（binomial tree）来实现归约操作，每个进程最多执行 log(P) 次局部归约和消息收发。

### MPI 实现中的算法选择

MPI 实现会根据消息大小和进程数自动选择最优算法：
- **小消息 + 少进程**：使用二项式树（binomial tree）
- **大消息 + 多进程**：使用环形（ring）或递归倍增（recursive doubling）
- **中等消息**：可能使用 Rabenseifner 算法（减少步骤数）

这些优化对用户透明——用户只需调用 `MPI_Bcast` 等函数，MPI 运行时会自动选择最优实现。

## 1. MPI_Bcast——广播

`MPI_Bcast` 将根进程的数据发送给通信子内所有其他进程。

```
树形广播过程（4个进程）：
轮次1:  根(0) ──数据──> 进程1
轮次2:  根(0) ──数据──> 进程2,  进程1 ──数据──> 进程3
完成:   所有进程都有数据
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 广播一个整数
    int data;
    if (rank == 0) {
        data = 42;  // 根进程准备数据
        printf("根进程广播 data = %d\n", data);
    }

    // 所有进程（包括根进程）调用 Bcast
    // buf: 数据缓冲区（输入输出参数）
    // count: 元素个数
    // datatype: 数据类型
    // root: 根进程 rank
    // comm: 通信子
    MPI_Bcast(&data, 1, MPI_INT, 0, MPI_COMM_WORLD);

    printf("进程%d 收到广播数据: %d\n", rank, data);

    // 广播数组
    int *arr = NULL;
    int arr_size = 100;
    if (rank == 0) {
        arr = (int *)malloc(arr_size * sizeof(int));
        for (int i = 0; i < arr_size; i++)
            arr[i] = i * i;
    } else {
        arr = (int *)malloc(arr_size * sizeof(int));
    }

    MPI_Bcast(arr, arr_size, MPI_INT, 0, MPI_COMM_WORLD);

    // 验证：所有进程都应该有相同的数据
    int correct = 1;
    for (int i = 0; i < arr_size; i++) {
        if (arr[i] != i * i) { correct = 0; break; }
    }
    printf("进程%d: 广播数组验证 %s\n", rank, correct ? "通过" : "失败");

    free(arr);
    MPI_Finalize();
    return 0;
}
```

## 2. MPI_Scatter——散射

`MPI_Scatter` 将根进程缓冲区中的数据**均匀分割**后分发给每个进程。每个进程接收 `sendcount` 个元素。

```
根进程(rank=0)                     各进程
[10, 20, 30, 40]  ──Scatter──>  rank 0: [10]
                                rank 1: [20]
                                rank 2: [30]
                                rank 3: [40]
```

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int *sendbuf = NULL;
    int recv_val;

    if (rank == 0) {
        sendbuf = (int *)malloc(size * sizeof(int));
        for (int i = 0; i < size; i++)
            sendbuf[i] = (i + 1) * 100;
        printf("根进程数据: ");
        for (int i = 0; i < size; i++) printf("%d ", sendbuf[i]);
        printf("\n");
    }

    // sendcount=1 表示每个进程接收 1 个元素
    // sendbuf 仅在根进程有意义
    MPI_Scatter(sendbuf, 1, MPI_INT,
                &recv_val, 1, MPI_INT,
                0, MPI_COMM_WORLD);

    printf("进程%d 收到: %d\n", rank, recv_val);

    free(sendbuf);
    MPI_Finalize();
    return 0;
}
```

### Scatterv——不均匀分发

当每个进程接收的数据量不同时，使用 `MPI_Scatterv`：

```c
// 假设有 7 个元素，分给 4 个进程：2, 2, 2, 1
int sendcounts[4] = {2, 2, 2, 1};
int displs[4]     = {0, 2, 4, 6};  // 每个进程数据在 sendbuf 中的起始偏移

int sendbuf[7] = {10, 20, 30, 40, 50, 60, 70};
int my_count = sendcounts[rank];
int *recvbuf = (int *)malloc(my_count * sizeof(int));

MPI_Scatterv(sendbuf, sendcounts, displs, MPI_INT,
             recvbuf, my_count, MPI_INT,
             0, MPI_COMM_WORLD);
```

## 3. MPI_Gather——收集

`MPI_Gather` 是 `MPI_Scatter` 的逆操作，将各进程的数据收集到根进程。

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int local_val = rank * 10 + 5;  // 每个进程准备自己的数据
    int *recvbuf = NULL;

    if (rank == 0)
        recvbuf = (int *)malloc(size * sizeof(int));

    MPI_Gather(&local_val, 1, MPI_INT,
               recvbuf, 1, MPI_INT,
               0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("根进程收集到: ");
        for (int i = 0; i < size; i++) printf("%d ", recvbuf[i]);
        printf("\n");
        free(recvbuf);
    }

    MPI_Finalize();
    return 0;
}
```

### Gatherv——不均匀收集

```c
// 各进程发送不同数量的元素
int my_count = rank + 1;  // rank 0 发 1 个，rank 1 发 2 个...
int *my_data = (int *)malloc(my_count * sizeof(int));
for (int i = 0; i < my_count; i++) my_data[i] = rank * 100 + i;

int *recvbuf = NULL;
int *recvcounts = NULL, *displs = NULL;

if (rank == 0) {
    recvcounts = (int *)malloc(size * sizeof(int));
    displs = (int *)malloc(size * sizeof(int));
    int total = 0;
    for (int i = 0; i < size; i++) {
        recvcounts[i] = i + 1;
        displs[i] = total;
        total += recvcounts[i];
    }
    recvbuf = (int *)malloc(total * sizeof(int));
}

MPI_Gatherv(my_data, my_count, MPI_INT,
            recvbuf, recvcounts, displs, MPI_INT,
            0, MPI_COMM_WORLD);

if (rank == 0) {
    printf("Gatherv 收集到的数据: ");
    int total = 0;
    for (int i = 0; i < size; i++) total += recvcounts[i];
    for (int i = 0; i < total; i++) printf("%d ", recvbuf[i]);
    printf("\n");
    free(recvbuf); free(recvcounts); free(displs);
}
free(my_data);
```

### Allgather——全收集

`MPI_Allgather` 将收集结果广播给所有进程，每个进程都能获得完整的数据：

```c
int local_val = rank * 10 + 5;
int *all_recv = (int *)malloc(size * sizeof(int));

MPI_Allgather(&local_val, 1, MPI_INT,
              all_recv, 1, MPI_INT, MPI_COMM_WORLD);

// 此时所有进程的 all_recv 中都有完整数据
printf("进程%d 看到全部数据: ", rank);
for (int i = 0; i < size; i++) printf("%d ", all_recv[i]);
printf("\n");
free(all_recv);
```

## 4. MPI_Reduce——归约

`MPI_Reduce` 对各进程的数据执行指定的归约操作（如求和、求最大值等），结果存放在根进程。

```
rank 0: 10  ─┐
rank 1: 20  ─┼─Reduce(求和)──>  根进程: 100
rank 2: 30  ─┤
rank 3: 40  ─┘
```

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local_val = (double)(rank + 1);  // 1.0, 2.0, 3.0, ...
    double global_sum, global_max, global_min;

    // MPI_SUM: 求和
    MPI_Reduce(&local_val, &global_sum, 1, MPI_DOUBLE,
               MPI_SUM, 0, MPI_COMM_WORLD);

    // MPI_MAX: 求最大值
    MPI_Reduce(&local_val, &global_max, 1, MPI_DOUBLE,
               MPI_MAX, 0, MPI_COMM_WORLD);

    // MPI_MIN: 求最小值
    MPI_Reduce(&local_val, &global_min, 1, MPI_DOUBLE,
               MPI_MIN, 0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("Sum = %.1f, Max = %.1f, Min = %.1f\n",
               global_sum, global_max, global_min);
    }

    MPI_Finalize();
    return 0;
}
```

### 常用归约操作

| 操作 | 含义 | C 数据类型 |
|------|------|-----------|
| `MPI_SUM` | 求和 | 整数、浮点 |
| `MPI_MAX` | 求最大值 | 整数、浮点 |
| `MPI_MIN` | 求最小值 | 整数、浮点 |
| `MPI_PROD` | 求积 | 整数、浮点 |
| `MPI_LAND` | 逻辑与 | 整数 |
| `MPI_LOR` | 逻辑或 | 整数 |
| `MPI_LXOR` | 逻辑异或 | 整数 |
| `MPI_BAND` | 按位与 | 整数 |
| `MPI_BOR` | 按位或 | 整数 |
| `MPI_BXOR` | 按位异或 | 整数 |
| `MPI_MAXLOC` | 最大值及其位置 | 结构体 |
| `MPI_MINLOC` | 最小值及其位置 | 结构体 |

### 自定义归约操作

```c
// 定义自定义归约：计算向量的逐元素最大值
void my_max_vec(void *in, void *inout, int *len, MPI_Datatype *dtype) {
    double *a = (double *)in;
    double *b = (double *)inout;
    for (int i = 0; i < *len; i++) {
        if (a[i] > b[i]) b[i] = a[i];
    }
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double local[3] = {(double)rank, (double)rank*2, (double)rank*3};
    double global[3];

    MPI_Op my_op;
    MPI_Op_create(my_max_vec, 1 /* commutative */, &my_op);

    MPI_Reduce(local, global, 3, MPI_DOUBLE, my_op, 0, MPI_COMM_WORLD);

    if (rank == 0)
        printf("自定义归约结果: [%.1f, %.1f, %.1f]\n", global[0], global[1], global[2]);

    MPI_Op_free(&my_op);
    MPI_Finalize();
    return 0;
}
```

## 5. MPI_Allreduce——全归约

`MPI_Allreduce` 将归约结果广播给所有进程，等价于 `Reduce` + `Bcast`，但实现更高效。

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

// 并行计算 PI——使用 Allreduce 汇总局部和
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 100000000;
    double h = 1.0 / N;
    double local_sum = 0.0;

    for (long long i = rank; i < N; i += size) {
        double x = h * (i + 0.5);
        local_sum += 4.0 / (1.0 + x * x);
    }

    double global_sum;
    MPI_Allreduce(&local_sum, &global_sum, 1, MPI_DOUBLE,
                  MPI_SUM, MPI_COMM_WORLD);

    double pi = global_sum * h;
    // 所有进程都得到了正确的 pi 值
    printf("进程%d: PI ≈ %.15f\n", rank, pi);

    MPI_Finalize();
    return 0;
}
```

### Allreduce 的典型应用

```c
// 分布式梯度下降中的梯度同步
// 每个进程计算本地梯度，然后对所有进程求平均
double local_grad[n_params];
double global_grad[n_params];

compute_local_gradient(local_grad, data_chunk);

// 对所有进程的梯度求和
MPI_Allreduce(local_grad, global_grad, n_params, MPI_DOUBLE,
              MPI_SUM, MPI_COMM_WORLD);

// 除以进程数得到平均梯度
for (int i = 0; i < n_params; i++)
    global_grad[i] /= size;

// 所有进程用相同的梯度更新模型参数
update_model(global_grad);
```

## 6. MPI_Scan——前缀和

`MPI_Scan` 计算对所有进程数据的前缀归约（prefix sum），在并行排序、并行搜索中广泛使用。

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int val = rank + 1;  // 1, 2, 3, 4
    int prefix_sum;
    MPI_Scan(&val, &prefix_sum, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    // rank 0: 1,  rank 1: 3,  rank 2: 6,  rank 3: 10
    printf("进程%d: 前缀和 = %d\n", rank, prefix_sum);

    // Exclusive scan（不包含自身）——MPI 标准不直接提供
    // 需要手动实现：先做 inclusive scan，再减去自身值
    int inclusive, exclusive;
    MPI_Scan(&val, &inclusive, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
    exclusive = inclusive - val;
    printf("进程%d: exclusive 前缀和 = %d\n", rank, exclusive);

    MPI_Finalize();
    return 0;
}
```

### 前缀和在并行排序中的应用

```c
// 并行基数排序中，用前缀和计算每个进程的输出偏移
// 例如：统计每个进程中 key=3 的元素个数
int my_count = count_elements_with_key(local_data, local_n, key);
int prefix_count;
MPI_Scan(&my_count, &prefix_count, 1, MPI_INT, MPI_SUM, MPI_COMM_WORLD);
// prefix_count 就是当前进程（含自身）之前所有进程中 key=3 的元素总数
// 用于确定全局输出数组中的写入位置
```

## 7. 完整实战：矩阵-向量乘法

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * 并行矩阵-向量乘法 y = A * x
 * 矩阵 A 按行分块，每个进程持有连续的若干行
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 8;  // 矩阵大小 N x N，向量长度 N

    // 进程0 初始化完整矩阵和向量
    double *A = NULL, *x = NULL;
    if (rank == 0) {
        A = (double *)malloc(N * N * sizeof(double));
        x = (double *)malloc(N * sizeof(double));
        for (int i = 0; i < N; i++) {
            x[i] = 1.0;  // 向量全为1
            for (int j = 0; j < N; j++)
                A[i * N + j] = (double)(i + j);
        }
    } else {
        x = (double *)malloc(N * sizeof(double));
    }

    // 广播向量 x 给所有进程
    MPI_Bcast(x, N, MPI_DOUBLE, 0, MPI_COMM_WORLD);

    // 计算每个进程分配的行数
    int rows_per_proc = N / size;
    int remainder = N % size;
    int my_rows = rows_per_proc + (rank < remainder ? 1 : 0);

    // Scatterv 分发行（处理不均匀分配）
    int *sendcounts = NULL, *displs = NULL;
    if (rank == 0) {
        sendcounts = (int *)malloc(size * sizeof(int));
        displs = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int r = rows_per_proc + (i < remainder ? 1 : 0);
            sendcounts[i] = r * N;
            displs[i] = offset;
            offset += r * N;
        }
    }

    double *local_A = (double *)malloc(my_rows * N * sizeof(double));
    MPI_Scatterv(A, sendcounts, displs, MPI_DOUBLE,
                 local_A, my_rows * N, MPI_DOUBLE,
                 0, MPI_COMM_WORLD);

    // 本地计算：local_y = local_A * x
    double *local_y = (double *)calloc(my_rows, sizeof(double));
    for (int i = 0; i < my_rows; i++)
        for (int j = 0; j < N; j++)
            local_y[i] += local_A[i * N + j] * x[j];

    // 收集结果到进程0
    double *y = NULL;
    int *recvcounts = NULL, *rdispls = NULL;
    if (rank == 0) {
        y = (double *)malloc(N * sizeof(double));
        recvcounts = (int *)malloc(size * sizeof(int));
        rdispls = (int *)malloc(size * sizeof(int));
        int offset = 0;
        for (int i = 0; i < size; i++) {
            int r = rows_per_proc + (i < remainder ? 1 : 0);
            recvcounts[i] = r;
            rdispls[i] = offset;
            offset += r;
        }
    }

    MPI_Gatherv(local_y, my_rows, MPI_DOUBLE,
                y, recvcounts, rdispls, MPI_DOUBLE,
                0, MPI_COMM_WORLD);

    if (rank == 0) {
        printf("结果向量 y = A * x:\n");
        for (int i = 0; i < N; i++)
            printf("  y[%d] = %.1f\n", i, y[i]);

        // 验证：每行的和应为 i*N + N*(N-1)/2
        int correct = 1;
        for (int i = 0; i < N; i++) {
            double expected = 0;
            for (int j = 0; j < N; j++) expected += (double)(i + j);
            if (y[i] != expected) { correct = 0; break; }
        }
        printf("验证: %s\n", correct ? "通过" : "失败");

        free(A); free(y); free(sendcounts); free(displs);
        free(recvcounts); free(rdispls);
    }

    free(x); free(local_A); free(local_y);
    MPI_Finalize();
    return 0;
}
```

**Makefile**：

```makefile
CC = mpicc
CFLAGS = -Wall -O2 -lm

matvec: matvec.c
	$(CC) $(CFLAGS) -o $@ $<

run: matvec
	mpirun -np 4 ./matvec

clean:
	rm -f matvec

.PHONY: run clean
```

## 8. 集合通信性能分析

### 通信开销模型

集合通信的性能可以用以下模型描述：

```
T = alpha + n * beta
```

其中：
- `alpha`：延迟（latency），单位为微秒，表示启动通信的固定开销
- `n`：消息大小（字节数）
- `beta`：传输时间每字节（1/带宽）

### 各操作的渐近时间复杂度

| 操作 | 算法 | 时间复杂度 |
|------|------|-----------|
| `MPI_Bcast` | 二项式树 | O(log P)（小消息）|
| `MPI_Reduce` | 二项式树 | O(log P)（小消息）|
| `MPI_Allreduce` | 递归倍增 | O(log P)（小消息）|
| `MPI_Scatter` | 二项式树 | O(log P)（小消息）|
| `MPI_Gather` | 二项式树 | O(log P)（小消息）|
| `MPI_Alltoall` | 全交换 | O(P) |

### 性能对比：手动实现 vs MPI 集合通信

```c
// 手动实现广播（慢）
if (rank == 0) {
    for (int i = 1; i < size; i++)
        MPI_Send(data, N, MPI_INT, i, 0, comm);
} else {
    MPI_Recv(data, N, MPI_INT, 0, 0, comm, &status);
}
// 时间: O(P) 轮通信

// MPI_Bcast（快）
MPI_Bcast(data, N, MPI_INT, 0, comm);
// 时间: O(log P) 轮通信（树形算法）
```

在 64 个进程的集群上，手动广播耗时约 64 次消息传输，而 MPI_Bcast 仅需约 6 次。

## 9. 常见错误与调试

### 错误1：所有进程必须调用集合通信

```c
// 错误！只有 rank 0 调用了 Bcast
if (rank == 0) {
    MPI_Bcast(&data, 1, MPI_INT, 0, comm);  // 死锁！其他进程没有调用
}
// 正确做法：所有进程都必须调用
MPI_Bcast(&data, 1, MPI_INT, 0, comm);
```

### 错误2：通信子不一致

```c
// 错误！不同进程使用不同通信子
MPI_Comm comm;
if (rank < 2)
    MPI_Comm_split(MPI_COMM_WORLD, 0, rank, &comm);

// 后面所有进程都调用 Bcast，但 rank>=2 的 comm 未初始化
MPI_Bcast(&data, 1, MPI_INT, 0, comm);  // 未定义行为
```

### 错误3：Root 不匹配

```c
// 所有进程必须指定相同的 root
if (rank == 0)
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
else if (rank == 1)
    MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 1, comm);  // 错误！root 不一致
```

### 错误4：Send/Recv 与集合通信混用

```c
// 危险！在集合通信之间插入点对点通信
MPI_Bcast(&data, 1, MPI_INT, 0, comm);
MPI_Send(&val, 1, MPI_INT, 1, 0, comm);  // 可能干扰下一个集合通信
MPI_Reduce(&local, &global, 1, MPI_DOUBLE, MPI_SUM, 0, comm);
// 规则：集合通信之间不应该有未完成的点对点通信
```

## 10. 真实应用场景

### 10.1 MPI 在深度学习中的应用

分布式深度学习框架（如 Horovod）大量使用 MPI 集合通信：

- **数据并行训练**：每个 GPU 计算一个 mini-batch 的梯度，使用 `MPI_Allreduce` 求全局平均梯度
- **模型并行训练**：使用 `MPI_Scatter` 分发不同层到不同 GPU
- **参数服务器**：使用 `MPI_Gather` 收集各 worker 的梯度，`MPI_Bcast` 分发更新后的参数

### 10.2 MPI 在计算流体力学（CFD）中的应用

- 使用 `MPI_Scatterv` 分发非均匀网格到各进程
- 边界交换使用 `MPI_Sendrecv` 交换相邻子域的 ghost cell
- 全局物理量（质量守恒、能量守恒）使用 `MPI_Allreduce` 计算
- 结果输出使用 `MPI_Gatherv` 收集到 I/O 进程

### 10.3 MPI 在图算法中的应用

PageRank、BFS 等图算法中：
- `MPI_Allgather` 用于同步各进程计算的部分 rank 值
- `MPI_Allreduce` 用于计算全局收敛条件
- `MPI_Scan` 用于并行前缀计算（如并行排序中的偏移量计算）
