# MPI 非阻塞通信

阻塞式 `MPI_Send`/`MPI_Recv` 在完成前会阻塞调用进程。非阻塞通信允许发起通信后立即返回，让程序在等待数据传输的同时执行计算，实现**通信与计算的重叠**——这是在大规模并行计算中获得高性能的关键技术。

## 0. 底层原理

### 阻塞通信的问题

```
阻塞发送的时间线：
进程0: [计算] [MPI_Send ────阻塞等待网络传输完成────] [计算]
                      ↑ 网络传输期间 CPU 空闲！
```

### 非阻塞通信的原理

非阻塞通信在底层的工作方式：

1. **注册通信请求**：`MPI_Isend` 向 MPI 运行时注册一个通信请求（包含缓冲区地址、目标、标签等），然后立即返回。MPI 后台线程或硬件（如 InfiniBand 网卡）负责实际的数据传输。
2. **用户继续计算**：在数据传输的同时，用户程序可以执行与通信无关的计算任务。
3. **检查/等待完成**：用户调用 `MPI_Wait` 或 `MPI_Test` 检查通信是否完成。如果完成，释放内部资源；如果未完成，`MPI_Wait` 阻塞等待，`MPI_Test` 立即返回。

```
非阻塞通信的时间线：
进程0: [计算] [MPI_Isend 立即返回] [计算（与通信重叠）] [MPI_Wait] [计算]
                      ↑ 注册请求                ↑ 网络传输在后台进行
```

### MPI_Request 对象

`MPI_Request` 是一个不透明的句柄，指向 MPI 内部的通信状态记录。每个非阻塞操作（Isend/Irecv）都会创建一个 Request。用户**必须**通过 Wait、Test 或 Cancel 来完成每个 Request，否则会导致资源泄漏。

### Eager vs Rendezvous 与非阻塞的关系

- **Eager 协议**（小消息）：`MPI_Isend` 发起后数据立即被复制到接收端内部缓冲区，发送端的 Request 很快变为"完成"。
- **Rendezvous 协议**（大消息）：`MPI_Isend` 只发送控制消息，实际数据传输等待接收端准备好。此时非阻塞的优势更明显——发送端可以在等待期间做其他计算。

## 1. MPI_Isend / MPI_Irecv

非阻塞发送和接收函数立即返回一个 `MPI_Request` 对象，用于后续查询完成状态。

```c
/*
 * int MPI_Isend(const void *buf, int count, MPI_Datatype datatype,
 *               int dest, int tag, MPI_Comm comm, MPI_Request *request);
 *
 * int MPI_Irecv(void *buf, int count, MPI_Datatype datatype,
 *               int source, int tag, MPI_Comm comm, MPI_Request *request);
 */
```

### 完整示例：双边非阻塞通信

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) {
        if (rank == 0) fprintf(stderr, "需要至少 2 个进程\n");
        MPI_Finalize();
        return 1;
    }

    double send_data[1000], recv_data[1000];
    MPI_Request req_send, req_recv;
    MPI_Status status;

    // 初始化发送数据
    for (int i = 0; i < 1000; i++)
        send_data[i] = (double)(rank * 1000 + i);

    // 发起非阻塞通信（立即返回！）
    if (rank == 0) {
        MPI_Isend(send_data, 1000, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(recv_data, 1000, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD, &req_recv);
    } else if (rank == 1) {
        MPI_Isend(send_data, 1000, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req_send);
        MPI_Irecv(recv_data, 1000, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &req_recv);
    }

    // ---- 通信正在进行中，可以执行计算 ----
    double local_sum = 0.0;
    for (int i = 0; i < 1000; i++)
        local_sum += send_data[i];
    printf("进程%d: 局部计算完成 sum=%.1f（通信仍在后台进行）\n", rank, local_sum);

    // 等待通信完成
    MPI_Wait(&req_send, &status);
    MPI_Wait(&req_recv, &status);

    printf("进程%d: 非阻塞通信完成, 收到前5个: %.0f %.0f %.0f %.0f %.0f\n",
           rank, recv_data[0], recv_data[1], recv_data[2], recv_data[3], recv_data[4]);

    MPI_Finalize();
    return 0;
}
```

**Makefile**：

```makefile
CC = mpicc
CFLAGS = -Wall -O2

nonblock: nonblock.c
	$(CC) $(CFLAGS) -o $@ $<

run: nonblock
	mpirun -np 2 ./nonblock

clean:
	rm -f nonblock

.PHONY: run clean
```

## 2. MPI_Wait 与 MPI_Waitall

### MPI_Wait

`MPI_Wait(&request, &status)` 阻塞调用进程，直到指定的非阻塞操作完成。

```c
MPI_Request req;
MPI_Status status;

MPI_Isend(buf, N, MPI_INT, dest, tag, comm, &req);

// 做一些其他工作...
do_something_else();

// 等待发送完成
MPI_Wait(&req, &status);
// 此时 buf 可以安全修改或释放
```

### MPI_Waitall

`MPI_Waitall(count, array_of_requests, array_of_statuses)` 等待所有请求完成：

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 4) {
        if (rank == 0) fprintf(stderr, "需要至少 4 个进程\n");
        MPI_Finalize();
        return 1;
    }

    int N = 1000;
    double *buf_a = (double *)malloc(N * sizeof(double));
    double *buf_b = (double *)malloc(N * sizeof(double));
    double *buf_c = (double *)malloc(N * sizeof(double));
    double *buf_d = (double *)malloc(N * sizeof(double));

    MPI_Request requests[4];
    MPI_Status statuses[4];

    // 同时发起多个非阻塞通信
    MPI_Irecv(buf_a, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, &requests[0]);
    MPI_Irecv(buf_b, N, MPI_DOUBLE, 1, 1, MPI_COMM_WORLD, &requests[1]);
    MPI_Isend(buf_c, N, MPI_DOUBLE, 2, 2, MPI_COMM_WORLD, &requests[2]);
    MPI_Isend(buf_d, N, MPI_DOUBLE, 3, 3, MPI_COMM_WORLD, &requests[3]);

    // 等待全部 4 个通信完成
    MPI_Waitall(4, requests, statuses);

    // 此时所有缓冲区都已就绪
    // 检查每个接收操作实际接收了多少数据
    for (int i = 0; i < 2; i++) {
        int count;
        MPI_Get_count(&statuses[i], MPI_DOUBLE, &count);
        printf("请求%d: 接收了 %d 个 double\n", i, count);
    }

    free(buf_a); free(buf_b); free(buf_c); free(buf_d);
    MPI_Finalize();
    return 0;
}
```

### MPI_Waitany

`MPI_Waitany` 等待任意一个请求完成，返回完成的索引：

```c
MPI_Request requests[4];
// ... 发起 4 个非阻塞操作 ...

int completed[4] = {0, 0, 0, 0};
for (int done = 0; done < 4; done++) {
    int index;
    MPI_Status status;
    MPI_Waitany(4, requests, &index, &status);
    completed[index] = 1;
    printf("请求 %d 先完成了（来自 rank %d）\n",
           index, status.MPI_SOURCE);
}
```

### MPI_Waitsome

`MPI_Waitsome` 等待至少一个请求完成：

```c
int outcount;
int indices[4];
MPI_Status statuses[4];
MPI_Waitsome(4, requests, &outcount, indices, statuses);
printf("有 %d 个请求完成了: ", outcount);
for (int i = 0; i < outcount; i++)
    printf("%d ", indices[i]);
printf("\n");
```

## 3. MPI_Test——非阻塞测试

`MPI_Test` 检查请求是否已完成，但不阻塞：

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) { MPI_Finalize(); return 1; }

    double data = (double)rank;
    double recv_data;
    MPI_Request req;

    // 非阻塞接收
    MPI_Irecv(&recv_data, 1, MPI_DOUBLE, 1 - rank, 0, MPI_COMM_WORLD, &req);

    // 轮询检查通信是否完成，同时做其他工作
    int flag = 0;
    int poll_count = 0;
    MPI_Status status;

    while (!flag) {
        poll_count++;
        MPI_Test(&req, &flag, &status);
        if (!flag) {
            // 通信尚未完成，执行其他计算
            do_some_background_work();
        }
    }

    printf("进程%d: 轮询 %d 次后通信完成，收到 %.1f\n",
           rank, poll_count, recv_data);

    MPI_Finalize();
    return 0;
}
```

### MPI_Testall / MPI_Testany / MPI_Testsome

与 Wait 系列类似，但不阻塞：

```c
int all_done;
MPI_Status statuses[4];
MPI_Testall(4, requests, &all_done, statuses);
if (all_done) {
    printf("所有通信已完成\n");
} else {
    printf("还有通信未完成\n");
}
```

## 4. 通信与计算重叠——核心应用场景

这是非阻塞通信最核心的价值：在数据传输期间执行有用的计算。

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <math.h>

// 模拟耗时计算
void do_computation(double *data, int n) {
    for (int i = 0; i < n; i++)
        data[i] = sin(data[i]) * cos(data[i]) + sqrt(fabs(data[i]) + 1.0);
}

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 100000;
    double *local_data = (double *)malloc(N * sizeof(double));
    double *boundary_buf = (double *)malloc(N * sizeof(double));
    MPI_Request req;

    // 初始化
    for (int i = 0; i < N; i++)
        local_data[i] = (double)(rank + i);

    int left = (rank - 1 + size) % size;
    int right = (rank + 1) % size;

    double t_start = MPI_Wtime();

    // 1. 发起非阻塞接收（等待邻居数据）
    MPI_Irecv(boundary_buf, N, MPI_DOUBLE, left, 0, MPI_COMM_WORLD, &req);

    // 2. 同时执行本地计算（与通信重叠！）
    do_computation(local_data, N);
    printf("进程%d: 本地计算完成（通信仍在后台进行）\n", rank);

    // 3. 发送数据给右邻居
    MPI_Send(local_data, N, MPI_DOUBLE, right, 0, MPI_COMM_WORLD);

    // 4. 等待接收完成
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    double t_end = MPI_Wtime();
    printf("进程%d: 总耗时 %.4f 秒\n", rank, t_end - t_start);

    free(local_data);
    free(boundary_buf);
    MPI_Finalize();
    return 0;
}
```

### 性能分析：何时非阻塞通信有意义

非阻塞通信的价值取决于**计算时间**与**通信时间**的比值：

```
场景1: 计算时间 > 通信时间（非阻塞优势大）
  阻塞:  [通信████████] [计算████████████████]
  非阻塞:[Isend] [计算████████████████] [Wait]  ← 总时间更短

场景2: 计算时间 ≈ 通信时间（非阻塞有优势）
  阻塞:  [通信████████] [计算████████]
  非阻塞:[Isend] [计算████████] [Wait]  ← 可能完全重叠

场景3: 计算时间 < 通信时间（非阻塞优势小）
  阻塞:  [通信████████████████] [计算████]
  非阻塞:[Isend] [计算████] [Wait████████████]  ← 改善有限
```

### 基准测试代码

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    if (size < 2) { MPI_Finalize(); return 1; }

    // 测试不同消息大小的延迟和带宽
    int sizes[] = {1, 8, 64, 512, 4096, 32768, 262144, 1048576, 4194304};
    int num_sizes = sizeof(sizes) / sizeof(int);
    int repeats = 100;

    if (rank == 0) {
        printf("%12s %12s %12s\n", "消息大小", "延迟(us)", "带宽(MB/s)");
        printf("----------------------------------------\n");
    }

    for (int s = 0; s < num_sizes; s++) {
        int msg_size = sizes[s];
        char *send_buf = (char *)malloc(msg_size);
        char *recv_buf = (char *)malloc(msg_size);
        memset(send_buf, rank, msg_size);

        MPI_Barrier(MPI_COMM_WORLD);
        double t_start = MPI_Wtime();

        for (int r = 0; r < repeats; r++) {
            if (rank == 0) {
                MPI_Send(send_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD);
                MPI_Recv(recv_buf, msg_size, MPI_BYTE, 1, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            } else if (rank == 1) {
                MPI_Recv(recv_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
                MPI_Send(send_buf, msg_size, MPI_BYTE, 0, 0, MPI_COMM_WORLD);
            }
        }

        double t_end = MPI_Wtime();
        double elapsed = (t_end - t_start) / repeats;  // 单次往返时间
        double latency_us = elapsed / 2.0 * 1e6;       // 单程延迟（微秒）
        double bandwidth = msg_size / (elapsed / 2.0) / 1e6;  // MB/s

        if (rank == 0)
            printf("%12d %12.2f %12.2f\n", msg_size, latency_us, bandwidth);

        free(send_buf); free(recv_buf);
    }

    MPI_Finalize();
    return 0;
}
```

典型输出（InfiniBand 网络）：

```
     消息大小      延迟(us)    带宽(MB/s)
----------------------------------------
           1        1.50        0.67
           8        1.52        5.26
          64        1.60       40.00
         512        1.80      284.44
        4096        2.50     1638.40
       32768        5.20     6301.54
      262144       25.00    10485.76
     1048576       85.00    12335.48
     4194304      300.00    13981.01
```

## 5. 拓扑感知的非阻塞通信

在大规模程序中，了解网络拓扑可以优化通信模式：

```c
// 网格通信：每个进程与上下左右邻居通信
int rank, size, dims[2], periods[2], coords[2];
MPI_Comm grid_comm;

dims[0] = dims[1] = 0;
periods[0] = periods[1] = 1;  // 周期性边界
MPI_Dims_create(size, 2, dims);
MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &grid_comm);
MPI_Cart_coords(grid_comm, rank, 2, coords);

int up, down, left, right;
MPI_Cart_shift(grid_comm, 0, 1, &up, &down);
MPI_Cart_shift(grid_comm, 1, 1, &left, &right);

double buf_up[N], buf_down[N], buf_left[N], buf_right[N];
MPI_Request reqs[4];

// 同时与四个邻居通信
MPI_Irecv(buf_up, N, MPI_DOUBLE, up, 0, grid_comm, &reqs[0]);
MPI_Irecv(buf_down, N, MPI_DOUBLE, down, 1, grid_comm, &reqs[1]);
MPI_Isend(local_data, N, MPI_DOUBLE, up, 1, grid_comm, &reqs[2]);
MPI_Isend(local_data, N, MPI_DOUBLE, down, 0, grid_comm, &reqs[3]);

// 在等待通信时执行本地计算
compute_interior(grid, N);

// 等待所有边界通信完成
MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

// 现在可以计算边界区域
compute_boundary(grid, buf_up, buf_down, buf_left, buf_right);
```

## 6. 常见陷阱与调试

### 陷阱1：缓冲区过早修改

```c
// 错误！Isend 发起后立即修改缓冲区可能导致数据损坏
MPI_Isend(buf, N, MPI_DOUBLE, dest, tag, comm, &req);
buf[0] = 999.0;  // 危险：通信可能还没读取完 buf

// 正确做法：等通信完成再修改
MPI_Wait(&req, MPI_STATUS_IGNORE);
buf[0] = 999.0;  // 安全
```

### 陷阱2：Request 泄漏

```c
// 每个 Isend/Irecv 都必须有对应的 Wait/Test/Cancel
// 否则 MPI_Request 对象不会被释放，造成资源泄漏
MPI_Request req;
MPI_Irecv(buf, N, MPI_INT, src, tag, comm, &req);
// ... 忘记 MPI_Wait → 泄漏！程序长时间运行会耗尽资源
```

### 陷阱3：接收缓冲区过小

```c
// 发送 100 个 int，但接收缓冲区只有 50 个 int
int send_buf[100], recv_buf[50];
MPI_Request req;
MPI_Irecv(recv_buf, 100, MPI_INT, src, tag, comm, &req);  // 缓冲区溢出！
// 正确：确保接收缓冲区足够大
```

### 陷阱4：重复 Wait 同一个 Request

```c
MPI_Request req;
MPI_Isend(buf, N, MPI_INT, dest, tag, comm, &req);
MPI_Wait(&req, MPI_STATUS_IGNORE);  // 第一次 Wait 成功
MPI_Wait(&req, MPI_STATUS_IGNORE);  // 未定义行为！Request 已被释放
```

### 陷阱5：在 Wait 之前修改接收缓冲区

```c
MPI_Request req;
MPI_Irecv(buf, N, MPI_INT, src, tag, comm, &req);
// 错误：在 Wait 之前使用 buf 的值
printf("buf[0] = %d\n", buf[0]);  // buf 的内容尚未确定！
MPI_Wait(&req, MPI_STATUS_IGNORE);
printf("buf[0] = %d\n", buf[0]);  // 现在可以安全访问
```

### 陷阱6：阻塞与非阻塞混用同一消息

```c
// 不要用阻塞 Recv 接收同一个消息
MPI_Request req;
MPI_Irecv(buf, N, MPI_INT, src, tag, comm, &req);
MPI_Recv(buf, N, MPI_INT, src, tag, comm, &status);  // 错误！两个 Recv 匹配同一条消息
```

### 调试技巧

```bash
# 使用 MPICH 的内置请求跟踪
MPIR_CVAR_DEBUG_HOLD=1 mpirun -np 4 ./program

# 使用 MUST 检测非阻塞通信错误（泄漏、重复使用等）
mustrun -np 4 ./program

# 使用 Intel Trace Analyzer 可视化通信模式
# 编译时加 -tcollect
mpicc -tcollect -o program program.c
# 运行后生成 .stf 文件，用 ITA 打开
```

## 7. 非阻塞集合通信（MPI-3）

MPI-3 引入了非阻塞集合通信，允许在集合通信期间执行计算：

```c
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    double local_val = (double)(rank + 1);
    double global_sum;

    MPI_Request req;

    // 发起非阻塞 Allreduce
    MPI_Iallreduce(&local_val, &global_sum, 1, MPI_DOUBLE,
                   MPI_SUM, MPI_COMM_WORLD, &req);

    // 在 Allreduce 进行中执行其他计算
    do_independent_work();

    // 等待 Allreduce 完成
    MPI_Wait(&req, MPI_STATUS_IGNORE);

    printf("进程%d: sum = %.1f\n", rank, global_sum);

    MPI_Finalize();
    return 0;
}
```

### 非阻塞集合通信函数

MPI-3 提供了几乎所有集合通信的非阻塞版本：

```c
MPI_Ibcast(...)
MPI_Iscatter(...)
MPI_Igather(...)
MPI_Iallgather(...)
MPI_Ireduce(...)
MPI_Iallreduce(...)
MPI_Iscan(...)
MPI_Iexscan(...)
MPI_Ireduce_scatter(...)
MPI_Ireduce_scatter_block(...)
MPI_Ialltoall(...)
```

## 8. 完整实战：Jacobi 迭代并行求解

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <mpi.h>

/*
 * 并行 Jacobi 迭代求解二维拉普拉斯方程
 * 使用非阻塞通信重叠边界交换与内部计算
 *
 * 网格大小: N x N
 * 按行分块，每个进程负责 N/size 行
 * 每次迭代需要与上下邻居交换边界行
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 200;               // 网格大小
    int max_iter = 1000;       // 最大迭代次数
    double tol = 1e-6;         // 收敛阈值

    int local_rows = N / size;  // 每个进程的行数（简化，假设整除）
    int total_rows = local_rows + 2;  // 加上上下 ghost 行

    // 分配网格（包括 ghost 行）
    double **grid = (double **)malloc(total_rows * sizeof(double *));
    double **new_grid = (double **)malloc(total_rows * sizeof(double *));
    for (int i = 0; i < total_rows; i++) {
        grid[i] = (double *)calloc(N, sizeof(double));
        new_grid[i] = (double *)calloc(N, sizeof(double));
    }

    // 边界条件：上边界 = 1.0，其余 = 0.0
    if (rank == 0)
        for (int j = 0; j < N; j++)
            grid[1][j] = 1.0;  // 第一行（index 1，index 0 是 ghost）

    int top = (rank > 0) ? rank - 1 : MPI_PROC_NULL;
    int bottom = (rank < size - 1) ? rank + 1 : MPI_PROC_NULL;

    MPI_Request reqs[4];
    double t_start = MPI_Wtime();
    int iter;
    double max_diff;

    for (iter = 0; iter < max_iter; iter++) {
        // 1. 非阻塞交换边界行
        MPI_Irecv(grid[0], N, MPI_DOUBLE, top, 0, MPI_COMM_WORLD, &reqs[0]);
        MPI_Irecv(grid[total_rows - 1], N, MPI_DOUBLE, bottom, 1, MPI_COMM_WORLD, &reqs[1]);
        MPI_Isend(grid[1], N, MPI_DOUBLE, top, 1, MPI_COMM_WORLD, &reqs[2]);
        MPI_Isend(grid[local_rows], N, MPI_DOUBLE, bottom, 0, MPI_COMM_WORLD, &reqs[3]);

        // 2. 在等待边界数据时，先计算内部区域（第 3 行到第 local_rows-1 行）
        for (int i = 2; i < local_rows; i++) {
            for (int j = 1; j < N - 1; j++) {
                new_grid[i][j] = 0.25 * (grid[i-1][j] + grid[i+1][j] +
                                          grid[i][j-1] + grid[i][j+1]);
            }
        }

        // 3. 等待边界通信完成
        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // 4. 计算边界行（现在 ghost 数据已就绪）
        for (int j = 1; j < N - 1; j++) {
            new_grid[1][j] = 0.25 * (grid[0][j] + grid[2][j] +
                                      grid[1][j-1] + grid[1][j+1]);
            new_grid[local_rows][j] = 0.25 * (grid[local_rows-1][j] +
                                                grid[local_rows+1][j] +
                                                grid[local_rows][j-1] +
                                                grid[local_rows][j+1]);
        }

        // 5. 计算最大变化量
        max_diff = 0.0;
        for (int i = 1; i <= local_rows; i++)
            for (int j = 1; j < N - 1; j++) {
                double diff = fabs(new_grid[i][j] - grid[i][j]);
                if (diff > max_diff) max_diff = diff;
            }

        // 6. 全局求最大变化量
        double global_max_diff;
        MPI_Allreduce(&max_diff, &global_max_diff, 1, MPI_DOUBLE,
                      MPI_MAX, MPI_COMM_WORLD);

        // 7. 交换网格指针
        double **tmp = grid;
        grid = new_grid;
        new_grid = tmp;

        // 8. 检查收敛
        if (global_max_diff < tol) {
            if (rank == 0)
                printf("Jacobi 迭代在第 %d 步收敛 (diff=%.2e)\n",
                       iter + 1, global_max_diff);
            break;
        }
    }

    double t_end = MPI_Wtime();
    if (rank == 0)
        printf("总耗时: %.4f 秒 (%d 次迭代)\n", t_end - t_start, iter);

    // 清理
    for (int i = 0; i < total_rows; i++) {
        free(grid[i]);
        free(new_grid[i]);
    }
    free(grid);
    free(new_grid);

    MPI_Finalize();
    return 0;
}
```

**Makefile**：

```makefile
CC = mpicc
CFLAGS = -Wall -O2 -lm

jacobi: jacobi.c
	$(CC) $(CFLAGS) -o $@ $<

run4: jacobi
	mpirun -np 4 ./jacobi

run16: jacobi
	mpirun -np 16 ./jacobi

clean:
	rm -f jacobi

.PHONY: run4 run16 clean
```

## 9. 真实应用场景

### 9.1 MPI 在天气预报中的应用

全球天气预报模型（如 ECMWF 的 IFS）使用非阻塞通信实现计算与通信重叠：

- 每个时间步，需要与 6 个邻居（三维网格上的上下左右前后）交换边界数据
- 使用 `MPI_Isend`/`MPI_Irecv` 同时发起所有边界通信
- 在等待通信期间，计算网格内部的物理过程（辐射传输、对流参数化等）
- 最后 `MPI_Waitall` 确保所有边界数据就绪后，计算边界区域

这种重叠策略在万核规模的超算上可以节省 20%-40% 的通信等待时间。

### 9.2 MPI 在计算流体力学（CFD）中的应用

OpenFOAM 等 CFD 软件使用非阻塞通信处理非结构化网格：

- 每个子域的边界面数量不均匀，通信量不规则
- 使用 `MPI_Waitsome` 处理异步完成的边界交换
- 计算内部单元格的通量，与边界交换重叠

### 9.3 分布式深度学习

在数据并行训练中：
- 前向传播和反向传播是纯计算
- 梯度同步（Allreduce）是通信
- 使用非阻塞 Allreduce，在梯度计算的同时启动通信
- 某些框架甚至将梯度计算与参数更新流水线化，利用 `MPI_Waitany` 处理按层完成的梯度同步
