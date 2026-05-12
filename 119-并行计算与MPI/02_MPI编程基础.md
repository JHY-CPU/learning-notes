# MPI 编程基础

MPI（Message Passing Interface）是分布式内存并行计算的标准编程接口，定义了一套进程间消息传递的函数规范。MPI 程序由多个独立的进程组成，每个进程拥有自己的地址空间，进程之间通过显式的消息传递来交换数据。

## 1. MPI 环境初始化与终止

### 1.1 底层原理

`MPI_Init` 不仅仅是"初始化"那么简单。它在底层执行以下操作：

1. **解析命令行参数**：提取 MPI 相关的启动参数（如 `--np`、`--host` 等），并将剩余参数回传给用户程序。
2. **建立通信基础设施**：在 TCP/IP 集群上，MPI 运行时（如 MPICH、Open MPI）会为每个进程建立 TCP 连接或注册 RDMA（远程直接内存访问）通道。
3. **创建默认通信子**：生成 `MPI_COMM_WORLD`，包含所有启动的进程，每个进程获得唯一的 rank 编号。
4. **分配内部资源**：为每个进程分配内部缓冲区、请求队列、标签匹配表等数据结构。

`MPI_Finalize` 则负责释放上述所有资源，关闭网络连接，确保所有挂起的消息传输完成。

在 OS 层面，`mpirun` 启动时会通过 SSH 或进程管理器（如 SLURM 的 `srun`、Hydra）在各节点上 fork 出工作进程，每个工作进程加载用户程序并调用 `MPI_Init`。

### 1.2 Hello World 完整示例

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    // MPI_Init 必须是第一个 MPI 调用，初始化 MPI 环境
    // 会解析命令行参数，创建通信基础设施
    MPI_Init(&argc, &argv);

    // 获取当前进程在 MPI_COMM_WORLD 中的编号（rank）
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 获取总进程数
    int size;
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 获取处理器名称（用于调试和诊断）
    char hostname[MPI_MAX_PROCESSOR_NAME];
    int name_len;
    MPI_Get_processor_name(hostname, &name_len);

    printf("Hello from process %d of %d, running on %s\n",
           rank, size, hostname);

    // MPI_Finalize 终止 MPI 环境，释放资源
    // 必须是最后一个 MPI 调用
    MPI_Finalize();
    return 0;
}
```

**Makefile**：

```makefile
CC = mpicc
CFLAGS = -Wall -O2
TARGETS = hello_mpi

all: $(TARGETS)

hello_mpi: hello_mpi.c
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -f $(TARGETS)

.PHONY: all clean
```

编译和运行：

```bash
mpicc -Wall -O2 -o hello_mpi hello_mpi.c
mpirun -np 4 ./hello_mpi
# 输出（顺序可能不同）：
# Hello from process 0 of 4, running on node01
# Hello from process 2 of 4, running on node01
# Hello from process 1 of 4, running on node01
# Hello from process 3 of 4, running on node01
```

注意：`printf` 的输出顺序是不确定的，因为各进程并行执行。如果需要有序输出，应使用点对点通信来协调。

### 1.3 MPI 启动参数详解

```bash
# 指定 8 个进程
mpirun -np 8 ./program

# 指定主机列表
mpirun -np 4 -host node01,node02 ./program

# 绑定进程到 CPU 核心（提高缓存局部性）
mpirun -np 4 --bind-to core ./program

# 显示详细的运行时信息（用于调试）
mpirun -np 4 --report-bindings ./program

# 使用 hostfile
mpirun -hostfile hosts.txt -np 16 ./program
# hosts.txt 内容：
# node01 slots=4
# node02 slots=4
# node03 slots=4
# node04 slots=4
```

## 2. 通信子（Communicator）

### 2.1 概念详解

通信子（Communicator）定义了一组可以互相通信的进程集合，以及通信的上下文（context）。上下文确保不同通信子之间的消息不会互相干扰——即使两个通信子包含相同的进程集合，它们的消息也是隔离的。

MPI 提供了三个预定义通信子：
- `MPI_COMM_WORLD`：包含所有启动的进程
- `MPI_COMM_SELF`：只包含调用进程自身
- `MPI_COMM_NULL`：空通信子，用于错误处理

每个通信子内部，进程有一个唯一的整数编号（rank），范围从 0 到 size-1。Rank 是进程在该通信子中的逻辑标识，不等同于物理处理器编号。

### 2.2 通信子拆分示例

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 创建通信子：将 MPI_COMM_WORLD 中的进程按奇偶分成两个组
    MPI_Comm new_comm;
    int color = rank % 2;  // 奇偶分组
    int key = rank;        // 在新组中的排序依据

    MPI_Comm_split(MPI_COMM_WORLD, color, key, &new_comm);

    int new_rank, new_size;
    MPI_Comm_rank(new_comm, &new_rank);
    MPI_Comm_size(new_comm, &new_size);

    printf("World rank %d -> 新通信子 rank %d (size %d, color %d)\n",
           rank, new_rank, new_size, color);

    // 在子通信子内执行集合通信（只有同 color 的进程参与）
    int local_val = rank;
    int sub_sum;
    MPI_Reduce(&local_val, &sub_sum, 1, MPI_INT, MPI_SUM, 0, new_comm);
    if (new_rank == 0)
        printf("color=%d 组的 sum = %d\n", color, sub_sum);

    // 释放自定义通信子
    MPI_Comm_free(&new_comm);
    MPI_Finalize();
    return 0;
}
```

运行 `mpirun -np 6 ./comm_split` 的典型输出：

```
World rank 0 -> 新通信子 rank 0 (size 3, color 0)
World rank 1 -> 新通信子 rank 0 (size 3, color 1)
World rank 2 -> 新通信子 rank 1 (size 3, color 0)
World rank 3 -> 新通信子 rank 1 (size 3, color 1)
World rank 4 -> 新通信子 rank 2 (size 3, color 0)
World rank 5 -> 新通信子 rank 2 (size 3, color 1)
color=0 组的 sum = 6
color=1 组的 sum = 9
```

### 2.3 通信子的常见用途

| 场景 | 说明 |
|------|------|
| 任务并行 | 不同子组执行不同类型的计算 |
| 主从模式 | 主进程在全局通信子中，工作进程在子通信子中 |
| 域分解 | 按空间区域划分，每组负责一个子域的计算和边界交换 |
| I/O 专用组 | 指定少量进程专门负责文件读写 |

## 3. 点对点通信

### 3.1 阻塞式发送与接收

`MPI_Send` 和 `MPI_Recv` 是最基本的点对点通信函数。

```c
/*
 * int MPI_Send(const void *buf, int count, MPI_Datatype datatype,
 *              int dest, int tag, MPI_Comm comm);
 *
 * buf:      发送缓冲区地址
 * count:    数据元素个数
 * datatype: 数据类型（MPI_INT, MPI_DOUBLE 等）
 * dest:     目标进程的 rank
 * tag:      消息标签（用于区分不同消息）
 * comm:     通信子
 */
```

#### 底层实现机制

MPI_Send 的实际行为取决于消息大小和实现：

- **短消息（Eager 协议）**：当消息较小时（通常 < 几 KB），MPI 直接将数据复制到接收端的内部缓冲区，发送端立即返回。这种方式不需要接收端事先调用 `MPI_Recv`。
- **长消息（Rendezvous 协议）**：当消息较大时，发送端先发送一个"请求发送"控制消息，接收端调用 `MPI_Recv` 后回复"准备好"，然后发送端才开始传输实际数据。这避免了大量数据的不必要复制。
- **RDMA 路径**：在 InfiniBand 等高速网络上，大消息可能直接通过 RDMA（远程直接内存访问）传输，绕过操作系统内核，由网卡硬件直接读写对方内存。

`MPI_Recv` 在底层维护一个**接收队列**。每次调用时，MPI 检查队列中是否有匹配的消息（根据 source、tag、comm 匹配）。如果有，立即完成；如果没有，进程阻塞等待。

### 3.2 完整示例：进程间数据交换

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
        if (rank == 0)
            fprintf(stderr, "需要至少2个进程\n");
        MPI_Finalize();
        return 1;
    }

    int data[5];
    MPI_Status status;

    if (rank == 0) {
        // 进程0：初始化数据并发送
        for (int i = 0; i < 5; i++)
            data[i] = (i + 1) * 10;  // {10, 20, 30, 40, 50}

        MPI_Send(data, 5, MPI_INT, 1, 99, MPI_COMM_WORLD);
        printf("进程0: 已发送数据 {%d, %d, %d, %d, %d}\n",
               data[0], data[1], data[2], data[3], data[4]);
    } else if (rank == 1) {
        // 进程1：接收数据
        MPI_Recv(data, 5, MPI_INT, 0, 99, MPI_COMM_WORLD, &status);
        printf("进程1: 已接收数据 {%d, %d, %d, %d, %d}\n",
               data[0], data[1], data[2], data[3], data[4]);

        // 查询实际接收的元素数量
        int count;
        MPI_Get_count(&status, MPI_INT, &count);
        printf("进程1: 实际接收 %d 个 int 元素\n", count);

        // 查询发送方 rank 和消息标签
        printf("进程1: 消息来自 rank %d, 标签 %d\n",
               status.MPI_SOURCE, status.MPI_TAG);
    }

    MPI_Finalize();
    return 0;
}
```

### 3.3 常用数据类型映射

| MPI 数据类型 | 对应 C 类型 | 大小（典型） |
|-------------|------------|-------------|
| `MPI_CHAR` | `char` | 1 字节 |
| `MPI_SHORT` | `short` | 2 字节 |
| `MPI_INT` | `int` | 4 字节 |
| `MPI_LONG` | `long` | 4/8 字节 |
| `MPI_LONG_LONG` | `long long` | 8 字节 |
| `MPI_FLOAT` | `float` | 4 字节 |
| `MPI_DOUBLE` | `double` | 8 字节 |
| `MPI_UNSIGNED_CHAR` | `unsigned char` | 1 字节 |
| `MPI_UNSIGNED` | `unsigned int` | 4 字节 |
| `MPI_BYTE` | 任意字节 | 1 字节 |

### 3.4 Sendrecv——安全地同时发送和接收

当两个进程需要交换数据时，如果都使用 `MPI_Send` + `MPI_Recv`，可能导致死锁：

```
死锁场景：
进程0: MPI_Send(to=1) → 阻塞等待进程1接收
进程1: MPI_Send(to=0) → 阻塞等待进程0接收
结果：双方都在等对方，永久阻塞
```

`MPI_Sendrecv` 在单次调用中同时完成发送和接收，内部使用临时缓冲区避免死锁：

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 每个进程与邻居交换数据，形成环形通信
    int send_to = (rank + 1) % size;
    int recv_from = (rank - 1 + size) % size;
    int send_val = rank * 100;
    int recv_val;

    MPI_Sendrecv(&send_val, 1, MPI_INT, send_to, 0,
                 &recv_val, 1, MPI_INT, recv_from, 0,
                 MPI_COMM_WORLD, MPI_STATUS_IGNORE);

    printf("进程%d: 发送 %d 给进程%d, 从进程%d 接收 %d\n",
           rank, send_val, send_to, recv_from, recv_val);

    MPI_Finalize();
    return 0;
}
```

### 3.5 MPI_ANY_SOURCE 与 MPI_ANY_TAG

```c
// 使用通配符接收来自任意源的消息
MPI_Recv(buf, count, datatype, MPI_ANY_SOURCE, MPI_ANY_TAG, comm, &status);
// 通过 status 获取实际的发送方和标签
int actual_source = status.MPI_SOURCE;
int actual_tag = status.MPI_TAG;
```

注意：`MPI_ANY_SOURCE` 在非阻塞通信中使用时要格外小心，可能干扰公平性调度。

### 3.6 常见错误与调试

#### 死锁场景一：交替 Send/Recv

```c
// 错误！可能导致死锁
if (rank == 0) {
    MPI_Send(data, N, MPI_INT, 1, 0, comm);  // 先发
    MPI_Recv(data, N, MPI_INT, 1, 0, comm, &status);  // 后收
} else if (rank == 1) {
    MPI_Send(data, N, MPI_INT, 0, 0, comm);  // 先发 ← 死锁！
    MPI_Recv(data, N, MPI_INT, 0, 0, comm, &status);  // 后收
}
// 两个进程都卡在 Send，都等对方 Recv

// 修正方法：让一侧先收后发
if (rank == 0) {
    MPI_Send(data, N, MPI_INT, 1, 0, comm);
    MPI_Recv(data, N, MPI_INT, 1, 0, comm, &status);
} else if (rank == 1) {
    MPI_Recv(data, N, MPI_INT, 0, 0, comm, &status);  // 先收
    MPI_Send(data, N, MPI_INT, 0, 0, comm);  // 后发
}
// 或使用 MPI_Sendrecv 一劳永逸地解决
```

#### 死锁场景二：缓冲区大小不匹配

```c
// 发送端发送 100 个 int
MPI_Send(data, 100, MPI_INT, 1, 0, comm);

// 接收端只分配了 50 个 int 的缓冲区 ← 缓冲区溢出！
int buf[50];
MPI_Recv(buf, 100, MPI_INT, 0, 0, comm, &status);  // 内存损坏
```

#### 调试技巧

```bash
# 使用 MPICH 的内置错误检查
mpirun -np 4 ./program 2>&1 | tee mpi_output.log

# 使用 TotalView 或 DDT 调试 MPI 程序
totalview mpirun -a -np 4 ./program

# 使用 MUST（MPI correctness checker）检测错误
mustrun -np 4 ./program

# 使用 Valgrind 检测内存错误
mpirun -np 2 valgrind --tool=memcheck ./program
```

## 4. 实战：并行计算 PI 值

利用数值积分法计算 pi，将积分区间分配给各进程：

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    long long N = 100000000;  // 积分总步数
    double h = 1.0 / N;      // 步长
    double local_sum = 0.0;

    // 每个进程处理 [rank, N) 中步长为 size 的子集
    for (long long i = rank; i < N; i += size) {
        double x = h * (i + 0.5);
        local_sum += 4.0 / (1.0 + x * x);
    }
    local_sum *= h;

    if (rank == 0) {
        double total_sum = local_sum;
        double tmp;
        for (int i = 1; i < size; i++) {
            MPI_Recv(&tmp, 1, MPI_DOUBLE, i, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
            total_sum += tmp;
        }
        printf("PI ≈ %.15f (误差 %.2e)\n", total_sum, total_sum - 3.141592653589793);
    } else {
        MPI_Send(&local_sum, 1, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD);
    }

    MPI_Finalize();
    return 0;
}
```

> 更高效的做法是使用归约操作（见第三章），这里演示点对点通信的用法。

## 5. 真实应用场景

### 5.1 MPI 在天气预报中的应用

全球天气预报（如欧洲中期天气预报中心 ECMWF 的 IFS 模型）使用 MPI 进行大规模并行计算：

- **域分解**：将地球表面划分为规则的网格，每个 MPI 进程负责一个子区域的计算
- **边界交换**：相邻区域之间通过 `MPI_Sendrecv` 交换边界层的数据（温度、气压、风速等）
- **集合通信**：使用 `MPI_Allreduce` 计算全局物理量（如全球总质量守恒检查）
- **I/O 并行**：指定专用 I/O 进程组，通过 `MPI_File_write_all` 并行输出结果

一个典型的全球天气模型可能使用 10,000+ 个 MPI 进程运行在超算上，每 6 小时做一次全球积分。

### 5.2 MPI 在分子动力学中的应用

LAMMPS（Large-scale Atomic/Molecular Massively Parallel Simulator）是最著名的 MPI 分子动力学模拟软件：

- 每个 MPI 进程管理空间中一个子区域内的原子
- 每个时间步，需要交换跨越子区域边界的原子信息
- 通过 `MPI_Allreduce` 计算全局温度、压力等热力学量

## 6. MPI vs OpenMP vs Pthreads 对比

| 特性 | MPI | OpenMP | Pthreads |
|------|-----|--------|----------|
| **内存模型** | 分布式内存 | 共享内存 | 共享内存 |
| **并行粒度** | 进程级 | 线程级 | 线程级 |
| **通信方式** | 显式消息传递 | 隐式（共享变量） | 显式同步原语 |
| **编程难度** | 较高 | 较低 | 高 |
| **可扩展性** | 极好（万级以上节点） | 受限于单节点核心数 | 受限于单节点核心数 |
| **适用场景** | 集群、超算 | 单节点多核 | 需要精细控制线程 |
| **典型用例** | 气候模拟、CFD | 图像处理、矩阵运算 | 服务器、嵌入式 |
| **数据分布** | 手动管理 | 自动（共享地址空间） | 手动管理 |

**选择建议**：
- 超过单节点规模 → 用 MPI
- 单节点多核加速 → 用 OpenMP（更简单）
- 需要精细线程控制 → 用 Pthreads
- 超算上的混合编程 → MPI（跨节点）+ OpenMP（节点内）
