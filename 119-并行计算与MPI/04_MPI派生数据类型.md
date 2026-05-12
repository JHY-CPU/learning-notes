# MPI 派生数据类型

MPI 允许用户定义自定义数据类型，用于高效发送非连续数据或复合结构，避免多次调用 `MPI_Send`。派生数据类型是 MPI 中最强大也最容易被误解的特性之一。

## 0. 为什么需要派生类型

### 问题一：结构体的 padding 问题

假设要发送一个结构体：

```c
struct Particle {
    double x, y, z;      // offset 0, 8, 16 — 共 24 字节
    int id;              // offset 24 — 4 字节
    // 编译器在此插入 4 字节 padding，使 mass 对齐到 8 字节边界
    double mass;         // offset 32 — 8 字节
};
// sizeof(struct Particle) = 40 字节（而非 36 字节）
```

如果直接用 `MPI_Send(&p, sizeof(p), MPI_BYTE, ...)` 发送，不同平台的 padding 可能不同，接收端解析会出错。MPI 派生数据类型精确描述内存布局，屏蔽平台差异。

### 问题二：非连续数据的发送

```
矩阵列：存储为行优先，要发送一列数据
内存布局: [1, 2, 3, | 4, 5, 6, | 7, 8, 9]
           ^        ^         ^
          第1列     第2列      第3列
要发送第2列: 需要取第2、5、8个元素（每隔3个取1个）
```

不使用派生类型就需要手动拷贝到连续缓冲区再发送。使用 `MPI_Type_vector` 可以直接描述这种非连续布局，MPI 实现可以在底层优化数据搬运。

### 问题三：性能优势

```
方案 A（无派生类型）：
  用户代码 → malloc 拷贝缓冲区 → 拷贝数据到连续缓冲区 → MPI_Send → 释放缓冲区
  开销：额外内存分配 + 数据拷贝

方案 B（使用派生类型）：
  用户代码 → MPI_Send（带派生类型）→ MPI 内部优化传输
  开销：无额外内存分配，可能直接 scatter-gather DMA
```

## 1. MPI_Type_contiguous——连续类型

`MPI_Type_contiguous` 从现有类型创建一个由 `count` 个连续元素组成的新类型。这是最简单的派生类型。

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 创建由 5 个连续 double 组成的类型
    MPI_Datatype vec5_type;
    MPI_Type_contiguous(5, MPI_DOUBLE, &vec5_type);
    MPI_Type_commit(&vec5_type);

    double data[5] = {1.1, 2.2, 3.3, 4.4, 5.5};

    if (rank == 0) {
        MPI_Send(data, 1, vec5_type, 1, 0, MPI_COMM_WORLD);
        printf("进程0: 发送了 5 个 double（作为 1 个 vec5_type）\n");
    } else if (rank == 1) {
        double buf[5];
        MPI_Recv(buf, 1, vec5_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("进程1 收到: ");
        for (int i = 0; i < 5; i++) printf("%.1f ", buf[i]);
        printf("\n");
    }

    MPI_Type_free(&vec5_type);
    MPI_Finalize();
    return 0;
}
```

### 底层实现

`MPI_Type_contiguous(5, MPI_DOUBLE, &new_type)` 在 MPI 内部创建了一个类型描述符：

```
new_type 描述:
  起始偏移: 0
  元素数: 5
  基础类型: MPI_DOUBLE (8字节)
  总跨度: 40 字节
```

发送时，MPI 知道需要从起始地址连续读取 40 字节。这与直接发送 `MPI_DOUBLE` 的 count=5 效果相同，但在集合通信等场景中，派生类型提供了更好的抽象。

## 2. MPI_Type_vector——向量类型

`MPI_Type_vector` 用于选取等间隔的块，适合矩阵的行/列操作。

```
原数组: [0 1 2 3 4 5 6 7 8 9 10 11]
         ^     ^     ^     ^
blocklength=1, stride=3, count=4 → 选取 {0, 3, 6, 9}
```

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 创建矩阵列类型：提取 4x3 矩阵的第二列
    MPI_Datatype col_type;
    // count=4 行, blocklength=1 每块 1 个元素, stride=3 每行 3 个元素
    MPI_Type_vector(4, 1, 3, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    double matrix[12] = {
        1, 2, 3,
        4, 5, 6,
        7, 8, 9,
        10, 11, 12
    };

    if (rank == 0) {
        // 发送第二列: 2, 5, 8, 11
        MPI_Send(&matrix[1], 1, col_type, 1, 0, MPI_COMM_WORLD);
        printf("进程0: 发送了第二列 {2, 5, 8, 11}\n");
    } else if (rank == 1) {
        double col[4];
        MPI_Recv(col, 4, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("进程1 收到第二列: ");
        for (int i = 0; i < 4; i++) printf("%.0f ", col[i]);
        printf("\n");
    }

    MPI_Type_free(&col_type);
    MPI_Finalize();
    return 0;
}
```

### 发送多列

```c
// 发送矩阵的第1列和第2列（blocklength=2）
MPI_Datatype two_cols;
// count=4 行, blocklength=2 每行取2个, stride=3 每行3个元素
MPI_Type_vector(4, 2, 3, MPI_DOUBLE, &two_cols);
MPI_Type_commit(&two_cols);

// 发送起点为 matrix[0]，即从第一列开始
MPI_Send(&matrix[0], 1, two_cols, dest, tag, comm);
// 接收端将收到: 1,2, 4,5, 7,8, 10,11（24 字节 = 8 个 double）
```

### Type_hvector——字节跨度版本

`MPI_Type_vector` 的 stride 以元素为单位。如果需要以字节为单位的 stride（例如处理有 padding 的结构体数组），使用 `MPI_Type_create_hvector`：

```c
// 处理包含 padding 的结构体数组中的某个字段
struct Record {
    double value;   // 8 字节
    int flags;      // 4 字节
    char pad[4];    // 4 字节 padding → sizeof = 16 字节
};

MPI_Datatype value_type;
// stride = 16 字节（一个 Record 的大小）
MPI_Type_create_hvector(N, 1, sizeof(struct Record), MPI_DOUBLE, &value_type);
MPI_Type_commit(&value_type);

// 发送所有 Record 的 value 字段
MPI_Send(&records[0].value, 1, value_type, dest, tag, comm);
```

## 3. MPI_Type_indexed——索引类型

`MPI_Type_indexed` 允许为每个块指定不同的偏移量，比 `MPI_Type_vector` 更灵活。

```
原数组: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
偏移:   {2, 5, 7}  块长: {1, 2, 1}
选取:   {2}, {5,6}, {7} → 总共 4 个元素
```

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    // 选取数组中的特定元素：第2,5,6,7个（偏移量2,5,6,7）
    int blocklengths[3] = {1, 2, 1};  // 每个块的元素数
    int displacements[3] = {2, 5, 7}; // 每个块的起始偏移

    MPI_Datatype indexed_type;
    MPI_Type_indexed(3, blocklengths, displacements, MPI_INT, &indexed_type);
    MPI_Type_commit(&indexed_type);

    int data[10] = {100, 200, 300, 400, 500, 600, 700, 800, 900, 1000};

    if (rank == 0) {
        MPI_Send(data, 1, indexed_type, 1, 0, MPI_COMM_WORLD);
    } else if (rank == 1) {
        int buf[4];  // 总共 1+2+1=4 个元素
        MPI_Recv(buf, 4, MPI_INT, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("收到: %d %d %d %d\n", buf[0], buf[1], buf[2], buf[3]);
        // 输出: 300 600 700 800
    }

    MPI_Type_free(&indexed_type);
    MPI_Finalize();
    return 0;
}
```

### 应用：发送稀疏矩阵的非零元素

```c
// 稀疏矩阵 CSR 格式中，发送每一行的非零元素
int row = 5;
int nnz = row_ptr[row+1] - row_ptr[row];  // 第 row 行的非零元个数
int start = row_ptr[row];

int *disps = (int *)malloc(nnz * sizeof(int));
int *blens = (int *)malloc(nnz * sizeof(int));
for (int i = 0; i < nnz; i++) {
    disps[i] = start + i;
    blens[i] = 1;
}

MPI_Datatype row_nz_type;
MPI_Type_indexed(nnz, blens, disps, MPI_DOUBLE, &row_nz_type);
MPI_Type_commit(&row_nz_type);

MPI_Send(values, 1, row_nz_type, dest, tag, comm);
free(disps); free(blens);
MPI_Type_free(&row_nz_type);
```

## 4. MPI_Type_create_struct——结构体类型

`MPI_Type_create_struct` 用于精确描述结构体的内存布局，是处理复合数据结构的终极方案。

```c
#include <stdio.h>
#include <stddef.h>
#include <mpi.h>

struct Particle {
    double x, y, z;      // 位置 (3 个 double)
    int id;              // 粒子编号 (1 个 int)
    double mass;         // 质量 (1 个 double)
};

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    struct Particle p;

    // 定义每个字段的个数
    int blocklengths[3] = {3, 1, 1};

    // 定义每个字段的类型
    MPI_Datatype types[3] = {MPI_DOUBLE, MPI_INT, MPI_DOUBLE};

    // 定义每个字段相对于结构体起始的偏移量
    // 使用 offsetof 宏精确获取，避免手动计算 padding
    MPI_Aint displacements[3];
    displacements[0] = offsetof(struct Particle, x);
    displacements[1] = offsetof(struct Particle, id);
    displacements[2] = offsetof(struct Particle, mass);

    MPI_Datatype particle_type;
    MPI_Type_create_struct(3, blocklengths, displacements, types, &particle_type);
    MPI_Type_commit(&particle_type);

    if (rank == 0) {
        p.x = 1.0; p.y = 2.0; p.z = 3.0;
        p.id = 42;
        p.mass = 9.81;
        MPI_Send(&p, 1, particle_type, 1, 0, MPI_COMM_WORLD);
        printf("进程0: 发送粒子 id=%d pos=(%.1f,%.1f,%.1f) mass=%.2f\n",
               p.id, p.x, p.y, p.z, p.mass);
    } else if (rank == 1) {
        MPI_Recv(&p, 1, particle_type, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        printf("进程1: 收到粒子 id=%d pos=(%.1f,%.1f,%.1f) mass=%.2f\n",
               p.id, p.x, p.y, p.z, p.mass);
    }

    MPI_Type_free(&particle_type);
    MPI_Finalize();
    return 0;
}
```

### 发送粒子数组

```c
// 发送一个粒子数组（N 个连续的 Particle）
int N = 100;
MPI_Datatype particle_array;
MPI_Type_contiguous(N, particle_type, &particle_array);
MPI_Type_commit(&particle_array);

MPI_Send(particles, 1, particle_array, dest, tag, comm);

MPI_Type_free(&particle_array);
```

### 处理复杂嵌套结构

```c
struct Cell {
    int id;
    double coords[3];        // 3 个 double
    struct Particle *atoms;  // 指针不能直接发送！
    int num_atoms;
};

// 正确做法：分别发送标量和数组字段
// 标量部分
int blens[2] = {1, 3};
MPI_Datatype dtypes[2] = {MPI_INT, MPI_DOUBLE};
MPI_Aint disps[2];
disps[0] = offsetof(struct Cell, id);
disps[1] = offsetof(struct Cell, coords);

MPI_Datatype cell_header;
MPI_Type_create_struct(2, blens, disps, dtypes, &cell_header);
MPI_Type_commit(&cell_header);

// 先发送头部，再发送原子数组
MPI_Send(&cell, 1, cell_header, dest, 0, comm);
MPI_Send(cell.atoms, cell.num_atoms, particle_type, dest, 1, comm);

MPI_Type_free(&cell_header);
```

## 5. MPI_Type_create_resized——调整类型边界

当类型中的偏移量不是从 0 开始时，或者需要调整类型的 extent（跨度）时，`MPI_Type_create_resized` 非常有用。

```c
// 典型场景：为 Gather/Scatter 准备列类型
// 矩阵 A 按列存储，每列之间间隔一整行
double A[4][3];  // 4 行 3 列

MPI_Datatype col_type, resized_col;
MPI_Type_vector(4, 1, 3, MPI_DOUBLE, &col_type);

// 调整 extent 为 1 个 double（而非原来的 4*3=12 个 double）
MPI_Type_create_resized(col_type, 0, sizeof(double), &resized_col);
MPI_Type_commit(&resized_col);
MPI_Type_free(&col_type);

// 现在可以用 Gather 收集每列了
double local_col[4];
// 每个进程计算一列，放入 local_col
MPI_Gather(local_col, 4, MPI_DOUBLE,
           A, 1, resized_col,  // 每个进程贡献 1 个 resized_col
           0, comm);
```

## 6. MPI_Pack / MPI_Unpack——打包解包

当派生类型过于复杂时，可以用 `MPI_Pack` 手动将非连续数据打包到连续缓冲区中。这是一种"最后手段"——通常派生类型更高效。

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    double a = 3.14;
    int b[3] = {10, 20, 30};
    char c[] = "hello";

    // 计算打包后的总大小
    int pack_size = MPI_Pack_size(1, MPI_DOUBLE, MPI_COMM_WORLD, NULL)
                  + MPI_Pack_size(3, MPI_INT, MPI_COMM_WORLD, NULL)
                  + MPI_Pack_size(6, MPI_CHAR, MPI_COMM_WORLD, NULL);

    char *buf = (char *)malloc(pack_size);
    int position = 0;

    if (rank == 0) {
        // 打包：按顺序将数据写入连续缓冲区
        MPI_Pack(&a, 1, MPI_DOUBLE, buf, pack_size, &position, MPI_COMM_WORLD);
        MPI_Pack(b, 3, MPI_INT, buf, pack_size, &position, MPI_COMM_WORLD);
        MPI_Pack(c, 6, MPI_CHAR, buf, pack_size, &position, MPI_COMM_WORLD);

        // 使用 MPI_PACKED 类型发送
        MPI_Send(buf, position, MPI_PACKED, 1, 0, MPI_COMM_WORLD);
        printf("进程0: 打包了 %d 字节\n", position);
    } else if (rank == 1) {
        MPI_Recv(buf, pack_size, MPI_PACKED, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);

        double a2;
        int b2[3];
        char c2[6];

        // 解包（顺序必须与打包一致）
        position = 0;
        MPI_Unpack(buf, pack_size, &position, &a2, 1, MPI_DOUBLE, MPI_COMM_WORLD);
        MPI_Unpack(buf, pack_size, &position, b2, 3, MPI_INT, MPI_COMM_WORLD);
        MPI_Unpack(buf, pack_size, &position, c2, 6, MPI_CHAR, MPI_COMM_WORLD);

        printf("进程1 解包结果: a=%.2f b=[%d,%d,%d] c=%s\n",
               a2, b2[0], b2[1], b2[2], c2);
    }

    free(buf);
    MPI_Finalize();
    return 0;
}
```

### Pack vs 派生类型

| 特性 | Pack/Unpack | 派生类型 |
|------|------------|---------|
| 灵活性 | 高（可发送任意混合数据） | 中（需要预定义类型） |
| 性能 | 较低（额外数据拷贝） | 高（可能零拷贝） |
| 内存开销 | 需要额外打包缓冲区 | 无需额外缓冲区 |
| 使用场景 | 偶尔发送复杂数据 | 频繁发送固定格式数据 |
| 代码复杂度 | 简单直接 | 需要类型定义和管理 |

## 7. 类型的提交与释放

所有派生数据类型必须经过以下完整生命周期：

```c
MPI_Datatype my_type;

// 1. 创建：使用各种 Type_create_* 函数
MPI_Type_contiguous(10, MPI_DOUBLE, &my_type);

// 2. 提交：通知 MPI 系统该类型已就绪
//    MPI 可能在此时进行内部优化（如预计算 scatter-gather 表）
MPI_Type_commit(&my_type);

// 3. 使用：在 MPI_Send、MPI_Recv、MPI_File_write 等函数中使用
MPI_Send(buffer, 1, my_type, dest, tag, comm);
MPI_File_write(fh, buffer, 1, my_type, &status);

// 4. 释放：释放 MPI 内部资源
//    即使忘记释放，MPI_Finalize 也会清理，但显式释放是好习惯
MPI_Type_free(&my_type);
```

### 调试派生类型

```c
// 获取类型的 extent（起始偏移到结束的跨度）
MPI_Aint lb, extent;
MPI_Type_get_extent(my_type, &lb, &extent);
printf("类型 lb=%ld, extent=%ld 字节\n", (long)lb, (long)extent);

// 获取类型的 true extent（忽略 MPI_LB/MPI_ULB 标记）
MPI_Aint true_lb, true_extent;
MPI_Type_get_true_extent(my_type, &true_lb, &true_extent);
printf("true_lb=%ld, true_extent=%ld 字节\n", (long)true_lb, (long)true_extent);

// 获取类型的大小（实际数据字节数，不含 padding）
int type_size;
MPI_Type_size(my_type, &type_size);
printf("类型大小: %d 字节\n", type_size);
```

## 8. 完整实战：并行矩阵转置

```c
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>

/*
 * 并行矩阵转置
 * 矩阵按行分块存储，每个进程持有一些行
 * 转置后每个进程持有的是原矩阵的一些列
 */
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int N = 4;  // 矩阵 N x N（为简化取小值）
    int rows_per_proc = N / size;

    // 每个进程初始化自己的行块
    double *local_rows = (double *)malloc(rows_per_proc * N * sizeof(double));
    for (int i = 0; i < rows_per_proc; i++)
        for (int j = 0; j < N; j++)
            local_rows[i * N + j] = (double)(rank * rows_per_proc + i) * N + j;

    printf("进程%d 的原始行块:\n", rank);
    for (int i = 0; i < rows_per_proc; i++) {
        for (int j = 0; j < N; j++)
            printf("%4.0f ", local_rows[i * N + j]);
        printf("\n");
    }

    // 创建列块类型：从行块中提取一列
    MPI_Datatype col_block, resized_col;
    MPI_Type_vector(rows_per_proc, 1, N, MPI_DOUBLE, &col_block);
    MPI_Type_create_resized(col_block, 0, sizeof(double), &resized_col);
    MPI_Type_commit(&resized_col);
    MPI_Type_free(&col_block);

    // 分配转置后的存储
    double *transposed = (double *)malloc(N * rows_per_proc * sizeof(double));

    // Alltoall 交换：每个进程将自己行块的列分发给其他进程
    MPI_Alltoall(local_rows, rows_per_proc, resized_col,
                 transposed, rows_per_proc, MPI_DOUBLE,
                 MPI_COMM_WORLD);

    printf("进程%d 的转置结果:\n", rank);
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < rows_per_proc; j++)
            printf("%4.0f ", transposed[i * rows_per_proc + j]);
        printf("\n");
    }

    MPI_Type_free(&resized_col);
    free(local_rows);
    free(transposed);
    MPI_Finalize();
    return 0;
}
```

**Makefile**：

```makefile
CC = mpicc
CFLAGS = -Wall -O2

transpose: transpose.c
	$(CC) $(CFLAGS) -o $@ $<

run: transpose
	mpirun -np 4 ./transpose

clean:
	rm -f transpose

.PHONY: run clean
```

## 9. 性能分析

### 派生类型 vs 手动拷贝的性能对比

```c
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);
    int rank;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);

    int N = 1000;
    int cols = 1000;
    double *matrix = (double *)malloc(N * cols * sizeof(double));
    double *col_buf = (double *)malloc(N * sizeof(double));
    double start, elapsed;

    // 初始化矩阵
    for (int i = 0; i < N * cols; i++) matrix[i] = (double)i;

    // 方法1: 派生类型发送列
    MPI_Datatype col_type;
    MPI_Type_vector(N, 1, cols, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int rep = 0; rep < 100; rep++) {
        if (rank == 0)
            MPI_Send(&matrix[500], 1, col_type, 1, 0, MPI_COMM_WORLD);
        else if (rank == 1)
            MPI_Recv(col_buf, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
    }
    elapsed = MPI_Wtime() - start;
    if (rank == 0) printf("派生类型: %.6f 秒 (100次)\n", elapsed);

    // 方法2: 手动拷贝到连续缓冲区再发送
    MPI_Barrier(MPI_COMM_WORLD);
    start = MPI_Wtime();
    for (int rep = 0; rep < 100; rep++) {
        if (rank == 0) {
            for (int i = 0; i < N; i++)
                col_buf[i] = matrix[i * cols + 500];
            MPI_Send(col_buf, N, MPI_DOUBLE, 1, 0, MPI_COMM_WORLD);
        } else if (rank == 1) {
            MPI_Recv(col_buf, N, MPI_DOUBLE, 0, 0, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }
    }
    elapsed = MPI_Wtime() - start;
    if (rank == 0) printf("手动拷贝: %.6f 秒 (100次)\n", elapsed);

    MPI_Type_free(&col_type);
    free(matrix); free(col_buf);
    MPI_Finalize();
    return 0;
}
```

典型结果（取决于 MPI 实现和网络）：
- 派生类型可能略慢于手动拷贝（因为 MPI 内部也要做 scatter-gather）
- 但在大消息时差距缩小，且代码更简洁、可维护

## 10. 真实应用场景

### 10.1 天气预报中的不规则网格

天气模型使用非结构化网格（如 icosahedral grid），每个网格单元有不同数量的邻居。使用 `MPI_Type_indexed` 为每个进程定义其需要交换的边界单元类型，避免手动构建边界缓冲区。

### 10.2 分子动力学中的原子数据

在 LAMMPS 中，每个原子有位置（3 个 double）、速度（3 个 double）、力（3 个 double）、类型（1 个 int）、电荷（1 个 double）等属性。使用 `MPI_Type_create_struct` 定义原子类型，在进程间批量传输原子数据。

### 10.3 图像处理中的像素块

分布式图像处理中，使用 `MPI_Type_vector` 提取图像中的特定通道（如 RGB 中的 R 通道），或提取不连续的像素块进行局部处理。

## 11. MPI vs OpenMP vs Pthreads 数据传递对比

| 特性 | MPI 派生类型 | OpenMP | Pthreads |
|------|------------|--------|----------|
| 数据分布 | 显式描述内存布局 | 共享内存，无需描述 | 共享内存，无需描述 |
| 类型安全 | 编译时检查基础类型 | 无 | 无 |
| 跨节点 | 支持 | 不支持 | 不支持 |
| 复杂度 | 高（需要类型管理） | 低 | 低 |
| 性能 | 可能零拷贝 | 直接内存访问 | 直接内存访问 |

MPI 派生类型的核心价值在于：在分布式内存环境下，精确描述非连续数据布局，让 MPI 实现有机会做底层优化（如 RDMA scatter-gather），避免用户手动拷贝数据。
