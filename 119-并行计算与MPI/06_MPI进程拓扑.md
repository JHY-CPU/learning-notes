# MPI 进程拓扑

MPI 允许将通信子中的进程组织为逻辑拓扑结构（如网格、图），使代码更直观，同时帮助MPI运行时优化通信。

## 1. 笛卡尔拓扑

### 创建笛卡尔网格

`MPI_Cart_create` 将进程映射到一个 $d$ 维笛卡尔网格上。

```c
#include <stdio.h>
#include <mpi.h>

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 创建 2D 笛卡尔网格
    int ndims = 2;
    int dims[2] = {0, 0};          // 0 表示让MPI自动决定
    int periods[2] = {1, 1};       // 两个维度都周期性（环形边界）
    int reorder = 1;               // 允许MPI重新排列进程号以优化拓扑

    // 自动分解为近似正方形的网格
    MPI_Dims_create(size, ndims, dims);
    printf("网格维度: %d x %d\n", dims[0], dims[1]);

    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, ndims, dims, periods, reorder, &cart_comm);

    // 获取当前进程在网格中的坐标
    int coords[2];
    MPI_Cart_coords(cart_comm, rank, ndims, coords);
    printf("进程%d: 网格坐标 (%d, %d)\n", rank, coords[0], coords[1]);

    // 根据坐标获取 rank（双向查询）
    int cart_rank;
    MPI_Cart_rank(cart_comm, coords, &cart_rank);
    printf("进程%d: 笛卡尔rank = %d\n", rank, cart_rank);

    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
```

### 获取邻居进程

`MPI_Cart_shift` 获取指定维度上的前后邻居：

```c
int rank;
MPI_Comm_rank(cart_comm, &rank);

// dim=0: 第一个维度（行方向）
// disp=1: 移动1步
int source, dest;
MPI_Cart_shift(cart_comm, 0, 1, &source, &dest);
printf("进程%d: 行方向 - 上邻居=%d, 下邻居=%d\n", rank, source, dest);

// dim=1: 第二个维度（列方向）
MPI_Cart_shift(cart_comm, 1, 1, &source, &dest);
printf("进程%d: 列方向 - 左邻居=%d, 右邻居=%d\n", rank, source, dest);
```

### 二维热传导——笛卡尔拓扑实战

```c
// 简化的二维热传导（Jacobi迭代）
#define ROWS 100
#define COLS 100

int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    int dims[2] = {0, 0};
    int periods[2] = {0, 0};  // 非周期边界
    MPI_Dims_create(size, 2, dims);
    MPI_Comm cart_comm;
    MPI_Cart_create(MPI_COMM_WORLD, 2, dims, periods, 1, &cart_comm);

    int coords[2];
    MPI_Cart_coords(cart_comm, rank, 2, coords);

    // 每个进程负责子网格
    int local_rows = ROWS / dims[0];
    int local_cols = COLS / dims[1];
    double *grid = (double *)calloc((local_rows + 2) * (local_cols + 2), sizeof(double));
    double *new_grid = (double *)calloc((local_rows + 2) * (local_cols + 2), sizeof(double));

    // 获取四个邻居
    int up, down, left, right;
    MPI_Cart_shift(cart_comm, 0, 1, &up, &down);
    MPI_Cart_shift(cart_comm, 1, 1, &left, &right);

    // 创建列数据类型用于水平方向交换
    MPI_Datatype col_type;
    MPI_Type_vector(local_rows, 1, local_cols + 2, MPI_DOUBLE, &col_type);
    MPI_Type_commit(&col_type);

    for (int iter = 0; iter < 1000; iter++) {
        MPI_Request reqs[4];

        // 交换边界数据
        MPI_Irecv(&grid[1 * (local_cols + 2) + 0], 1, col_type,
                  left, 0, cart_comm, &reqs[0]);
        MPI_Isend(&grid[1 * (local_cols + 2) + 1], 1, col_type,
                  left, 1, cart_comm, &reqs[1]);
        MPI_Irecv(&grid[1 * (local_cols + 2) + (local_cols + 1)], 1, col_type,
                  right, 1, cart_comm, &reqs[2]);
        MPI_Isend(&grid[1 * (local_cols + 2) + local_cols], 1, col_type,
                  right, 0, cart_comm, &reqs[3]);

        MPI_Waitall(4, reqs, MPI_STATUSES_IGNORE);

        // Jacobi 迭代（内部点先算，边界交换完再算边界）
        for (int i = 1; i <= local_rows; i++)
            for (int j = 1; j <= local_cols; j++)
                new_grid[i * (local_cols + 2) + j] = 0.25 * (
                    grid[(i-1) * (local_cols + 2) + j] +
                    grid[(i+1) * (local_cols + 2) + j] +
                    grid[i * (local_cols + 2) + (j-1)] +
                    grid[i * (local_cols + 2) + (j+1)]);

        // 交换 grid 和 new_grid
        double *tmp = grid;
        grid = new_grid;
        new_grid = tmp;
    }

    free(grid);
    free(new_grid);
    MPI_Type_free(&col_type);
    MPI_Comm_free(&cart_comm);
    MPI_Finalize();
    return 0;
}
```

## 2. 图拓扑

`MPI_Graph_create` 支持任意图结构，适合不规则通信模式。

```c
int main(int argc, char *argv[]) {
    MPI_Init(&argc, &argv);

    int rank, size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &size);

    // 定义一个4节点的图：0-1, 0-2, 1-3, 2-3（菱形）
    int nnodes = 4;
    int index[4] = {2, 4, 6, 8};    // 每个节点的边累计数
    int edges[8] = {1, 2,            // 节点0连接1,2
                    0, 3,            // 节点1连接0,3
                    0, 3,            // 节点2连接0,3
                    1, 2};           // 节点3连接1,2

    MPI_Comm graph_comm;
    MPI_Graph_create(MPI_COMM_WORLD, nnodes, index, edges, 1, &graph_comm);

    // 查询邻居
    int nneighbors;
    MPI_Graph_neighbors_count(graph_comm, rank, &nneighbors);
    int *neighbors = (int *)malloc(nneighbors * sizeof(int));
    MPI_Graph_neighbors(graph_comm, rank, nneighbors, neighbors);

    printf("进程%d 的邻居: ", rank);
    for (int i = 0; i < nneighbors; i++) printf("%d ", neighbors[i]);
    printf("\n");

    free(neighbors);
    MPI_Comm_free(&graph_comm);
    MPI_Finalize();
    return 0;
}
```

## 3. MPI_Dims_create——自动维度分解

`MPI_Dims_create` 自动将 `nprocs` 个进程分配到 `ndims` 维网格中：

```c
int dims[3] = {0, 0, 0};  // 全部设0表示自动决定
MPI_Dims_create(24, 3, dims);
// dims 可能变成 {2, 3, 4} 或类似组合，使网格尽量接近正方体
printf("24个进程分解为: %d x %d x %d\n", dims[0], dims[1], dims[2]);
```

## 4. 坐标与Rank的互相转换

```c
// rank → 坐标
int coords[2];
MPI_Cart_coords(cart_comm, rank, 2, coords);

// 坐标 → rank
int computed_rank;
MPI_Cart_rank(cart_comm, coords, &computed_rank);
// computed_rank == rank
```
