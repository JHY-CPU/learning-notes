# Shared Memory优化

## 共享内存概述

共享内存（Shared Memory）位于SM芯片上，是CUDA性能优化的核心工具。同一线程块内的所有线程可以高效地共享数据，延迟约为全局内存的1/100。

```
SM内部结构:
┌──────────────────────────────────┐
│  Register File (65536个32位寄存器) │
│  ┌──────────────────────────┐    │
│  │  Shared Memory (48-164KB) │    │  ← 线程块内共享
│  └──────────────────────────┘    │
│  ┌────┐ ┌────┐ ┌────┐ ┌────┐   │
│  │Core│ │Core│ │Core│ │Core│... │  ← CUDA核心
│  └────┘ └────┘ └────┘ └────┘   │
└──────────────────────────────────┘
           ↓ 全局内存 (HBM)
```

## Bank Conflict

共享内存被分为32个bank（与warp大小一致），连续地址映射到连续bank：

```
Bank分布:
Address  0  4  8  12 ... 124   (32个bank)
Bank     0  1  2  3  ... 31

正常访问（无bank conflict）：
线程0读bank0，线程1读bank1，... 线程31读bank31  → 1个周期完成

Bank conflict示例（2路bank conflict）：
线程0读bank0，线程1读bank0，...  → 需要串行访问，延迟加倍
```

### Bank Conflict的产生与解决

```cpp
// 产生bank conflict的例子
__global__ void bank_conflict_bad(float* data) {
    __shared__ float s_data[256];
    int tid = threadIdx.x;

    s_data[tid] = data[tid];
    __syncthreads();

    // 2路bank conflict: 相邻线程间隔2个位置访问
    float val = s_data[tid * 2];  // 线程0读0,线程1读2,线程2读4...
    //                 0→bank0, 2→bank2, 4→bank4... 实际没有conflict
    // 但如果写成 s_data[tid * 32] 则完全冲突
}

// 避免bank conflict：padding技巧
__global__ void no_bank_conflict(float* data) {
    // +1 padding打破bank对齐
    __shared__ float s_data[256 + 1];
    int tid = threadIdx.x;

    s_data[tid] = data[tid];
    __syncthreads();

    // 现在相邻线程访问的地址映射到不同bank
    float val = s_data[tid * 32];
}

// 经典的转置bank conflict
__global__ void transpose_conflict(const float* input, float* output, int n) {
    __shared__ float tile[32][32];  // 读取时无conflict，写入时有conflict

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    // 读取：线程0读bank0，线程1读bank1... 无conflict
    tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    __syncthreads();

    // 写入转置：线程0写bank0，线程1写bank0... 有conflict!
    // 读取转置时也有conflict
    output[x * n + y] = tile[threadIdx.x][threadIdx.y];
}

// 解决：padding数组的列
__global__ void transpose_no_conflict(const float* input, float* output, int n) {
    __shared__ float tile[32][32 + 1];  // +1 padding

    int x = blockIdx.x * 32 + threadIdx.x;
    int y = blockIdx.y * 32 + threadIdx.y;

    tile[threadIdx.y][threadIdx.x] = input[y * n + x];
    __syncthreads();

    // 转置读取：threadIdx.x作为行索引，每行有33个元素
    // 同一warp内相邻线程读bank0,bank1,...bank31,bank0（下一行的bank0）
    // 但因为第33个元素属于下一个warp，所以当前warp内无conflict
    output[x * n + y] = tile[threadIdx.x][threadIdx.y];
}
```

## Tiling技术：矩阵乘法优化

Tiling（分块）是使用共享内存的经典优化技术。

### Naive矩阵乘法（全局内存版本）

```cpp
// 直接从全局内存读取，性能差
__global__ void matmul_naive(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // C[M x N] = A[M x K] * B[K x N]
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < N) {
        float sum = 0.0f;
        for (int k = 0; k < K; k++) {
            // 每次迭代读取全局内存2次
            sum += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = sum;
    }
}
```

### Tiled矩阵乘法（共享内存版本）

```cpp
#define TILE_SIZE 16

__global__ void matmul_tiled(const float* A, const float* B, float* C,
                              int M, int N, int K) {
    // 共享内存tile
    __shared__ float s_A[TILE_SIZE][TILE_SIZE];
    __shared__ float s_B[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;

    float sum = 0.0f;

    // 遍历所有tile
    for (int tile = 0; tile < (K + TILE_SIZE - 1) / TILE_SIZE; tile++) {
        // 加载A的一个tile
        int a_col = tile * TILE_SIZE + threadIdx.x;
        if (row < M && a_col < K) {
            s_A[threadIdx.y][threadIdx.x] = A[row * K + a_col];
        } else {
            s_A[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 加载B的一个tile
        int b_row = tile * TILE_SIZE + threadIdx.y;
        if (b_row < K && col < N) {
            s_B[threadIdx.y][threadIdx.x] = B[b_row * N + col];
        } else {
            s_B[threadIdx.y][threadIdx.x] = 0.0f;
        }

        // 同步：确保tile加载完成
        __syncthreads();

        // 计算这个tile的贡献
        // 从共享内存读取（快速），而不是全局内存
        for (int k = 0; k < TILE_SIZE; k++) {
            sum += s_A[threadIdx.y][k] * s_B[k][threadIdx.x];
        }

        // 同步：确保计算完成再加载下一个tile
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}
```

### 性能对比分析

```
Naive版本:
- 每个C元素读取A的K个元素 + B的K个元素 = 2K次全局内存读取
- 总全局内存访问: M * N * 2K

Tiled版本（tile大小T）:
- 每个tile加载 T² 个A元素 + T² 个B元素
- tile数量: K/T
- 每个线程的全局内存访问: 2K/T
- 减少比例: T倍 (T=16时，减少16倍全局内存访问)
```

## 向量点积优化

```cpp
// 使用共享内存的向量点积
__global__ void dot_product_shared(const float* A, const float* B, float* result, int N) {
    __shared__ float s_data[256];

    int tid = threadIdx.x;
    int idx = threadIdx.x + blockIdx.x * blockDim.x;

    // 每个线程计算局部乘积
    float local_sum = 0.0f;
    // Grid-stride loop
    for (int i = idx; i < N; i += blockDim.x * gridDim.x) {
        local_sum += A[i] * B[i];
    }

    s_data[tid] = local_sum;
    __syncthreads();

    // 归约求和（详见09_归约算法.md）
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    // 第一个线程写结果
    if (tid == 0) {
        atomicAdd(result, s_data[0]);
    }
}
```

## 动态共享内存的灵活使用

```cpp
// 可以在运行时决定共享内存大小
__global__ void flexible_tiled(float* A, float* B, float* C,
                                int M, int N, int K, int tileSize) {
    // 动态共享内存
    extern __shared__ float s_mem[];
    float* s_A = s_mem;
    float* s_B = s_mem + tileSize * tileSize;

    int row = blockIdx.y * tileSize + threadIdx.y;
    int col = blockIdx.x * tileSize + threadIdx.x;

    float sum = 0.0f;

    for (int t = 0; t < (K + tileSize - 1) / tileSize; t++) {
        int aIdx = t * tileSize + threadIdx.x;
        s_A[threadIdx.y * tileSize + threadIdx.x] =
            (row < M && aIdx < K) ? A[row * K + aIdx] : 0.0f;

        int bIdx = t * tileSize + threadIdx.y;
        s_B[threadIdx.y * tileSize + threadIdx.x] =
            (bIdx < K && col < N) ? B[bIdx * N + col] : 0.0f;

        __syncthreads();

        for (int k = 0; k < tileSize; k++) {
            sum += s_A[threadIdx.y * tileSize + k] * s_B[k * tileSize + threadIdx.x];
        }
        __syncthreads();
    }

    if (row < M && col < N) {
        C[row * N + col] = sum;
    }
}

// 启动时指定共享内存大小
void launch_flexible(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
    int tileSize = 16;
    dim3 threads(tileSize, tileSize);
    dim3 blocks((N + tileSize - 1) / tileSize, (M + tileSize - 1) / tileSize);
    size_t sharedMem = 2 * tileSize * tileSize * sizeof(float);
    flexible_tiled<<<blocks, threads, sharedMem>>>(d_A, d_B, d_C, M, N, K, tileSize);
}
```

## 卷积中的共享内存tiling

```cpp
// 一维卷积的共享内存优化
#define MASK_SIZE 5
#define BLOCK_SIZE 256

__global__ void conv1d_shared(const float* input, const float* mask,
                               float* output, int N) {
    // 共享内存需要包含halo区域
    __shared__ float s_input[BLOCK_SIZE + MASK_SIZE - 1];

    int tid = threadIdx.x;
    int idx = blockIdx.x * BLOCK_SIZE + tid;
    int halo_start = blockIdx.x * BLOCK_SIZE - MASK_SIZE / 2;

    // 加载主区域
    if (idx < N) {
        s_input[tid + MASK_SIZE / 2] = input[idx];
    }

    // 加载左侧halo
    if (tid < MASK_SIZE / 2) {
        int halo_idx = halo_start + tid;
        s_input[tid] = (halo_idx >= 0) ? input[halo_idx] : 0.0f;
    }

    // 加载右侧halo
    int right_halo_tid = BLOCK_SIZE + tid;
    if (tid < MASK_SIZE / 2 && right_halo_tid + MASK_SIZE / 2 < BLOCK_SIZE + MASK_SIZE - 1) {
        int halo_idx = halo_start + right_halo_tid;
        s_input[right_halo_tid] = (halo_idx < N) ? input[halo_idx] : 0.0f;
    }

    __syncthreads();

    // 计算卷积
    if (idx < N) {
        float sum = 0.0f;
        for (int i = 0; i < MASK_SIZE; i++) {
            sum += s_input[tid + i] * mask[i];
        }
        output[idx] = sum;
    }
}
```

## 小结

1. 共享内存是芯片上的高速存储，延迟约5个周期，远优于全局内存
2. Bank conflict会降低共享内存带宽，通过padding技巧可以避免
3. Tiling技术将数据分块加载到共享内存，大幅减少全局内存访问
4. 矩阵乘法的tiled实现是最经典的优化案例，可以减少T倍全局内存访问
5. `__syncthreads()`确保共享内存操作的正确同步
