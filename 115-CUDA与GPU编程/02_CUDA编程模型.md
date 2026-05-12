# CUDA编程模型

## Grid/Block/Thread层次结构

CUDA采用三级线程组织方式来映射到GPU硬件：

```
Grid
 ├── Block (0,0,0)   Block (1,0,0)   ...  Block (Bx-1, By-1, Bz-1)
 │   ├── Thread(0,0,0)
 │   ├── Thread(1,0,0)
 │   ├── ...
 │   └── Thread(Tx-1, Ty-1, Tz-1)
 │
 └── ...
```

每个维度的关键信息：
- **Grid**：由多个Block组成，维度上限取决于GPU型号（通常各维度65535）
- **Block**：由多个Thread组成，线程总数上限通常为1024
- **Thread**：最小执行单元

## 内置变量

CUDA提供了一系列内置变量供kernel使用：

| 变量 | 类型 | 含义 |
|------|------|------|
| `threadIdx.x/y/z` | `uint3` | 线程在Block内的索引 |
| `blockIdx.x/y/z` | `uint3` | Block在Grid内的索引 |
| `blockDim.x/y/z` | `dim3` | 每个Block的维度大小 |
| `gridDim.x/y/z` | `dim3` | Grid的维度大小 |

## Kernel函数基础

Kernel是运行在GPU上的函数，使用`__global__`修饰符声明。

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// 最简单的kernel：每个线程打印自己的坐标
__global__ void hello_kernel() {
    printf("Block(%d,%d,%d) Thread(%d,%d,%d)\n",
           blockIdx.x, blockIdx.y, blockIdx.z,
           threadIdx.x, threadIdx.y, threadIdx.z);
}

int main() {
    // 启动2个Block，每个Block有4个线程
    dim3 grid(2);        // 2个Block
    dim3 block(4);       // 每个Block 4个线程

    hello_kernel<<<grid, block>>>();
    cudaDeviceSynchronize();  // 等待GPU完成

    return 0;
}
```

## 启动配置：<<<gridDim, blockDim>>>

Kernel的启动语法为 `kernel<<<gridDim, blockDim, sharedMemSize, stream>>>`。

```cpp
// 1D配置：最常见
kernel<<<256, 128>>>();    // 256个Block，每个128个线程，共32768个线程

// 2D配置：处理图像等二维数据
dim3 grid(16, 16);         // 16x16 = 256个Block
dim3 block(32, 32);        // 32x32 = 1024个线程/Block
kernel<<<grid, block>>>();

// 3D配置：处理体积数据
dim3 grid(8, 8, 8);        // 512个Block
dim3 block(8, 8, 8);       // 512个线程/Block
kernel<<<grid, block>>>();

// 带共享内存和流
kernel<<<grid, block, 4096, stream>>>();
//                         ↑     ↑
//                    共享内存(bytes)  CUDA流
```

## dim3类型

`dim3`是CUDA提供的三维向量类型，用于指定Grid和Block的维度：

```cpp
dim3 a;            // 默认(1, 1, 1)
dim3 b(10);        // (10, 1, 1)
dim3 c(10, 20);    // (10, 20, 1)
dim3 d(10, 20, 30); // (10, 20, 30)
```

## 完整示例：向量加法

向量加法是CUDA编程的"Hello World"，展示完整的GPU编程流程：

```cpp
#include <cstdio>
#include <cuda_runtime.h>

// Kernel：每个线程计算一个元素的加法
__global__ void vector_add(const float* A, const float* B, float* C, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        C[idx] = A[idx] + B[idx];
    }
}

int main() {
    int N = 1000000;
    size_t bytes = N * sizeof(float);

    // 1. 分配主机内存
    float* h_A = (float*)malloc(bytes);
    float* h_B = (float*)malloc(bytes);
    float* h_C = (float*)malloc(bytes);

    // 初始化数据
    for (int i = 0; i < N; i++) {
        h_A[i] = (float)i;
        h_B[i] = (float)(i * 2);
    }

    // 2. 分配设备内存
    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, bytes);
    cudaMalloc(&d_B, bytes);
    cudaMalloc(&d_C, bytes);

    // 3. 数据从主机复制到设备
    cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, bytes, cudaMemcpyHostToDevice);

    // 4. 配置kernel启动参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    // 5. 启动kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);

    // 6. 检查错误
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("Kernel launch error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    // 7. 等待GPU完成并将结果拷回主机
    cudaDeviceSynchronize();
    cudaMemcpy(h_C, d_C, bytes, cudaMemcpyDeviceToHost);

    // 8. 验证结果
    bool correct = true;
    for (int i = 0; i < N; i++) {
        if (h_C[i] != h_A[i] + h_B[i]) {
            correct = false;
            break;
        }
    }
    printf("Result: %s\n", correct ? "PASS" : "FAIL");

    // 9. 释放内存
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
    free(h_A);
    free(h_B);
    free(h_C);

    return 0;
}
```

## CUDA函数修饰符

CUDA提供四种函数修饰符：

| 修饰符 | 执行位置 | 调用位置 | 用途 |
|--------|----------|----------|------|
| `__global__` | GPU | CPU或GPU | Kernel函数 |
| `__device__` | GPU | GPU | 设备端辅助函数 |
| `__host__` | CPU | CPU | 普通C++函数（默认） |
| `__device__ __host__` | 两端均可 | 两端均可 | 通用函数 |

```cpp
// __device__函数：只能在GPU上调用，被kernel调用
__device__ float square(float x) {
    return x * x;
}

// __host__ __device__函数：CPU和GPU都能使用
__host__ __device__ float max_val(float a, float b) {
    return a > b ? a : b;
}

__global__ void apply_square(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        data[idx] = square(data[idx]);  // 调用__device__函数
    }
}
```

## 错误检查宏

在实际项目中，推荐使用宏封装CUDA API调用的错误检查：

```cpp
#include <cstdio>
#include <cuda_runtime.h>

#define CUDA_CHECK(call)                                                    \
    do {                                                                    \
        cudaError_t err = call;                                             \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "CUDA Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

#define CUDA_KERNEL_CHECK()                                                 \
    do {                                                                    \
        cudaError_t err = cudaGetLastError();                               \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Kernel Error at %s:%d - %s\n",                \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
        err = cudaDeviceSynchronize();                                      \
        if (err != cudaSuccess) {                                           \
            fprintf(stderr, "Sync Error at %s:%d - %s\n",                  \
                    __FILE__, __LINE__, cudaGetErrorString(err));           \
            exit(EXIT_FAILURE);                                             \
        }                                                                   \
    } while (0)

// 使用示例
int main() {
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));

    kernel<<<4, 256>>>(d_data, 1024);
    CUDA_KERNEL_CHECK();

    CUDA_CHECK(cudaFree(d_data));
    return 0;
}
```

## 线程块大小选择指南

选择线程块大小的经验法则：

1. **通常是warp大小的倍数**：32、64、128、256、512、1024
2. **128或256是常见最佳选择**：兼顾占用率和资源使用
3. **考虑资源限制**：
   - 每个SM的最大线程数（通常2048）
   - 每个SM的最大Block数（通常16-32）
   - 每个线程的寄存器数限制
   - 共享内存大小限制

```cpp
// 计算占用率的基本方法
int device;
cudaGetDevice(&device);
cudaDeviceProp prop;
cudaGetDeviceProperties(&prop, device);

int maxThreadsPerSM = prop.maxThreadsPerMultiProcessor;  // 通常2048
int maxBlocksPerSM = prop.maxBlocksPerMultiProcessor;
int warpSize = prop.warpSize;  // 32

printf("Device: %s\n", prop.name);
printf("SM count: %d\n", prop.multiProcessorCount);
printf("Max threads per SM: %d\n", maxThreadsPerSM);
printf("Max blocks per SM: %d\n", maxBlocksPerSM);
printf("Shared memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);
```

## 小结

1. CUDA的Grid/Block/Thread三级层次对应了软件抽象到硬件映射的关系
2. 正确计算线程索引是CUDA编程的基础技能
3. 使用错误检查宏是编写健壮CUDA程序的必备实践
4. 线程块大小应选择32的倍数，128-256通常是好的起点
