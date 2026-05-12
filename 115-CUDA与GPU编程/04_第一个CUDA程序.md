# 第一个CUDA程序

## 开发环境配置

在开始编写CUDA程序之前，需要安装NVIDIA CUDA Toolkit和兼容的GPU驱动。

```bash
# 检查CUDA版本
nvcc --version
# 输出示例:
# nvcc: NVIDIA (R) Cuda compiler driver
# Copyright (c) 2005-2024 NVIDIA Corporation
# Built on Wed_Apr_17_19:19:55_PDT_2024
# Cuda compilation tools, release 12.4, V12.4.131

# 检查GPU信息
nvidia-smi
# 输出示例:
# +-----------------------------------------------------------------------------+
# | NVIDIA-SMI 550.54.15    Driver Version: 550.54.15    CUDA Version: 12.4     |
# |-------------------------------+----------------------+----------------------+
# | GPU  Name        Persistence-M| Bus-Id        Disp.A | Volatile Uncorr. ECC |
# | Fan  Temp  Perf  Pwr:Usage/Cap|         Memory-Usage | GPU-Util  Compute M. |
# |===============================+======================+======================|
# |   0  NVIDIA A100-SXM...  Off  | 00000000:00:04.0 Off |                    0 |
# | N/A   32C    P0    50W / 400W |      0MiB / 81920MiB |      0%      Default |
# +-------------------------------+----------------------+----------------------+

# 查看GPU计算能力
nvidia-smi --query-gpu=compute_cap --format=csv
# 输出: compute_cap
#        8.0

# 列出系统中所有GPU及其详细信息
nvidia-smi -L
# GPU 0: NVIDIA A100-SXM4-80GB (UUID: GPU-xxxx)
```

### 环境变量配置

```bash
# 添加到 ~/.bashrc 或 ~/.zshrc
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH

# 验证安装
nvcc --version && nvidia-smi
```

### IDE配置建议

- **VS Code**: 安装 "Nsight Visual Studio Code Edition" 和 "CUDA" 扩展
- **CLion**: 内置CUDA支持，创建项目时选择"CUDA Executable"
- **命令行**: 直接使用nvcc，配合gdb进行调试

## Hello World Kernel

最简单的CUDA程序，理解kernel的基本结构和执行流程。

```cpp
// hello.cu - 最简CUDA程序
#include <cstdio>
#include <cuda_runtime.h>

// __global__修饰符表示这是一个kernel函数
// kernel运行在GPU上，从CPU端调用
__global__ void hello_from_gpu() {
    printf("Hello from GPU! Block: %d, Thread: %d\n",
           blockIdx.x, threadIdx.x);
}

int main() {
    printf("Hello from CPU!\n");

    // <<<2, 4>>> 表示启动2个Block，每个Block有4个线程
    // 共8个线程同时执行这个kernel
    hello_from_gpu<<<2, 4>>>();

    // 等待GPU上所有操作完成
    // GPU是异步执行的，不等待可能在结果返回前就退出了
    cudaDeviceSynchronize();

    // 检查kernel执行是否出错
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("CUDA error: %s\n", cudaGetErrorString(err));
        return 1;
    }

    printf("Back to CPU!\n");
    return 0;
}
```

编译和运行：

```bash
nvcc -o hello hello.cu
./hello
```

预期输出（线程执行顺序不确定）：

```
Hello from CPU!
Hello from GPU! Block: 0, Thread: 0
Hello from GPU! Block: 0, Thread: 1
Hello from GPU! Block: 0, Thread: 2
Hello from GPU: Block: 0, Thread: 3
Hello from GPU! Block: 1, Thread: 0
Hello from GPU! Block: 1, Thread: 1
Hello from GPU! Block: 1, Thread: 2
Hello from GPU! Block: 1, Thread: 3
Back to CPU!
```

**注意**：GPU端printf的输出顺序是不确定的，因为不同warp/Block的执行顺序由硬件调度器决定。

### CUDA函数修饰符速查

| 修饰符 | 执行位置 | 调用位置 | 用途 |
|--------|----------|----------|------|
| `__global__` | GPU | CPU或GPU（动态并行） | Kernel函数 |
| `__device__` | GPU | GPU | 设备端辅助函数 |
| `__host__` | CPU | CPU | 主机端函数（默认） |
| `__host__ __device__` | CPU和GPU | CPU和GPU | 两边都能编译的函数 |
| `__noinline__` | GPU | GPU | 禁止内联（调试用） |
| `__forceinline__` | GPU | GPU | 强制内联 |

## nvcc编译过程详解

nvcc编译CUDA代码分为多个阶段：

```
hello.cu (混合C++/CUDA代码)
    │
    ├─→ [nvcc前端] → 使用CUDA词法分析器分离host代码和device代码
    │
    ├─→ [cicc] → 将device代码编译为PTX（Parallel Thread Execution虚拟汇编）
    │              PTX是设备无关的中间表示
    │
    ├─→ [ptxas] → 将PTX编译为cubin（特定GPU架构的二进制指令，即SASS）
    │
    ├─→ [fatbinary] → 可选：打包多个架构的cubin到一个fatbin中
    │
    └─→ [host编译器(g++/cl)] → 编译host代码，链接cubin生成最终可执行文件
            │
            ▼
        hello (可执行文件，包含host二进制 + GPU二进制)
```

### PTX vs SASS

- **PTX**（Parallel Thread Execution）：设备无关的虚拟汇编，类似LLVM IR。新GPU可以通过JIT编译运行旧PTX代码（向前兼容）。
- **SASS**（Streaming ASSembler）：特定GPU架构的真实机器码，性能最优但不具备兼容性。

```bash
# 查看PTX代码
nvcc -ptx -o kernel.ptx kernel.cu
cat kernel.ptx
# 输出:
# .version 8.4
# .target sm_80
# .address_size 64
# .visible .entry _Z13hello_from_gpuv() {
#    .reg .b32 %r<10>;
#    ...
# }

# 查看SASS代码（需要cuobjdump）
nvcc -cubin -o kernel.cubin kernel.cu
cuobjdump -sass kernel.cubin
# 输出:
# _Z13hello_from_gpuv:
#  MOV R1, c[0x0][0x28]
#  ...
```

### 常用编译选项详解

```bash
# ========== 架构相关 ==========
# 指定目标架构（提高性能，减少JIT编译开销）
nvcc -arch=sm_75 -o hello hello.cu       # Turing (RTX 20系列, T4)
nvcc -arch=sm_80 -o hello hello.cu       # Ampere (A100)
nvcc -arch=sm_86 -o hello hello.cu       # Ampere (RTX 30系列)
nvcc -arch=sm_89 -o hello hello.cu       # Ada Lovelace (RTX 40系列)
nvcc -arch=sm_90 -o hello hello.cu       # Hopper (H100)

# 同时支持多个架构（fatbin）- 更大的二进制但兼容更多GPU
nvcc -gencode arch=compute_70,code=compute_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -o multi_arch hello.cu

# ========== 优化相关 ==========
nvcc -O3 -o hello hello.cu               # 最高优化（生产环境推荐）
nvcc -O0 -o hello hello.cu               # 无优化（调试用）
nvcc --use_fast_math -o hello hello.cu   # 快速数学函数（精度略有损失）
nvcc -ftz=true -o hello hello.cu         # 将denormal浮点数视为零
nvcc -prec-div=false -o hello hello.cu   # 降低除法精度换取速度
nvcc -prec-sqrt=false -o hello hello.cu  # 降低开方精度换取速度

# ========== 调试相关 ==========
nvcc -G -g -o hello hello.cu             # 完整调试信息（用于cuda-gdb/nsight）
nvcc -lineinfo -o hello hello.cu         # 行号信息（Nsight Compute用，不影响优化）

# ========== 信息输出 ==========
nvcc --keep -o hello hello.cu            # 保留所有中间文件
nvcc --ptxas-options=-v -o hello hello.cu # 显示每个kernel的寄存器/共享内存使用
# 输出示例:
# ptxas info    : Compiling entry function '_Z13hello_from_gpuv'
# ptxas info    : Function properties for _Z13hello_from_gpuv
#     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
# ptxas info    : Used 8 registers, 40 bytes cmem[0]

nvcc -Xptxas -dlcm=cg -o hello hello.cu # 全局内存默认走L2缓存
nvcc -Xptxas -dlcm=ca -o hello hello.cu # 全局内存默认走L1缓存

# ========== 多文件项目 ==========
nvcc -c kernel1.cu -o kernel1.o
nvcc -c kernel2.cu -o kernel2.o
nvcc kernel1.o kernel2.o -o app          # 链接
```

## 完整的向量加法程序

这是一个完整的CUDA程序，包含错误检查、内存管理、数据验证等生产级代码规范。

```cpp
// vector_add.cu - 完整的向量加法程序（带错误检查）
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cuda_runtime.h>

// ========== CUDA错误检查宏 ==========
#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d - %s\n",              \
                __FILE__, __LINE__, cudaGetErrorString(err));       \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// 或者更详细的版本:
#define CUDA_CHECK_LAST() do {                                      \
    cudaError_t err = cudaGetLastError();                           \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d - %s: %s\n",          \
                __FILE__, __LINE__, "kernel launch",                \
                cudaGetErrorString(err));                           \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
    err = cudaDeviceSynchronize();                                  \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d - %s: %s\n",          \
                __FILE__, __LINE__, "kernel execution",             \
                cudaGetErrorString(err));                           \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

#define N (1 << 20)  // 1M元素

// ========== Kernel定义 ==========
__global__ void vector_add(const float* __restrict__ a,
                           const float* __restrict__ b,
                           float* __restrict__ c,
                           int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        c[idx] = a[idx] + b[idx];
    }
}

// ========== CPU参考实现（用于验证正确性）==========
void vector_add_cpu(const float* a, const float* b, float* c, int n) {
    for (int i = 0; i < n; i++) {
        c[i] = a[i] + b[i];
    }
}

// ========== 主函数 ==========
int main() {
    const size_t bytes = N * sizeof(float);

    // 1. 分配主机内存
    float* h_a = (float*)malloc(bytes);
    float* h_b = (float*)malloc(bytes);
    float* h_c = (float*)malloc(bytes);        // GPU结果
    float* h_c_ref = (float*)malloc(bytes);    // CPU参考结果

    // 2. 初始化主机数据
    for (int i = 0; i < N; i++) {
        h_a[i] = (float)(rand() % 1000) / 100.0f;
        h_b[i] = (float)(rand() % 1000) / 100.0f;
    }

    // 3. 分配设备内存
    float *d_a, *d_b, *d_c;
    CUDA_CHECK(cudaMalloc(&d_a, bytes));
    CUDA_CHECK(cudaMalloc(&d_b, bytes));
    CUDA_CHECK(cudaMalloc(&d_c, bytes));

    // 4. 将数据从主机复制到设备
    CUDA_CHECK(cudaMemcpy(d_a, h_a, bytes, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_b, h_b, bytes, cudaMemcpyHostToDevice));

    // 5. 配置kernel执行参数
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    printf("Launching kernel: %d blocks × %d threads = %d total threads\n",
           blocksPerGrid, threadsPerBlock, blocksPerGrid * threadsPerBlock);

    // 6. 启动kernel
    vector_add<<<blocksPerGrid, threadsPerBlock>>>(d_a, d_b, d_c, N);
    CUDA_CHECK_LAST();  // 检查kernel启动和执行错误

    // 7. 将结果从设备复制回主机
    CUDA_CHECK(cudaMemcpy(h_c, d_c, bytes, cudaMemcpyDeviceToHost));

    // 8. CPU参考计算
    vector_add_cpu(h_a, h_b, h_c_ref, N);

    // 9. 验证结果
    float max_error = 0.0f;
    int error_count = 0;
    for (int i = 0; i < N; i++) {
        float err = fabsf(h_c[i] - h_c_ref[i]);
        if (err > 1e-5f) {
            error_count++;
            if (error_count <= 5) {
                printf("Error at [%d]: GPU=%.6f, CPU=%.6f, diff=%.6f\n",
                       i, h_c[i], h_c_ref[i], err);
            }
        }
        if (err > max_error) max_error = err;
    }

    if (error_count == 0) {
        printf("PASSED! Max error: %.2e\n", max_error);
    } else {
        printf("FAILED! %d errors, max error: %.2e\n", error_count, max_error);
    }

    // 10. 释放资源
    CUDA_CHECK(cudaFree(d_a));
    CUDA_CHECK(cudaFree(d_b));
    CUDA_CHECK(cudaFree(d_c));
    free(h_a);
    free(h_b);
    free(h_c);
    free(h_c_ref);

    return 0;
}
```

编译运行：
```bash
nvcc -O3 -arch=sm_80 -o vector_add vector_add.cu
./vector_add
# 输出:
# Launching kernel: 4096 blocks × 256 threads = 1048576 total threads
# PASSED! Max error: 0.00e+00
```

### `__restrict__`关键字的作用

`__restrict__`告诉编译器该指针不会与其他指针重叠，允许编译器更激进地优化（如将load合并为向量load）。在GPU上这通常能带来5-15%的性能提升。

### cudaMemcpy方向枚举

| 枚举值 | 方向 | 典型用途 |
|--------|------|----------|
| `cudaMemcpyHostToDevice` | CPU -> GPU | 传输输入数据 |
| `cudaMemcpyDeviceToHost` | GPU -> CPU | 取回计算结果 |
| `cudaMemcpyDeviceToDevice` | GPU -> GPU | GPU内部数据搬运 |
| `cudaMemcpyDefault` | 自动判断 | 统一内存（Unified Memory）使用 |

## 计时与性能测量

使用CUDA事件精确测量GPU执行时间——这是推荐的GPU计时方式，精度可达0.5微秒。

```cpp
// timing.cu - CUDA事件计时 + 带宽和算力测量
#include <cstdio>
#include <cuda_runtime.h>

// 一个简单的内存带宽测试kernel
__global__ void scale_add_kernel(float* __restrict__ a,
                                 const float* __restrict__ b,
                                 float scale, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        // 2 FLOP: 一次乘法 + 一次加法
        // 3次内存访问: 读a, 读b, 写a
        a[idx] = a[idx] * scale + b[idx];
    }
}

// 一个计算密集型kernel（用于测量算力）
__global__ void compute_kernel(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        float x = data[idx];
        // 200次FMA操作 = 400 FLOP
        // 内存访问: 1读 + 1写 = 2次 = 8 bytes
        // 算术强度: 400 / 8 = 50 FLOP/byte (远高于A100拐点9.75)
        #pragma unroll 10
        for (int i = 0; i < 200; i++) {
            x = fmaf(x, 1.001f, 0.001f);  // fused multiply-add
        }
        data[idx] = x;
    }
}

int main() {
    const int N = 1 << 26;  // 256M元素 = 1GB
    const size_t bytes = N * sizeof(float);
    float *d_data, *d_data2;
    cudaMalloc(&d_data, bytes);
    cudaMalloc(&d_data2, bytes);

    // 初始化
    cudaMemset(d_data, 0, bytes);
    cudaMemset(d_data2, 1, bytes);

    // 创建CUDA事件
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // ======== 测试1: 内存带宽 ========
    int threads = 256;
    int blocks = (N + threads - 1) / threads;

    // 预热（消除首次启动的JIT编译和上下文初始化开销）
    scale_add_kernel<<<blocks, threads>>>(d_data, d_data2, 0.5f, N);
    cudaDeviceSynchronize();

    const int NUM_ITERS = 20;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        scale_add_kernel<<<blocks, threads>>>(d_data, d_data2, 0.5f, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= NUM_ITERS;

    // scale_add: 读a(4B) + 读b(4B) + 写a(4B) = 12 bytes/元素
    float bw_measured = (3.0f * bytes) / (ms * 1e6);  // GB/s
    float bw_peak = 2000.0f;  // A100峰值2 TB/s

    // 算力: 2 FLOP/元素
    float gflops = (2.0f * N) / (ms * 1e6);  // GFLOPS

    printf("=== Memory-bound kernel (scale_add) ===\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  Bandwidth: %.1f GB/s (%.1f%% of peak %.0f GB/s)\n",
           bw_measured, bw_measured / bw_peak * 100, bw_peak);
    printf("  Compute: %.1f GFLOPS\n", gflops);
    printf("  Arithmetic intensity: %.2f FLOP/byte\n\n",
           2.0f * N / (3.0f * bytes));

    // ======== 测试2: 计算密集 ========
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        compute_kernel<<<blocks, threads>>>(d_data, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    cudaEventElapsedTime(&ms, start, stop);
    ms /= NUM_ITERS;

    // compute_kernel: 200 FMA = 400 FLOP/线程
    // 内存: 读4B + 写4B = 8 bytes/线程
    float compute_gflops = (400.0f * N) / (ms * 1e6);  // GFLOPS
    float compute_tflops = compute_gflops / 1000.0f;    // TFLOPS
    float compute_bw = (2.0f * bytes) / (ms * 1e6);     // GB/s

    printf("=== Compute-bound kernel ===\n");
    printf("  Time: %.3f ms\n", ms);
    printf("  Compute: %.1f GFLOPS (%.2f TFLOPS)\n", compute_gflops, compute_tflops);
    printf("  Theoretical FP32 peak: 19.5 TFLOPS (A100)\n");
    printf("  Efficiency: %.1f%%\n", compute_tflops / 19.5f * 100);
    printf("  Bandwidth: %.1f GB/s (should be low for compute-bound)\n", compute_bw);
    printf("  Arithmetic intensity: %.1f FLOP/byte (well above ridge point 9.75)\n",
           400.0f * N / (2.0f * bytes));

    // 清理
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    cudaFree(d_data);
    cudaFree(d_data2);

    return 0;
}
```

编译运行：
```bash
nvcc -O3 -arch=sm_80 -o timing timing.cu
./timing
# 典型输出 (A100):
# === Memory-bound kernel (scale_add) ===
#   Time: 0.782 ms
#   Bandwidth: 1535.3 GB/s (76.8% of peak 2000 GB/s)
#   Compute: 2718.9 GFLOPS
#   Arithmetic intensity: 0.17 FLOP/byte
#
# === Compute-bound kernel ===
#   Time: 1.435 ms
#   Compute: 71780.1 GFLOPS (71.8 TFLOPS)
#   Theoretical FP32 peak: 19.5 TFLOPS (A100)
#   Efficiency: 368.2%
#   Bandwidth: 712.7 GB/s (should be low for compute-bound)
#   Arithmetic intensity: 50.0 FLOP/byte (well above ridge point 9.75)
```

### CUDA事件 vs CPU计时

| 特性 | CUDA事件 | CPU计时（std::chrono） |
|------|----------|----------------------|
| 精度 | ~0.5微秒 | ~100纳秒（但不准确） |
| 测量对象 | GPU执行时间 | 包含CPU开销和GPU排队时间 |
| 异步性 | 正确处理 | 无法正确处理异步kernel |
| 推荐度 | GPU计时首选 | 仅用于CPU端代码 |

## 设备属性查询

在运行时获取GPU信息，用于自适应配置（例如根据GPU型号选择最优Block大小）。

```cpp
// device_query.cu - 完整的GPU设备信息查询
#include <cstdio>
#include <cuda_runtime.h>

void print_device_info() {
    int deviceCount;
    cudaGetDeviceCount(&deviceCount);
    printf("Found %d CUDA device(s)\n\n", deviceCount);

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop;
        cudaGetDeviceProperties(&prop, i);

        printf("========================================\n");
        printf("Device %d: %s\n", i, prop.name);
        printf("========================================\n");

        // 基本信息
        printf("Compute Capability: %d.%d\n", prop.major, prop.minor);
        printf("CUDA Driver/Runtime: ");
        int driverVersion, runtimeVersion;
        cudaDriverGetVersion(&driverVersion);
        cudaRuntimeGetVersion(&runtimeVersion);
        printf("%d.%d / %d.%d\n",
               driverVersion / 1000, driverVersion % 100,
               runtimeVersion / 1000, runtimeVersion % 100);

        // 内存
        printf("\n--- Memory ---\n");
        printf("Total Global Memory: %.2f GB (%zu bytes)\n",
               prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0),
               prop.totalGlobalMem);
        printf("Shared Memory per Block: %zu bytes\n", prop.sharedMemPerBlock);
        printf("Shared Memory per SM: %zu bytes\n", prop.sharedMemPerMultiprocessor);
        printf("L2 Cache Size: %d bytes (%d KB)\n",
               prop.l2CacheSize, prop.l2CacheSize / 1024);
        printf("Constant Memory: %zu bytes\n", prop.totalConstMem);
        printf("Registers per Block: %d\n", prop.regsPerBlock);
        printf("Registers per SM: %d\n", prop.regsPerMultiprocessor);

        // 计算资源
        printf("\n--- Compute Resources ---\n");
        printf("MultiProcessor (SM) Count: %d\n", prop.multiProcessorCount);
        printf("CUDA Cores per SM: ");
        // 根据架构推算核心数
        int cores_per_sm = 0;
        switch (prop.major) {
            case 6: cores_per_sm = (prop.minor == 0) ? 64 : 128; break; // Pascal
            case 7: cores_per_sm = (prop.minor == 0) ? 64 : 128; break; // Volta/Turing
            case 8: cores_per_sm = (prop.minor == 0) ? 64 : 128; break; // Ampere
            case 9: cores_per_sm = 128; break;                          // Hopper
            default: cores_per_sm = 0; break;
        }
        printf("%d (estimated, arch %d.%d)\n", cores_per_sm, prop.major, prop.minor);
        printf("Total CUDA Cores: %d\n", cores_per_sm * prop.multiProcessorCount);
        printf("Warp Size: %d threads\n", prop.warpSize);
        printf("Max Threads per Block: %d\n", prop.maxThreadsPerBlock);
        printf("Max Threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
        printf("Max Warps per SM: %d\n",
               prop.maxThreadsPerMultiProcessor / prop.warpSize);
        printf("Max Block Dimensions: %d x %d x %d\n",
               prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
        printf("Max Grid Dimensions: %d x %d x %d\n",
               prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
        printf("Max Blocks per SM: %d\n", prop.maxBlocksPerMultiProcessor);

        // 时钟频率
        printf("\n--- Clock Rates ---\n");
        printf("GPU Clock Rate: %.2f MHz (Boost: %.2f GHz)\n",
               prop.clockRate / 1000.0, prop.clockRate / 1e6);
        printf("Memory Clock Rate: %.2f MHz\n", prop.memoryClockRate / 1000.0);
        printf("Memory Bus Width: %d bits\n", prop.memoryBusWidth);

        // 计算理论带宽
        float bw = 2.0f * prop.memoryBusWidth / 8.0f  // DDR: x2
                   * prop.memoryClockRate / 1e6;       // MHz -> GHz -> bytes/cycle -> GB/s
        printf("Theoretical Memory Bandwidth: %.1f GB/s\n", bw);

        // 其他
        printf("\n--- Other ---\n");
        printf("Concurrent Kernels: %s\n",
               prop.concurrentKernels ? "Yes" : "No");
        printf("Unified Addressing: %s\n",
               prop.unifiedAddressing ? "Yes" : "No");
        printf("Managed Memory: %s\n",
               prop.managedMemory ? "Yes" : "No");
        printf("Page Migration: %s\n",
               prop.pageableMemoryAccess ? "Yes (Pageable)" : "No (Pinned only)");
        printf("ECC Enabled: %s\n", prop.ECCEnabled ? "Yes" : "No");
        printf("Integrated: %s\n", prop.integrated ? "Yes" : "No");
        printf("Can Map Host Memory: %s\n",
               prop.canMapHostMemory ? "Yes" : "No");
        printf("\n");
    }
}

// 自动选择最优配置
void auto_config(int N, int& blocks, int& threads) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    // 策略: 使用每SM的最大线程数的1/4作为Block大小
    // 并确保占用率足够
    threads = 256;  // 通用安全值
    int max_blocks = (N + threads - 1) / threads;

    // 限制grid大小避免过度配置
    int max_sm_blocks = prop.multiProcessorCount * (prop.maxThreadsPerMultiProcessor / threads);
    blocks = (max_blocks < max_sm_blocks * 4) ? max_blocks : max_sm_blocks * 4;

    printf("Auto config for N=%d: %d blocks x %d threads\n", N, blocks, threads);
    printf("  Estimated occupancy: %.1f%%\n",
           (float)(blocks * threads) / (prop.multiProcessorCount * prop.maxThreadsPerMultiProcessor) * 100);
}

int main() {
    print_device_info();

    printf("=== Auto Configuration ===\n");
    int blocks, threads;
    auto_config(1000000, blocks, threads);
    auto_config(100000000, blocks, threads);

    return 0;
}
```

编译运行：
```bash
nvcc -O3 -arch=sm_80 -o device_query device_query.cu
./device_query
```

### 计算理论峰值带宽的公式

```
理论带宽 = 内存时钟频率(MHz) × 内存总线宽度(bits) / 8 × 2(DDR)
例如A100: 1215 MHz × 5120 bits / 8 × 2 = 1555 GB/s (实际标称2 TB/s含boost)
```

## CUDA错误处理最佳实践

```cpp
// error_handling.cu - 完整的错误处理框架
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

// 方式1: 宏定义（最常用，附带文件和行号信息）
#define CUDA_CHECK(call) do {                                       \
    cudaError_t err = (call);                                       \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "CUDA error at %s:%d - %s: %s\n",          \
                __FILE__, __LINE__, #call, cudaGetErrorString(err));\
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// 方式2: 检查kernel执行错误（kernel是异步的，需要额外同步）
#define CUDA_CHECK_KERNEL() do {                                    \
    cudaError_t err = cudaGetLastError();                           \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "Kernel launch error: %s\n",                \
                cudaGetErrorString(err));                           \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
    err = cudaDeviceSynchronize();                                  \
    if (err != cudaSuccess) {                                       \
        fprintf(stderr, "Kernel execution error: %s\n",             \
                cudaGetErrorString(err));                           \
        exit(EXIT_FAILURE);                                         \
    }                                                               \
} while(0)

// 方式3: 使用回调函数（适合异步流）
void CUDART_CB errorCallback(cudaStream_t stream, cudaError_t status, void* userData) {
    if (status != cudaSuccess) {
        fprintf(stderr, "Async CUDA error in stream %p: %s\n",
                (void*)stream, cudaGetErrorString(status));
        // 可以在这里设置全局错误标志
    }
}

// 常见错误类型及诊断
void demonstrate_common_errors() {
    // 1. 未初始化的设备指针
    float* d_ptr;
    // cudaFree(d_ptr);  // 错误! 未分配的指针

    // 2. 越界访问
    float* d_data;
    cudaMalloc(&d_data, 100 * sizeof(float));
    // kernel<<<1, 256>>>(d_data, 256);  // 错误! 线程数超过数组大小
    // kernel<<<1, 256>>>(d_data, 100);  // 正确: 需要边界检查

    // 3. 内存溢出
    size_t huge = (size_t)1 << 40;  // 1TB
    // cudaMalloc(&d_ptr, huge);  // 错误! 超过显存容量

    // 4. 忘记同步
    // kernel<<<...>>>(...);
    // cudaMemcpy(h_ptr, d_ptr, ...);  // 错误! kernel可能还没执行完
    // cudaDeviceSynchronize();        // 正确: 先同步

    // 5. 在条件分支中使用__syncthreads()
    // 见"原子操作与同步"章节

    cudaFree(d_data);
}

// 查看当前CUDA错误状态
void check_cuda_status(const char* location) {
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("[%s] CUDA error: %s\n", location, cudaGetErrorString(err));
    } else {
        printf("[%s] CUDA OK\n", location);
    }
}

int main() {
    // 演示正确的错误检查
    float* d_data;
    CUDA_CHECK(cudaMalloc(&d_data, 1024 * sizeof(float)));

    // 检查分配后状态
    check_cuda_status("after malloc");

    CUDA_CHECK(cudaFree(d_data));
    printf("All checks passed!\n");

    return 0;
}
```

## 编译器优化技巧深入

```bash
# ========== 使用--use_fast_math启用快速数学函数 ==========
# 包含以下优化:
#   - __fmul_rn -> __fmul_rz (四舍五入->截断)
#   - 使用CUDA内部函数替代标准数学函数
#   - 启用-ftz (flush denormal to zero)
#   - 启用-prec-div=false, -prec-sqrt=false
nvcc -use_fast_math -o fast_math_test fast_math.cu

# ========== 生成PTX汇编代码 ==========
nvcc -ptx -o kernel.ptx kernel.cu
cat kernel.ptx  # 查看虚拟汇编

# ========== 使用-keep生成所有中间文件 ==========
nvcc -keep -o test test.cu
# 生成: test.cpp1.ii (预处理后的C++)
#       test.cudafe1.gpu (分离后的GPU代码)
#       test.cudafe1.c (分离后的CPU代码)
#       test.ptx (虚拟汇编)
#       test.sm_80.cubin (二进制)
#       test.fatbin (打包的fat二进制)

# ========== 使用fatbin包含多种架构 ==========
nvcc -gencode arch=compute_50,code=compute_50 \
     -gencode arch=compute_60,code=sm_60 \
     -gencode arch=compute_70,code=sm_70 \
     -gencode arch=compute_80,code=sm_80 \
     -gencode arch=compute_86,code=sm_86 \
     -o multi_arch multi_arch.cu
# 结果: 单个可执行文件支持多代GPU，运行时自动选择最优二进制

# ========== 查看kernel的资源使用 ==========
nvcc --ptxas-options=-v -o test test.cu 2>&1 | grep "ptxas info"
# 输出:
# ptxas info    : 0 bytes gmem
# ptxas info    : Compiling entry function '_Z14vector_add...'
# ptxas info    : Function properties for _Z14vector_add...
#     0 bytes stack frame, 0 bytes spill stores, 0 bytes spill loads
# ptxas info    : Used 16 registers, 40 bytes cmem[0]
```

### 寄存器使用与占用率的关系

kernel使用的寄存器数量直接影响SM上能驻留多少线程。例如A100每SM有65536个寄存器：

- kernel使用16个寄存器/线程: 最大驻留线程 = 65536/16 = 4096（但受max 2048限制）= 2048（100%占用率）
- kernel使用32个寄存器/线程: 最大驻留线程 = 65536/32 = 2048（100%占用率）
- kernel使用64个寄存器/线程: 最大驻留线程 = 65536/64 = 1024（50%占用率）
- kernel使用128个寄存器/线程: 最大驻留线程 = 65536/128 = 512（25%占用率）

如果寄存器用量过高，可以用`-maxrregcount=N`限制：

```bash
nvcc -maxrregcount=32 -o test test.cu
# 警告: 限制寄存器可能导致溢出到local memory（慢）
# 要用--ptxas-options=-v检查spill数量
```

## Unified Memory（统一内存）简介

从CUDA 6.0 / Kepler架构开始，NVIDIA引入了统一内存（Managed Memory），允许CPU和GPU共享同一指针，避免手动cudaMemcpy。

```cpp
// unified_memory.cu - 使用统一内存简化编程
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add_kernel(float* a, float* b, float* c, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) c[idx] = a[idx] + b[idx];
}

int main() {
    const int N = 1 << 20;
    const size_t bytes = N * sizeof(float);

    // 使用cudaMallocManaged代替cudaMalloc
    // 分配的内存CPU和GPU都能访问
    float *a, *b, *c;
    cudaMallocManaged(&a, bytes);
    cudaMallocManaged(&b, bytes);
    cudaMallocManaged(&c, bytes);

    // CPU端初始化（在CPU内存中）
    for (int i = 0; i < N; i++) {
        a[i] = (float)i;
        b[i] = (float)(i * 2);
    }

    // 在kernel执行前预取数据到GPU（可选但推荐）
    int device = 0;
    cudaMemPrefetchAsync(a, bytes, device);
    cudaMemPrefetchAsync(b, bytes, device);
    cudaMemPrefetchAsync(c, bytes, device);

    // 启动kernel - 直接使用指针，不需要cudaMemcpy
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    add_kernel<<<blocks, threads>>>(a, b, c, N);

    // 等待kernel完成
    cudaDeviceSynchronize();

    // 预取结果回CPU（可选但推荐）
    cudaMemPrefetchAsync(c, bytes, cudaCpuDeviceId);

    // CPU端直接读取结果
    float max_err = 0.0f;
    for (int i = 0; i < N; i++) {
        float expected = (float)i + (float)(i * 2);
        float err = fabsf(c[i] - expected);
        if (err > max_err) max_err = err;
    }
    printf("Max error: %.2e %s\n", max_err, max_err < 1e-5 ? "PASS" : "FAIL");

    // 释放统一内存
    cudaFree(a);
    cudaFree(b);
    cudaFree(c);

    return 0;
}
```

### Unified Memory vs 手动cudaMemcpy

| 特性 | 手动cudaMemcpy | Unified Memory |
|------|---------------|----------------|
| 编程复杂度 | 需要管理4个指针（h_*, d_*） | 只需1个指针 |
| 性能控制 | 完全控制传输时机和大小 | 依赖硬件页面迁移 |
| 页面大小 | N/A | GPU: 2MB大页面 / CPU: 4KB |
| 双精度支持 | 任意硬件 | 需要Pascal+ (CC 6.0+) |
| 生产环境 | 首选（性能可预测） | 适合原型开发和不规则访问模式 |
| 故障处理 | 显式错误 | 可能导致页面错误暂停 |

## 完整项目模板

```bash
# 项目结构:
# my_cuda_project/
# ├── CMakeLists.txt          # CMake构建文件
# ├── src/
# │   ├── main.cu             # 主程序
# │   ├── kernels.cu          # CUDA kernels
# │   └── kernels.h           # kernel声明
# ├── include/
# │   └── utils.h             # 工具函数
# └── build/                  # 构建目录
```

```cmake
# CMakeLists.txt - 最小化CUDA项目
cmake_minimum_required(VERSION 3.18)
project(my_cuda_project LANGUAGES CXX CUDA)

set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# 选择目标GPU架构
set(CMAKE_CUDA_ARCHITECTURES 70 75 80 86 89)

add_executable(my_app
    src/main.cu
    src/kernels.cu
)

# 链接CUDA运行时（默认自动链接）
target_include_directories(my_app PRIVATE include)

# 可选: 启用快速数学
# target_compile_options(my_app PRIVATE $<$<COMPILE_LANGUAGE:CUDA>:--use_fast_math>)
```

```bash
# CMake构建
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES="80"
make -j$(nproc)
./my_app
```

## 小结

1. `__global__`函数是CUDA kernel，从CPU调用，在GPU上执行。`<<<gridDim, blockDim>>>`指定线程配置
2. nvcc编译过程：host代码 → g++编译，device代码 → PTX → SASS，最终合并为fatbin
3. `cudaMemcpy`负责主机与设备间数据搬运，同步调用（内部会等待传输完成）
4. 使用CUDA事件进行精确的GPU计时，比CPU端计时更准确
5. 编译时指定`-arch=sm_XX`生成针对特定GPU优化的代码，避免运行时JIT开销
6. 完整的CUDA程序应包含：错误检查宏、预热、计时、结果验证
7. 统一内存（Unified Memory）可以简化编程但牺牲部分性能控制力
8. 寄存器使用量直接影响占用率，需要通过`--ptxas-options=-v`监控
