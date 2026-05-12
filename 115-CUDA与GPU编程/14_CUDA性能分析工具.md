# CUDA性能分析工具

## 性能分析概述

CUDA提供了多个性能分析工具，从高层次的系统视图到底层的硬件计数器分析：

```
工具层次:
┌─────────────────────────────┐
│  Nsight Systems             │ ← 系统级时间线分析
├─────────────────────────────┤
│  Nsight Compute             │ ← GPU kernel级详细分析
├─────────────────────────────┤
│  CUDA Profiler API          │ ← 代码中嵌入分析
├─────────────────────────────┤
│  nvprof (legacy)            │ ← 命令行分析工具
└─────────────────────────────┘
```

## Nsight Systems

Nsight Systems提供系统级的GPU和CPU时间线分析。

```bash
# 基本使用
nsys profile ./my_cuda_app

# 指定输出文件和追踪选项
nsys profile --trace=cuda,nvtx --output=my_report ./my_cuda_app

# 分析Python程序
nsys profile python train.py

# 查看生成的报告
nsys stats my_report.nsys-rep
```

### 在代码中添加NVTX标记

```cpp
#include <nvtx3/nvToolsExt.h>

__global__ void compute_kernel(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        data[idx] = data[idx] * 2.0f;
    }
}

void instrumented_code() {
    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // NVTX范围标记 - 在Nsight Systems中显示为带颜色的范围
    nvtxRangePush("Data Initialization");
    cudaMemset(d_data, 0, N * sizeof(float));
    nvtxRangePop();

    nvtxRangePush("Compute Phase");
    compute_kernel<<<N / 256, 256>>>(d_data, N);
    nvtxRangePop();

    // 使用RAII方式（C++推荐）
    // nvtx3::scoped_range range("MyFunction");

    // 带颜色的标记
    nvtxEventAttributes_t eventAttrib = {0};
    eventAttrib.version = NVTX_VERSION;
    eventAttrib.size = NVTX_EVENT_ATTRIB_STRUCT_SIZE;
    eventAttrib.colorType = NVTX_COLOR_ARGB;
    eventAttrib.color = 0xFF00FF00;  // 绿色
    eventAttrib.messageType = NVTX_MESSAGE_TYPE_ASCII;
    eventAttrib.message.ascii = "Custom Kernel";
    nvtxMarkEx(&eventAttrib);

    cudaFree(d_data);
}
```

## Nsight Compute

Nsight Compute提供详细的kernel级别分析，包括硬件计数器、内存访问模式等。

```bash
# 分析特定kernel
ncu ./my_cuda_app

# 收集所有指标
ncu --set full ./my_cuda_app

# 分析特定kernel名称
ncu --kernel-name "vector_add" ./my_cuda_app

# 指定要收集的指标
ncu --metrics "sm__throughput.avg.pct_of_peak_sustained_elapsed,\
gld_throughput,\
gst_throughput,\
shared_load_throughput,\
dram_read_throughput,\
l2_read_throughput" ./my_cuda_app

# 导出报告
ncu --export report.ncu-rep ./my_cuda_app
```

### 常用Nsight Compute指标

```
计算指标:
- sm__throughput.avg.pct_of_peak_sustained_elapsed  SM利用率
- sm__warps_active.avg.per_cycle_active             活跃warp数

内存指标:
- l1tex__t_sectors_pipe_lsu_mem_global_op_ld.sum    全局内存加载事务
- l1tex__t_sectors_pipe_lsu_mem_global_op_st.sum    全局内存存储事务
- l1tex__t_sectors_pipe_lsu_mem_shared_op_ld.sum    共享内存加载事务
- dram__throughput.avg.pct_of_peak_sustained_elapsed DRAM带宽利用率

控制流指标:
- smsp__thread_inst_executed_pred_on.avg            预测为true的指令比例
- smsp__sass_thread_inst_executed_op_control_pred_on.avg  分支效率
```

## CUDA Profiler API

在代码中嵌入分析控制，精确控制分析范围：

```cpp
#include <cuda_profiler_api.h>

void profile_specific_region() {
    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // 预热（不分析）
    cudaMemset(d_data, 0, N * sizeof(float));

    // 开始分析
    cudaProfilerStart();

    // 只有这个区域会被分析
    compute_kernel<<<N / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    // 结束分析
    cudaProfilerStop();

    // 后续代码不分析
    cudaFree(d_data);
}
```

## 代码内计时

使用CUDA事件进行精确的kernel级计时：

```cpp
#include <cstdio>
#include <cuda_runtime.h>

class CudaTimer {
public:
    CudaTimer() {
        cudaEventCreate(&start_);
        cudaEventCreate(&stop_);
    }

    ~CudaTimer() {
        cudaEventDestroy(start_);
        cudaEventDestroy(stop_);
    }

    void start() {
        cudaEventRecord(start_, 0);
    }

    float stop(const char* label = nullptr) {
        cudaEventRecord(stop_, 0);
        cudaEventSynchronize(stop_);
        float ms;
        cudaEventElapsedTime(&ms, start_, stop_);
        if (label) {
            printf("[Timer] %s: %.4f ms\n", label, ms);
        }
        return ms;
    }

private:
    cudaEvent_t start_, stop_;
};

// 使用示例
void benchmark_kernel() {
    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));
    cudaMemset(d_data, 0, N * sizeof(float));

    CudaTimer timer;

    // 预热
    compute_kernel<<<N / 256, 256>>>(d_data, N);
    cudaDeviceSynchronize();

    // 多次运行取平均
    const int numRuns = 100;
    timer.start();
    for (int i = 0; i < numRuns; i++) {
        compute_kernel<<<N / 256, 256>>>(d_data, N);
    }
    float total_ms = timer.stop("100 kernel runs");
    printf("Average kernel time: %.4f ms\n", total_ms / numRuns);

    // 计算带宽
    float bytes = 2.0f * N * sizeof(float) * numRuns;  // 读+写
    float gb_s = bytes / (total_ms / 1000.0f) / 1e9f;
    printf("Memory bandwidth: %.2f GB/s\n", gb_s);

    cudaFree(d_data);
}
```

## Occupancy Calculator

占用率（Occupancy）是实际活跃warp数与最大可能活跃warp数的比值：

```cpp
#include <cstdio>
#include <cuda_runtime.h>

void calculate_occupancy() {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    // 使用Occupancy API
    int blockSize = 256;
    int minGridSize, gridSize;

    // 计算最优grid大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &gridSize,
                                        compute_kernel, 0, 0);
    printf("Suggested grid size: %d, block size: %d\n", gridSize, blockSize);

    // 计算指定配置的占用率
    int numBlocks;
    cudaOccupancyMaxActiveBlocksPerMultiprocessor(&numBlocks,
                                                    compute_kernel,
                                                    blockSize, 0);

    int maxWarps = prop.maxThreadsPerMultiProcessor / prop.warpSize;
    int activeWarps = numBlocks * (blockSize / prop.warpSize);
    float occupancy = (float)activeWarps / maxWarps;

    printf("Device: %s\n", prop.name);
    printf("SM count: %d\n", prop.multiProcessorCount);
    printf("Max threads per SM: %d\n", prop.maxThreadsPerMultiProcessor);
    printf("Block size: %d\n", blockSize);
    printf("Active blocks per SM: %d\n", numBlocks);
    printf("Active warps per SM: %d / %d\n", activeWarps, maxWarps);
    printf("Occupancy: %.2f%%\n", occupancy * 100);
}

// 使用cudaOccupancyMaxPotentialBlockSizeWithFlags
void optimal_launch_config() {
    int blockSize, minGridSize, gridSize;

    // 自动选择最优的block大小和grid大小
    cudaOccupancyMaxPotentialBlockSize(&minGridSize, &gridSize,
                                        compute_kernel, 0, 0);

    blockSize = gridSize;
    // blockSize会是最大化占用率的最佳值

    int N = 1024 * 1024;
    float* d_data;
    cudaMalloc(&d_data, N * sizeof(float));

    // 使用最优配置启动
    compute_kernel<<<minGridSize, blockSize>>>(d_data, N);

    cudaFree(d_data);
}
```

## nvprof（传统命令行工具）

注意：nvprof在CUDA 12.0后已弃用，推荐使用Nsight系列工具。

```bash
# nvprof基本使用
nvprof ./my_cuda_app

# GPU时间线
nvprof --print-gpu-trace ./my_cuda_app

# 收集特定指标
nvprof --metrics gld_efficiency,gst_efficiency ./my_cuda_app

# 收集所有指标
nvprof --metrics all ./my_cuda_app

# 导出到CSV
nvprof --csv --log-file profile.csv ./my_cuda_app

# 使用Python分析
nvprof --export-profile timeline.nvprof ./my_cuda_app
# 然后用 nvprof --import-profile timeline.nvprof 分析
```

## 性能分析工作流程

```
1. 运行Nsight Systems（系统级分析）
   ├── 识别瓶颈：计算受限 vs 内存受限 vs 传输受限
   ├── 检查是否有不必要的同步
   └── 检查H2D/D2H传输是否可以重叠

2. 运行Nsight Compute（kernel级分析）
   ├── 检查SM利用率
   ├── 检查内存访问效率（合并访问、bank conflict）
   ├── 检查分支发散
   └── 检查占用率

3. 根据发现进行优化
   ├── 内存受限 → 优化合并访问、使用共享内存
   ├── 计算受限 → 减少算术强度、使用Tensor Core
   ├── 传输受限 → 使用流、异步传输、统一内存
   └── 占用率低 → 减少寄存器/共享内存使用
```

## 常见性能问题诊断

```cpp
// 使用CUDA assert检查异步错误
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("CUDA error: %s\n", cudaGetErrorString(err));
}

// 检查kernel是否正确完成
cudaDeviceSynchronize();
err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel error: %s\n", cudaGetErrorString(err));
}

// 设备端assert（CUDA 8.0+）
__global__ void kernel_with_assert(float* data, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    assert(idx < N);  // 如果失败，会打印错误并终止
    data[idx] = data[idx] * 2.0f;
}
```

## 小结

1. Nsight Systems用于系统级分析，识别CPU/GPU瓶颈和传输问题
2. Nsight Compute用于详细的kernel分析，检查内存访问和计算效率
3. 使用NVTX标记代码区域，在时间线上可视化执行流程
4. 占用率是重要的性能指标，但不是越高越好，需要平衡资源使用
5. 先做高层次分析确定瓶颈方向，再做低层次分析指导具体优化
