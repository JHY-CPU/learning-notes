# GPU计算模型概述

## CPU vs GPU 架构对比

CPU（中央处理器）和GPU（图形处理器）在设计哲学上有根本区别。CPU优化的是**延迟（Latency）**——让单个任务尽快完成；GPU优化的是**吞吐量（Throughput）**——单位时间内完成尽可能多的任务。

### CPU架构特点

- **少量强大的核心**：通常4-64个核心，每个核心拥有大容量缓存、复杂的控制逻辑
- **分支预测**：硬件级别的分支预测器（通常准确率95%+），减少控制冒险
- **乱序执行**：指令可以不按程序顺序执行，利用指令级并行（ILP），例如超标量执行可以每个周期发射多条指令
- **大容量缓存**：L1（~32KB, ~1ns延迟）/ L2（~256KB, ~3ns）/ L3（~30MB, ~10ns）缓存层次，隐藏DRAM延迟（~100ns）
- **超线程**：一个物理核心运行多个硬件线程，隐藏单线程的内存延迟

### GPU架构特点

- **大量简单核心**：通常数千个核心，每个核心功能简单，没有分支预测和乱序执行
- **高内存带宽**：GPU显存带宽可达1-2 TB/s，远超CPU的50-100 GB/s
- **SIMT执行模型**：Single Instruction, Multiple Threads，一个warp内32个线程执行相同指令
- **适合数据并行**：大量数据执行相同操作时效率极高
- **通过线程级并行隐藏延迟**：当一个warp在等数据时，调度器立即切换到另一个就绪的warp

```
CPU架构示意（简化）:                GPU架构示意（简化）:
┌─────────────────────┐            ┌─────────────────────────────────┐
│ Core0  Core1  Core2 │            │ SM0          SM1          SMn   │
│ ┌───┐  ┌───┐  ┌───┐│            │┌─────────┐ ┌─────────┐ ┌─────┐│
│ │ALU│  │ALU│  │ALU││            ││32 Cores │ │32 Cores │ │     ││
│ │   │  │   │  │   ││            ││SharedMem│ │SharedMem│ │ ... ││
│ │L1 │  │L1 │  │L1 ││            ││Register │ │Register │ │     ││
│ └───┘  └───┘  └───┘│            │└─────────┘ └─────────┘ └─────┘│
│     共享 L2/L3 缓存   │            │         全局显存 (HBM)          │
└─────────────────────┘            └─────────────────────────────────┘
```

### 设计哲学的本质差异

CPU设计者假设程序中存在大量不可预测的分支和数据依赖，因此用复杂硬件（分支预测器、乱序缓冲、大缓存）来尽可能降低单线程延迟。GPU设计者假设程序对大量数据执行相同的操作（数据并行），因此用极简的核心和庞大的数量来最大化同时进行的计算量，靠warp切换来隐藏内存延迟。

一个形象的类比：CPU像一个精通十八般武艺的大厨，能处理任何复杂的菜肴；GPU像一个有1000人的切菜团队，每人只会切一种菜，但同时开工效率极高。

## 流式多处理器（SM）深度解析

SM（Streaming Multiprocessor）是GPU的基本计算单元。一块GPU包含多个SM，每个SM是完整的计算子系统，包含：

- **CUDA核心（FP32/INT32 ALU）**：执行整数和单精度浮点运算，每个SM有32-192个
- **FP64单元**：双精度浮点运算，数量通常是FP32的1/2或1/64
- **Tensor Core**：矩阵乘加运算（Volta架构开始引入），专门加速深度学习
- **特殊功能单元（SFU）**：执行超越函数（sin、cos、exp、rsqrt等），通常每SM 4-8个
- **加载/存储单元（LD/ST）**：处理内存访问请求，通常每SM 32-64个（与核心数对应）
- **寄存器文件**：每个线程独享的寄存器，A100每SM有65536个32位寄存器
- **共享内存（Shared Memory）**：同一线程块内线程共享的低延迟内存，可编程划分为L1/Shared
- **L1缓存**：可配置与Shared Memory共享物理存储（通常合计164KB/SM）
- **Warp调度器**：每个SM通常有2-4个warp调度器，每个调度器可以每周期选出一个warp发射指令
- **常量缓存**：对只读常量数据的广播优化

### 主流GPU的SM配置对比

| GPU | 架构 | SM数量 | FP32核/SM | 总FP32核 | Tensor核/SM | 共享内存/SM | 寄存器/SM |
|-----|------|--------|-----------|----------|-------------|-------------|-----------|
| V100 | Volta | 80 | 64 | 5120 | 8 | 96KB | 65536 |
| A100 | Ampere | 108 | 64 | 6912 | 4 (第三代) | 164KB | 65536 |
| H100 | Hopper | 132 | 128 | 16896 | 4 (第四代) | 228KB | 65536 |
| RTX 4090 | Ada Lovelace | 128 | 128 | 16384 | 4 (第四代) | 128KB | 65536 |
| RTX 3080 | Ampere | 68 | 128 | 8704 | 4 (第三代) | 128KB | 65536 |

### SM内部详细结构（以A100为例）

```
A100 单个SM内部结构:
┌─────────────────────────────────────────────────────────────┐
│  Warp Scheduler 0          Warp Scheduler 1                 │
│  ┌──────────────────┐     ┌──────────────────┐             │
│  │ 选取ready的warp   │     │ 选取ready的warp   │             │
│  └────────┬─────────┘     └────────┬─────────┘             │
│           │                        │                        │
│  ┌────────▼────────────────────────▼─────────────────┐     │
│  │              执行单元 (64个FP32 + 64个INT32)       │     │
│  │  FP32 Pipe │ INT32 Pipe │ FP64 │ SFU │ LD/ST     │     │
│  │   (64)     │   (64)     │ (32) │ (4) │  (32)     │     │
│  │         Tensor Core (4个, 每个256 FLOP/cycle)     │     │
│  └───────────────────────────────────────────────────┘     │
│                                                             │
│  ┌──────────────────┐  ┌──────────────────────────────┐    │
│  │  寄存器文件        │  │  共享内存/L1缓存 (可配置)     │    │
│  │  65536 x 32-bit  │  │  164KB total:                │    │
│  │  最多2048线程/SM  │  │  配置1: 164KB SMEM + 0KB L1  │    │
│  │                  │  │  配置2: 100KB SMEM + 64KB L1  │    │
│  └──────────────────┘  └──────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

## 内存层次与延迟

GPU的内存层次对性能有决定性影响。以下是A100的各级内存延迟和带宽数据：

| 内存层次 | 容量（每SM） | 延迟（周期） | 带宽（每SM） | 范围 |
|----------|------------|------------|-------------|------|
| 寄存器 | 256KB (65536个32-bit) | 1 | ~192 TB/s | 单个线程 |
| 共享内存 | 最大164KB | 5-30 | ~80 TB/s | 同一Block内 |
| L1缓存 | 最大164KB（与共享内存共享） | ~30 | ~40 TB/s | 同一SM |
| L2缓存 | 40MB（全芯片共享） | ~200 | ~6 TB/s | 所有SM |
| 全局显存(HBM2e) | 40/80GB | ~400 | 2 TB/s (全芯片) | 所有SM + 主机 |

### 关键延迟隐藏机制

GPU不使用CPU那种复杂的乱序执行来隐藏延迟，而是使用**零开销上下文切换**：当一个warp遇到内存延迟时，warp调度器在**同一周期**切换到另一个已就绪的warp执行。要隐藏约400周期的全局内存延迟，理论上需要约400/32（每个warp的指令间隔）= 约12-20个活跃warp。实际上，A100每个SM最多可以同时驻留64个warp（2048线程），足以隐藏大部分延迟。

```
延迟隐藏示意（4个warp的情况）:
时间 →  ──────────────────────────────────────────►
Warp 0: [执行指令]...[等待内存]..................[执行指令]
Warp 1:           [执行指令]...[等待内存]............
Warp 2:                     [执行指令]...[等待内存]....
Warp 3:                               [执行指令]...[等待]
        ═══════════════════════════════════════════
SM实际: [W0][W1][W2][W3][W0][W1][W2][W3][W0][W1][W2]
         SM始终保持忙碌，零空闲周期！
```

## SIMT执行模型

SIMT（Single Instruction, Multiple Threads）是CUDA的核心执行模型。与传统的SIMD不同，SIMT允许线程有独立的分支路径——但同一warp内走不同分支时会付出性能代价。

### SIMT vs SIMD 的区别

| 特性 | SIMD | SIMT |
|------|------|------|
| 编程模型 | 显式向量操作 | 标量代码 + 隐式并行 |
| 分支处理 | 不支持分支发散 | 支持分支发散（有代价） |
| 线程独立性 | 无 | 每个线程有独立的寄存器和PC |
| 典型指令 | `vaddps ymm0, ymm1, ymm2` | `add.f32 r4, r1, r2` |
| 硬件实现 | 宽ALU | 32个窄ALU |

```cpp
// SIMT示例：不同线程可以走不同分支
__global__ void simt_example(float* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < n) {
        if (data[idx] > 0.0f) {
            // 分支A：线程走这条路
            data[idx] = sqrtf(data[idx]);
        } else {
            // 分支B：其他线程走这条路
            data[idx] = 0.0f;
        }
    }
}
```

**分支发散（Branch Divergence）的硬件细节**：当warp遇到条件分支时，硬件维护一个**活跃掩码（active mask）**。假设warp中线程{0,1,4,5}走分支A，线程{2,3,6,7}走分支B，则：
1. 先执行分支A：活跃掩码 = 0b110011，不活跃线程被禁用（不写回结果）
2. 再执行分支B：活跃掩码 = 0b001100，不活跃线程被禁用
3. 两个分支串行执行，总时间翻倍

```
分支发散执行示意:
Warp (线程 0-31):
  if (data[idx] > 0)    →  线程 0,1,4,5,8,9... 满足条件 (mask: 0x55555555)
    sqrt(data[idx])     →  这些线程执行, 其他线程挂起
  else                  →  线程 2,3,6,7,10,11... 满足条件 (mask: 0xAAAAAAAA)
    data[idx] = 0       →  这些线程执行, 其他线程挂起
  ─────────────────
  总执行时间 = max(分支A时间, 分支B时间) * 2
```

## Warp：GPU的最小调度单位

Warp是GPU执行的基本粒度，一个warp包含**32个线程**。理解warp行为是写出高效CUDA代码的前提。

### Warp的关键特性

1. **锁步执行（Lockstep）**：同一warp内的32个线程执行相同的指令
2. **分支发散代价**：如果warp内线程走不同分支，硬件串行执行每个分支路径，活跃掩码控制哪些线程真正执行
3. **warp调度**：SM在多个warp之间快速切换，零开销切换（没有寄存器保存/恢复），用于隐藏内存延迟
4. **warp投票**：`__all_sync()`和`__any_sync()`可以检查warp内所有线程的谓词
5. **warp洗牌**：`__shfl_sync()`等可以在warp内直接交换数据，不需要共享内存

```cpp
// warp分支发散示例
__global__ void warp_divergence_example(int* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 这里warp内一半线程走if，一半走else
    // 导致warp串行执行两个分支，效率减半
    if (idx % 2 == 0) {
        data[idx] = data[idx] * 2;  // warp中偶数线程
    } else {
        data[idx] = data[idx] + 1;  // warp中奇数线程
    }
}

// 优化：按warp对齐，避免warp内分支发散
__global__ void warp_aligned_branch(int* data, int n) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    // 整个warp要么全走if，要么全走else
    if (idx / 32 % 2 == 0) {
        data[idx] = data[idx] * 2;
    } else {
        data[idx] = data[idx] + 1;
    }
}
```

### 一个warp到底有多少线程在"真正"执行？

这个问题很关键。虽然一个warp有32个线程，但在A100上每个SM每周期可以执行一条FP32指令覆盖所有64个FP32核心（即2个warp的FP32指令）。实际的warp调度器每周期从就绪warp中选出1-2个warp发射到执行单元。关键在于：**只要warp数量足够多，SM就永远不会空闲**。

## 线程层次结构

CUDA的线程按层次组织，从大到小依次为Grid、Block、Thread、Warp：

```
Grid（网格）
├── Block(0,0)          Block(1,0)          ...
│   ├── Warp 0          ├── Warp 0
│   │   Thread 0-31     │   Thread 0-31
│   ├── Warp 1          ├── Warp 1
│   │   Thread 32-63    │   Thread 32-63
│   └── ...
│
└── Block(0,1)          Block(1,1)          ...
    ├── Warp 0          ├── Warp 0
    └── ...             └── ...
```

- **Thread**：最小执行单元，拥有自己的寄存器和程序计数器
- **Warp**：32个线程组成一个warp，硬件调度的最小粒度。一个Block内的线程按线性编号分组成warp
- **Block**：一组线程，共享Shared Memory，可以同步。Block内的线程保证驻留在同一SM上
- **Grid**：一组Block，启动一个Kernel时创建一个Grid。不同Block可以分配到不同SM

### Block大小与SM的关系

以A100为例：每SM最多2048个线程，最大1024线程/Block，最多16个Block驻留。如果一个Block有256个线程，则一个SM最多驻留2048/256 = 8个Block（但不超过16个Block限制）。合理选择Block大小（如128、256、512）直接影响**占用率（Occupancy）**，从而影响延迟隐藏效果。

### Occupancy（占用率）详解

占用率 = SM上实际活跃warp数 / SM最大支持warp数

A100每SM最多64个warp（2048线程）。如果Block大小为256（8个warp），每SM驻留8个Block（64个warp），占用率 = 100%。但如果每个线程使用太多寄存器（如128个），则2048 * 32 bit * 128 = 8MB，超过SM的256KB寄存器文件容量，实际驻留的线程数会减少，占用率下降。

```
占用率影响因素:
1. Block大小: 每Block的线程数必须能整除warpSize(32)
2. 寄存器用量: 每SM总寄存器 / 每线程寄存器 = 最大线程数
3. 共享内存用量: 每SM总共享内存 / 每Block共享内存 = 最大Block数
4. Block数量限制: 每SM最多16个Block

占用率 = min(线程数限制, Block数限制)的实际warp数 / 64
```

## CUDA核心 vs Tensor Core

| 特性 | CUDA Core | Tensor Core |
|------|-----------|-------------|
| 运算类型 | FP32/FP64标量运算 | FP16/BF16/INT8/FP8矩阵运算 |
| 典型操作 | `a*b+c` (标量FMA) | `D = A*B + C`（4x4矩阵乘加） |
| 吞吐量 | 1 FMA/cycle/core | 每TC每cycle 256 FP16 FMA |
| 主要用途 | 通用计算 | 深度学习训练/推理 |
| 首次引入 | 初代CUDA GPU | Volta架构（V100, 2017） |
| 精度支持 | FP32, FP64 | FP16, BF16, TF32, INT8, FP8 |
| 编程方式 | 直接CUDA代码 | WMMA API或cuBLAS/cutlass |

### Tensor Core工作原理

Tensor Core执行的是4x4矩阵乘加：D = A x B + C，其中A和B可以是FP16矩阵，C和D是FP16或FP32矩阵。硬件将4x4矩阵乘分解为多个小步骤在单周期内完成。一个A100 Tensor Core每周期可以完成256次FP16 FMA运算（8倍于FP32 CUDA Core的吞吐量）。

```
Tensor Core 4x4矩阵乘:
┌───┐   ┌───┐   ┌───┐     ┌───┐
│ A │ × │ B │ + │ C │  =  │ D │   (4x4矩阵，每个元素FP16)
└───┘   └───┘   └───┘     └───┘
  每个Tensor Core每周期执行:
  4×4×4 = 64次乘法 + 64次加法 = 128次运算
  FP16模式: ×2(两个TC并行) = 256 ops/cycle

A100全芯片 Tensor Core算力:
  108 SM × 4 TC/SM × 256 ops/cycle × 1.41 GHz = ~156 TFLOPS (FP16)
```

## 性能指标与屋顶线模型

### 峰值计算能力详解

```cpp
// 以A100为例计算理论峰值
// FP32 (with FP32 FMA counting as 2 ops):
//   108 SMs × 64 cores × 2 (FMA) × 1.41 GHz = 19.5 TFLOPS
//
// FP16 Tensor Core:
//   108 SMs × 4 TCs × 256 ops × 2 (FMA) × 1.41 GHz = 312 TFLOPS
//
// INT8 Tensor Core:
//   108 SMs × 4 TCs × 512 ops × 2 (FMA) × 1.41 GHz = 624 TOPS
//
// 显存带宽: 2 TB/s (HBM2e, 80GB版本)
```

### 屋顶线模型（Roofline Model）

屋顶线模型帮助判断程序是**计算受限（compute-bound）**还是**带宽受限（memory-bound）**。

```
性能(TFLOPS)
 ↑  312 ┤  ════════════════════════════  FP16 Tensor Core峰值
 │  19.5 ┤  ════════════════════════════  FP32峰值
 │       │         /
 │       │        /   计算受限区 (compute-bound)
 │       │       /
 │       │      /    拐点: 算术强度 = 峰值算力 / 带宽
 │       │     /     FP32: 19.5 TFLOPS / 2 TB/s = 9.75 FLOP/byte
 │       │    /
 │       │   /       带宽受限区 (memory-bound)
 │       │  /
 │       │ /
 │       │/
 └───────┴───────────────────────────────→ 算术强度 (FLOPS/Byte)
```

- **算术强度 = 计算量 / 数据传输量**（单位：FLOP/Byte）
- A100 FP32拐点: 19.5 TFLOPS / 2000 GB/s = 9.75 FLOP/Byte
- 矩阵乘法NxN: 计算量 = 2N^3 FLOP, 数据量 = 3N^2 × 4 bytes（读A,B + 写C），算术强度 = 2N / 12，当N足够大时算术强度高，是计算受限
- 逐元素加法: 计算量 = N FLOP, 数据量 = 3N × 4 bytes，算术强度 = 1/12 FLOP/Byte，远低于拐点，是带宽受限

### 实际测量带宽

```cpp
// 测量实际内存带宽并计算带宽利用率
#include <cstdio>
#include <cuda_runtime.h>

__global__ void copy_kernel(const float* __restrict__ src,
                            float* __restrict__ dst, int N) {
    int idx = threadIdx.x + blockIdx.x * blockDim.x;
    if (idx < N) {
        dst[idx] = src[idx];  // 一次读 + 一次写
    }
}

int main() {
    const int N = 1 << 26;  // 256M元素 = 1GB
    const size_t bytes = N * sizeof(float);
    float *d_src, *d_dst;
    cudaMalloc(&d_src, bytes);
    cudaMalloc(&d_dst, bytes);

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    // 预热
    copy_kernel<<<(N + 255) / 256, 256>>>(d_src, d_dst, N);
    cudaDeviceSynchronize();

    // 多次测量取平均
    const int NUM_ITERS = 10;
    cudaEventRecord(start);
    for (int i = 0; i < NUM_ITERS; i++) {
        copy_kernel<<<(N + 255) / 256, 256>>>(d_src, d_dst, N);
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);

    float ms;
    cudaEventElapsedTime(&ms, start, stop);
    ms /= NUM_ITERS;

    // 带宽计算: 一次读(4字节) + 一次写(4字节) = 8字节/元素
    float bw_measured = (2.0f * bytes) / (ms * 1e6);  // GB/s
    float bw_peak = 2000.0f;  // A100 HBM2e峰值 2 TB/s
    float efficiency = bw_measured / bw_peak * 100.0f;

    printf("Copy kernel: %.3f ms\n", ms);
    printf("Measured bandwidth: %.1f GB/s\n", bw_measured);
    printf("Theoretical peak: %.1f GB/s\n", bw_peak);
    printf("Efficiency: %.1f%%\n", efficiency);
    // 典型结果: ~1500-1700 GB/s (75-85% 效率)

    cudaFree(d_src);
    cudaFree(d_dst);
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    return 0;
}
```

编译运行：
```bash
nvcc -O3 -arch=sm_80 -o bandwidth_test bandwidth_test.cu
./bandwidth_test
# 预期输出:
# Copy kernel: 0.615 ms
# Measured bandwidth: 1626.0 GB/s
# Theoretical peak: 2000.0 GB/s
# Efficiency: 81.3%
```

## GPU计算 vs CPU计算：一个具体数字对比

以矩阵乘法 C = A × B (4096×4096 × FP32) 为例：

| 指标 | CPU (i9-13900K, AVX-512) | GPU (A100, cuBLAS) |
|------|--------------------------|-------------------|
| 理论FP32峰值 | 1.5 TFLOPS | 19.5 TFLOPS |
| 实际matmul算力 | ~0.8 TFLOPS (MKL) | ~15.2 TFLOPS (cuBLAS) |
| 效率 | ~53% | ~78% |
| 执行时间 | ~85 ms | ~3.5 ms |
| 加速比 | 1x | ~24x |

计算量 = 2 × 4096^3 = 137.4 GFLOP

## CUDA vs OpenCL vs ROCm vs Metal 对比

| 特性 | CUDA | OpenCL | ROCm/HIP | Metal |
|------|------|--------|----------|-------|
| 厂商 | NVIDIA only | 跨厂商 | AMD | Apple |
| 编程语言 | CUDA C/C++ | OpenCL C | HIP C++ | Metal Shading Language |
| API风格 | 专有API | 开放标准API | CUDA-like API | 专有API |
| 生态成熟度 | 最成熟 | 广泛但碎片化 | 追赶中 | Apple生态内优秀 |
| 深度学习支持 | cuDNN/TensorRT | 较弱 | MIOpen | Core ML |
| 调试工具 | Nsight/Compute Sanitizer | 厂商工具 | ROCgdb | Xcode GPU调试 |
| 矩阵加速 | Tensor Core | 无标准方案 | Matrix Core | Apple Neural Engine |
| 编译模型 | nvcc → PTX → SASS | 运行时编译 | hipcc → AMDGPU | metal编译器 |
| 适合场景 | AI/HPC首选 | 跨平台需求 | AMD GPU用户 | Apple平台开发 |

### 代码风格对比

```cpp
// CUDA
__global__ void vecAdd(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}
// 调用: vecAdd<<<(n+255)/256, 256>>>(a, b, c, n);

// HIP (几乎与CUDA相同，可移植)
__global__ void vecAdd(float* a, float* b, float* c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) c[i] = a[i] + b[i];
}
// 调用: hipLaunchKernelGGL(vecAdd, (n+255)/256, 256, 0, 0, a, b, c, n);

// OpenCL (更冗长)
__kernel void vecAdd(__global float* a, __global float* b,
                     __global float* c, int n) {
    int i = get_global_id(0);
    if (i < n) c[i] = a[i] + b[i];
}
// 需要手动创建上下文、命令队列、编译内核...

// Metal
kernel void vecAdd(device float* a [[buffer(0)]],
                   device float* b [[buffer(1)]],
                   device float* c [[buffer(2)]],
                   uint i [[thread_position_in_grid]]) {
    c[i] = a[i] + b[i];
}
```

## Profiling入门：Nsight Compute

理解GPU程序性能需要使用NVIDIA的profiling工具。

```bash
# Nsight Compute (推荐): 分析单个kernel
ncu --set full -o my_report ./my_cuda_app

# 查看报告
ncu-ui my_report.ncu-rep

# 快速概览模式
ncu --metrics sm__throughput.avg.pct_of_peak,smsp__pipe_tensor_op_active.avg.pct_of_peak_active ./my_cuda_app

# Nsight Systems: 系统级时间线分析
nsys profile -o my_timeline ./my_cuda_app
nsys-ui my_timeline.qdrep
```

### 常用性能指标解读

| 指标 | 含义 | 好的值 |
|------|------|--------|
| SM Occupancy | SM上活跃warp比例 | > 50% |
| DRAM Throughput | 显存带宽利用率 | > 60% |
| Compute (SM) Throughput | 计算单元利用率 | > 70% |
| L1/L2 Hit Rate | 缓存命中率 | L1 > 80%, L2 > 50% |
| Warp Execution Efficiency | warp无分支发散的比例 | > 90% |
| Stall Long Scoreboard | 等待内存操作的比例 | 越低越好 |
| Stall Barrier | 等待__syncthreads()的比例 | 越低越好 |

## 小结

1. GPU通过大量简单核心和高带宽内存实现高吞吐量并行计算，设计哲学与CPU截然不同
2. SM是GPU的核心计算单元，包含CUDA核心、Tensor Core、共享内存、寄存器文件等完整子系统
3. GPU内存层次（寄存器~1周期、共享内存~5周期、全局内存~400周期）对性能有决定性影响
4. SIMT模型是CUDA执行的基础，warp（32线程）是最小调度粒度，分支发散会导致串行执行
5. 通过屋顶线模型判断程序瓶颈（计算受限 vs 带宽受限），指导优化方向
6. Tensor Core提供远超CUDA Core的矩阵运算吞吐量，是深度学习加速的关键
7. 占用率、内存带宽利用率、分支发散率是三个最核心的性能指标
