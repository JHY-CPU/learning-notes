# 03-数据级并行 (DLP)

数据级并行通过对多个数据元素同时执行相同操作来提升计算吞吐率，是现代处理器（CPU SIMD扩展与GPU）的重要加速手段。

---

## 1. SIMD概念与历史

### 1.1 SIMD基本思想

**SIMD (Single Instruction, Multiple Data)**：一条指令同时对多个数据元素执行相同操作。Flynn分类法中与SISD、MISD、MIMD并列的并行模式。

```
SISD:  ADD R1, R2        → 一个操作处理一对数据
SIMD:  VADD V1, V2[0..7] → 一个操作处理8对数据
```

### 1.2 Intel SIMD演进

| 指令集 | 年份 | 寄存器宽度 | 典型用途 |
|--------|------|-----------|----------|
| MMX | 1997 | 64位 (MM0~MM7) | 多媒体整数运算 |
| SSE | 1999 | 128位 (XMM0~XMM7) | 单精度浮点 |
| SSE2~SSE4 | 2001~2006 | 128位 | 双精度浮点、整数扩展 |
| AVX | 2011 | 256位 (YMM0~YMM15) | 科学计算 |
| AVX2 | 2013 | 256位 | 整数向量化 |
| AVX-512 | 2016 | 512位 (ZMM0~ZMM31) | 高性能计算、AI推理 |

### 1.3 ARM NEON

ARM的SIMD扩展，广泛用于移动设备：

- 128位向量寄存器 (V0~V31)
- 支持8/16/32/64位整数和单/双精度浮点
- ARMv8引入SVE (Scalable Vector Extension)，向量长度可变（128~2048位）

---

## 2. GPU架构基础

### 2.1 GPU vs CPU设计理念

| 特征 | CPU | GPU |
|------|-----|-----|
| 核心数 | 少量(4~64)复杂核心 | 大量(数千)简单核心 |
| 设计目标 | 低延迟 | 高吞吐率 |
| 缓存 | 大容量多级缓存 | 小缓存，高带宽内存 |
| 控制逻辑 | 复杂（乱序、分支预测） | 简单 |

### 2.2 SIMT (Single Instruction, Multiple Threads)

GPU的执行模型，SIMD的扩展：

- **Warp/Wavefront**：一组线程（NVIDIA: 32线程，AMD: 64线程）执行同一指令
- 所有线程同步执行同一指令流，但操作不同数据
- 分支发散 (Branch Divergence)：同一Warp内不同线程走向不同分支时，串行执行各分支

### 2.3 流多处理器 (SM / Streaming Multiprocessor)

GPU的基本计算单元，包含：

- 多个CUDA核心 / 流处理器 (SP)
- 共享的寄存器文件
- 共享内存 (Shared Memory)
- Warp调度器
- 特殊功能单元 (SFU) 用于超越函数

---

## 3. CUDA编程模型

### 3.1 基本概念

CUDA是NVIDIA的GPU通用计算编程平台：

```
Grid（网格）
├── Block（线程块）  ← blockDim（线程块维度）
│   ├── Thread（线程）← threadIdx（线程索引）
│   └── ...
└── ...
```

### 3.2 内存层次

| 内存类型 | 作用域 | 延迟 | 容量 |
|----------|--------|------|------|
| 寄存器 | 单线程 | 1周期 | ~255/线程 |
| 共享内存 | 线程块内 | ~5周期 | ~48KB/SM |
| 常量内存 | 全局（只读） | ~5周期(缓存命中) | 64KB |
| 全局内存 | 全局 | ~400周期 | 数GB |
| 纹理内存 | 全局（只读） | 缓存优化 | 绑定到全局 |

### 3.3 CUDA核函数示例

```cuda
__global__ void vectorAdd(float *a, float *b, float *c, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) c[idx] = a[idx] + b[idx];
}

// 调用
int threadsPerBlock = 256;
int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(a, b, c, n);
```

### 3.4 性能优化要点

- **合并访问 (Coalesced Access)**：同一Warp的线程访问连续全局内存地址
- **共享内存Bank冲突**：避免多个线程同时访问同一Bank
- **占用率 (Occupupancy)**：活跃Warp数与最大Warp数之比
- **异步传输**：使用CUDA Stream实现计算与传输重叠

---

## 4. OpenCL基础

### 4.1 概述

OpenCL (Open Computing Language) 是跨平台的异构计算框架，支持CPU、GPU、FPGA等多种设备。

### 4.2 编程模型

- **Platform → Device → Context → Command Queue → Kernel**
- 内核用OpenCL C语言编写（C99子集）
- 主机端使用C/C++ API管理设备和内存

### 4.3 与CUDA的对比

| 方面 | CUDA | OpenCL |
|------|------|--------|
| 平台 | 仅NVIDIA GPU | 跨平台 |
| 生态 | 成熟完善 | 通用但分散 |
| 调试工具 | nvcc, nsight | 厂商提供 |
| 学习曲线 | 相对简单 | 样板代码多 |

---

## 5. 向量处理器

### 5.1 经典向量机

早期超级计算机（如Cray-1）采用的架构：

- **向量寄存器**：存储长向量（64~128个元素）
- **向量功能单元**：流水化的加法、乘法、访存单元
- **向量链 (Chaining)**：类似转发，将一个向量单元的输出直接送入下一个

### 5.2 现代向量处理器

RISC-V的V扩展 (RVV) 重新引入向量指令：

- 向量长度寄存器 (vl) 确定操作长度
- 支持多种数据宽度 (SEW) 和分组 (LMUL)
- 便于自动向量化编译器生成代码

---

## 6. 数据并行的编程模式

### 6.1 Map模式

对数据集中的每个元素独立应用相同函数：

```
Map(f, [a, b, c, d]) = [f(a), f(b), f(c), f(d)]
```

### 6.2 Reduce模式

将数据集通过二元操作归约为单个值：

```
Reduce(+, [a, b, c, d]) = ((a + b) + c) + d
```

GPU中通常使用共享内存实现高效的并行归约。

### 6.3 Scan (前缀和)

计算数据集的部分和，在流压缩、排序等算法中广泛使用。

### 6.4 Stencil模式

用邻域数据的加权组合更新每个网格点，广泛用于图像处理和物理仿真。

---

## 7. GPGPU计算

### 7.1 通用GPU计算的概念

利用GPU的大规模并行计算能力处理通用计算任务，而不仅限于图形渲染：

- **适合的工作负载**：高算术密度、可大规模并行、低分支复杂度
- **典型应用**：深度学习训练/推理、科学模拟、密码学、金融建模

### 7.2 深度学习中的GPGPU

- **矩阵乘法**：cuBLAS库利用Tensor Core加速GEMM
- **卷积**：cuDNN库针对卷积运算深度优化
- **混合精度训练**：FP16/BF16计算 + FP32累加平衡速度与精度

### 7.3 GPGPU的性能模型

**计算密度** = 浮点运算数 / 内存访问字节数

- 当计算密度 > Roofline拐点时：计算受限 (Compute-Bound)
- 当计算密度 < Roofline拐点时：带宽受限 (Memory-Bound)

---

## 8. 矩阵运算的并行化

### 8.1 矩阵-向量乘法 (GEMV)

$$y = Ax$$

- 每行独立计算，天然并行
- 访存模式：A按行读取，x按列重复读取（可利用缓存）

### 8.2 矩阵-矩阵乘法 (GEMM)

$$C = A \times B$$

经典的优化策略：

- **分块 (Tiling/Blocking)**：将大矩阵分解为小块，提高缓存命中率
```
for ii in range(0, N, TILE):
  for jj in range(0, N, TILE):
    for kk in range(0, N, TILE):
      C[ii:ii+TILE][jj:jj+TILE] += A[ii:ii+TILE][kk:kk+TILE] * B[kk:kk+TILE][jj:jj+TILE]
```

- **寄存器分块**：进一步将Tile分解到寄存器级别
- **向量化**：使用SIMD指令一次计算多个元素

### 8.3 矩阵转置

朴素转置会导致严重的Bank冲突（GPU共享内存中）。使用Padding（在每行末尾多留一个元素）可消除冲突：`__shared__ float tile[TILE_SIZE][TILE_SIZE + 1];`

### 8.4 现代硬件矩阵加速

- **NVIDIA Tensor Core**：每周期执行4x4矩阵乘加 (D = A*B + C)，支持FP16/BF16/INT8/TF32/FP8
- **Intel AMX (Advanced Matrix Extensions)**：Tile架构，支持INT8/BF16矩阵运算
- **ARM SME (Scalable Matrix Extension)**：流式矩阵扩展
