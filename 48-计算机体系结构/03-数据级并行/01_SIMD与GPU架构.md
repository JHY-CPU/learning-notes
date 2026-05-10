# SIMD与GPU架构 - 计算机体系结构笔记


### 1.1 Flynn 分类法


| 类型 | 指令流 | 数据流 | 代表 |
| --- | --- | --- | --- |
| **SISD** | 单 | 单 | 传统单核处理器 |
| **SIMD** | 单 | 多 | SSE/AVX, GPU |
| **MISD** | 多 | 单 | 容错系统（罕见） |
| **MIMD** | 多 | 多 | 多核处理器 |


### 1.2 数据级并行（DLP）


对大量数据元素执行相同的操作。SIMD是实现数据级并行的主要方式。


```
SISD: 循环4次
  for (i=0; i<4; i++)
      C[i] = A[i] + B[i];

  SIMD: 一条指令处理4个数据
  C[0:3] = A[0:3] + B[0:3]  // 一条向量加法指令
```


| 指令集 | 年份 | 寄存器宽度 | 寄存器数 | 关键特性 |
| --- | --- | --- | --- | --- |
| **MMX** | 1997 | 64 bit | 8 | 整数SIMD |
| **SSE** | 1999 | 128 bit | 16 (XMM) | 单精度浮点 |
| **SSE2** | 2001 | 128 bit | 16 | 双精度浮点、整数 |
| **AVX** | 2011 | 256 bit | 16 (YMM) | 256位浮点运算 |
| **AVX2** | 2013 | 256 bit | 16 | 整数SIMD扩展 |
| **AVX-512** | 2016 | 512 bit | 32 (ZMM) | 512位，掩码寄存器 |


### 2.1 AVX 示例


> **Example:** **标量代码（8次加法）：**
>
>
> ```
> for (int i = 0; i < 8; i++)
>     c[i] = a[i] + b[i];
> ```
>
>
> **AVX 代码（1条指令）：**
>
>
> ```
> __m256 va = _mm256_load_ps(a);    // 加载8个float
> __m256 vb = _mm256_load_ps(b);
> __m256 vc = _mm256_add_ps(va, vb); // 8个float同时相加
> _mm256_store_ps(c, vc);
> ```


### 3.1 向量处理器模型


```
标量处理器:           向量处理器:
  ┌──────────┐         ┌──────────┐
  │ ALU (1个) │         │ 向量ALU  │
  │ 处理1个数据│         │ 处理N个  │──▶ 一个结果向量
  └──────────┘         │ 数据并行 │
                       └──────────┘

  示例 (向量加法):
  标量: LD→ADD→ST 循环8次 = 24条指令
  向量: VLD→VADD→VST = 3条向量指令
```


### 3.2 向量长度寄存器（VL）


RISC-V 向量扩展（RVV）引入可变长度向量，硬件根据 VL 决定实际处理的数据长度。


### 3.3 收集-分散（Gather-Scatter）


- **Gather**
   ：从不连续的内存地址收集数据到向量寄存器
- **Scatter**
   ：将向量寄存器的数据分散存储到不连续的地址


### 4.1 GPU vs CPU 设计哲学


| 特性 | CPU | GPU |
| --- | --- | --- |
| 核心数 | 4-64 | 数千 |
| 单核性能 | 高（复杂控制逻辑） | 低（简单控制） |
| 缓存 | 大容量多级缓存 | 小容量共享内存 |
| 延迟优化 | 是 | 否 |
| 吞吐量优化 | 否 | 是 |
| 适用场景 | 复杂分支、延迟敏感 | 大规模数据并行 |


### 4.2 GPU 层次结构


```
┌─────────────────────────────────────────┐
  │              GPU (如 A100)               │
  │  ┌─────────┐ ┌─────────┐ ┌─────────┐   │
  │  │ SM 0    │ │ SM 1    │ │ SM N    │   │
  │  │ ┌─────┐ │ │ ┌─────┐ │ │ ┌─────┐ │   │
  │  │ │Core │ │ │ │Core │ │ │ │Core │ │   │
  │  │ │Core │ │ │ │Core │ │ │ │Core │ │   │
  │  │ │Core │ │ │ │Core │ │ │ │Core │ │   │
  │  │ └─────┘ │ │ └─────┘ │ │ └─────┘ │   │
  │  │ Shared  │ │ Shared  │ │ Shared  │   │
  │  │ Memory  │ │ Memory  │ │ Memory  │   │
  │  │ L1 Cach │ │ L1 Cach │ │ L1 Cach │   │
  │  └─────────┘ └─────────┘ └─────────┘   │
  │           L2 Cache (共享)                │
  │           HBM (高带宽内存)               │
  └─────────────────────────────────────────┘
```


### 4.3 关键概念


| 概念 | 含义 | 类比 |
| --- | --- | --- |
| **SM** (Streaming Multiprocessor) | GPU的基本计算单元，包含多个CUDA核心 | CPU的一个核心 |
| **CUDA Core** | 执行单个线程的标量运算单元 | ALU |
| **Warp** | 32个线程组成一个warp，SIMT执行 | SIMD操作 |
| **Thread Block** | 一组协作的线程，共享shared memory | 线程组 |
| **Grid** | 一个kernel的所有block | 整个并行任务 |


SIMT（Single Instruction, Multiple Thread）是 NVIDIA GPU 的执行模型，每个线程有自己的寄存器状态，但同一 warp 内的 32 个线程执行相同的指令。


### 5.1 分支处理（Divergence）


```
代码:
  if (threadIdx.x < 16)
      A();  // 前16个线程执行
  else
      B();  // 后16个线程执行

  Warp执行过程:
  1. 所有32个线程执行条件判断
  2. 设置执行掩码: [111...000]
  3. 执行A() — 后16个线程被禁用
  4. 切换掩码: [000...111]
  5. 执行B() — 前16个线程被禁用
  → 有效吞吐量减半!
```

**Warp Divergence**
是GPU编程的重要性能考虑。尽量避免同一 warp 内的线程走不同分支。

| 内存类型 | 容量 | 带宽 | 延迟 | 可见范围 |
| --- | --- | --- | --- | --- |
| 寄存器 | ~256KB/SM | 最高 | 1 cycle | 单个线程 |
| Shared Memory | 48-164KB/SM | ~10 TB/s | ~5 cycles | Block内 |
| L1 Cache | ~128KB/SM | 高 | ~30 cycles | SM内 |
| L2 Cache | 40-64MB | 中 | ~200 cycles | 全GPU |
| Global Memory (HBM) | 16-80GB | ~2 TB/s | ~400 cycles | 全GPU+Host |


### 6.1 Shared Memory 与 Bank Conflict


Shared memory 被分为多个 bank（通常 32 个），若同一 warp 中多个线程访问同一 bank 的不同地址，产生 **bank conflict**，导致串行化。


> **Example:** **无bank conflict：**线程 i 访问 bank i — 并行访问
>
>
> **2-way bank conflict：**线程 i 访问 bank (i%16) — 每个bank被2个线程访问
>
>
> **解决**：padding — 在数组行末添加填充元素错开bank映射


<!-- Converted from: 01_SIMD与GPU架构.html -->
