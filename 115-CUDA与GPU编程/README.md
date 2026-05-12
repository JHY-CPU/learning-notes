# CUDA与GPU编程学习笔记

本模块系统性地介绍CUDA并行计算与GPU编程的核心知识，从基础架构到深度学习中的高级优化技巧。

## 目录

| 编号 | 主题 | 文件 |
|------|------|------|
| 01 | GPU计算模型概述 | [01_GPU计算模型概述.md](01_GPU计算模型概述.md) |
| 02 | CUDA编程模型 | [02_CUDA编程模型.md](02_CUDA编程模型.md) |
| 03 | CUDA内存模型 | [03_CUDA内存模型.md](03_CUDA内存模型.md) |
| 04 | 第一个CUDA程序 | [04_第一个CUDA程序.md](04_第一个CUDA程序.md) |
| 05 | 线程组织与索引计算 | [05_线程组织与索引计算.md](05_线程组织与索引计算.md) |
| 06 | Shared Memory优化 | [06_Shared_Memory优化.md](06_Shared_Memory优化.md) |
| 07 | 全局内存合并访问 | [07_全局内存合并访问.md](07_全局内存合并访问.md) |
| 08 | 原子操作与同步 | [08_原子操作与同步.md](08_原子操作与同步.md) |
| 09 | 归约算法 | [09_归约算法.md](09_归约算法.md) |
| 10 | 卷积在GPU上的实现 | [10_卷积在GPU上的实现.md](10_卷积在GPU上的实现.md) |
| 11 | 流与异步执行 | [11_流与异步执行.md](11_流与异步执行.md) |
| 12 | 统一内存与托管内存 | [12_统一内存与托管内存.md](12_统一内存与托管内存.md) |
| 13 | cuBLAS与cuDNN使用 | [13_CUBLAS_CUDNN使用.md](13_CUBLAS_CUDNN使用.md) |
| 14 | CUDA性能分析工具 | [14_CUDA性能分析工具.md](14_CUDA性能分析工具.md) |
| 15 | 深度学习中的GPU优化 | [15_深度学习中的GPU优化.md](15_深度学习中的GPU优化.md) |
| 16 | Warp级编程 | [16_Warp级编程.md](16_Warp级编程.md) |
| 17 | 动态并行 | [17_动态并行.md](17_动态并行.md) |
| 18 | 多GPU编程 | [18_多GPU编程.md](18_多GPU编程.md) |

## 学习路径建议

**入门阶段（01-05）**：理解GPU架构与CUDA编程模型，能够编写基础kernel。

**进阶阶段（06-09）**：掌握内存优化、同步机制和经典并行算法。

**应用阶段（10-13）**：将CUDA应用到卷积、矩阵运算等实际场景，学会使用cuBLAS/cuDNN库。

**高级阶段（14-18）**：性能分析调优、深度学习优化、Warp级编程、多GPU协作。

## 环境要求

- NVIDIA GPU（Compute Capability 3.5+）
- CUDA Toolkit 11.0+
- GCC/G++ 7.0+（Linux）或 MSVC 2019+（Windows）
- 推荐IDE：VS Code + Nsight Extension

## 编译示例

```bash
nvcc -o hello hello.cu
nvcc -O3 -arch=sm_80 -o optimized kernel.cu
```
