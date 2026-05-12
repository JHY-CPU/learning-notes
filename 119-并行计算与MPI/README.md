# 119-并行计算与MPI

本模块系统整理并行计算的核心理论与主流编程模型，涵盖 MPI、OpenMP、Pthreads 以及分布式并行框架。

## 目录

| 编号 | 文件 | 主题 |
|------|------|------|
| 01 | `01_并行计算概述.md` | Flynn分类法、共享/分布式内存、加速比与Amdahl定律 |
| 02 | `02_MPI编程基础.md` | MPI初始化、通信子、点对点通信 |
| 03 | `03_MPI集合通信.md` | 广播、散射、收集、归约操作 |
| 04 | `04_MPI派生数据类型.md` | 连续/向量/结构体类型、打包/解包 |
| 05 | `05_MPI非阻塞通信.md` | 异步通信、通信与计算重叠 |
| 06 | `06_MPI进程拓扑.md` | 笛卡尔/图拓扑、虚拟拓扑用于网格计算 |
| 07 | `07_OpenMP编程.md` | 并行for、归约、调度策略、critical/atomic |
| 08 | `08_OpenMP高级特性.md` | 任务并行、SIMD、线程亲和性、嵌套并行 |
| 09 | `09_Pthreads多线程.md` | 线程创建/回收、互斥锁、条件变量 |
| 10 | `10_并行算法设计.md` | 并行前缀和、并行排序、并行矩阵运算 |
| 11 | `11_分布式并行框架.md` | MapReduce、Spark RDD、Ray框架 |
| 12 | `12_并行性能分析.md` | 加速比、效率、可扩展性、性能分析工具 |

## 环境要求

- **编译器**: GCC 9+ 或 Intel ICC
- **MPI**: OpenMPI 4.x 或 MPICH 3.x
- **OpenMP**: 编译器内置支持（GCC/Clang/ICC 均可）
- **Python**: 3.8+（用于 Spark/Ray 示例）

## 编译与运行

```bash
# MPI 程序
mpicc -o mpi_demo 02_MPI编程基础.c -lm
mpirun -np 4 ./mpi_demo

# OpenMP 程序
gcc -fopenmp -o omp_demo 07_OpenMP编程.c
./omp_demo

# Pthreads 程序
gcc -pthread -o thread_demo 09_Pthreads多线程.c
./thread_demo
```
