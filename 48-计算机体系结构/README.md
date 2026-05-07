# 46-计算机体系结构

本目录系统整理计算机体系结构核心知识，涵盖从指令级并行到多处理器系统的完整体系。

## 目录结构

| 序号 | 目录 | 主要内容 |
|------|------|----------|
| 01 | [指令级并行](./01-指令级并行/README.md) | 流水线技术、超标量处理器、乱序执行、分支预测、推测执行、VLIW/EPIC、超线程 |
| 02 | [存储层次与缓存](./02-存储层次与缓存/README.md) | 存储层次结构、缓存映射与替换、写策略、缓存一致性协议（MESI/MOESI）、虚拟内存与TLB、预取 |
| 03 | [数据级并行](./03-数据级并行/README.md) | SIMD指令集（MMX/SSE/AVX/NEON）、GPU架构与SIMT、CUDA/OpenCL、向量处理器、GPGPU |
| 04 | [多处理器系统](./04-多处理器系统/README.md) | SMP/NUMA、互连网络、内存一致性模型、同步原语、多线程、片上网络NoC、异构计算 |

## 学习路线建议

```
指令级并行 ──→ 存储层次与缓存 ──→ 数据级并行 ──→ 多处理器系统
 (ILP基础)      (性能瓶颈分析)     (SIMD/GPU)      (多核/异构)
```

## 参考教材

- Patterson & Hennessy, *Computer Organization and Design* (RISC-V Edition)
- Hennessy & Patterson, *Computer Architecture: A Quantitative Approach* (6th Ed.)
- David Culler, *Parallel Computer Architecture*
