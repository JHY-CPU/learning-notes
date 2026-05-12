# 113-游戏性能优化

## 模块概述

本模块系统讲解游戏引擎性能优化的各方面技术，涵盖CPU、GPU、内存、加载等核心优化方向。

## 目录

1. CPU性能分析与优化 - Profiler使用、热点定位、SIMD优化
2. Draw Call合批技术 - Static/Dynamic Batching、GPU Instancing、SRP Batcher
3. 遮挡剔除(Occlusion Culling) - PVS、软件光栅化剔除、Precomputed Visibility
4. LOD层次细节 - 模型LOD、LOD Group、CrossFade、HLOD方案
5. 内存管理与优化 - 内存池、对象池、GC优化、纹理内存预算
6. 渲染优化策略 - Overdraw减少、Early-Z、RenderTexture复用
7. 物理优化 - 碰撞层矩阵、休眠机制、简化碰撞体
8. 脚本与逻辑优化 - 缓存组件引用、协程优化、避免GC分配
9. 加载优化 - 异步加载、流式加载、预加载策略
10. 多线程与Job System - Unity Job System、ECS Burst、UE Task Graph
