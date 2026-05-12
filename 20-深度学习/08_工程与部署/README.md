# 08_工程与部署

> 从PyTorch基础到生产级部署的全链路工程实践。覆盖数据加载、模块设计、模型保存/加载、分布式训练（DDP/FSDP/DeepSpeed）、混合精度、模型压缩（剪枝/量化/蒸馏）、推理优化（ONNX/TensorRT）与服务化部署（Triton/TorchServe）。

---

## 基础知识

- **前置知识**：03_神经网络核心; Python 工程经验; CUDA 基础概念
- **关联目录**：05_NLP与序列模型（LLM 部署）; 04_计算机视觉（CV 模型部署）
- **笔记数量**：共 20 篇

---

## 内容结构

#### PyTorch基础

张量操作、DataLoader、自定义Dataset、nn.Module、Autograd、模型保存/加载

| 编号 | 笔记 |
|------|------|
| 00 | [PyTorch 张量操作与广播机制 (Broadcasting)](0_PyTorch 张量操作与广播机制 (Broadcasting).md) |
| 01 | [Dataset 与 DataLoader 高效数据加载](1_Dataset 与 DataLoader 高效数据加载.md) |
| 02 | [自定义 Dataset 类实现与增强策略](2_自定义 Dataset 类实现与增强策略.md) |
| 03 | [nn.Module 模块化设计与参数管理](3_nn.Module 模块化设计与参数管理.md) |
| 04 | [动态计算图与 Autograd 引擎底层逻辑](4_动态计算图与 Autograd 引擎底层逻辑.md) |
| 05 | [模型保存与加载：state_dict 最佳实践](5_模型保存与加载：state_dict 最佳实践.md) |

#### 分布式训练

DataParallel/DDP对比、FSDP、DeepSpeed ZeRO、混合精度AMP

| 编号 | 笔记 |
|------|------|
| 06 | [分布式训练基础：DataParallel vs DistributedDataParallel](6_分布式训练基础：DataParallel vs DistributedDataParallel.md) |
| 07 | [FSDP (Fully Sharded Data Parallel) 原理](7_FSDP (Fully Sharded Data Parallel) 原理.md) |
| 08 | [Deepspeed 集成与 ZeRO 优化技术](8_Deepspeed 集成与 ZeRO 优化技术.md) |
| 09 | [混合精度训练 (AMP) 的 PyTorch 实现](9_混合精度训练 (AMP) 的 PyTorch 实现.md) |

#### 模型压缩

剪枝、知识蒸馏、PTQ量化、QAT量化感知训练

| 编号 | 笔记 |
|------|------|
| 10 | [模型压缩：剪枝 (Pruning) 技术与稀疏性](10_模型压缩：剪枝 (Pruning) 技术与稀疏性.md) |
| 11 | [知识蒸馏 (Distillation) 的工程化流程](11_知识蒸馏 (Distillation) 的工程化流程.md) |
| 12 | [量化技术：PTQ (训练后量化) 基础](12_量化技术：PTQ (训练后量化) 基础.md) |
| 13 | [QAT (量化感知训练) 伪量化节点插入](13_QAT (量化感知训练) 伪量化节点插入.md) |

#### 推理优化与部署

ONNX格式转换、TensorRT层融合、Triton Server、TorchServe

| 编号 | 笔记 |
|------|------|
| 14 | [ONNX 格式转换与算子兼容性处理](14_ONNX 格式转换与算子兼容性处理.md) |
| 15 | [TensorRT 加速：层融合 (Layer Fusion) 优化](15_TensorRT 加速：层融合 (Layer Fusion) 优化.md) |
| 16 | [Triton Inference Server 服务化部署](16_Triton Inference Server 服务化部署.md) |
| 17 | [TorchServe 模型版本管理与批处理](17_TorchServe 模型版本管理与批处理.md) |

#### 移动端与监控

移动端部署（ExecuTorch）、模型监控与漂移检测

| 编号 | 笔记 |
|------|------|
| 18 | [移动端部署：PyTorch Mobile 与 CoreML](18_移动端部署：PyTorch Mobile 与 CoreML.md) |
| 19 | [模型监控与漂移检测 (Drift Detection)](19_模型监控与漂移检测 (Drift Detection).md) |
---

## 学习建议

1. 按编号顺序阅读每个子主题内的笔记，因为内部存在递进关系
2. 每个子主题完成后，尝试用「深度学习关联」部分串联知识点
3. 代码示例可以直接复制运行（需要 PyTorch 和 transformers 库）
4. 遇到数学推导不熟悉时，回到 01_数学基础 查阅对应基础

---

*本 README 由笔记元数据自动生成。*
