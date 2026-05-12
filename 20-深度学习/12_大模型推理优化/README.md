# 大模型推理优化

## 一、推理瓶颈

大模型推理的主要瓶颈：
- **内存带宽**：KV Cache占用大量显存
- **计算量**：自回归解码每步都需要前向传播
- **延迟**：用户等待时间敏感

---

## 二、KV Cache优化

### 2.1 PagedAttention

借鉴操作系统虚拟内存，将KV Cache分成固定大小的page：
- 减少内存碎片
- 支持请求间KV Cache共享（如beam search）
- vLLM框架的核心创新

### 2.2 MQA / GQA

- **MHA**：每个头有独立的K、V
- **MQA**：所有头共享一组K、V（减少Cache）
- **GQA**：分组共享K、V（折中方案）

### 2.3 KV Cache量化

将FP16的KV Cache量化为INT8/INT4，减少内存占用。

---

## 三、注意力优化

### 3.1 Flash Attention

IO感知的注意力计算：
- 分块计算，利用SRAM
- 避免 materialization 完整注意力矩阵
- 显著减少显存访问

### 3.2 Flash Attention 2/3

进一步优化并行度和计算效率。

---

## 四、量化

### 4.1 权重量化

- **GPTQ**：训练后量化（PTQ），INT4/INT8
- **AWQ**：激活感知权重量化
- **GGUF**：llama.cpp的量化格式

### 4.2 量化精度

| 位宽 | 精度损失 | 速度 | 显存 |
|------|----------|------|------|
| FP16 | 基准 | 1x | 1x |
| INT8 | 轻微 | ~1.5x | 0.5x |
| INT4 | 可接受 | ~2x | 0.25x |

---

## 五、投机解码

小模型快速生成草稿，大模型并行验证：
- 草稿模型：小模型，快速但不精确
- 验证模型：大模型，并行验证多个token
- 保证输出与大模型一致
- 速度提升2-3倍

---

## 六、连续批处理

- 动态合并多个请求
- 交错执行不同请求的prefill和decode
- 提高GPU利用率

---

## 七、分布式推理

- **Tensor Parallelism**：将单层切分到多GPU
- **Pipeline Parallelism**：不同层在不同GPU
- **Expert Parallelism**：MoE模型的专家并行
