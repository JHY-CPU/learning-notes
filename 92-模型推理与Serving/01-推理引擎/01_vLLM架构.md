# vLLM架构 - 模型推理与Serving

*PagedAttention、Continuous Batching、KV Cache管理等核心创新，vLLM如何成为LLM推理的事实标准*

性能对比（论文数据）

- 相比 HuggingFace Transformers: 吞吐量提升
   **14x-24x**
- 相比 HuggingFace TGI: 吞吐量提升
   **2.2x-3.5x**
- KV Cache利用率从 ~50% 提升到
   **<4% 浪费**
- 支持在同等硬件上服务
   **4x-6x 更多并发请求**

vLLM 调度器 (Scheduler)

- **Running队列**
   : 当前正在处理的请求
- **Waiting队列**
   : 等待GPU资源的新请求
- **Swapped队列**
   : 因显存不足被换出到CPU的请求
- **调度逻辑**
   :

   1. Running队列中的请求生成一个token
   2. 如果请求完成（遇到EOS或达到max_tokens），从Running中移除
   3. 如果有空闲KV Cache块，从Waiting中选择新请求加入Running
   4. 如果显存不足，将低优先级请求Swap到CPU

分块预填充机制

- 将长prompt的prefill切分为多个chunk（如每chunk 512 tokens）
- 每个调度step中，prefill chunk和decode请求
   **混合调度**
- 保证decode请求的延迟不会被长prefill阻塞
- 代价：prefill阶段的吞吐量略有下降（约5-10%）
- 收益：P99延迟显著降低（特别是有长输入请求时）

## PagedAttention 核心原理

传统注意力机制为每个请求预分配最大长度的 KV Cache，导致严重的内存浪费。

```
传统方式：
请求1: [KV████░░░░░░░░░░]  预分配16块，实际用4块 → 浪费75%
请求2: [KV██████░░░░░░░░]  预分配16块，实际用6块 → 浪费62.5%

PagedAttention：
请求1: [KV████]  使用4个物理块
请求2: [KV██████] 使用6个物理块
块表记录逻辑块到物理块的映射，无预分配浪费
```

PagedAttention 借鉴操作系统的虚拟内存/分页机制：
- 将 KV Cache 分为固定大小的块(如 16 tokens)
- 通过块表(Block Table)维护逻辑块到物理块的映射
- 按需分配物理块，消除碎片化浪费
- 支持 Copy-on-Write，共享相同前缀的请求可以共享 KV 块

## Continuous Batching

传统静态批处理：一批中所有请求必须同时完成，短请求等待长请求。

```
静态批处理：
Step 1: [req1-T1, req2-T1, req3-T1] → 全部计算
Step 2: [req1-T2, req2-T2, req3-T2] → 全部计算
... req1 完成后仍在等待 req2, req3

Continuous Batching：
Step 1: [req1-T1, req2-T1, req3-T1]
Step 2: [req1-T2, req2-T2, req3-T3] → req3 完成
Step 3: [req1-T3, req2-T3, req4-T1] → 新请求 req4 立即加入
```

优势：
- 吞吐量提升 2-10x
- 请求完成即释放资源
- 新请求无需等待整批完成

## vLLM 支持的模型

| 模型类型 | 代表模型 | 特点 |
|----------|---------|------|
| Decoder-only | LLaMA, Mistral, Qwen | 最常见，原生支持 |
| MoE | Mixtral, Qwen-MoE | 专家并行优化 |
| 多模态 | LLaVA, Qwen-VL | 图文理解 |
| Embedding | BGE, E5 | 向量生成 |

## vLLM 使用示例

```python
from vllm import LLM, SamplingParams

# 初始化引擎
llm = LLM(
    model="Qwen/Qwen2-7B",
    tensor_parallel_size=2,  # 2卡张量并行
    gpu_memory_utilization=0.9,
    max_model_len=8192,
)

# 批量推理
prompts = ["解释量子计算", "什么是深度学习"]
sampling_params = SamplingParams(temperature=0.7, max_tokens=512)
outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    print(output.outputs[0].text)

# 兼容 OpenAI API
# vllm.entrypoints.openai.api_server --model Qwen/Qwen2-7B --port 8000
# curl http://localhost:8000/v1/completions -d '{"model":"Qwen/Qwen2-7B","prompt":"..."}'
```

## 与其他推理引擎对比

| 引擎 | 核心优势 | 适用场景 |
|------|---------|---------|
| vLLM | PagedAttention，吞吐量最高 | 高并发在线服务 |
| TGI (HuggingFace) | 生态集成好，部署简单 | 快速上线 |
| TensorRT-LLM | NVIDIA GPU 极致优化 | NVIDIA 专属部署 |
| SGLang | RadixAttention，结构化生成 | 复杂 Prompt 场景 |
| llama.cpp | CPU/边缘推理 | 本地部署/资源受限 |


<!-- Converted from: 01_vLLM架构.html -->
