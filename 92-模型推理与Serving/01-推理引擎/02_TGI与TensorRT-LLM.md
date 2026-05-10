# TGI与TensorRT-LLM - 模型推理与Serving

*HuggingFace TGI 和 NVIDIA TensorRT-LLM 两大推理引擎的架构、特性与选型对比*

TGI 主要特性

| 特性 | 说明 |
| --- | --- |
| **Rust Tokenizer** | 高性能Rust实现的tokenizer服务，避免GIL瓶颈 |
| **Flash Attention** | 集成Flash Attention 2，加速prefill阶段 |
| **Continuous Batching** | 支持连续批处理，GPU利用率高 |
| **量化支持** | GPTQ、AWQ、EETQ、BitsAndBytes等量化方法 |
| **Tensor Parallel** | 多GPU张量并行推理 |
| **水印** | 集成A Watermark for LLMs进行文本水印 |
| **Guided Generation** | 支持JSON Schema、正则表达式约束生成 |
| **推测采样** | 支持Medusa等推测解码方法 |
| **GRPC + HTTP** | 同时提供gRPC和REST API接口 |

FP8 量化推理

H100 GPU 引入了对 FP8 (Float8) 的原生支持，TensorRT-LLM 充分利用这一特性：


- **FP8 E4M3**
   : 4位指数 + 3位尾数，适合前向推理
- **FP8 E5M2**
   : 5位指数 + 2位尾数，适合梯度计算
- **性能提升**
   : 相比FP16推理速度提升 1.5x-2x
- **精度损失**
   : 在大多数任务上精度损失小于 0.5%
- **自动校准**
   : TensorRT-LLM 自动进行量化校准（无需手动提供校准数据集）

三大推理引擎对比

| 维度 | vLLM | TGI | TensorRT-LLM |
| --- | --- | --- | --- |
| **开发者** | UC Berkeley | HuggingFace | NVIDIA |
| **语言** | Python + CUDA | Rust + Python | C++ + Python |
| **核心创新** | PagedAttention | HF生态集成 | 编译优化 |
| **吞吐量** | 高 | 中高 | 最高 |
| **首Token延迟** | 中 | 中 | 低 |
| **易用性** | 高 (pip install) | 高 (Docker) | 中 (需要编译) |
| **模型支持** | 广泛 | HF模型为主 | 主流模型 |
| **量化** | AWQ/GPTQ/FP8 | GPTQ/AWQ/BnB | FP8/INT8/INT4 |
| **多LoRA** | 支持 | 支持 | 支持 |
| **硬件锁定** | NVIDIA/AMD | NVIDIA/Intel | NVIDIA only |
| **部署复杂度** | 低 | 低 | 高 |
| **适用场景** | 通用推理 | HF生态快速部署 | 极致性能要求 |

Llama 2 7B 性能参考 (A100-80G)

| 引擎 | 吞吐量 (tokens/s) | P50延迟 (ms/token) | P99延迟 (ms/token) |
| --- | --- | --- | --- |
| HF Transformers | ~300 | ~50 | ~120 |
| TGI | ~2500 | ~15 | ~35 |
| vLLM | ~3500 | ~12 | ~30 |
| TensorRT-LLM | ~4500 | ~8 | ~20 |


** 数据为典型场景参考值，实际性能受配置、输入长度等因素影响*


<!-- Converted from: 02_TGI与TensorRT-LLM.html -->
