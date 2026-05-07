# 6_GGUF 格式与 llama.cpp

## 1. llama.cpp 简介

llama.cpp 由 Georgi Gerganov 开发，是**纯 C/C++ 实现的 LLM 推理引擎**，最大的特点是**无需 GPU，能在 CPU 上高效运行大模型**。

```
llama.cpp 生态:

llama.cpp (C/C++ 推理引擎)
  ├── GGUF 模型格式
  ├── 量化工具 (llama-quantize)
  ├── 服务器 (llama-server)
  ├── 绑定: Python, Go, Rust, Node.js, ...
  └── 上游: Ollama, LM Studio, GPT4All, ...

核心优势:
  ✓ 跨平台 (CPU/GPU/手机/嵌入式)
  ✓ 零依赖 (不需要 PyTorch/CUDA)
  ✓ 丰富的量化级别 (2-8 bit)
  ✓ 极致优化 (SIMD, Metal, Vulkan, CUDA)
```

## 2. GGUF 格式

GGUF (GGML Universal Format) 是 llama.cpp 的模型文件格式，替代了旧的 GGML/GGMF 格式。

```
GGUF 文件结构:

┌─────────────────────────────┐
│  Magic Number (GGUF)        │  文件标识
├─────────────────────────────┤
│  Version                    │  格式版本
├─────────────────────────────┤
│  Tensor Count               │  张量数量
├─────────────────────────────┤
│  Metadata Key-Value Pairs   │  模型元数据
│  - general.architecture     │  (llama, mistral, qwen, ...)
│  - llama.context_length     │  上下文长度
│  - llama.embedding_length   │  嵌入维度
│  - llama.block_count        │  层数
│  - llama.attention.head_count│  注意力头数
│  - ...                      │
├─────────────────────────────┤
│  Tensor Info                │  张量信息
│  - name, shape, dtype, offset│
├─────────────────────────────┤
│  Alignment Padding          │  对齐填充
├─────────────────────────────┤
│  Tensor Data                │  实际权重数据
└─────────────────────────────┘
```

## 3. 量化级别

```python
"""
GGUF 量化级别 (Q 格式):

级别    │ 位宽 │ 描述                   │ 质量  │ 大小(vs FP16)
────────┼──────┼────────────────────────┼───────┼──────────────
F32     │ 32   │ 全精度                 │ 100%  │ 200%
F16     │ 16   │ 半精度                 │ ~100% │ 100%
BF16    │ 16   │ Brain Float            │ ~100% │ 100%
Q8_0    │  8   │ 8-bit 量化             │ ~99%  │ 50%
Q6_K    │  6   │ 6-bit k-quant          │ ~98%  │ 38%
Q5_K_M  │ ~5.5 │ 5-bit 中等 k-quant     │ ~97%  │ 34%
Q5_K_S  │  5   │ 5-bit 小 k-quant       │ ~96%  │ 33%
Q4_K_M  │ ~4.5 │ 4-bit 中等 k-quant     │ ~95%  │ 27%
Q4_K_S  │  4   │ 4-bit 小 k-quant       │ ~93%  │ 25%
Q3_K_M  │ ~3.5 │ 3-bit 中等 k-quant     │ ~90%  │ 22%
Q3_K_S  │  3   │ 3-bit 小 k-quant       │ ~85%  │ 20%
Q2_K    │  2   │ 2-bit k-quant          │ ~80%  │ 16%

K-Quant 技术:
  对不同层使用不同的量化精度
  - 注意力层: 使用更高精度 (Q6_K)
  - FFN 层: 使用更低精度 (Q4_K)
  - 输出层: 保持高精度 (Q8_0)
  → 在相同大小下获得更好质量
"""
```

## 4. 模型转换

```bash
# 1. 从 HuggingFace 格式转换为 GGUF
python convert_hf_to_gguf.py /path/to/model --outfile model-f16.gguf

# 2. 量化 (多种方式)
# 方式 A: 使用 llama-quantize
./llama-quantize model-f16.gguf model-q4_k_m.gguf Q4_K_M

# 方式 B: 一步完成转换+量化
python convert_hf_to_gguf.py /path/to/model \
    --outfile model-q4_k_m.gguf \
    --outtype q4_k_m

# 3. 可用的量化类型
./llama-quantize --help
# 列出所有支持的量化类型
```

## 5. llama.cpp API 使用

### 5.1 Python 绑定 (llama-cpp-python)

```python
from llama_cpp import Llama

# 加载 GGUF 模型
llm = Llama(
    model_path="model-q4_k_m.gguf",
    n_ctx=4096,          # 上下文长度
    n_batch=512,          # 批处理大小
    n_gpu_layers=0,       # GPU 卸载层数 (0=纯CPU)
    verbose=False
)

# 基础对话
output = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "你是一个有用的助手。"},
        {"role": "user", "content": "解释量子计算的基本原理"}
    ],
    max_tokens=512,
    temperature=0.7,
    stream=False
)

print(output["choices"][0]["message"]["content"])

# 流式输出
for chunk in llm.create_chat_completion(
    messages=[{"role": "user", "content": "写一首诗"}],
    max_tokens=256,
    stream=True
):
    delta = chunk["choices"][0].get("delta", {})
    if "content" in delta:
        print(delta["content"], end="", flush=True)
```

### 5.2 服务端部署

```bash
# 启动 OpenAI 兼容的 API 服务器
./llama-server \
    -m model-q4_k_m.gguf \
    --host 0.0.0.0 \
    --port 8080 \
    -c 4096 \              # 上下文长度
    -t 8 \                 # 线程数
    -ngl 0 \               # GPU 层数
    --chat-template chatml # 聊天模板

# API 调用 (与 OpenAI API 兼容)
curl http://localhost:8080/v1/chat/completions \
    -H "Content-Type: application/json" \
    -d '{
        "model": "model-q4_k_m",
        "messages": [{"role": "user", "content": "Hello!"}],
        "max_tokens": 128
    }'
```

## 6. 性能优化

```python
# llama.cpp 性能调优参数

config = {
    # 线程设置
    "n_threads": 8,           # CPU 线程数 = 物理核心数

    # 批处理
    "n_batch": 512,           # Prompt 处理批大小
    "n_ubatch": 256,          # 统一批大小

    # 上下文
    "n_ctx": 4096,            # 上下文长度
    "n_keep": 0,              # 保留的初始 token 数

    # GPU 卸载
    "n_gpu_layers": 0,        # 卸载到 GPU 的层数
    "main_gpu": 0,            # 主 GPU

    # 内存
    "use_mmap": True,         # 内存映射加载
    "use_mlock": False,       # 锁定内存

    # 采样
    "temperature": 0.7,
    "top_p": 0.9,
    "top_k": 40,
    "repeat_penalty": 1.1,
}

# 性能基准参考 (Apple M2 Ultra, 7B Q4_K_M):
# - Prompt 处理: ~150 tok/s
# - Token 生成: ~30 tok/s

# 性能基准参考 (Intel i7-13700K, 7B Q4_K_M):
# - Prompt 处理: ~80 tok/s
# - Token 生成: ~15 tok/s
```

## 7. GPU 卸载 (Partial Offloading)

```python
# 当 GPU 内存不足以放下整个模型时，可以部分卸载
llm = Llama(
    model_path="model-q4_k_m.gguf",
    n_gpu_layers=20,  # 将 20 层卸载到 GPU
    # 剩余层在 CPU 上计算
    # 适用于: GPU 内存 < 模型大小的场景
)

# 常见配置:
# 7B  Q4: 全部 35 层可放入 6GB GPU
# 13B Q4: 全部 40 层可放入 10GB GPU
# 70B Q4: 需要 ~40GB, 可部分卸载到 24GB GPU
```

## 8. Ollama 简介

Ollama 是基于 llama.cpp 的**用户友好封装**，简化了模型管理和部署。

```bash
# 安装: brew install ollama (macOS) 或下载安装包

# 拉取并运行模型
ollama run llama3.1:8b

# 列出可用模型
ollama list

# 自定义 Modelfile
cat > Modelfile << 'EOF'
FROM llama3.1:8b
PARAMETER temperature 0.7
PARAMETER num_ctx 4096
SYSTEM 你是一个专业的编程助手。
EOF

ollama create mymodel -f Modelfile

# API 服务 (默认 http://localhost:11434)
curl http://localhost:11434/api/chat -d '{
    "model": "llama3.1:8b",
    "messages": [{"role": "user", "content": "Hello"}]
}'
```

## 总结

llama.cpp + GGUF 格式是**本地部署 LLM 的黄金标准**，核心优势在于跨平台、零依赖和丰富的量化级别。K-Quant 技术通过**对不同层使用不同量化精度**实现了质量与大小的最佳平衡。Ollama 进一步降低了使用门槛，让每个人都能在本地运行大模型。
