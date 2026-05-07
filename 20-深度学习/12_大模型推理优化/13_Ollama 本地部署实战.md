# 13_Ollama 本地部署实战

## 1. Ollama 概述

Ollama 是最流行的**本地 LLM 运行工具**，底层基于 llama.cpp，提供了极简的命令行和 API 接口。

```
Ollama 优势:
  ✓ 一行命令安装和运行
  ✓ 自动模型管理 (拉取/删除/列表)
  ✓ OpenAI 兼容 API
  ✓ 支持 Mac/Linux/Windows
  ✓ 自动 GPU 检测和加速
  ✓ Modelfile 自定义模型
```

## 2. 安装与基础使用

```bash
# 安装
# macOS: brew install ollama
# Linux: curl -fsSL https://ollama.com/install.sh | sh
# Windows: 下载 ollama.com 安装包

# 启动服务 (后台运行)
ollama serve

# 拉取模型
ollama pull llama3.1:8b        # 8B 参数
ollama pull llama3.1:70b       # 70B 参数
ollama pull qwen2:7b           # 通义千问
ollama pull codellama:13b      # 代码专用
ollama pull mistral:7b         # Mistral

# 运行模型 (交互式)
ollama run llama3.1:8b

# 列出已安装模型
ollama list

# 删除模型
ollama rm llama3.1:8b

# 查看模型信息
ollama show llama3.1:8b --modelfile
```

## 3. 模型选择指南

```
模型推荐 (按用途):

通用对话:
  llama3.1:8b     - 最佳平衡 (4.7GB)
  qwen2:7b        - 中文优秀 (4.4GB)
  mistral:7b      - 欧洲模型 (4.1GB)

代码生成:
  codellama:13b   - 代码专用 (7.4GB)
  deepseek-coder:6.7b - 代码 (3.8GB)
  starcoder2:7b   - 多语言代码 (4.0GB)

中文场景:
  qwen2:7b        - 中文理解 (4.4GB)
  glm4:9b         - 智谱 (5.5GB)
  yi:9b           - 零一万物 (5.2GB)

小型/边缘:
  tinyllama:1.1b  - 超小 (637MB)
  phi3:mini       - 微软小模型 (2.2GB)
  gemma2:2b       - 谷歌小模型 (1.6GB)
```

## 4. API 使用

```python
import requests
import json

# Ollama API (默认 http://localhost:11434)

# 1. 简单对话
def chat(prompt: str, model: str = "llama3.1:8b") -> str:
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": 0.7,
                "num_predict": 512,
            }
        }
    )
    return response.json()["response"]

# 2. 多轮对话
def multi_turn(messages: list, model: str = "llama3.1:8b") -> str:
    response = requests.post(
        "http://localhost:11434/api/chat",
        json={
            "model": model,
            "messages": messages,
            "stream": False
        }
    )
    return response.json()["message"]["content"]

# 使用示例
messages = [
    {"role": "system", "content": "你是一个有帮助的助手。"},
    {"role": "user", "content": "什么是 Transformer？"},
]
answer = multi_turn(messages)
print(answer)

# 3. 流式输出
def stream_chat(prompt: str, model: str = "llama3.1:8b"):
    response = requests.post(
        "http://localhost:11434/api/generate",
        json={"model": model, "prompt": prompt, "stream": True},
        stream=True
    )
    for line in response.iter_lines():
        if line:
            data = json.loads(line)
            print(data.get("response", ""), end="", flush=True)
            if data.get("done"):
                break

# 4. OpenAI 兼容接口
import openai

client = openai.OpenAI(
    base_url="http://localhost:11434/v1",
    api_key="ollama"  # 随意填写
)

response = client.chat.completions.create(
    model="llama3.1:8b",
    messages=[{"role": "user", "content": "Hello!"}],
    max_tokens=256,
)
print(response.choices[0].message.content)
```

## 5. Modelfile 自定义

```dockerfile
# Modelfile - 自定义模型配置
FROM llama3.1:8b

# 系统提示词
SYSTEM """
你是一个专业的Python编程助手。
你的回答应该：
1. 包含可运行的代码
2. 有清晰的注释
3. 遵循 PEP8 规范
"""

# 参数设置
PARAMETER temperature 0.5
PARAMETER top_p 0.9
PARAMETER top_k 40
PARAMETER repeat_penalty 1.1
PARAMETER num_ctx 4096

# 停止词
PARAMETER stop "```"
PARAMETER stop "Human:"
```

```bash
# 创建自定义模型
ollama create mycoder -f Modelfile

# 使用自定义模型
ollama run mycoder
```

## 6. 硬件适配与性能调优

```python
"""
硬件需求 (以 7B Q4 模型为例):

最低配置:
  - 内存: 8GB RAM
  - 显卡: 无 (纯 CPU, ~5-10 tok/s)
  - 磁盘: 5GB

推荐配置:
  - 内存: 16GB+ RAM
  - 显卡: RTX 3060 12GB+ (~30-50 tok/s)
  - 磁盘: SSD, 20GB+

高性能配置:
  - 内存: 32GB+ RAM
  - 显卡: RTX 4090 24GB (~80-100 tok/s)
  - 磁盘: NVMe SSD

Apple Silicon:
  - M1/M2 8GB:  7B ~10-15 tok/s
  - M1/M2 16GB: 7B ~20-30 tok/s
  - M2 Ultra:   7B ~40-60 tok/s, 70B ~8-12 tok/s
"""
```

```bash
# 性能调优参数
OLLAMA_NUM_PARALLEL=4      # 并行请求数
OLLAMA_MAX_LOADED_MODELS=2 # 最大加载模型数
OLLAMA_GPU_LAYERS=35       # GPU 卸载层数 (0=纯CPU)

# 环境变量设置
export OLLAMA_NUM_PARALLEL=4
export OLLAMA_HOST=0.0.0.0:11434
```

## 7. 实际应用示例

```python
# 示例: 基于 Ollama 的本地 RAG 系统
from langchain_community.llms import Ollama
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import OllamaEmbeddings

# 初始化
llm = Ollama(model="llama3.1:8b")
embeddings = OllamaEmbeddings(model="nomic-embed-text")

# 构建向量数据库
vectorstore = Chroma.from_documents(
    documents=docs,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# RAG 查询
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

def rag_query(question: str) -> str:
    relevant_docs = retriever.invoke(question)
    context = "\n".join(doc.page_content for doc in relevant_docs)

    prompt = f"""基于以下上下文回答问题。如果找不到答案，说明不知道。

上下文:
{context}

问题: {question}
"""
    return llm.invoke(prompt)
```

## 8. 常见问题排查

```bash
# 1. GPU 未被检测到
ollama list  # 检查模型是否显示 GPU 层数

# macOS: 确保使用 Metal 后端
# Linux: 检查 CUDA 驱动
nvidia-smi

# 2. 内存不足
# 使用更小的量化版本
ollama pull llama3.1:8b-q4_0  # 最小量化

# 3. 速度太慢
# 增加 GPU 卸载层数
OLLAMA_GPU_LAYERS=35 ollama run llama3.1:8b

# 4. 端口冲突
OLLAMA_HOST=0.0.0.0:11435 ollama serve
```

## 总结

Ollama 是本地运行 LLM 的最佳入门工具，**一行命令安装、一行命令运行**。对于个人开发、隐私敏感场景和离线使用，Ollama + GGUF 模型是最实用的方案。通过 Modelfile 可以定制模型行为，通过 API 可以集成到应用中。
