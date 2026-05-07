# 12_TGI 与推理服务化

## 1. 推理服务化概述

LLM 推理服务化是将模型部署为**可伸缩、高可用的 API 服务**，核心框架包括 HuggingFace TGI、vLLM、TensorRT-LLM 等。

```
推理服务化架构:

客户端 → Load Balancer → API Gateway → 推理实例集群
                                              │
                        ┌──────────────────────┤
                        │           │          │
                    Instance₁  Instance₂  Instance₃
                    (GPU 0-1)   (GPU 2-3)  (GPU 4-5)

关键能力:
  ✓ 高吞吐 (连续批处理)
  ✓ 低延迟 (流式输出)
  ✓ 高可用 (多实例 + 负载均衡)
  ✓ 可扩展 (自动扩缩容)
```

## 2. HuggingFace TGI

### 2.1 基本使用

```bash
# Docker 部署 (最简单)
docker run --gpus all \
    -p 8080:80 \
    -v ~/.cache/huggingface:/data \
    ghcr.io/huggingface/text-generation-inference:latest \
    --model-id meta-llama/Llama-2-7b-hf \
    --num-shard 1 \
    --max-input-length 2048 \
    --max-total-tokens 4096 \
    --max-batch-total-tokens 8192 \
    --max-batch-prefill-tokens 4096

# API 调用
curl localhost:8080/generate \
    -X POST \
    -H 'Content-Type: application/json' \
    -d '{"inputs": "量子计算的基本原理是", "parameters": {"max_new_tokens": 256}}'
```

### 2.2 TGI 核心特性

```
TGI 特性:

1. Flash Attention 2
   自动使用优化的注意力 kernel

2. 连续批处理
   iteration-level batching

3. 量化支持
   GPTQ, AWQ, EETQ, bitsandbytes

4. 流式输出 (SSE)
   Server-Sent Events 实时返回 token

5. 工具调用
   支持 OpenAI 兼容的 function calling

6. 健康检查
   /health, /health/live, /health/ready

7. 指标监控
   Prometheus metrics endpoint
```

## 3. 流式输出

```python
import requests
import json

def stream_generate(prompt: str, api_url: str = "http://localhost:8080"):
    """流式生成"""
    response = requests.post(
        f"{api_url}/generate_stream",
        json={
            "inputs": prompt,
            "parameters": {
                "max_new_tokens": 512,
                "temperature": 0.7,
                "top_p": 0.9,
                "repetition_penalty": 1.1,
            }
        },
        stream=True
    )

    for line in response.iter_lines():
        if line:
            line = line.decode("utf-8")
            if line.startswith("data:"):
                data = json.loads(line[5:])
                if "token" in data:
                    print(data["token"]["text"], end="", flush=True)
                if data.get("generated_text"):
                    break

# OpenAI 兼容的流式 API
def openai_stream(prompt: str):
    response = requests.post(
        "http://localhost:8000/v1/chat/completions",
        json={
            "model": "llama-2-7b",
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": 256,
            "stream": True
        },
        stream=True
    )

    for line in response.iter_lines():
        if line and line.startswith(b"data: ") and line != b"data: [DONE]":
            data = json.loads(line[6:])
            content = data["choices"][0]["delta"].get("content", "")
            print(content, end="", flush=True)
```

## 4. 负载均衡

```python
import random
from typing import List

class LoadBalancer:
    """推理服务负载均衡"""

    def __init__(self, instances: List[str]):
        self.instances = instances
        self.request_counts = {inst: 0 for inst in instances}

    def round_robin(self) -> str:
        """轮询"""
        return min(self.instances, key=lambda x: self.request_counts[x])

    def least_connections(self, active_connections: dict) -> str:
        """最少连接"""
        return min(active_connections, key=active_connections.get)

    def weighted_random(self, weights: dict) -> str:
        """加权随机"""
        return random.choices(
            list(weights.keys()),
            weights=list(weights.values())
        )[0]

    def route(self, request) -> str:
        """路由请求"""
        # 根据请求特征选择实例
        estimated_tokens = len(request["inputs"].split()) * 2

        if estimated_tokens > 1000:
            # 长请求路由到专用实例
            return self.route_to_large_instance()
        else:
            return self.round_robin()
```

## 5. 自动扩缩容

```python
class AutoScaler:
    """基于指标的自动扩缩容"""

    def __init__(self, config: dict):
        self.min_replicas = config.get("min", 1)
        self.max_replicas = config.get("max", 10)
        self.target_gpu_util = config.get("target_gpu_utilization", 0.8)
        self.target_queue_length = config.get("target_queue_length", 10)

    def should_scale(self, metrics: dict) -> str:
        """决定是否扩缩容"""
        gpu_util = metrics.get("gpu_utilization", 0)
        queue_len = metrics.get("pending_requests", 0)
        current_replicas = metrics.get("replicas", 1)

        if gpu_util > self.target_gpu_util and current_replicas < self.max_replicas:
            return "scale_up"
        elif gpu_util < self.target_gpu_util * 0.5 and current_replicas > self.min_replicas:
            return "scale_down"
        elif queue_len > self.target_queue_length and current_replicas < self.max_replicas:
            return "scale_up"

        return "no_change"

    def calculate_new_replicas(self, action: str, current: int) -> int:
        if action == "scale_up":
            return min(current + 1, self.max_replicas)
        elif action == "scale_down":
            return max(current - 1, self.min_replicas)
        return current
```

## 6. 监控指标

```python
# Prometheus 指标

# 关键指标:
"""
推理延迟:
  tgi_request_duration_seconds        # 请求总延迟
  tgi_request_inference_duration_seconds  # 推理延迟
  tgi_request_queue_duration_seconds    # 排队延迟

吞吐量:
  tgi_request_count                   # 请求数
  tgi_generated_tokens_total          # 生成 token 总数

批处理:
  tgi_batch_size                      # 当前批大小
  tgi_batch_inference_count           # 批推理次数

内存:
  tgi_cache_memory_usage_bytes        # KV Cache 内存
  tgi_queue_size                      # 排队请求数
"""

# 监控面板 (Grafana)
class InferenceMonitor:
    def collect_metrics(self, instance_url: str) -> dict:
        """收集推理实例指标"""
        response = requests.get(f"{instance_url}/metrics")
        return self.parse_prometheus_metrics(response.text)

    def alert(self, metric: str, value: float, threshold: float):
        """告警"""
        if value > threshold:
            print(f"[ALERT] {metric} = {value} > {threshold}")
```

## 7. 多模型服务

```python
class ModelRouter:
    """多模型路由"""

    def __init__(self):
        self.models = {
            "gpt-4-level": {
                "endpoint": "http://llama-70b:8080",
                "cost_per_1k": 0.03
            },
            "fast": {
                "endpoint": "http://llama-7b:8080",
                "cost_per_1k": 0.001
            },
            "code": {
                "endpoint": "http://codellama:8080",
                "cost_per_1k": 0.002
            }
        }

    def route(self, request: dict) -> str:
        """根据请求选择模型"""
        # 显式指定模型
        if "model" in request:
            return self.models[request["model"]]["endpoint"]

        # 根据任务类型路由
        task_type = self.classify_task(request["messages"])
        if task_type == "coding":
            return self.models["code"]["endpoint"]
        elif task_type == "simple_qa":
            return self.models["fast"]["endpoint"]
        else:
            return self.models["gpt-4-level"]["endpoint"]
```

## 8. 部署架构对比

```
┌──────────────┬────────────────┬────────────────┬────────────────┐
│    框架       │      TGI       │     vLLM       │ TensorRT-LLM   │
├──────────────┼────────────────┼────────────────┼────────────────┤
│ 开发者        │ HuggingFace    │ UC Berkeley    │ NVIDIA         │
│ 连续批处理    │ ✓              │ ✓              │ ✓              │
│ 量化          │ GPTQ/AWQ/BnB   │ GPTQ/AWQ/FP8   │ INT8/INT4/FP8  │
│ Tensor并行   │ ✓              │ ✓              │ ✓              │
│ 流式输出      │ ✓              │ ✓              │ ✓              │
│ 前缀缓存      │ ✗              │ ✓              │ ✓              │
│ OpenAI兼容   │ ✓              │ ✓              │ 需适配          │
│ 易用性        │ 高(Docker)     │ 高             │ 低(需编译)     │
│ 性能          │ 好             │ 最佳           │ 最佳(需调优)   │
└──────────────┴────────────────┴────────────────┴────────────────┘
```

## 总结

推理服务化将模型从**脚本**提升为**生产服务**。TGI 以 Docker 镜像方式提供开箱即用的推理服务，vLLM 以 Python 库方式提供更灵活的定制能力。流式输出、连续批处理、自动扩缩容和监控是生产部署的必备能力。
