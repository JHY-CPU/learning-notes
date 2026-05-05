# 18_TorchServe 模型版本管理与批处理

## 核心概念

- **TorchServe**：由 PyTorch 和 AWS 联合开发的生产级模型服务框架，专门针对 PyTorch 模型优化。提供 HTTP/gRPC 推理接口、RESTful 管理 API 和开箱即用的模型版本管理功能，是 PyTorch 生态的首选部署方案。
- **模型存档 (MAR, Model ARchive)**：TorchServe 使用 `.mar` 文件作为模型打包格式。MAR 文件将模型权重 (state_dict)、模型定义代码 (model.py)、推理处理程序 (handler.py) 和依赖配置文件打包为单个 zip 文件，便于分发和版本管理。
- **自定义 Handler**：TorchServe 的核心扩展点。通过继承 `base_handler` 实现 `preprocess`、`inference`、`postprocess` 三个方法，分别对应请求数据预处理、模型推理和结果后处理。Handler 决定了推理流水线的完整逻辑。
- **模型版本管理 (Model Versioning)**：TorchServe 的 `management API` 支持在同一端口中注册同一模型的多个版本（如 v1.0, v2.0），并通过 `version` 参数指定默认版本、最小版本和可路由版本。新版本可以灰度上线，旧版本可以随时回滚。
- **批量推理 (Batch Inference)**：TorchServe 支持服务端自动累积请求凑 batch，通过 `batch_size` 和 `max_batch_delay` 配置。Handler 需要实现 `handle` 方法以支持批处理输入（list of requests）。
- **工作线程 (Workers)**：每个模型可以配置多个 worker 进程（或线程），通过 `minWorkers` 和 `maxWorkers` 控制。TorchServe 根据负载自动伸缩 worker 数量，实现弹性推理。

## 数学推导

TorchServe 的批处理策略可以通过排队论分析。假设请求到达过程是泊松过程，速率 $\lambda$，批处理窗口 $T_w$，目标 batch size $B$。

**批处理形成概率**：在时间窗口 $T_w$ 内到达 k 个请求的概率为：
$$
P(N(T_w) = k) = \frac{(\lambda T_w)^k e^{-\lambda T_w}}{k!}
$$

期望 batch size：
$$
\mathbb{E}[B] = \sum_{k=1}^{\infty} k \cdot P(N(T_w) = k) = \lambda T_w \cdot \frac{1 - e^{-\lambda T_w}}{\lambda T_w} \quad \text{(truncated)}
$$

更实际地，由于最大 batch size 限制为 $B_{\max}$：
$$
\mathbb{E}[B_{\text{eff}}] = \sum_{k=1}^{B_{\max}-1} k P(k) + B_{\max} \sum_{k=B_{\max}}^{\infty} P(k)
$$

**延迟分析**：
$$
\text{平均延迟} = \underbrace{T_w/2}_{\text{等待延迟}} + \underbrace{\mathbb{E}[S(B)]}_{\text{推理延迟}}
$$

其中 $S(B)$ 是 batch size 为 $B$ 时的推理时间。对于 Transformer 模型，$S(B) \approx c_0 + c_1 B$（近似线性扩展）。

## 直观理解

- **TorchServe = 模型部署的"交钥匙工程"**：相比 Triton 的灵活性需要较多配置，TorchServe 更像是"拎包入住"——一个命令就能把 PyTorch 模型部署为 RESTful API。特别适合 PyTorch 用户快速上线推理服务。
- **Model Versioning = 应用的版本管理**：类似于手机 App 的版本更新——v1.0 发布后，v1.1 先给 10% 的灰度用户（通过版本路由），确认无问题后全量发布。如果出问题，一键回滚到 v1.0。
- **最佳实践**：对于纯 PyTorch 模型部署，TorchServe 的 PyTorch 原生支持（Eager Mode、TorchScript、DDP）提供了最佳框架兼容性；对于多框架混合部署，选择 Triton。
- **常见陷阱**：自定义 Handler 需要在 `__init__` 中加载模型、tokenizer 等资源（仅一次），而非在每次请求时加载；`preprocess` 和 `postprocess` 在 CPU 上执行，`inference` 在 GPU 上执行——理解这一数据流是性能调优的关键。

## 代码示例

```python
# ========== 1. 自定义 Handler ==========
# 文件: custom_handler.py
from torchserve import base_handler
import torch
import torch.nn.functional as F
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification

class TextClassificationHandler(base_handler.BaseHandler):
    def __init__(self):
        super().__init__()
        self.tokenizer = None
        self.model = None
        self.device = None
        self.initialized = False

    def initialize(self, context):
        """模型初始化：只在服务启动时调用一次"""
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        properties = context.system_properties
        model_dir = properties.get("model_dir")

        # 加载模型和 tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()
        self.initialized = True

    def preprocess(self, data):
        """请求预处理：从 HTTP 请求中提取并编码文本"""
        batch_texts = []
        for row in data:
            # data 格式: [{"body": {"text": "..."}}, ...]
            text = row.get("body", {}).get("text", "")
            if isinstance(text, bytes):
                text = text.decode("utf-8")
            batch_texts.append(text)

        # 批量编码
        inputs = self.tokenizer(
            batch_texts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        return inputs

    def inference(self, inputs):
        """模型推理：接收预处理后的数据，返回 logits"""
        with torch.no_grad():
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
            outputs = self.model(**inputs)
            return outputs.logits

    def postprocess(self, logits):
        """后处理：将模型输出转换为 HTTP 响应格式"""
        probs = F.softmax(logits, dim=-1)
        predictions = probs.argmax(dim=-1).tolist()
        scores = probs.max(dim=-1).values.tolist()
        return [{"prediction": p, "score": s} for p, s in zip(predictions, scores)]

# ========== 2. 批量推理 Handler ==========
class BatchTextHandler(base_handler.BaseHandler):
    """支持动态批处理的 Handler"""
    def preprocess(self, data):
        return [item.get("body", {}).get("text", "") for item in data]

    def inference(self, texts):
        # texts 可能包含多个请求
        if isinstance(texts, list):
            texts = [t if isinstance(t, str) else "" for t in texts]
            inputs = self.tokenizer(texts, padding=True, truncation=True,
                                     return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            return outputs.logits
        return super().inference(texts)

# ========== 3. 打包和部署 ==========
"""
# 3.1 创建 MAR 文件
torch-model-archiver \\
  --model-name text_classifier \\
  --version 1.0 \\
  --serialized-file model/pytorch_model.bin \\
  --handler custom_handler.py \\
  --extra-files "model/config.json,model/vocab.txt" \\
  --requirements-file requirements.txt \\
  --export-path model_store/

# 3.2 启动 TorchServe
torchserve --start \\
  --model-store model_store/ \\
  --models text_classifier=text_classifier.mar \\
  --ncs  # 无模型缓存

# 3.3 注册模型（带版本）
curl -X POST "localhost:8081/models?url=text_classifier_v2.mar&model_name=text_classifier&version=2.0"

# 3.4 设置默认版本
curl -X PUT "localhost:8081/models/text_classifier/version/2.0/set-default"

# 3.5 推理请求
curl -X POST "localhost:8080/predictions/text_classifier" \\
  -H "Content-Type: application/json" \\
  -d '{"text": "This movie was fantastic!"}'
"""

# ========== 4. Docker 部署 ==========
"""
FROM pytorch/torchserve:latest

COPY model_store/ /home/model-server/model-store/
COPY config.properties /home/model-server/config.properties

# config.properties
# inference_address=http://0.0.0.0:8080
# management_address=http://0.0.0.0:8081
# metrics_address=http://0.0.0.0:8082
# model_store=/home/model-server/model-store
# default_workers_per_model=2
# default_response_timeout=120

CMD ["torchserve", "--start", "--ts-config", "/home/model-server/config.properties", "--models", "all"]
"""

# ========== 5. 批处理配置 ==========
"""
# 在 config.properties 中设置：
batch_size=4
max_batch_delay=100    # 最多等待 100ms
default_workers_per_model=4

# 或在注册模型时指定：
# curl -X POST "localhost:8081/models?url=model.mar&batch_size=4&max_batch_delay=100"
"""

# ========== 6. 使用 TorchServe 的 Python SDK ==========
def torchserve_client_example():
    import requests
    import json

    # 推理请求
    response = requests.post(
        "http://localhost:8080/predictions/text_classifier",
        json={"text": "I love this product!"},
        headers={"Content-Type": "application/json"}
    )
    result = response.json()
    print(f"Prediction: {result}")

    # 管理 API: 获取模型状态
    status = requests.get("http://localhost:8081/models/text_classifier")
    print(f"Model status: {status.json()}")

    # 管理 API: 设置 worker 数
    requests.put(
        "http://localhost:8081/models/text_classifier",
        params={"min_worker": 2, "max_worker": 8}
    )
```

## 深度学习关联

- **PyTorch 优先的 MLOps 部署方案**：对于使用 PyTorch 训练的团队，TorchServe 是最自然的部署选择。它与 PyTorch 生态的深度集成（TorchScript、TorchVision、TorchText 等）使得从训练到部署的转换无缝衔接。在 MLflow 中注册模型后，可以通过 `torchserve` 插件一键部署。
- **灰度发布与 A/B 测试**：TorchServe 的模型版本管理支持精细的流量控制。可以将新模型的某个版本指定给特定客户端（通过自定义 `version` 路由），实现 A/B 测试。典型实践是：新模型 v2.0 先部署到 staging 环境验证，然后将 5% 流量切到 v2.0 进行线上 A/B 测试，确认指标后逐步放量。
- **可观测性与自动扩缩**：TorchServe 内置了与 Prometheus 和 Grafana 的集成，暴露每个模型的请求延迟、吞吐量、worker 状态等指标。结合 Kubernetes HPA (Horizontal Pod Autoscaler) 和 TorchServe 的自动 worker 伸缩，可以实现 GPU 推理服务的完整弹性伸缩方案。
