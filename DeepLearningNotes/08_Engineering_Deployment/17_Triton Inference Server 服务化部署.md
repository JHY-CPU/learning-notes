# 17_Triton Inference Server 服务化部署

## 核心概念

- **Triton Inference Server**：NVIDIA 出品的高性能推理服务器，支持多种模型框架（TensorRT、ONNX、PyTorch、TensorFlow、自定义 Python Backend）和多种硬件（GPU、CPU、ARM）。提供 HTTP/gRPC 接口和原生 C API，是生产级推理部署的标准方案。
- **模型仓库 (Model Repository)**：Triton 的模型存储结构。每个模型在仓库中有独立目录，包含 `config.pbtxt`（模型配置）和按版本命名的子目录（如 `1/model.plan`）。Triton 自动管理模型的版本发现和加载。
- **并发模型执行 (Concurrent Model Execution)**：Triton 可以在一张 GPU 上同时执行多个模型实例（多个模型副本），通过 `instance_group` 配置控制。对于小模型，多实例并行可以显著提高 GPU 利用率和吞吐量。
- **动态批处理 (Dynamic Batching)**：Triton 在服务端自动将多个请求合并为一个批次 (batch) 执行，然后拆分为独立响应返回。通过 `max_batch_size` 和 `preferred_batch_size` 控制，在延迟和吞吐量之间取得平衡。
- **BLS (Business Logic Scripting)**：Triton 的 Python Backend 支持在推理流水线中执行自定义业务逻辑，包括数据预处理/后处理、多模型组合推理、条件分支等。BLS 使得复杂的推理 DAG（有向无环图）可以在 Triton 内完成。
- **模型集成 (Model Ensemble)**：Triton 支持将多个模型组合成流水线 (ensemble)，前一个模型的输出自动作为后一个模型的输入。Ensemble 配置在 `config.pbtxt` 中声明，由 Triton 调度器自动执行。

## 数学推导

Triton 的动态批处理可以建模为一个排队论问题。假设推理请求到达率为 $\lambda$，每个请求的处理时间为 $S$（平均），服务率为 $\mu = 1/S$。

**无批处理 (单请求处理)**：
$$
\text{吞吐量} = \mu, \quad \text{平均延迟} = \frac{1}{\mu - \lambda} \quad (\text{M/M/1 队列})
$$

**有批处理 (最大 batch size = B，最大等待时间 = T_w)**：
$$
\text{吞吐量} = B \cdot \mu \cdot P(\text{batch size} = B) + \cdots
$$

批处理带来的吞吐量提升源于摊销了 GPU kernel launch 开销：

$$
\text{Amdahl 定律加速：} \quad \text{Speedup} = \frac{T_{\text{serial}} + T_{\text{parallel}}}{T_{\text{serial}} + T_{\text{parallel}} / B}
$$

其中 $B$ 是 batch size，$T_{\text{serial}}$ 是串行开销（预处理、后处理、I/O），$T_{\text{parallel}}$ 是可并行计算的时间。对于 Transformer 模型，$T_{\text{serial}}$ 占比小，批处理可以获得接近 $B$ 倍的吞吐量提升。

## 直观理解

- **Triton = 智能餐厅的厨房**：客人（客户端）通过服务员（HTTP/gRPC 接口）下单。厨房有一个大厨（GPU），但 Triton 会智能地等一会儿再开始做菜——如果 2 毫秒内有其他客人点了相似的菜（同模型请求），就合并在一起做（动态批处理），一锅出比一锅一锅做效率高得多。
- **模型集成 = 流水线作业**：一个复杂的推理可能需要经过"预处理模型 -> 主模型 -> 后处理模型"。Triton 的 ensemble 就像工厂流水线——每个工位完成自己的工作后自动传给下一个工位，不需要人工搬运。
- **最佳实践**：对于延迟敏感的应用，禁用动态批处理或设置极短的 `max_queue_delay`（如 100μs）；对于吞吐量优先的业务（如离线批量推理），使用大 batch 和较大延迟。
- **常见陷阱**：`instance_group` 的数量不是越多越好——过多的模型副本会争抢 GPU 显存和计算资源，反而降低吞吐量。对于大模型，设置为 1；对于小模型（如 ResNet-50），设置为 2-4。

## 代码示例

```python
# ========== 1. Triton 模型仓库结构 ==========
"""
triton_model_repo/
├── resnet50/
│   ├── config.pbtxt          # 模型配置
│   ├── 1/
│   │   └── model.plan        # TensorRT 引擎 / ONNX 模型
│   └── 2/
│       └── model.plan        # 版本 2 (可选)
├── bert_ensemble/
│   ├── config.pbtxt          # Ensemble 配置
│   └── 1/
│       └── model.plan
└── preprocess/
    ├── config.pbtxt
    └── 1/
        └── model.py          # Python Backend 脚本
"""

# ========== 2. config.pbtxt 配置示例 ==========
# 文件: triton_model_repo/resnet50/config.pbtxt
"""
name: "resnet50"
platform: "tensorrt_plan"      # 或 onnxruntime_onnx, pytorch_libtorch
max_batch_size: 64             # 最大 batch size
input [
  {
    name: "input"
    data_type: TYPE_FP32
    dims: [3, 224, 224]
  }
]
output [
  {
    name: "output"
    data_type: TYPE_FP32
    dims: [1000]
  }
]
dynamic_batching {
  preferred_batch_size: [8, 16, 32]  # 优选的 batch size
  max_queue_delay_microseconds: 100   # 等待 100μs 凑 batch
}
instance_group [
  {
    count: 2                    # 2 个模型副本
    kind: KIND_GPU
    gpus: [0]
  }
]
"""

# ========== 3. Python Backend ==========
# 文件: triton_model_repo/preprocess/1/model.py
import triton_python_backend_utils as pb_utils
import numpy as np
import torch
from torchvision import transforms

class TritonPythonModel:
    def initialize(self, args):
        """模型初始化 (启动时调用一次)"""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    def execute(self, requests):
        """处理推理请求 (每个请求调用一次)"""
        responses = []
        for request in requests:
            # 获取输入
            input_tensor = pb_utils.get_input_tensor_by_name(request, "raw_image")
            input_data = input_tensor.as_numpy()  # shape: [B, H, W, C], dtype=uint8

            # 预处理
            batch = []
            for i in range(input_data.shape[0]):
                img = input_data[i]
                # numpy -> tensor -> normalize
                tensor = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
                tensor = self.transform(tensor)
                batch.append(tensor)
            batch = torch.stack(batch).cpu().numpy()

            # 创建输出张量
            output_tensor = pb_utils.Tensor("preprocessed_input", batch)
            responses.append(pb_utils.InferenceResponse(output_tensors=[output_tensor]))

        return responses

    def finalize(self):
        """清理 (服务关闭时调用)"""
        pass

# ========== 4. 客户端请求（Python）===========
import tritonclient.http as httpclient
import tritonclient.grpc as grpcclient

def triton_client_example():
    client = httpclient.InferenceServerClient(url="localhost:8000")

    # 服务健康检查
    assert client.is_server_live()
    assert client.is_model_ready("resnet50")

    # 准备输入
    image = np.random.randn(1, 3, 224, 224).astype(np.float32)
    input_tensor = httpclient.InferInput("input", image.shape, "FP32")
    input_tensor.set_data_from_numpy(image)

    # 推理
    result = client.infer(model_name="resnet50", inputs=[input_tensor])
    output = result.as_numpy("output")
    print(f"Output shape: {output.shape}, Top-1: {output.argmax()}")

# ========== 5. 启动 Triton Server ==========
"""
Docker 启动命令:
docker run --gpus=all -p 8000:8000 -p 8001:8001 -p 8002:8002 \\
  -v /path/to/model_repo:/models \\
  nvcr.io/nvidia/tritonserver:23.12-py3 \\
  tritonserver --model-repository=/models

# 使用 Prometheus 监控
tritonserver --model-repository=/models --metrics-port=8002

# 查看模型状态
curl localhost:8000/v2/models/resnet50
"""

# ========== 6. 性能分析工具 ==========
def perf_analyzer_example():
    """
    命令行:
    perf_analyzer -m resnet50 -u localhost:8001 \\
      --concurrency-range 1:4:1 \\
      --shape input:3,224,224 \\
      --measurement-interval 10000

    # 输出:
    # Request concurrency: 1
    #   Throughput: 312 infer/sec
    #   Latency: 2.41 msec (avg)
    """
    pass
```

## 深度学习关联

- **生产级 MLOps 部署标准**：Triton Inference Server 是 MLOps 部署阶段的标准组件。典型的部署架构为：模型仓库 (S3/NFS) -> Triton Server (Kubernetes Pod) -> 负载均衡 (Istio/NGINX) -> 客户端。每次模型更新通过 GitOps 流水线自动更新 S3 中的模型文件，Triton 通过 model control API 动态加载新版本。
- **多模型编排与资源复用**：在真实业务中，通常需要部署多个模型（如物体检测 + 文字识别 + 排序模型）在同一组 GPU 上。Triton 的 `instance_group` 和 `rate_limiter` 配置允许精细控制每个模型的 GPU 资源分配，避免资源竞争。这在 MLflow 部署到生产环境时通常需要手动调整。
- **可观测性 (Observability)**：Triton 内置了 Prometheus 指标导出（请求延迟、吞吐量、GPU 利用率、排队长度等），与 Grafana 集成可实现完整的推理服务监控面板。结合自定义的模型漂移检测器，可以在指标异常时自动触发模型回滚或重新训练。
