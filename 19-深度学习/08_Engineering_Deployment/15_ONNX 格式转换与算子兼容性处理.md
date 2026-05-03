# 15_ONNX 格式转换与算子兼容性处理

## 核心概念

- **ONNX (Open Neural Network Exchange)**：由 Microsoft 和 Facebook 联合推出的开放神经网络交换格式，旨在实现不同深度学习框架之间的模型互操作性。ONNX 定义了一套标准化的算子集合和计算图表示格式，使模型可以在 PyTorch、TensorFlow、ONNX Runtime、TensorRT 等不同平台间自由迁移。
- **torch.onnx.export**：PyTorch 的 ONNX 导出入口，通过 tracing（跟踪执行路径）或 scripting（编译代码）将 PyTorch 模型转换为 ONNX 格式。核心参数包括 `input_names`、`output_names`、`dynamic_axes`（动态轴）和 `opset_version`（算子集版本）。
- **算子兼容性 (Op Compatibility)**：不同 ONNX opset 版本支持的算子集合不同，高版本支持更多现代算子（如 `LayerNorm`, `GELU`）。当 PyTorch 的算子无法直接映射到 ONNX 算子时，会自动回退到复合算子或自定义算子，导致导出失败或效率低下。
- **动态轴 (Dynamic Axes)**：指定 ONNX 模型中哪些维度在推理时可变（如 batch size、序列长度）。对于生产级部署至关重要，因为推理请求的 batch size 通常是动态的。
- **ONNX Runtime**：Microsoft 开发的高性能 ONNX 推理引擎，支持跨平台执行（CPU/GPU/ARM）和多种优化技术（图优化、算子融合、内存复用）。是 ONNX 模型部署的推荐执行引擎。
- **模型验证与精度对齐**：ONNX 导出后需验证 PyTorch 和 ONNX Runtime 的推理输出是否一致（通常容忍 1e-3 ~ 1e-5 的数值误差）。不一致通常由算子实现细节差异或动态控制流未正确 trace 导致。

## 数学推导

ONNX 计算图是一个静态的有向无环图 (DAG)，其中节点是算子，边是张量。对于一个 $L$ 层网络，ONNX 的序列化表示可以写为：

$$
\text{Graph} = (\mathcal{V}, \mathcal{E}, \mathcal{A})
$$

其中 $\mathcal{V}$ 是算子节点集，$\mathcal{E}$ 是张量边集，$\mathcal{A}$ 是属性集（如卷积的 kernel_size、padding）。

算子兼容性取决于 opset 版本对应的算子规范。例如，`LayerNormalization` 算子在不同 opset 中的支持情况：

- opset 1-16: 不支持 LayerNorm（需用数学算子组合）
- opset 17+: 原生支持 LayerNormalization 算子

算子映射的数学本质是寻找一个函数 $\Phi$，将 PyTorch 的算子语义映射到 ONNX 算子：

$$
\Phi: \mathcal{O}_{\text{PyTorch}} \to \mathcal{O}_{\text{ONNX}}
$$

当 $\mathcal{O}_{\text{PyTorch}}$ 没有直接对应的 ONNX 算子时，需要分解为多个 ONNX 算子的组合：

$$
\Phi(o) = o_1 \circ o_2 \circ \cdots \circ o_k \quad \text{where } o_i \in \mathcal{O}_{\text{ONNX}}
$$

例如 PyTorch 的 `F.scaled_dot_product_attention` 需要分解为矩阵乘、softmax、scale 等多个 ONNX 算子。

## 直观理解

- **ONNX = 通用翻译器**：PyTorch 说的是"法语"，TensorFlow 说的是"德语"，ONNX 相当于"世界语"——一个中间语言。你把法语文章（PyTorch 模型）翻译成世界语，说德语的人（TensorFlow Runtime）就可以读懂。不需要大家都学对方的语言。
- **算子兼容性 = 翻译字典的完整程度**：ONNX 的 opset 版本相当于翻译字典的版本。新版翻译词典包含了更多词汇的翻译（更多算子），老版本的词典可能需要用多个简单词组合来表达一个复杂词（算子分解）。
- **最佳实践**：导出时选择最高的 opset 版本（取决于目标推理引擎的支持范围）；使用 `dynamic_axes` 支持动态 batch size；导出后务必进行精度对齐验证。
- **常见陷阱**：`if` 语句和 `for` 循环的 trace 问题——PyTorch 的 `torch.jit.trace` 只会跟随一条执行路径，因此 trace 得到的 ONNX 模型不会包含条件分支；如果 forward 中有依赖输入的控制流，必须使用 `torch.jit.script` 而非 `trace`。

## 代码示例

```python
import torch
import torch.nn as nn
import onnx
import onnxruntime as ort
import numpy as np

# ========== 1. 基础 ONNX 导出 ==========
class SimpleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 64, 3, padding=1)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.relu(self.bn(self.conv(x)))
        x = self.pool(x).view(x.size(0), -1)
        return self.fc(x)

model = SimpleModel().eval()
dummy_input = torch.randn(1, 3, 224, 224)

torch.onnx.export(
    model,
    dummy_input,
    "simple_model.onnx",
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "output": {0: "batch_size"}
    },
    opset_version=18,  # 使用最新的 opset
)

# ========== 2. ONNX 模型验证 ==========
# 2.1 检查模型结构
onnx_model = onnx.load("simple_model.onnx")
onnx.checker.check_model(onnx_model)
print(f"ONNX 模型 Graph 节点数: {len(onnx_model.graph.node)}")

# 2.2 精度对齐验证
ort_session = ort.InferenceSession("simple_model.onnx")
def verify_onnx(model, ort_session, dummy_input):
    # PyTorch 推理
    with torch.no_grad():
        torch_output = model(dummy_input)

    # ONNX Runtime 推理
    ort_inputs = {ort_session.get_inputs()[0].name: dummy_input.numpy()}
    ort_output = ort_session.run(None, ort_inputs)[0]

    # 比较
    diff = np.abs(torch_output.numpy() - ort_output).max()
    print(f"PyTorch vs ONNX Runtime 最大差异: {diff:.6f}")
    assert diff < 1e-4, "精度对齐失败！"
    return True

verify_onnx(model, ort_session, dummy_input)

# ========== 3. 处理特殊算子 ==========
class ModelWithCustomOp(nn.Module):
    """包含 PyTorch 特有操作的模型"""
    def forward(self, x):
        # F.scaled_dot_product_attention 在 opset 中可能需要分解
        q, k, v = x.chunk(3, dim=-1)
        attn = (q @ k.transpose(-2, -1)) / (k.size(-1) ** 0.5)
        attn = attn.softmax(dim=-1)
        return attn @ v

# 3.1 使用自定义导出符号
from torch.onnx.symbolic_registry import register_op

def symbolic_gelu(g, input):
    """注册自定义 GELU 符号"""
    # 使用 ONNX 现有的 Gelu 算子（opset 20+）
    return g.op("Gelu", input)

# 注册自定义符号（如果 PyTorch 不自带）
# register_op("gelu", symbolic_gelu, "", 18)

# 3.2 导出时设置自定义操作
def export_with_custom_ops():
    """当需要在不支持某算子的 ONNX Runtime 中运行时，用基础算子组合"""
    class CustomGELU(nn.Module):
        def forward(self, x):
            return x * 0.5 * (1.0 + torch.erf(x / 1.4142135623730951))

    model = nn.Sequential(nn.Linear(768, 768), CustomGELU())
    dummy = torch.randn(1, 768)

    torch.onnx.export(
        model, dummy, "gelu_model.onnx",
        opset_version=14,  # 使用较低版本的 opset
        do_constant_folding=True,
    )
    print("GELU 模型导出成功（使用 erf 组合实现）")

export_with_custom_ops()

# ========== 4. ONNX Runtime 优化 ==========
def optimize_with_ort():
    session_options = ort.SessionOptions()
    session_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    session_options.optimized_model_filepath = "optimized_model.onnx"

    # 创建优化后的 session
    session = ort.InferenceSession(
        "simple_model.onnx",
        session_options,
        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
    )
    print(f"优化后的 session providers: {session.get_providers()}")

# ========== 5. 批量推理 ==========
def batch_inference():
    session = ort.InferenceSession("simple_model.onnx")
    input_name = session.get_inputs()[0].name

    # 动态 batch：可以传入任意 batch size
    for batch_size in [1, 4, 8, 16]:
        data = np.random.randn(batch_size, 3, 224, 224).astype(np.float32)
        outputs = session.run(None, {input_name: data})
        print(f"Batch {batch_size}: output shape {outputs[0].shape}")

# ========== 6. ONNX 编辑 ==========
def edit_onnx():
    model = onnx.load("simple_model.onnx")
    graph = model.graph

    # 查看所有输入
    for input_tensor in graph.input:
        print(f"输入: {input_tensor.name}, shape: {[d.dim_value for d in input_tensor.type.tensor_type.shape.dim]}")

    # 修改模型元数据
    model.doc_string = "Edited ONNX model for deployment"
    onnx.save(model, "edited_model.onnx")

# 清理
import os
for f in ["simple_model.onnx", "gelu_model.onnx", "edited_model.onnx", "optimized_model.onnx"]:
    if os.path.exists(f):
        os.remove(f)
```

## 深度学习关联

- **CI/CD 中的 ONNX 验证流水线**：在生产 MLOps 流水线中，ONNX 导出和验证应作为模型注册前的自动化步骤。典型流程为：PyTorch 训练 -> 导出 ONNX -> 精度对齐验证 -> 如果失败则自动报警 -> 通过后注册到模型库。这一流水线可以使用 GitHub Actions 或 GitLab CI 配合 pytest 自动化。
- **跨平台部署的核心桥梁**：ONNX 是连接 PyTorch 训练和多种推理引擎的桥梁。在云端，ONNX -> ONNX Runtime/TensorRT；在移动端，ONNX -> CoreML/MediaPipe；在边缘端，ONNX -> ONNX Runtime OpenCL。统一的 ONNX 表示避免了为每个平台重复导出和适配。
- **ONNX 与 Triton Inference Server**：NVIDIA Triton 原生支持 ONNX 模型后端，可以直接加载 ONNX 模型并提供服务。Triton 会自动选择和优化 ONNX Runtime 的 execution provider（TensorRT、CUDA、CPU），实现高效推理。这是实际生产部署中最常见的组合之一。
