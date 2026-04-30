# 19_移动端部署：PyTorch Mobile 与 CoreML

## 核心概念

- **PyTorch Mobile**：PyTorch 的移动端部署方案，支持在 Android 和 iOS 设备上直接运行 PyTorch 模型。核心组件包括：`torchscript`（模型序列化格式）、`libtorch`（移动端 C++ 运行时库）和 `pytorch_android_lite`（Android Java API）。
- **TorchScript**：PyTorch 模型的静态图表示，是 PyTorch Mobile 的模型格式。通过 `torch.jit.trace` 或 `torch.jit.script` 将动态 PyTorch 模型转换为静态 TorchScript 图，移除了 Python 依赖，可在没有 Python 运行时的环境中执行。
- **模型量化与优化**：移动端部署前必须进行模型压缩。典型流程包括：FP32 -> FP16 量化（iOS 上的 CoreML）或 FP32 -> INT8 量化（Android 上的 NNAPI）。模型大小的目标通常控制在 50MB 以下。
- **CoreML**：Apple 的原生机器学习框架，支持在 iOS 设备上高效执行模型推理。CoreML 利用 Apple Neural Engine (ANE) 和 GPU 实现硬件加速。PyTorch 模型可以通过 `torch.onnx.export` -> ONNX -> CoreML 工具链转换，或使用 `coremltools` 直接转换。
- **模型大小优化技术**：移动端部署的核心挑战是存储和内存限制。关键优化包括：量化（INT8/FP16）、剪枝、知识蒸馏、权重共享和采用 MobileNet/EfficientNet-Lite 等轻量级架构。
- **NNAPI (Android Neural Networks API)**：Android 平台的硬件加速推理 API，支持在支持 NN 加速的 SoC（如 Qualcomm Hexagon、MediaTek APU）上运行量化模型。

## 数学推导

移动端模型部署的延迟预算分析。一个典型的移动端推理任务需要在以下约束内完成：

- 帧处理类任务（实时视频/相机）：延迟 < 33ms (30 FPS)
- 交互式任务（文本分类/OCR）：延迟 < 100ms
- 后台任务（图像标签/推荐）：延迟 < 500ms

在移动设备 GPU/ANE 上，推理延迟可以建模为：

$$
T_{\text{total}} = T_{\text{load}} + T_{\text{preprocess}} + \sum_{l=1}^L T_l(B) + T_{\text{postprocess}}
$$

其中 $T_l(B)$ 是第 $l$ 层在 batch size $B=1$ 时的推理时间。在移动端，$B=1$ 是常态。

**CoreML 的 ANE 加速效果**：

$$
\text{Speedup}_{\text{ANE}} = \frac{T_{\text{CPU}}}{T_{\text{ANE}}}
$$

对于卷积层，ANE 加速通常在 5-20 倍之间。但 ANE 对非卷积算子（如 LayerNorm、Softmax）的加速有限，这些操作会成为瓶颈。

**模型大小与精度权衡**：假设原始 FP32 模型大小为 $M_{\text{FP32}}$，INT8 量化后：

$$
M_{\text{INT8}} = \frac{M_{\text{FP32}}}{4}, \quad \text{精度损失} \approx 0.5\text{-}1\%
$$

知识蒸馏可使学生模型大小缩小 60-90% 而保持 95% 以上的相对精度。

## 直观理解

- **移动端部署 = 把大象装进冰箱**：训练好的模型（大象）需要经过层层"瘦身"才放得进手机（冰箱）。TorchScript 删除了 Python 解释器这个"大冰箱"，量化把 FP32 数据（整箱牛奶）压缩为 INT8（浓缩奶），剪枝扔掉不需要的"脂肪"。
- **CoreML vs PyTorch Mobile**：CoreML 是 iPhone 的"原生接口"——像用 iPhone 相机拍照一样，又快又稳定但只能在 Apple 设备上用；PyTorch Mobile 是"万能转接头"——兼容性好但性能不如原生。对于 iOS 设备，优先使用 CoreML；对于跨平台应用（Android + iOS），使用 PyTorch Mobile。
- **最佳实践**：对于 iOS-only 应用，使用 `coremltools` 直接转换；对于跨平台应用，使用 PyTorch Mobile + ONNX Runtime Mobile；始终在真机而非模拟器上测试推理速度（模拟器没有 GPU/ANE 加速）。
- **常见陷阱**：手机 GPU 的显存带宽远小于服务器 GPU，因此重计算量/少参数的模型（如 Depthwise Conv）比轻计算量/多参数的模型（如大矩阵乘）更适合移动端。

## 代码示例

```python
import torch
import torch.nn as nn
import torchvision.models as models

# ========== 1. PyTorch -> TorchScript（移动端部署第一步）===========
class MobileModel(nn.Module):
    def __init__(self, num_classes=1000):
        super().__init__()
        # 使用轻量级架构
        self.backbone = models.mobilenet_v3_small(pretrained=True)
        self.backbone.classifier[3] = nn.Linear(1024, num_classes)

    def forward(self, x):
        return self.backbone(x)

model = MobileModel().eval()
dummy_input = torch.randn(1, 3, 224, 224)

# 方法 1: Trace (推荐，不支持控制流)
traced_model = torch.jit.trace(model, dummy_input)
traced_model.save("mobilenet_v3.pt")

# 方法 2: Script (支持控制流，但限制较多)
scripted_model = torch.jit.script(model)
scripted_model.save("mobilenet_v3_scripted.pt")

# 优化：删除冗余操作
optimized = torch.jit.optimize_for_inference(traced_model)
optimized.save("mobilenet_v3_optimized.pt")

# 量化：移动端必须量化（INT8）
quantized_model = torch.quantization.quantize_dynamic(
    model, {nn.Linear}, dtype=torch.qint8
)
scripted_quant = torch.jit.script(quantized_model)
scripted_quant.save("mobilenet_v3_quantized.pt")

import os
for f in ["mobilenet_v3.pt", "mobilenet_v3_scripted.pt",
          "mobilenet_v3_optimized.pt", "mobilenet_v3_quantized.pt"]:
    if os.path.exists(f):
        size = os.path.getsize(f) / 1024**2
        print(f"{f}: {size:.2f} MB")

# ========== 2. PyTorch -> CoreML (iOS) ==========
def export_to_coreml():
    """使用 coremltools 将 PyTorch 模型转换为 CoreML"""
    try:
        import coremltools as ct
    except ImportError:
        print("需要安装 coremltools: pip install coremltools")
        return

    model = models.mobilenet_v3_small(pretrained=True).eval()

    # 方法 1: 通过 TorchScript -> CoreML
    traced = torch.jit.trace(model, torch.randn(1, 3, 224, 224))

    # 转换为 CoreML
    mlmodel = ct.convert(
        traced,
        inputs=[ct.TensorType(name="input", shape=(1, 3, 224, 224))],
        outputs=[ct.TensorType(name="output")],
        convert_to="mlprogram",  # 使用 CoreML 程序格式（支持 ANE）
        minimum_deployment_target=ct.target.iOS16,
        compute_precision=ct.precision.FLOAT16,  # FP16 精度
    )

    # 添加元数据
    mlmodel.author = "MLOps Team"
    mlmodel.short_description = "MobileNetV3 Small Classifier"
    mlmodel.version = "1.0"

    mlmodel.save("MobileNetV3.mlpackage")
    print("CoreML 模型导出成功")

    # 模型大小对比
    mlmodel_size = os.path.getsize("MobileNetV3.mlpackage") / 1024**2
    print(f"CoreML 模型大小: {mlmodel_size:.2f} MB")

# ========== 3. Android 部署（PyTorch Mobile Java API）===========
"""
// build.gradle 依赖
dependencies {
    implementation 'org.pytorch:pytorch_android_lite:2.1.0'
    implementation 'org.pytorch:pytorch_android_torchvision_lite:2.1.0'
}

// Java/Kotlin 代码
import org.pytorch.Module;
import org.pytorch.Tensor;
import org.pytorch.torchvision.TensorImageUtils;

public class Classifier {
    private Module model;

    public Classifier(String modelPath) {
        model = Module.load(modelPath);
    }

    public int classify(Bitmap bitmap) {
        // 预处理图像
        Tensor input = TensorImageUtils.bitmapToFloat32Tensor(
            bitmap,
            TensorImageUtils.TORCHVISION_NORM_MEAN,
            TensorImageUtils.TORCHVISION_NORM_STD
        );

        // 推理
        Tensor output = model.forward(IValue.from(input)).toTensor();

        // 后处理
        float[] scores = output.getDataAsFloatArray();
        return argmax(scores);
    }
}
"""

# ========== 4. iOS 部署（CoreML Swift API）===========
"""
// Xcode 中直接导入 .mlpackage 文件

import CoreML
import Vision

// 创建模型
guard let model = try? VNCoreMLModel(for: MobileNetV3().model) else { return }

// 推理请求
let request = VNCoreMLRequest(model: model) { request, error in
    guard let results = request.results as? [VNClassificationObservation],
          let topResult = results.first else { return }
    DispatchQueue.main.async {
        print("预测: \(topResult.identifier), 置信度: \(topResult.confidence)")
    }
}

// 执行推理
let handler = VNImageRequestHandler(cgImage: image)
try? handler.perform([request])
"""

# ========== 5. 移动端推理基准测试 ==========
def mobile_benchmark():
    """模拟移动端推理的性能评估"""
    import time

    model = models.mobilenet_v3_small(pretrained=True).eval()

    # 量化
    quantized = torch.quantization.quantize_dynamic(
        model, {nn.Linear}, dtype=torch.qint8
    )

    # 推理速度测试
    dummy = torch.randn(1, 3, 224, 224)
    models_to_test = {"FP32": model, "INT8 Dynamic": quantized}

    for name, m in models_to_test.items():
        with torch.no_grad():
            t0 = time.time()
            for _ in range(50):
                m(dummy)
            avg = (time.time() - t0) / 50
        size = sum(p.numel() * p.element_size() for p in m.parameters()) / 1024**2
        print(f"{name}: {avg*1000:.2f} ms, {size:.2f} MB")
```

## 深度学习关联

- **CI/CD 中的移动端部署流水线**：移动端模型的部署需要一个完整的 CI/CD 流水线：模型训练 -> 剪枝/蒸馏 -> TorchScript 导出 -> 量化 -> 平台特定转换（CoreML/PyTorch Mobile/NNAPI） -> 集成到移动 App -> A/B 测试。每次模型更新使用 MLflow 记录转换参数和基准测试结果，确保模型质量。
- **联邦学习与设备端学习**：移动端设备不仅是推理执行端，也可以参与模型训练。联邦学习 (Federated Learning) 技术在手机上使用本地数据微调模型，然后将加密的梯度更新回传服务器。PyTorch Mobile 支持这种"设备端训练"模式，但需要额外的内存和电池优化。
- **移动端 MLOps 的挑战**：与云端不同，移动端部署面临碎片化（数千种 Android 设备型号、不同的 SoC 和 NPU）、版本管控（App Store 审核周期长、用户更新延迟大）、冷启动优化（模型加载时间）等独特挑战。监控方案通常使用 Firebase Analytics 或自建 SDK 收集客户端的推理延迟和精度数据。
