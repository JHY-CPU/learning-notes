import os

titles = [
    "01_PyTorch 张量操作与广播机制 (Broadcasting)",
    "02_Dataset 与 DataLoader 高效数据加载",
    "03_自定义 Dataset 类实现与增强策略",
    "04_nn.Module 模块化设计与参数管理",
    "05_动态计算图与 Autograd 引擎底层逻辑",
    "06_模型保存与加载：state_dict 最佳实践",
    "07_分布式训练基础：DataParallel vs DistributedDataParallel",
    "08_FSDP (Fully Sharded Data Parallel) 原理",
    "09_Deepspeed 集成与 ZeRO 优化技术",
    "10_混合精度训练 (AMP) 的 PyTorch 实现",
    "11_模型压缩：剪枝 (Pruning) 技术与稀疏性",
    "12_知识蒸馏 (Distillation) 的工程化流程",
    "13_量化技术：PTQ (训练后量化) 基础",
    "14_QAT (量化感知训练) 伪量化节点插入",
    "15_ONNX 格式转换与算子兼容性处理",
    "16_TensorRT 加速：层融合 (Layer Fusion) 优化",
    "17_Triton Inference Server 服务化部署",
    "18_TorchServe 模型版本管理与批处理",
    "19_移动端部署：PyTorch Mobile 与 CoreML",
    "20_模型监控与漂移检测 (Drift Detection)"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
