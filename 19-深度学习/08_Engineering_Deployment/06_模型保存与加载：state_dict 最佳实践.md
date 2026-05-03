# 06_模型保存与加载：state_dict 最佳实践

## 核心概念

- **state_dict**：PyTorch 中模型的"内存快照"，是一个 Python 字典，将每层的可学习参数（权重、偏置）和缓冲区（BN 的 running_mean/std）的名称映射到对应的张量。它是模型持久化的标准格式。
- **torch.save 与 torch.load**：`torch.save(obj, f)` 使用 Python 的 pickle 序列化任意对象到磁盘；`torch.load(f, map_location)` 反序列化加载。建议仅保存 state_dict 而非完整模型对象，以避免 pickle 的兼容性问题和安全风险。
- **strict 加载**：`model.load_state_dict(state_dict, strict=True)` 要求 state_dict 的键与模型完全匹配。当进行迁移学习或加载部分权重时，需设置 `strict=False` 并手动检查缺失/多余的键。
- **检查点 (Checkpoint)**：训练过程中除了模型权重外，还需保存优化器状态、当前 epoch、学习率调度器状态等，以便从中断处恢复训练。通常打包为字典：`{"epoch": epoch, "model": model.state_dict(), "optimizer": opt.state_dict(), "scheduler": sch.state_dict()}`。
- **分布式保存**：在 DDP/FSDP 中，通常只在 rank 0 上保存，或使用 `torch.distributed.barrier()` 确保保存完成后再加载。FSDP 还需 `summon_full_params()` 上下文才能收集完整权重。
- **模型权重转换 (Weight Conversion)**：在不同框架间（PyTorch <-> HuggingFace <-> JAX）迁移权重时，需要根据命名规则进行键名映射和维度转置。这是模型部署中的常见痛点。

## 数学推导

state_dict 的存储本质上是一个参数空间的离散化映射。对于一个具有 $L$ 层的网络，参数空间为：

$$
\Theta = \{\theta_1, \theta_2, ..., \theta_L\}
$$

state_dict 定义了一个从字符串键到张量值的双射：

$$
\text{state\_dict}: \{\text{key}_l\} \to \{\theta_l\}, \quad l=1,...,L
$$

加载过程即是从磁盘恢复该映射并赋值到模型参数：

$$
\theta_l \leftarrow \text{state\_dict}[\text{key}_l], \quad \forall l
$$

当存在预训练权重 $\Theta_{\text{pretrain}}$ 和当前模型参数 $\Theta_{\text{model}}$ 时，迁移学习对应着部分赋值操作：

$$
\theta_l^{\text{(model)}} \leftarrow \begin{cases}
\theta_l^{\text{(pretrain)}} & \text{if } \text{key}_l^{\text{(model)}} = \text{key}_m^{\text{(pretrain)}} \text{ and } \text{shape matches} \\
\theta_l^{\text{(model)}} & \text{(random init) otherwise}
\end{cases}
$$

## 直观理解

- **state_dict = 模型的 DNA**：它完整记录了模型每个组件的基因信息（权重值），不包含生命活动（forward 逻辑）。你可以把这个 DNA 移植到不同的"身体"（模型架构）上，只要基因序列匹配。
- **最佳实践**：始终保存和加载 `model.state_dict()` 而非整个 `model` 对象。前者是纯张量字典，跨 Python/PyTorch 版本兼容性好；后者绑定了具体的类定义，版本变化时极易出错。
- **常见陷阱**：在 DDP 包装后，模型参数名会加上 `module.` 前缀。保存时使用 `model.module.state_dict()` 或加载时通过 `strict=False` 并手动 strip 前缀来处理。
- **经验法则**：每个实验的 checkpoint 目录结构建议为 `checkpoints/{run_id}/epoch_{epoch:04d}_step_{step:07d}.pt`，并配合 MLflow 等实验管理工具记录最佳模型路径。

## 代码示例

```python
import torch
import torch.nn as nn
import os

# 1. 基础保存与加载
model = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

# 保存 state_dict
torch.save(model.state_dict(), "model_weights.pt")
print(f"state_dict keys: {list(model.state_dict().keys())}")

# 加载 state_dict
model2 = nn.Sequential(
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)
model2.load_state_dict(torch.load("model_weights.pt", map_location="cpu"))

# 2. 完整的训练检查点
checkpoint = {
    "epoch": 10,
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": torch.optim.Adam(model.parameters()).state_dict(),
    "loss": 0.035,
    "best_metric": 0.942,
}
torch.save(checkpoint, "checkpoint_epoch10.pt")

# 恢复训练
ckpt = torch.load("checkpoint_epoch10.pt", map_location="cpu")
model.load_state_dict(ckpt["model_state_dict"])
optimizer = torch.optim.Adam(model.parameters())
optimizer.load_state_dict(ckpt["optimizer_state_dict"])
start_epoch = ckpt["epoch"] + 1
print(f"从 epoch {start_epoch} 恢复训练")

# 3. 迁移学习：加载部分权重
class PretrainedModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Linear(784, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
        )
        self.classifier = nn.Linear(128, 10)

pretrained = PretrainedModel()

# 假设只有 features 部分有预训练权重
pretrained_state = torch.load("pretrained_features.pt", map_location="cpu")
# 加载时跳过不匹配的键
missing, unexpected = model.load_state_dict(pretrained_state, strict=False)
print(f"Missing keys: {missing}")      # 当前模型有但 state_dict 没有的
print(f"Unexpected keys: {unexpected}")  # state_dict 有但当前模型没有的

# 4. 处理 DDP 的 module. 前缀
if torch.cuda.device_count() > 1:
    model_ddp = nn.DataParallel(model)
    # 方案 A: 保存时剥离前缀
    torch.save(model_ddp.module.state_dict(), "ddp_weights.pt")
    # 方案 B: 加载时创建新字典
    raw_state = model_ddp.state_dict()  # 包含 "module." 前缀
    clean_state = {k.replace("module.", ""): v for k, v in raw_state.items()}
    model.load_state_dict(clean_state)

# 5. Safetensors 格式（推荐用于生产环境）
# pip install safetensors
# from safetensors.torch import save_file, load_file
# save_file(model.state_dict(), "model.safetensors")
# model.load_state_dict(load_file("model.safetensors"))

# 6. 最佳实践：自动保存最佳模型
best_metric = 0.0
for epoch in range(100):
    # ... 训练代码 ...
    val_metric = 0.95  # 假设值
    if val_metric > best_metric:
        best_metric = val_metric
        torch.save(model.state_dict(), "best_model.pt")
        print(f"保存最佳模型 (epoch {epoch}, metric {val_metric:.4f})")

# 清理生成的临时文件
for f in ["model_weights.pt", "checkpoint_epoch10.pt", "best_model.pt"]:
    if os.path.exists(f):
        os.remove(f)
```

## 深度学习关联

- **模型注册表 (Model Registry)**：在 MLOps 平台（MLflow Model Registry、SageMaker Model Registry）中，每个模型版本除了存储 state_dict 外，还需记录 Python 环境依赖（`conda.yaml`）、模型签名（输入/输出 schema）和性能指标。保存 model.state_dict() 后，配合 `torch.jit.script` 或 ONNX 导出为可部署格式是标准流程。
- **增量训练与模型回滚**：生产系统中的模型需要支持增量更新（基于新数据微调后发布新版本）。通过版本化的 state_dict 保存和回滚机制（如 S3 版本控制），可以在新模型指标下降时快速恢复到上一版本。
- **联邦学习中的安全聚合**：在联邦学习场景中，每个客户端的 state_dict 梯度更新通过安全聚合（Secure Aggregation）协议加密后发送到中心服务器。服务器对加密后的 state_dict 差分进行加权平均，再将聚合后的全局 state_dict 分发到各客户端。
