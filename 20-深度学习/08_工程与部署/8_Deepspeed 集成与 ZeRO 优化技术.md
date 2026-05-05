# 09_Deepspeed 集成与 ZeRO 优化技术

## 核心概念

- **ZeRO (Zero Redundancy Optimizer)**：Microsoft 提出的分布式训练内存优化技术。核心思想是在分布式训练中消除数据冗余——传统 DDP 中每个 GPU 都存有完整的模型参数、梯度和优化器状态，ZeRO 将它们合理地分布到各 GPU 上。
- **ZeRO-Stage 1 (优化器状态分片)**：将优化器状态（如 Adam 的动量和方差）均分到各 GPU。每个 GPU 只更新其分片对应的参数，通信量减少 4 倍（相比 DDP）。显存节省约 2-4 倍。
- **ZeRO-Stage 2 (梯度分片)**：在 Stage 1 基础上，进一步将梯度均分到各 GPU。每个 GPU 在 backward 完成后只保留其分片的梯度，其余释放。显存节省约 4-6 倍。
- **ZeRO-Stage 3 (参数分片)**：在 Stage 2 基础上，将模型参数也均分到各 GPU。forward 和 backward 时通过 all-gather 实时收集参数，计算完后释放。显存节省约 8 倍（取决于 GPU 数量），与模型规模无关。
- **ZeRO-Offload**：将优化器状态和梯度卸载到 CPU 内存（甚至 NVMe SSD），进一步释放 GPU 显存。适合 GPU 显存有限但 CPU 内存充裕的环境。
- **DeepSpeed Engine**：DeepSpeed 的核心组件，封装了 ZeRO 优化、混合精度训练、梯度裁剪、学习率调度等功能，通过简单的 JSON 配置文件即可启用，对用户代码侵入极小。

## 数学推导

ZeRO-3 的内存分布分析。设模型参数为 $\Phi$，梯度为 $\Phi$，Adam 优化器状态为 $2\Phi$（动量和方差），总冗余存储为 $4\Phi$ 每 GPU。

**DDP 每 GPU 显存占用**：
$$
M_{\text{DDP}} = 4\Phi = \Phi_{\text{param}} + \Phi_{\text{grad}} + 2\Phi_{\text{optim}}
$$

**ZeRO-Stage 3 每 GPU 显存占用（N 个 GPU）**：
$$
M_{\text{ZeRO-3}} = \underbrace{\frac{\Phi}{N}}_{\text{参数分片}} + \underbrace{\frac{\Phi}{N}}_{\text{梯度分片}} + \underbrace{\frac{2\Phi}{N}}_{\text{优化器状态分片}} + \underbrace{\alpha \Phi}_{\text{暂存缓冲}}
$$

DeepSpeed 的通信量分析。以 Stage 3 为例，每个模型单元需要：
- forward: 1 次 all-gather（收集完整参数）
- backward: 1 次 all-gather（再次收集参数用于梯度计算）+ 1 次 reduce-scatter（归约梯度）

每单元通信量：$2 \times \frac{N-1}{N} \cdot \Phi_{\text{unit}}$（all-gather）+ $\frac{N-1}{N} \cdot \Phi_{\text{unit}}$（reduce-scatter）。

总通信量是 DDP 的约 1.5 倍，但通过通信与计算重叠可以基本掩盖额外开销。

## 直观理解

- **ZeRO 的三个阶段 = 仓库管理的不同粒度**：Stage 1 是"每个人共享优化器状态书"（不常用，共享更新计划），Stage 2 是"共享成绩单"（梯度），Stage 3 是"共享教材"（参数）。Stage 3 是最彻底的共享，每个人只拿着自己的几页书，上课（forward）时向同学借其他部分。
- **DeepSpeed 的易用性**：DeepSpeed 的最大优势是低侵入性——只需修改 3-5 行 Python 代码加一个 JSON 配置文件，就能让单卡跑不动的模型在多卡上跑起来。
- **最佳实践**：先尝试 ZeRO-2（通常够用且通信开销小），如果显存不足再升级到 ZeRO-3。对于 >10B 参数的模型，直接使用 ZeRO-3 + Offload。
- **常见陷阱**：ZeRO-3 会显著增加通信量，在 InfiniBand 带宽不足（如仅使用以太网）时可能比 DDP 更慢。使用 `activation_checkpointing` 与 ZeRO-3 叠加可进一步节省显存。

## 代码示例

```python
import torch
import torch.nn as nn
import deepspeed
from deepspeed.ops.adam import DeepSpeedCPUAdam
import argparse
import json

# ========== 1. 定义模型 ==========
class DeepSpeedModel(nn.Module):
    def __init__(self, vocab_size=32000, dim=4096, num_layers=4):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, dim)
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim, dim * 4),
                nn.GELU(),
                nn.Linear(dim * 4, dim),
            ) for _ in range(num_layers)
        ])
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, vocab_size)

    def forward(self, x):
        x = self.embed(x)
        for layer in self.layers:
            x = x + layer(self.norm(x))
        return self.head(x)

# ========== 2. DeepSpeed JSON 配置文件 ==========
ds_config = {
    "train_batch_size": 32,
    "train_micro_batch_size_per_gpu": 4,
    "gradient_accumulation_steps": 8,
    "optimizer": {
        "type": "AdamW",
        "params": {
            "lr": 3e-4,
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01
        }
    },
    "scheduler": {
        "type": "WarmupLR",
        "params": {
            "warmup_min_lr": 0,
            "warmup_max_lr": 3e-4,
            "warmup_num_steps": 1000
        }
    },
    "zero_optimization": {
        "stage": 3,
        "offload_optimizer": {
            "device": "cpu",
            "pin_memory": True
        },
        "offload_param": {
            "device": "cpu",
            "pin_memory": True
        },
        "overlap_comm": True,
        "contiguous_gradients": True,
        "reduce_bucket_size": 5e8,
        "stage3_prefetch_bucket_size": 5e8,
        "stage3_param_persistence_threshold": 1e6,
        "sub_group_size": 1e9,
        "stage3_max_live_parameters": 1e9,
        "stage3_max_reuse_distance": 1e9,
    },
    "fp16": {
        "enabled": True,
        "auto_cast": True,
        "loss_scale": 0,
        "initial_scale_power": 16,
        "loss_scale_window": 1000,
        "hysteresis": 2,
        "min_loss_scale": 1
    },
    "gradient_clipping": 1.0,
    "steps_per_print": 100,
    "wall_clock_breakdown": False,
}

# ========== 3. 使用 DeepSpeed 初始化引擎 ==========
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_rank", type=int, default=0)
    args = parser.parse_args()

    model = DeepSpeedModel()
    # 模拟数据
    dataset = torch.randint(0, 32000, (1000, 512))

    # DeepSpeed 引擎初始化（自动处理 DDP、ZeRO、AMP）
    model_engine, optimizer, _, _ = deepspeed.initialize(
        args=args,
        model=model,
        model_parameters=model.parameters(),
        config_params=ds_config,  # 也可以传 JSON 文件路径
    )

    # 训练循环（与普通 PyTorch 几乎一样）
    for epoch in range(3):
        for step in range(0, len(dataset), model_engine.train_micro_batch_size_per_gpu()):
            batch = dataset[step:step + model_engine.train_micro_batch_size_per_gpu()].cuda()
            outputs = model_engine(batch)
            labels = batch.clone()
            loss = nn.CrossEntropyLoss()(outputs.view(-1, 32000), labels.view(-1))

            model_engine.backward(loss)   # DeepSpeed 封装的 backward
            model_engine.step()            # DeepSpeed 封装的 optimizer step

            if step % 100 == 0:
                # DeepSpeed 自动处理了日志输出中的 loss 缩放
                print(f"Epoch {epoch}, Step {step}, Loss: {loss.item():.4f}")

        # 保存 checkpoint
        model_engine.save_checkpoint(save_dir="./ds_checkpoint", tag=f"epoch_{epoch}")

    # 在 ZeRO-3 下保存完整权重
    if model_engine.global_rank == 0:
        # 需要创建一个临时模型来收集完整权重
        full_model = DeepSpeedModel()
        # engine.load_checkpoint 附带完整权重恢复
        model_engine.save_checkpoint(save_dir="./ds_final", tag="final")

# ========== 4. ZeRO 阶段切换演示 ==========
def zero_stage_demo(stage=2):
    """切换 ZeRO stage 只需修改 JSON 中的 stage 字段"""
    config = ds_config.copy()
    config["zero_optimization"]["stage"] = stage
    # ZeRO-1: stage=1, 只分片优化器状态
    # ZeRO-2: stage=2, 分片优化器 + 梯度
    # ZeRO-3: stage=3, 全分片
    return config

if __name__ == "__main__":
    main()

# 启动命令:
# deepspeed train_deepspeed.py \
#   --deepspeed \
#   --deepspeed_config ds_config.json \
#   --local_rank 0

# 或指定 GPU 数:
# deepspeed --num_gpus=4 train_deepspeed.py
```

## 深度学习关联

- **LLM 训练的行业标准**：DeepSpeed ZeRO 是当前大语言模型训练的行业标准方案，Hugging Face Transformers 的 `Trainer` 类原生集成了 DeepSpeed。在训练 LLaMA、GPT、BLOOM 等模型时，DeepSpeed 的 ZeRO-3 配合 Flash Attention 是标准配置。
- **MLOps 中的多阶段训练流水线**：在生产环境中，训练流水线通常分为预训练（ZeRO-3 + Offload，数百 GPU）和微调（ZeRO-2，32-64 GPU）两个阶段。DeepSpeed 的 checkpoint 兼容性使得不同阶段间的模型转换无缝衔接。
- **模型评估与持续集成 (CI)**：在模型部署到生产环境前，需要在 CI 流水线中使用 DeepSpeed 的 ZeRO-Inference 进行大规模评估。ZeRO-Inference 可以在推理时利用 ZeRO 的分片机制，使单 GPU 能够运行数十亿参数的模型进行验证测试。
