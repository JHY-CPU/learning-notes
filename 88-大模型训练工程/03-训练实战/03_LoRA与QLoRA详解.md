# LoRA与QLoRA详解 - 大模型训练工程

*低秩适配 / 量化低秩适配 / AdaLoRA / DoRA*


## LoRA 与 QLoRA 详解


低秩适配 / 量化低秩适配 / AdaLoRA / DoRA

[88-大模型训练工程](../index.html)
>
[03-训练实战](./)
>
            03_LoRA与QLoRA详解

### 目录


1. [LoRA 数学原理](#lora-math)
2. [LoRA 实现详解](#lora-impl)
3. [Rank 选择策略](#rank)
4. [Target Modules 选择](#target)
5. [QLoRA 详解](#qlora)
6. [AdaLoRA](#adalora)
7. [DoRA](#dora)
8. [方法对比与选型](#comparison)


## 1. LoRA 数学原理


LoRA（Low-Rank Adaptation）的核心思想是：预训练模型的权重更新矩阵是低秩的，可以用两个小矩阵的乘积来近似。


### 1.1 数学公式


$$
原始线性层: h = W₀ · x

                LoRA 适配:  h = W₀ · x + ΔW · x = W₀ · x + (B · A) · x

                其中:
                  W₀ ∈ R^{d×k}  (冻结的预训练权重)
                  A  ∈ R^{r×k}  (降维矩阵, r ≪ min(d,k))
                  B  ∈ R^{d×r}  (升维矩阵)
                  r  = LoRA rank (通常 4~64)
                  ΔW = B · A    (权重更新的低秩近似)
$$


### 1.2 图解


```
LoRA 结构图:

输入 x ∈ R^k
     │
     ├──────────────────────┐
     │                      │
     ▼                      ▼
┌──────────┐         ┌──────────┐
│ W₀ ∈ R^{d×k} │         │ A ∈ R^{r×k}  │  ← 降维
│ (冻结)   │         │ (可训练)  │
└────┬─────┘         └────┬─────┘
     │                     │
     │                     ▼
     │              ┌──────────┐
     │              │ B ∈ R^{d×r}  │  ← 升维
     │              │ (可训练)  │
     │              └────┬─────┘
     │                   │
     │           × (α/r)  │  ← 缩放因子
     │                   │
     ▼                   ▼
  W₀·x         +   B·A·x    =  输出 h

前向: h = W₀·x + (α/r)·B·A·x

初始化: A ~ Gaussian(0, σ²), B = 0
→ 训练开始时 ΔW = 0, 不影响原始模型行为
```


### 1.3 参数量分析


$$
原始参数量: d × k
                LoRA 参数量: d × r + r × k = r × (d + k)

                压缩比 = r × (d + k) / (d × k) = r × (1/d + 1/k)

                例: d=4096, k=4096, r=16
                LoRA 参数 = 16 × 8192 = 131,072
                原始参数 = 16,777,216
                压缩比 = 0.78% (仅需不到 1% 的参数!)
$$


## 2. LoRA 实现详解


### 2.1 手动实现


```
python
import torch
import torch.nn as nn
import math

class LoRALinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 16,           # LoRA rank
        lora_alpha: int = 32,   # 缩放因子
        lora_dropout: float = 0.05,
        merge_weights: bool = False,
    ):
        super().__init__()
        self.r = r
        self.lora_alpha = lora_alpha
        self.scaling = lora_alpha / r  # 缩放因子 α/r
# 冻结的原始权重
        self.weight = nn.Parameter(torch.zeros(out_features, in_features))
        self.weight.requires_grad = False # 冻结!
# LoRA 参数
if r > 0:
            self.lora_A = nn.Parameter(
                torch.randn(r, in_features) * (1.0 / math.sqrt(r))
            )  # 高斯初始化
            self.lora_B = nn.Parameter(
                torch.zeros(out_features, r)
            )  # 零初始化 (确保初始 ΔW = 0)
            self.lora_dropout = nn.Dropout(lora_dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 原始线性变换 (frozen)
        result = torch.nn.functional.linear(x, self.weight)

        # LoRA 分支
if self.r > 0:
            lora_x = self.lora_dropout(x)
            lora_out = lora_x @ self.lora_A.T()  # [batch, r]
            lora_out = lora_out @ self.lora_B.T()  # [batch, out]
            result = result + lora_out * self.scaling

        return result

    def merge(self):
        # 合并 LoRA 权重到原始权重 (推理时使用)
if self.r > 0:
            self.weight.data += (self.lora_B @ self.lora_A) * self.scaling
            # 合并后可以删除 LoRA 参数
def unmerge(self):
        # 恢复分离状态
if self.r > 0:
            self.weight.data -= (self.lora_B @ self.lora_A) * self.scaling
```


### 2.2 PEFT 库使用


```
python
from peft import LoraConfig, get_peft_model, TaskType
from peft import PeftModel, PeftConfig

# 训练
config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=16,
    lora_alpha=32,
    target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
    lora_dropout=0.05,
    bias="none",
    modules_to_save=None,
)
model = get_peft_model(base_model, config)

# 保存 (仅保存 LoRA 权重, ~几十 MB)
model.save_pretrained("./lora_weights")

# 加载
model = PeftModel.from_pretrained(base_model, "./lora_weights")

# 合并权重 (推理优化)
merged_model = model.merge_and_unload()
```


## 3. Rank 选择策略


Rank（秩）是 LoRA 最关键的超参数，控制低秩近似的表达能力。


### 3.1 Rank 的影响


| Rank | 参数量 (4096×4096层) | 表达能力 | 过拟合风险 | 适用场景 |
| --- | --- | --- | --- | --- |
| r=4 | 32K (0.19%) | 弱 | 低 | 简单任务、数据少 |
| r=8 | 65K (0.39%) | 中等 | 低 | 一般任务 |
| r=16 | 131K (0.78%) | 较强 | 中 | 通用推荐 |
| r=32 | 262K (1.56%) | 强 | 中高 | 复杂任务、数据多 |
| r=64 | 524K (3.13%) | 很强 | 高 | 接近 Full FT 效果 |


### 3.2 选择建议


> **Note:** **经验法则：**
>
> - 简单分类/问答任务：r=4~8
> - 通用指令跟随：r=16~32（大多数场景的推荐值）
> - 需要接近 Full FT 效果：r=64+
> - 数据量大时可适当增大 rank
> - 数据量小时应减小 rank 防止过拟合
> - LoRA alpha 通常设为 rank 的 2 倍


## 4. Target Modules 选择


### 4.1 Transformer 模块分析


```
Transformer Block 中的线性层:

┌─────────────────────────────────────┐
│          Transformer Block          │
│                                     │
│  ┌─── Attention ─────────────────┐  │
│  │  q_proj: [d × d]              │  │
│  │  k_proj: [d × d]              │  │
│  │  v_proj: [d × d]              │  │
│  │  o_proj: [d × d]              │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌─── FFN ───────────────────────┐  │
│  │  gate_proj: [d × 4d]          │  │
│  │  up_proj:   [d × 4d]          │  │
│  │  down_proj: [4d × d]          │  │
│  └───────────────────────────────┘  │
│                                     │
│  ┌─── Embeddings ────────────────┐  │
│  │  embed_tokens: [vocab × d]    │  │
│  │  lm_head: [d × vocab]         │  │
│  └───────────────────────────────┘  │
└─────────────────────────────────────┘

注意力层参数: 4 × d²
FFN层参数: 3 × d × 4d = 12d² (FFN 参数是注意力的 3 倍)
```


### 4.2 Target Modules 策略对比


| 策略 | 模块 | LoRA 参数占比 | 效果 |
| --- | --- | --- | --- |
| 最小 | q_proj, v_proj | ~0.2% | 基础，效果有限 |
| 注意力全部 | q_proj, k_proj, v_proj, o_proj | ~0.4% | 较好 |
| 全部线性层 | 注意力 + FFN 所有 | ~0.8% | 最佳 (推荐) |
| 含嵌入 | 全部线性层 + embed + head | ~1%+ | 适配新词表 |


> **Tip:** **推荐：**
> 大多数场景下，将所有线性层作为 target modules 效果最好。实验证明，LoRA 在 FFN 层的适配同样重要（甚至比注意力层更显著），因为 FFN 层参数量更大。


## 5. QLoRA 详解


QLoRA 在 LoRA 的基础上，将基础模型量化为 4-bit，进一步降低显存需求，使得在单卡上微调超大模型成为可能。


### 5.1 核心创新


1. **NF4 量化：**
   一种针对正态分布权重优化的 4-bit 数据类型
2. **双量化 (Double Quantization)：**
   对量化常数本身再做量化，进一步节省显存
3. **分页优化器 (Paged Optimizers)：**
   利用 NVIDIA 统一内存机制，在 GPU 显存不足时自动将优化器状态卸载到 CPU


### 5.2 NF4 量化原理


```
NF4 (Normal Float 4-bit) 量化:

标准 FP4 量化:
  0000 0001 0010 0011 0100 0101 0110 0111
  均匀分布在 [-max, +max] 之间

NF4 量化 (针对正态分布优化):
  假设权重服从 N(0, σ²), 将正态分布等分成 16 个区间
  每个区间包含等概率的值
  → 非均匀量化, 但信息论最优

量化公式:
  NF4 值 = quantile(正态分布 CDF, 等间隔点)
  这保证了每个量化值覆盖等概率的数值区间
```


### 5.3 QLoRA 实现


```
python
import torch
from transformers import (
    AutoModelForCausalLM,
    BitsAndBytesConfig,
    TrainingArguments
)
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training

# 1. 配置 4-bit 量化
bnb_config = BitsAndBytesConfig(
    load_in_4bit=True,                    # 4-bit 量化加载
    bnb_4bit_quant_type="nf4",             # 使用 NF4 量化类型
    bnb_4bit_compute_dtype=torch.bfloat16,  # 计算精度 (反量化后)
    bnb_4bit_use_double_quant=True,        # 双量化 (节省 ~0.4GB/7B)
)

# 2. 加载量化模型
model = AutoModelForCausalLM.from_pretrained(
    "meta-llama/Llama-2-70b-hf",           # 70B 模型!
    quantization_config=bnb_config,
    device_map="auto",                     # 自动分配到可用 GPU
    trust_remote_code=True,
)

# 3. 准备模型进行 k-bit 训练
model = prepare_model_for_kbit_training(model)

# 4. 配置 LoRA
lora_config = LoraConfig(
    r=64,                                   # 可以使用较大 rank
    lora_alpha=128,
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                    "gate_proj", "up_proj", "down_proj"],
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
)

model = get_peft_model(model, lora_config)
model.print_trainable_parameters()
# trainable: ~1.6% for 70B model with r=64
# 5. 训练 (单卡 A100 80GB 可以微调 70B 模型!)
training_args = TrainingArguments(
    output_dir="./qlora_output",
    num_train_epochs=3,
    per_device_train_batch_size=4,
    gradient_accumulation_steps=4,
    learning_rate=2e-4,
    bf16=True,
    optim="paged_adamw_32bit",            # 分页优化器
)
```


### 5.4 显存对比


| 方法 | 7B 模型 | 13B 模型 | 70B 模型 |
| --- | --- | --- | --- |
| Full FT (BF16) | ~60 GB | ~120 GB | ~600 GB |
| LoRA (BF16) | ~20 GB | ~40 GB | ~160 GB |
| QLoRA (NF4) | **~6 GB** | **~10 GB** | **~48 GB** |


> **Note:** **QLoRA 的突破：**
> QLoRA 让单卡 24GB 的消费级 GPU（如 RTX 3090/4090）可以微调 7B 模型，单卡 48GB（如 A6000）可以微调 30B+ 模型。这大大降低了大模型微调的门槛。


## 6. AdaLoRA


AdaLoRA（Adaptive LoRA）通过自适应调整不同模块的 LoRA rank，在总参数量不变的情况下，将更多的参数分配给更重要的模块。


### 6.1 核心思想


```
LoRA vs AdaLoRA:

LoRA (固定 rank):
  Attention q_proj: r=16  ← 所有模块相同
  Attention k_proj: r=16
  Attention v_proj: r=16
  FFN gate_proj:    r=16
  FFN up_proj:      r=16

AdaLoRA (自适应 rank):
  Attention q_proj: r=32  ← 重要模块分配更多
  Attention k_proj: r=8   ← 不重要模块减少
  Attention v_proj: r=24
  FFN gate_proj:    r=16
  FFN up_proj:      r=8

总参数量相同, 但分配更合理
```


### 6.2 实现


```
python
from peft import AdaLoraConfig, get_peft_model

config = AdaLoraConfig(
    task_type=TaskType.CAUSAL_LM,
    target_r=8,              # 目标平均 rank
    init_r=12,               # 初始 rank (稍大)
    tinit=200,               # 前 200 步不裁剪
    tfinal=1000,             # 第 1000 步完成裁剪
    deltaT=10,               # 每 10 步调整一次
    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"],
    lora_alpha=32,
    orth_reg_weight=0.5,     # 正交正则化权重
)

model = get_peft_model(base_model, config)
```


## 7. DoRA


DoRA（Weight-Decomposed Low-Rank Adaptation）将权重分解为幅度和方向两个分量，用 LoRA 只适配方向分量，效果优于标准 LoRA。


### 7.1 原理


$$
标准 LoRA:  W' = W₀ + B·A

                DoRA:      W' = m · (W₀/||W₀|| + B·A/||W₀+B·A||) · ||W₀||

                即: W' = m · (W₀ + B·A) / ||W₀ + B·A|| × ||W₀||

                m 是可学习的幅度向量, LoRA 只改变方向
$$


### 7.2 使用


```
python
# PEFT 支持 DoRA
from peft import LoraConfig

config = LoraConfig(
    r=16,
    lora_alpha=32,
    use_dora=True,             # 启用 DoRA
    target_modules=["q_proj", "v_proj"],
    lora_dropout=0.05,
)
# DoRA 比标准 LoRA 多约 0.1% 的参数 (幅度向量)
# 但在许多任务上效果显著更好
```


> **Tip:** **DoRA 的优势：**
> 实验证明，DoRA 在各种 rank 下都优于标准 LoRA，且在低 rank（r=4,8）时优势更明显。这使得 DoRA 特别适合显存受限的场景。


## 8. 方法对比与选型


### 8.1 综合对比


| 方法 | 基础精度 | 额外参数 | 显存需求 | 效果 | 推理开销 |
| --- | --- | --- | --- | --- | --- |
| Full FT | BF16 | 0% | 最高 | 最优 | 无 |
| LoRA | BF16 | 0.1-1% | 中 | 接近 FT | 可合并为零 |
| QLoRA | NF4 | 0.1-1% | 最低 | 接近 LoRA | 需保留量化 |
| AdaLoRA | BF16 | 自适应 | 中 | 优于 LoRA | 可合并 |
| DoRA | BF16 | 0.1-1%+ | 中 | 优于 LoRA | 可合并 |


### 8.2 选型建议


> **Tip:** **场景推荐：**
>
> 1. **算力充足 + 追求极致效果：**
>    Full Fine-tuning
> 2. **通用微调（推荐）：**
>    LoRA (r=16, all linear layers) + DoRA
> 3. **显存受限：**
>    QLoRA (NF4, r=8~16)
> 4. **参数预算紧张：**
>    AdaLoRA（自动优化参数分配）
> 5. **推理部署：**
>    LoRA/DoRA 训练 → 合并权重后部署


### 8.3 推荐配置汇总


```
python
# 最佳实践配置 (综合推荐)
lora_config = LoraConfig(
    r=16,                          # 通用 rank
    lora_alpha=32,                 # 2 × rank
    lora_dropout=0.05,             # 轻度正则化
    use_dora=True,                 # 启用 DoRA
    target_modules=[               # 全部线性层
"q_proj", "k_proj", "v_proj", "o_proj",
        "gate_proj", "up_proj", "down_proj"
    ],
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)

# 显存受限时的配置
qlora_config = LoraConfig(
    r=8,                           # 较小 rank
    lora_alpha=16,
    lora_dropout=0.05,
    target_modules=["q_proj", "v_proj"],  # 最小模块
    bias="none",
    task_type=TaskType.CAUSAL_LM,
)
# 配合 BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4")
```

大模型训练工程 - LoRA与QLoRA详解 | 最后更新: 2025年


<!-- Converted from: 03_LoRA与QLoRA详解.html -->
