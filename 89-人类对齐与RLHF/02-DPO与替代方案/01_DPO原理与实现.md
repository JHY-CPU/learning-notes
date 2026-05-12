# DPO原理与实现 - 人类对齐与RLHF

*Direct Preference Optimization——无需 Reward Model 和 RL 训练的偏好对齐方法，从 RLHF 到 closed-form 解的完整数学推导*

## 一、从 RLHF 到 DPO 的数学推导

RLHF 优化问题

```
max_π E_{x~D, y~π(y|x)}[r(x,y)] - β * D_KL(π(y|x) || π_ref(y|x))

展开 KL 散度：
= max_π E[r(x,y)] - β * E[log π(y|x) - log π_ref(y|x)]

= max_π E_{x,y}[r(x,y)/β + log π_ref(y|x) - log π(y|x)]

这等价于最小化：
= min_π E_{x,y}[log π(y|x) - (r(x,y)/β + log π_ref(y|x))]

令 Z(x) = Σ_y π_ref(y|x) * exp(r(x,y)/β)（配分函数）

可以证明最优策略为：
π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)
```

## 二、Reward 反解

```
由 π*(y|x) = π_ref(y|x) * exp(r(x,y)/β) / Z(x)

两边取 log：
log π*(y|x) = log π_ref(y|x) + r(x,y)/β - log Z(x)

整理得：
r(x,y) = β * [log π*(y|x) - log π_ref(y|x)] + β * log Z(x)

关键：配分函数 Z(x) 不依赖于 y！
因此在比较两个回答时，Z(x) 会消掉。
```

## 三、DPO 损失函数推导

```
P(y_w ≻ y_l | x) = σ(r(x,y_w) - r(x,y_l))

代入 r(x,y) = β * [log π(y|x) - log π_ref(y|x)] + β*log Z(x)：

= σ(β * [log π(y_w|x) - log π_ref(y_w|x) - log π(y_l|x) + log π_ref(y_l|x)])

= σ(β * log[π(y_w|x)/π_ref(y_w|x)] - β * log[π(y_l|x)/π_ref(y_l|x)])

定义隐式 Reward：
r_θ(x,y) = β * log[π_θ(y|x) / π_ref(y|x)]

DPO 损失（负对数似然）：
L_DPO(θ) = -E[log σ(β * (log π_θ(y_w|x)/π_ref(y_w|x) - log π_θ(y_l|x)/π_ref(y_l|x)))]

这就是 DPO 的最终形式——一个简单的二元交叉熵损失！
只需要策略模型 π_θ 和冻结的参考模型 π_ref，无需 RM 和 RL。
```

## 四、β 值的影响分析

| β 值 | 行为 | 适用场景 |
| --- | --- | --- |
| **β → 0** | 模型大幅偏向 chosen、远离 rejected | 偏好明确、数据质量高 |
| **β = 0.01 ~ 0.05** | 较强的偏好学习 | Chat 模型微调（常用） |
| **β = 0.1 ~ 0.2** | 温和的偏好学习 | 一般对齐任务 |
| **β → ∞** | 模型保持不变（π_θ ≈ π_ref） | 不做对齐 |

## 五、DPO vs PPO 深度对比

| 维度 | PPO (RLHF) | DPO |
| --- | --- | --- |
| **训练范式** | RM 训练 + RL 优化（两阶段） | 单阶段监督学习 |
| **模型数量** | 4 个（策略/参考/RM/Value Head） | 2 个（策略/参考） |
| **训练稳定性** | 不稳定，超参敏感 | 稳定，像正常微调 |
| **计算成本** | 高（需要在线采样 + RM 推理） | 低（离线数据 + 简单 loss） |
| **数据需求** | 需要在线生成数据 | 离线偏好对即可 |
| **探索能力** | 可以探索新的输出空间 | 受限于训练数据分布 |
| **分布外泛化** | 较好（在线采样覆盖更多分布） | 可能较差（离线数据有限） |
| **超参数** | 多（lr, clip, kl_coef, gae_lambda...） | 少（lr, beta） |
| **实现难度** | 高（需要工程优化） | 低（几行核心代码） |
| **最佳效果** | 理论上界更高 | 实践中常与 PPO 持平 |
| **适用场景** | 大规模生产训练 | 快速迭代、资源有限 |

## 六、DPO 家族方法对比

| 方法 | 损失函数 | 核心改进 |
| --- | --- | --- |
| **DPO** | -log σ(β * Δ) | Baseline，简单有效 |
| **IPO** | (Δ - 1/(2β))^2 | L2 损失防过拟合 |
| **cDPO** | 带 Label Smoothing 的 DPO | 噪声鲁棒性 |
| **SPPO** | 自博弈生成 + DPO | 无需人类标注 |
| **EXO** | Expert Optimization | 扩展到多专家 |
| **RDPO** | Regularized DPO | 更强正则化 |
| **Cal-DPO** | 校准的 DPO | 概率校准 |

## 七、Python 实战：DPO 训练

### 示例：手动实现 DPO 损失和训练

```python
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModelForCausalLM

# 1. DPO 损失函数
def dpo_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps, beta=0.1):
    """
    计算 DPO 损失

    Args:
        policy_chosen_logps: 策略模型对 chosen 的 log prob [B]
        policy_rejected_logps: 策略模型对 rejected 的 log prob [B]
        reference_chosen_logps: 参考模型对 chosen 的 log prob [B]
        reference_rejected_logps: 参考模型对 rejected 的 log prob [B]
        beta: 温度参数
    """
    # 隐式 reward
    chosen_rewards = beta * (policy_chosen_logps - reference_chosen_logps)
    rejected_rewards = beta * (policy_rejected_logps - reference_rejected_logps)

    # DPO 损失 = -log σ(r_chosen - r_rejected)
    logits = chosen_rewards - rejected_rewards
    loss = -torch.nn.functional.logsigmoid(logits).mean()

    # 额外指标
    chosen_reward_mean = chosen_rewards.mean().item()
    rejected_reward_mean = rejected_rewards.mean().item()
    reward_margin = chosen_reward_mean - rejected_reward_mean
    accuracy = (logits > 0).float().mean().item()

    return loss, {
        "chosen_reward": chosen_reward_mean,
        "rejected_reward": rejected_reward_mean,
        "reward_margin": reward_margin,
        "accuracy": accuracy,
    }

# 2. 计算序列 log probability
def get_batch_logps(model, input_ids, attention_mask, labels):
    """计算模型对完整序列的平均 log probability"""
    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
    logits = outputs.logits  # [B, T, V]

    # 对每个token计算log prob，只在label != -100的位置
    log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
    token_logps = torch.gather(log_probs, dim=-1,
                                index=labels.unsqueeze(-1)).squeeze(-1)

    # 只计算非 padding token 的 log prob
    mask = (labels != -100).float()
    seq_logps = (token_logps * mask).sum(dim=-1) / mask.sum(dim=-1)
    return seq_logps

# 3. DPO 训练循环
def train_dpo():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    tokenizer.pad_token = tokenizer.eos_token

    # 策略模型（可训练）
    policy_model = AutoModelForCausalLM.from_pretrained("gpt2")
    # 参考模型（冻结）
    ref_model = AutoModelForCausalLM.from_pretrained("gpt2")
    for param in ref_model.parameters():
        param.requires_grad = False

    optimizer = torch.optim.AdamW(policy_model.parameters(), lr=5e-7)
    beta = 0.1

    # 模拟偏好数据
    preference_data = [
        {"prompt": "什么是深度学习？",
         "chosen": "深度学习是机器学习的一个子领域，使用多层神经网络...",
         "rejected": "就是很深度的学习呗。"},
    ]

    for epoch in range(3):
        for item in preference_data:
            optimizer.zero_grad()

            # 构造输入
            chosen_text = item["prompt"] + item["chosen"]
            rejected_text = item["prompt"] + item["rejected"]

            chosen_enc = tokenizer(chosen_text, return_tensors="pt",
                                    truncation=True, max_length=256,
                                    padding="max_length")
            rejected_enc = tokenizer(rejected_text, return_tensors="pt",
                                      truncation=True, max_length=256,
                                      padding="max_length")

            # 计算 policy log probs
            policy_chosen = get_batch_logps(
                policy_model, chosen_enc["input_ids"],
                chosen_enc["attention_mask"], chosen_enc["input_ids"]
            )
            policy_rejected = get_batch_logps(
                policy_model, rejected_enc["input_ids"],
                rejected_enc["attention_mask"], rejected_enc["input_ids"]
            )

            # 计算 reference log probs（不计算梯度）
            with torch.no_grad():
                ref_chosen = get_batch_logps(
                    ref_model, chosen_enc["input_ids"],
                    chosen_enc["attention_mask"], chosen_enc["input_ids"]
                )
                ref_rejected = get_batch_logps(
                    ref_model, rejected_enc["input_ids"],
                    rejected_enc["attention_mask"], rejected_enc["input_ids"]
                )

            # DPO 损失
            loss, info = dpo_loss(policy_chosen, policy_rejected,
                                   ref_chosen, ref_rejected, beta=beta)
            loss.backward()
            optimizer.step()

            print(f"Epoch {epoch+1}, Loss: {loss.item():.4f}, "
                  f"Acc: {info['accuracy']:.2f}, "
                  f"Margin: {info['reward_margin']:.4f}")

# train_dpo()
```

### 示例：使用 TRL 库进行 DPO 训练

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import DPOTrainer, DPOConfig

# 加载模型
model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")

# 配置 DPO 训练
dpo_config = DPOConfig(
    output_dir="./dpo_checkpoint",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-7,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    gradient_accumulation_steps=4,
    logging_steps=10,
    evaluation_strategy="steps",
    save_strategy="steps",
)

# 数据集需要包含: prompt, chosen, rejected 三列
# from datasets import load_dataset
# dataset = load_dataset("Anthropic/hh-rlhf", split="train[:1000]")

# trainer = DPOTrainer(
#     model=model,
#     ref_model=None,  # 自动创建参考模型副本
#     args=dpo_config,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
# )
# trainer.train()
```

## 总结

- DPO 通过数学推导将 RLHF 的 RL 问题转化为简单的监督学习
- 核心公式：L_DPO = -E[log σ(β * (log π/π_ref)_chosen - (log π/π_ref)_rejected))]
- 只需要策略模型和参考模型（2个），不需要 RM 和 Value Head（4个）
- β 控制偏好学习强度，0.01-0.05 适合 Chat 模型微调
- 实践中 DPO 与 PPO 效果相当，但训练更简单、更稳定
- DPO 的主要限制是缺乏在线采样，对分布外泛化不如 PPO


<!-- Converted from: 01_DPO原理与实现.html -->
