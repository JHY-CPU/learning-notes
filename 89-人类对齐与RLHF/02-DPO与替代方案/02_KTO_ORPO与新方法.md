# KTO、ORPO与新方法 - 人类对齐与RLHF

*探索 DPO 之外的人类对齐新范式：Kahneman-Tversky Optimization、Odds Ratio Preference Optimization、SimPO、自博弈方法及 Alignment Tax*

## 一、KTO（Kahneman-Tversky Optimization）

KTO 基于前景理论（Prospect Theory），只需要二元标签（好/坏），不需要配对偏好数据。这使得 KTO 可以利用更广泛的数据来源。

### 核心公式

$$
L_{KTO}(θ) = E_{x,y}[w(y) · (1 - v_θ(x, y))]
$$

其中 v_θ 是隐式 reward 的 sigmoid 映射，w(y) 根据好/坏回答给予不同权重。KTO 的关键是利用前景理论中的损失厌恶——人们对损失的敏感度高于等量收益。

## 二、ORPO（Odds Ratio Preference Optimization）

赔率（Odds）与赔率比（Odds Ratio）

```
赔率（Odds）：
  odds(y|x) = P(y|x) / (1 - P(y|x))
  即"发生概率 / 不发生概率"

赔率比（Odds Ratio）：
  OR(y_w, y_l | x) = odds(y_w|x) / odds(y_l|x)
  即"好回答的赔率 / 坏回答的赔率"

  当 OR > 1 时，y_w 比 y_l 更可能被生成
  ORPO 目标：最大化好回答相对于坏回答的赔率比

  逐 token 计算赔率：
  odds(y|x) = Π_t [p(y_t|x, y_<t) / (1 - p(y_t|x, y_<t))]
```

ORPO 的最大优势：**完全不需要参考模型**。它将 SFT 和偏好优化合并为一步，损失函数为：

$$
L_{ORPO} = L_{SFT} + λ · L_{OR}
$$

其中 L_OR = -log σ(log OR(y_w, y_l))

## 三、SimPO（Simple Preference Optimization）

SimPO 使用隐式 reward 的长度归一化版本，解决 DPO 中序列长度对 reward 的影响：

$$
r_{SimPO}(x, y) = \frac{1}{|y|} \log π_θ(y|x)
$$

用平均 log prob 替代总 log prob，避免长回答天然获得更高 reward。

## 四、自博弈方法分类

| 方法 | 描述 | 代表工作 |
| --- | --- | --- |
| **SPIN** | 当前模型 vs SFT 模型的生成，通过自博弈消除幻觉 | SPIN (Chen et al., 2024) |
| **SPPO** | 自博弈生成偏好数据 + DPO 训练 | SPPO (Wu et al., 2024) |
| **IPO-Self** | 用自身不同温度采样的回答作为偏好对 | 理论分析 |
| **SPO** | 自我博弈偏好优化，两模型互相学习 | SPO 系列 |
| **RLAI Self-Play** | 类似 AlphaGo 的自我对弈范式 | DeepMind 研究 |

## 五、Alignment Tax

Alignment Tax 的表现形式

| 维度 | 下降表现 | 可能原因 |
| --- | --- | --- |
| **学术基准** | MMLU/GSM8K 等分数下降 | 对齐训练改变了知识表达 |
| **代码能力** | 代码生成能力退化 | 安全训练限制了代码输出 |
| **创造性** | 回答趋于保守、模板化 | 拒绝训练导致过度谨慎 |
| **多样性** | 输出多样性下降 | 偏好学习收敛到单一模式 |
| **多语言** | 非英语语言能力下降 | 偏好数据以英语为主 |

## 六、人类对齐方法全对比

| 方法 | 需要RM? | 需要参考模型? | 需要配对数据? | 训练阶段 | 复杂度 |
| --- | --- | --- | --- | --- | --- |
| **PPO (RLHF)** | 是 | 是 | 是（训练RM） | 3阶段 | 高 |
| **DPO** | 否 | 是 | 是 | 1阶段 | 低 |
| **KTO** | 否 | 是 | **否** | 1阶段 | 低 |
| **ORPO** | 否 | **否** | 是 | 1阶段 | 最低 |
| **SimPO** | 否 | 是 | 是 | 1阶段 | 低 |
| **IPO** | 否 | 是 | 是 | 1阶段 | 低 |
| **SPIN** | 否 | 是 | **否** | 迭代 | 中 |

## 七、Python 实战：ORPO 与 KTO 训练

### 示例：ORPO 损失函数实现

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

def orpo_loss(policy_chosen_logps, policy_rejected_logps, sft_weight=1.0, orpo_weight=1.0):
    """
    ORPO 损失 = SFT 损失 + λ * OR 损失

    Args:
        policy_chosen_logps: 策略模型对 chosen 序列的平均 log prob [B]
        policy_rejected_logps: 策略模型对 rejected 序列的平均 log prob [B]
        sft_weight: SFT 损失权重
        orpo_weight: OR 损失权重
    """
    # SFT 损失：最大化 chosen 的 log prob
    sft_loss = -policy_chosen_logps.mean()

    # 计算赔率比
    # odds = p / (1-p), log_odds = log_prob - log(1 - exp(log_prob))
    log_odds_chosen = policy_chosen_logps - torch.log1p(-torch.exp(policy_chosen_logps))
    log_odds_rejected = policy_rejected_logps - torch.log1p(-torch.exp(policy_rejected_logps))

    # log OR = log_odds_chosen - log_odds_rejected
    log_or = log_odds_chosen - log_odds_rejected

    # OR 损失
    or_loss = -F.logsigmoid(log_or).mean()

    total_loss = sft_weight * sft_loss + orpo_weight * or_loss

    # 计算指标
    accuracy = (log_or > 0).float().mean().item()

    return total_loss, {
        "sft_loss": sft_loss.item(),
        "or_loss": or_loss.item(),
        "accuracy": accuracy,
        "mean_log_or": log_or.mean().item(),
    }


def kto_loss(policy_chosen_logps, policy_rejected_logps,
             reference_chosen_logps, reference_rejected_logps,
             beta=0.1, desirable_weight=1.0, undesirable_weight=1.0):
    """
    KTO 损失函数：基于前景理论的非配对偏好优化

    不需要配对数据，只需好/坏回答的分别集合
    """
    # 隐式 KL
    chosen_kl = (policy_chosen_logps - reference_chosen_logps).mean().detach()
    rejected_kl = (policy_rejected_logps - reference_rejected_logps).mean().detach()

    # 好回答的损失
    chosen_reward = beta * (policy_chosen_logps - reference_chosen_logps)
    chosen_loss = desirable_weight * (1 - torch.sigmoid(chosen_reward - beta * chosen_kl))

    # 坏回答的损失
    rejected_reward = beta * (policy_rejected_logps - reference_rejected_logps)
    rejected_loss = undesirable_weight * (1 - torch.sigmoid(beta * rejected_kl - rejected_reward))

    loss = (chosen_loss.mean() + rejected_loss.mean()) / 2

    return loss, {
        "chosen_reward_mean": chosen_reward.mean().item(),
        "rejected_reward_mean": rejected_reward.mean().item(),
    }


def simpo_loss(policy_chosen_logps, policy_rejected_logps,
               reference_chosen_logps, reference_rejected_logps,
               beta=2.0, gamma_margin=0.5):
    """
    SimPO：长度归一化的 DPO
    使用平均 log prob 而非总 log prob，并加入 reward margin
    """
    # 隐式 reward（已由调用者做了长度归一化）
    chosen_rewards = beta * policy_chosen_logps
    rejected_rewards = beta * policy_rejected_logps

    # SimPO 损失：带 margin 的 DPO
    logits = chosen_rewards - rejected_rewards - gamma_margin
    loss = -F.logsigmoid(logits).mean()

    accuracy = (logits > 0).float().mean().item()

    return loss, {"accuracy": accuracy}
```

### 示例：使用 TRL 进行 ORPO 训练

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
from trl import ORPOTrainer, ORPOConfig

model = AutoModelForCausalLM.from_pretrained("facebook/opt-350m")
tokenizer = AutoTokenizer.from_pretrained("facebook/opt-350m")
tokenizer.pad_token = tokenizer.eos_token

orpo_config = ORPOConfig(
    output_dir="./orpo_checkpoint",
    per_device_train_batch_size=4,
    num_train_epochs=3,
    learning_rate=5e-6,
    beta=0.1,
    max_length=512,
    max_prompt_length=256,
    gradient_accumulation_steps=4,
    logging_steps=10,
)

# 数据集需要: prompt, chosen, rejected 三列
# trainer = ORPOTrainer(
#     model=model,
#     args=orpo_config,
#     train_dataset=train_dataset,
#     tokenizer=tokenizer,
# )
# trainer.train()
```

## 总结

- KTO 利用前景理论，只需要好/坏标签，不需要配对数据
- ORPO 将 SFT 和偏好优化合并为一步，完全不需要参考模型
- SimPO 通过长度归一化解决 DPO 中的长度偏好问题
- 自博弈方法（SPIN、SPPO）通过模型自身生成偏好数据
- Alignment Tax 是对齐训练不可避免的代价，需在安全和性能间权衡
- 选择对齐方法时应综合考虑数据格式、计算预算和性能需求


<!-- Converted from: 02_KTO_ORPO与新方法.html -->
