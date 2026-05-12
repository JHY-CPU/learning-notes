# PPO在RLHF中的应用 - 人类对齐与RLHF

*深入理解 Proximal Policy Optimization 算法在 RLHF 中的具体实现，涵盖 Clipped Surrogate 目标、GAE 优势估计、KL 惩罚及完整 Pipeline*

## 一、PPO-RLHF 完整 Pipeline

```
Step 1: 训练 Reward Model (RM)
  - 收集人类偏好数据：(prompt, chosen, rejected)
  - 训练 Bradley-Terry 模型：L = -E[log(sigma(r(chosen) - r(rejected)))]

Step 2: PPO 微调
  - 初始化：Policy = SFT Model, Value = RM (或单独训练)
  - 对每个 prompt：
    1. 用当前 Policy 生成回答
    2. 用 RM 打分得到 reward
    3. 计算 KL 惩罚：r_total = r_RM - beta * KL(Policy || SFT)
    4. 用 GAE 计算优势函数
    5. 执行 PPO 更新（多个 epoch）
    6. 更新 KL 系数 beta
```

## 二、KL 惩罚的三种实现方式

| 方式 | 公式 | 特点 |
| --- | --- | --- |
| **固定系数 KL** | r = r_RM - β * KL | 简单直接，但 β 难以调优 |
| **自适应 KL** | 根据实际 KL 调整 β | 自动调节，InstructGPT 使用 |
| **KL 预算** | 设置 KL 上限 | 约束总偏离量 |

## 三、GAE（广义优势估计）

```
GAE 公式：
  A_t = sum_{l=0}^{infinity} (gamma * lambda)^l * delta_{t+l}
  其中 delta_t = r_t + gamma * V(s_{t+1}) - V(s_t)

  gamma=1, lambda=0.0 -> TD(0) 优势：方差小但偏差大
  gamma=1, lambda=1.0 -> MC 优势：偏差小但方差大
  gamma=1, lambda=0.95 -> GAE：偏差-方差权衡
```

## 四、PPO-RLHF 关键超参数

| 超参数 | InstructGPT 值 | 作用 | 调优建议 |
| --- | --- | --- | --- |
| lr (PPO) | 9.65e-6 | 策略更新学习率 | 从 1e-6 开始，过大会崩溃 |
| clip_epsilon | 0.2 | 裁剪范围 | 0.1-0.3，越小越保守 |
| kl_coef (β) | 自适应, 目标 6.0 | KL 惩罚强度 | 用自适应控制器 |
| gamma | 1.0 | 折扣因子 | RLHF 中通常用 1.0 |
| gae_lambda | 0.95 | GAE 偏差-方差权衡 | 0.9-0.99 常用 |
| ppo_epochs | 1 | 每批数据重复训练次数 | 1-4，过多会过拟合 |
| vf_coef | 0.5 | Value loss 权重 | 0.1-1.0 |
| entropy_coef | 0.01 | 熵正则化 | 过小则模式坍塌 |
| batch_size | 512 prompts | 每步 PPO 使用的 prompt 数 | 越大越稳定 |
| rollout_size | 1 回答/prompt | 每个 prompt 生成的回答数 | 1-4 |

## 五、PPO-RLHF 开源框架

| 框架 | 特点 | 适用场景 |
| --- | --- | --- |
| **TRL (HuggingFace)** | 集成度高，API 友好，与 HF 生态无缝对接 | 中小规模实验 |
| **DeepSpeed-Chat** | ZeRO 优化，显存效率高，支持大规模训练 | 百亿级模型训练 |
| **OpenRLHF** | 高性能，支持 Ray 分布式，解耦 4 个模型 | 大规模生产训练 |
| **trlX** | 支持分布式 PPO，设计灵活 | 研究实验 |

## 六、常见问题与解决方案

| 问题 | 原因 | 解决方案 |
| --- | --- | --- |
| 策略崩溃 | 学习率太大 | 降低 lr 到 1e-6 |
| 模式坍塌 | 熵正则化不够 | 增大 entropy_coef |
| Reward hacking | RM 被 exploit | 定期更新 RM |
| 训练不稳定 | GAE 方差大 | 调整 gae_lambda |
| 过拟合 | ppo_epochs 太多 | 减少到 1-2 |

## 七、Python 实战：PPO-RLHF 训练

### 示例：使用 TRL 库进行 PPO 训练

```python
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead
from trl.core import LengthSampler

# 1. 配置 PPO
config = PPOConfig(
    model_name="gpt2",
    learning_rate=1.41e-5,
    batch_size=64,
    mini_batch_size=16,
    ppo_epochs=4,
    kl_penalty="kl",
    init_kl_coef=0.2,
    adap_kl_ctrl=True,
    target_kl=6.0,
    gamma=1.0,
    gae_lambda=0.95,
    cliprange=0.2,
    cliprange_value=0.2,
    vf_coef=0.5,
    log_with="wandb",
)

# 2. 加载模型
model = AutoModelForCausalLMWithValueHead.from_pretrained(config.model_name)
tokenizer = AutoTokenizer.from_pretrained(config.model_name)
tokenizer.pad_token = tokenizer.eos_token

# 3. 加载 Reward Model（已训练好的）
from transformers import AutoModelForSequenceClassification
reward_model = AutoModelForSequenceClassification.from_pretrained(
    "./rm_checkpoint", num_labels=1
)
reward_model.eval()

# 4. 初始化 PPO Trainer
ppo_trainer = PPOTrainer(
    config=config,
    model=model,
    ref_model=None,  # TRL 会自动创建参考模型
    tokenizer=tokenizer,
)

# 5. 定义 reward 函数
def compute_reward(prompt, response):
    """用 RM 对 (prompt, response) 打分"""
    inputs = tokenizer(prompt + response, return_tensors="pt",
                       truncation=True, max_length=512)
    with torch.no_grad():
        score = reward_model(**inputs).logits.item()
    # KL 惩罚由 PPOTrainer 自动处理
    return score

# 6. PPO 训练循环
generation_kwargs = {
    "max_new_tokens": 64,
    "do_sample": True,
    "temperature": 0.7,
    "top_k": 0,
    "top_p": 0.9,
    "pad_token_id": tokenizer.eos_token_id,
}

# 模拟 prompt 数据集
prompts = ["What is machine learning?", "Explain neural networks.",
           "How does backpropagation work?", "What is overfitting?"]

for epoch in range(3):
    for batch_start in range(0, len(prompts), config.batch_size):
        batch_prompts = prompts[batch_start:batch_start + config.batch_size]

        # Tokenize prompts
        query_tensors = [tokenizer.encode(p, return_tensors="pt").squeeze()
                         for p in batch_prompts]

        # 生成回答
        response_tensors = ppo_trainer.generate(
            query_tensors, **generation_kwargs
        )

        # 计算 reward
        rewards = []
        for q, r in zip(query_tensors, response_tensors):
            prompt_text = tokenizer.decode(q)
            response_text = tokenizer.decode(r[len(q):])
            rewards.append(torch.tensor(compute_reward(prompt_text, response_text)))

        # PPO 更新
        stats = ppo_trainer.step(query_tensors, response_tensors, rewards)

        print(f"Epoch {epoch+1}, Mean reward: {torch.stack(rewards).mean():.4f}")
        ppo_trainer.log_stats(stats, {}, rewards)
```

### 示例：手动实现 PPO Clipped Loss

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class PPOTrainerManual:
    """简化的PPO训练器，展示核心算法"""

    def __init__(self, policy_model, ref_model, value_model,
                 clip_epsilon=0.2, kl_coef=0.2):
        self.policy = policy_model
        self.ref_model = ref_model
        self.value_model = value_model
        self.clip_epsilon = clip_epsilon
        self.kl_coef = kl_coef

        # 冻结参考模型
        for param in self.ref_model.parameters():
            param.requires_grad = False

    def compute_gae(self, rewards, values, gamma=1.0, lam=0.95):
        """计算广义优势估计"""
        advantages = []
        gae = 0
        next_value = 0

        for t in reversed(range(len(rewards))):
            delta = rewards[t] + gamma * next_value - values[t]
            gae = delta + gamma * lam * gae
            advantages.insert(0, gae)
            next_value = values[t]

        advantages = torch.tensor(advantages)
        returns = advantages + torch.tensor(values)
        return advantages, returns

    def ppo_loss(self, logprobs_new, logprobs_old, advantages, rewards):
        """PPO Clipped 目标函数"""
        # 重要性采样比率
        ratio = torch.exp(logprobs_new - logprobs_old)

        # 未裁剪的 surrogate 目标
        surr1 = ratio * advantages

        # 裁剪后的 surrogate 目标
        surr2 = torch.clamp(ratio, 1 - self.clip_epsilon,
                             1 + self.clip_epsilon) * advantages

        # 取两者的最小值（保守更新）
        policy_loss = -torch.min(surr1, surr2).mean()

        # KL 惩罚
        kl_div = (logprobs_new - logprobs_old).mean()
        total_loss = policy_loss + self.kl_coef * kl_div

        return total_loss, {
            "policy_loss": policy_loss.item(),
            "kl_div": kl_div.item(),
            "mean_ratio": ratio.mean().item(),
            "max_ratio": ratio.max().item(),
        }

    def train_step(self, prompts, responses, rewards, old_logprobs):
        """单步PPO训练"""
        # 1. 计算新策略的 log prob
        new_logprobs = self._get_logprobs(self.policy, prompts, responses)

        # 2. 计算价值估计
        values = self._get_values(self.value_model, prompts)

        # 3. 计算 GAE 优势
        advantages, returns = self.compute_gae(rewards, values)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        # 4. 计算 PPO 损失
        loss, info = self.ppo_loss(new_logprobs, old_logprobs, advantages, rewards)

        return loss, info

    def _get_logprobs(self, model, prompts, responses):
        """获取模型对响应的log概率"""
        # 简化实现
        return torch.zeros(len(prompts))

    def _get_values(self, model, prompts):
        """获取价值估计"""
        return [0.0] * len(prompts)
```

## 总结

- PPO-RLHF 的核心是 Clipped Surrogate 目标 + GAE 优势估计 + KL 惩罚
- 自适应 KL 系数是训练稳定性的关键（InstructGPT 使用目标 KL=6.0）
- 使用 TRL 库可以大幅简化 PPO-RLHF 的实现
- 训练中要监控 KL 散度、reward 和策略比率，防止策略崩溃
- DPO 等替代方案省去了 PPO 的复杂性，但 PPO 在大规模训练中仍有理论优势


<!-- Converted from: 02_PPO在RLHF中的应用.html -->
