# 39_RLHF：基于人类反馈的强化学习对齐

## 核心概念

- **RLHF (Reinforcement Learning from Human Feedback)**：使用人类偏好反馈作为奖励信号，通过强化学习微调语言模型，使其输出与人类价值观和期望对齐。
- **三阶段流程**：
  1. 指令微调 (SFT)：在有监督的（指令，理想回复）数据上微调预训练模型
  2. 训练奖励模型 (RM)：收集人类对模型输出的偏好比较（A>B），训练一个打分模型
  3. 强化学习优化 (PPO)：使用奖励模型的评分作为奖励信号，用 PPO 算法微调 SFT 模型
- **奖励模型 (Reward Model)**：输入（指令，回复），输出一个标量分数。训练数据为人类标注的比较对——标注员从两个模型输出中选择更好的一个，RM 学习最大化正确比较的概率。
- **Bradley-Terry 模型**：奖励模型训练的偏好建模框架。假设人类偏好满足 $P(\text{response}_1 > \text{response}_2) = \sigma(r_1 - r_2)$，其中 $r_i$ 是奖励模型对第 $i$ 个回复的打分。
- **KL 散度惩罚**：在 PPO 目标中加入与初始 SFT 模型之间的 KL 散度惩罚，防止模型在追求高奖励时偏离太远（奖励劫持问题）。
- **对齐税 (Alignment Tax)**：RLHF 在提升有用性和安全性的同时，可能降低模型在某些基准上的性能（如创造性）。需要平衡对齐程度和通用能力。
- **InstructGPT/ChatGPT 的成功**：InstructGPT 展示了 RLHF 在 1.3B 模型上的有效性，ChatGPT 将其扩展到大规模模型并取得了商业成功。

## 数学推导

**奖励模型训练**（Bradley-Terry 损失）：
$$
\mathcal{L}_{RM} = -\mathbb{E}_{(x, y_w, y_l) \sim D}[\log \sigma(r_\phi(x, y_w) - r_\phi(x, y_l))]
$$

其中 $y_w$ 是人类偏好的回复，$y_l$ 是较差的回复，$r_\phi$ 是奖励模型。

**PPO 优化目标**：
$$
\mathcal{L}_{PPO} = \mathbb{E}_{(x, y) \sim \pi_{\theta}}[r_\phi(x, y)] - \beta \cdot \mathbb{D}_{KL}(\pi_{\theta} || \pi_{\text{SFT}})
$$

其中 $\pi_{\theta}$ 是正在优化的策略模型，$\pi_{\text{SFT}}$ 是初始的指令微调模型，$\beta$ 控制 KL 惩罚强度。

实际使用中，使用 PPO-clip 变体：
$$
\mathcal{L}_{\text{PPO-clip}} = \mathbb{E}\left[\min\left(\frac{\pi_{\theta}(y|x)}{\pi_{\text{old}}(y|x)} A, \text{clip}\left(\frac{\pi_{\theta}}{\pi_{\text{old}}}, 1-\epsilon, 1+\epsilon\right) A\right)\right]
$$

## 直观理解

- **RLHF 像"师傅带徒弟"**：师傅（奖励模型）看过很多"好回复和差回复"的对比，建立了一套审美标准。徒弟（策略模型）不断尝试写回复，师傅打分。徒弟为了高分不断改进——但也不能改得"太刻意"（KL 惩罚），要保持自然。
- **三阶段 = 从模仿到优化的成长之路**：SFT 阶段像学徒模仿师傅的每个动作（有监督学习），RM 训练像建立质量标准（什么是对什么是错），PPO 像在实际工作中磨练——通过不断尝试和反馈来精进技能。
- **KL 散度惩罚像"不忘初心"**：RLHF 的目标是提升模型表现，但如果不加限制，模型可能会不择手段追求高奖励——比如说一些"用户爱听但不符合事实"的话。KL 惩罚确保模型不偏离初始模型的核心能力——你可以在"说真话"的前提下表达，但不能为了讨好说谎。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

# 模拟 RLHF 中的组件

class RewardModel(nn.Module):
    """简单奖励模型"""
    def __init__(self, hidden_size=768):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

    def forward(self, hidden_states):
        return self.classifier(hidden_states).squeeze(-1)

# 模拟 Bradley-Terry 损失
def bradley_terry_loss(reward_better, reward_worse):
    """偏好排序损失"""
    return -torch.log(torch.sigmoid(reward_better - reward_worse) + 1e-8).mean()

# 模拟 KL 散度惩罚
def kl_penalty(log_probs_current, log_probs_ref):
    """计算 KL 散度：KL(pi_cur || pi_ref)"""
    return (torch.exp(log_probs_ref) * (log_probs_ref - log_probs_current)).sum(-1).mean()

# 训练数据模拟
batch_size = 4
hidden_size = 768
better_hidden = torch.randn(batch_size, hidden_size)
worse_hidden = torch.randn(batch_size, hidden_size)

# RM 训练
rm = RewardModel(hidden_size)
reward_better = rm(better_hidden)
reward_worse = rm(worse_hidden)
rm_loss = bradley_terry_loss(reward_better, reward_worse)
print(f"RM 训练损失: {rm_loss.item():.4f}")

# RLHF PPO 损失模拟
pi_log_probs = torch.randn(batch_size, 100)  # 当前策略的 log probs
ref_log_probs = torch.randn(batch_size, 100)  # 初始 SFT 模型的 log probs
rewards = torch.randn(batch_size)             # 奖励模型的评分

kl_loss = kl_penalty(pi_log_probs, ref_log_probs)
rl_loss = -rewards.mean()                     # 最大化奖励
beta = 0.1
total_loss = rl_loss + beta * kl_loss
print(f"RLHF PPO 损失: {total_loss.item():.4f} (RL: {rl_loss.item():.4f}, KL: {kl_loss.item():.4f})")
```

## 深度学习关联

- **AI 对齐的核心技术**：RLHF 是实现 AI 对齐（Alignment）的关键技术，确保 AI 系统的行为符合人类意图和价值观。这是当前大模型安全研究的核心议题。
- **RLHF 的局限与改进**：RLHF 存在奖励劫持、偏好标注不一致、过度优化等问题。后续工作提出了 DPO（直接偏好优化）、Reinforcement Learning from AI Feedback (RLAIF) 等替代方案。
- **从人类反馈到可扩展监督**：RLHF 是"可扩展监督"（Scalable Oversight）的一种形式——通过有限的昂贵人类反馈来训练奖励模型，再用奖励模型自动评估大量输出。这一范式对超级对齐（Superalignment）研究有重要启示。
