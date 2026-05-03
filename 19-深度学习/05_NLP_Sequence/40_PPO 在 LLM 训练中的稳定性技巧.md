# 40_PPO 在 LLM 训练中的稳定性技巧

## 核心概念

- **PPO (Proximal Policy Optimization)**：由 Schulman et al. (2017) 提出的强化学习算法，通过限制策略更新的幅度保证训练稳定性。在 LLM 对齐（RLHF）中作为标准优化算法。
- **Clipping 目标函数**：PPO 的核心创新——将新旧策略的比率 $\pi_{\theta}/\pi_{\text{old}}$ 限制在 $[1-\epsilon, 1+\epsilon]$ 范围内，防止单次更新过大导致策略崩溃。
- **重要性采样比率**：$r_t(\theta) = \frac{\pi_{\theta}(a_t|s_t)}{\pi_{\text{old}}(a_t|s_t)}$，衡量新策略相对于旧策略在动作 $a_t$ 上的概率变化。
- **Advantage 估计 (GAE)**：广义优势估计 (Generalized Advantage Estimation)——使用 TD-$\lambda$ 方法平衡偏差和方差，减少奖励稀疏性问题。在 LLM 的 RLHF 中，通常使用"每个 token 的奖励"或"整句奖励"。
- **KL 散度惩罚**：在 RLHF 中，PPO 目标包含与参考模型的 KL 散度惩罚项。这既保持了模型的语言能力，也防止"奖励劫持"（模型学到漏洞来欺骗奖励模型）。
- **价值函数 (Value Function)**：PPO 使用一个 critic 网络（通常与策略网络共享 Transformer 主体）来估计状态价值 $V(s)$，用于优势计算。Critic 的损失是均方误差。
- **mini-batch 和 epoch**：PPO 使用与经验回放类似的机制——收集一批经验数据后，在多个 epoch 上用小批量更新。每个 epoch 都要重新计算 old log prob 和 Advantage。
- **学习率预热和衰减**：LLM 的 PPO 训练对学习率非常敏感，通常使用线性预热 + 余弦衰减或线性衰减的调度策略。

## 数学推导

PPO 的 clipping 目标函数：
$$
\mathcal{L}_{\text{PPO}}(\theta) = \mathbb{E}_{t}\left[\min\left(r_t(\theta) \hat{A}_t, \text{clip}(r_t(\theta), 1-\epsilon, 1+\epsilon) \hat{A}_t\right)\right]
$$

其中 $\hat{A}_t$ 是优势估计（advantage），$r_t(\theta)$ 是重要性采样比率。

**GAE 优势估计**：
$$
\hat{A}_t^{\text{GAE}} = \sum_{l=0}^{\infty} (\gamma\lambda)^l \delta_{t+l}, \quad \delta_t = r_t + \gamma V(s_{t+1}) - V(s_t)
$$

其中 $\gamma$ 是折扣因子（LLM 中通常为 1），$\lambda$ 是 GAE 参数。

**LLM 中的 PPO 总损失**：
$$
\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{PPO}} + c_1 \mathcal{L}_{\text{Value}} - c_2 \mathcal{L}_{\text{KL}}
$$

其中 $\mathcal{L}_{\text{Value}} = (V_\theta(s) - \hat{R})^2$ 是价值函数损失，$\mathcal{L}_{\text{KL}} = \mathbb{D}_{KL}(\pi_{\theta} || \pi_{\text{ref}})$ 是 KL 散度。

## 直观理解

- **PPO 的 Clipping 像运动时的"保护杠"**：你不想一步迈得太大导致摔倒（策略崩溃），所以给每一步设置一个最大步长限制。Clipping 就是将策略更新限制在 $[1-\epsilon, 1+\epsilon]$ 范围——就像跑步机上的安全夹，防止速度变化过快。
- **KL 惩罚像"不能忘本"**：奖励模型可能说"用更多感叹号！"，模型学到在每句话后面加 10 个感叹号——但这样回复变得很奇怪。KL 惩罚确保模型的语言风格不会偏离人类自然语言太远。
- **PPO 多 epoch 更新像"把经验用透"**：收集了一批经验（几个对话），然后不是用一次就扔掉，而是在这批数据上反复学习多次（类似有监督学习的多 epoch 训练）。但每次都要重新计算新旧策略的概率比，确保更新幅度在安全范围内。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class PPOTrainer:
    """简化的 LLM PPO 训练器"""
    def __init__(self, policy_model, ref_model, value_model, epsilon=0.2, kl_beta=0.1):
        self.policy = policy_model
        self.ref = ref_model
        self.value = value_model
        self.epsilon = epsilon
        self.kl_beta = kl_beta

    def compute_loss(self, input_ids, old_log_probs, advantages, returns):
        # 当前策略的 log probs
        outputs = self.policy(input_ids)
        logits = outputs.logits
        new_log_probs = -F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            input_ids.view(-1),
            reduction='none'
        ).view(input_ids.shape)

        # 重要性采样比率
        ratio = torch.exp(new_log_probs - old_log_probs)

        # Clipped 目标
        clip_ratio = torch.clamp(ratio, 1 - self.epsilon, 1 + self.epsilon)
        policy_loss = -torch.min(ratio * advantages, clip_ratio * advantages).mean()

        # KL 散度惩罚
        with torch.no_grad():
            ref_outputs = self.ref(input_ids)
            ref_log_probs = -F.cross_entropy(
                ref_outputs.logits.view(-1, ref_outputs.logits.size(-1)),
                input_ids.view(-1),
                reduction='none'
            ).view(input_ids.shape)
        kl_div = torch.exp(ref_log_probs) * (ref_log_probs - new_log_probs)
        kl_loss = kl_div.sum(-1).mean()

        # 价值函数损失
        values = self.value(input_ids).squeeze(-1)
        value_loss = F.mse_loss(values, returns)

        total_loss = policy_loss + 0.5 * value_loss + self.kl_beta * kl_loss
        return total_loss, policy_loss, value_loss, kl_loss

    def update(self, input_ids, old_log_probs, advantages, returns, optimizer):
        optimizer.zero_grad()
        total_loss, pl, vl, kl = self.compute_loss(
            input_ids, old_log_probs, advantages, returns
        )
        total_loss.backward()
        # 梯度裁剪——另一个重要的稳定性技巧
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), max_norm=1.0)
        optimizer.step()
        return pl.item(), vl.item(), kl.item()

# 模拟使用（需要实际模型）
print("PPO 训练器定义完成")
print("关键稳定性技巧:")
print("  1. Clipping (eps={})".format(0.2))
print("  2. KL 散度惩罚 (beta={})".format(0.1))
print("  3. GAE 优势估计")
print("  4. 梯度裁剪 (max_norm=1.0)")
print("  5. 学习率预热")
```

## 深度学习关联

- **RLHF 的标配算法**：PPO 是 RLHF 中使用的标准 RL 算法，InstructGPT/ChatGPT 的成功离不开 PPO 的稳定性保证。它使得在数十亿参数的模型上应用强化学习成为可能。
- **PPO 的替代方案**：近年出现了无需强化学习的替代方案——DPO (Direct Preference Optimization) 直接在偏好数据上优化，避免了 PPO 的复杂超参数调优。但 PPO 仍然在复杂对齐场景（如安全约束）中不可替代。
- **训练基础设施的挑战**：LLM 的 PPO 训练需要同时维护四个模型（策略、参考、奖励、价值），显存消耗巨大。混合精度训练、模型并行、LoRA 等技术被用来降低训练成本。
