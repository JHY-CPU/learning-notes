# 61_机器翻译中的 Transformer 实现细节

## 核心概念
- **Transformer 在机器翻译中的应用**：Vaswani et al. (2017) 最初提出 Transformer 时就是在 WMT 2014 英-德翻译任务上验证的。它使用 Encoder-Decoder 架构，编码器读取源语言，解码器生成目标语言。
- **词嵌入 + 位置编码**：输入 token 通过 Embedding 层映射为向量，然后与位置编码逐元素相加。缩放因子 $\sqrt{d_{\text{model}}}$ 被应用于嵌入，使得嵌入的方差与位置编码匹配。
- **标签平滑 (Label Smoothing)**：在训练中，Transformer 使用标签平滑（$\epsilon_{ls} = 0.1$）替代硬标签。将目标分布从 one-hot 变为 soft label（$1 - \epsilon_{ls}$ 分配给正确类别，$\epsilon_{ls}/(K-1)$ 分配给其他类别），提升困惑度和 BLEU。
- **学习率调度 (Learning Rate Schedule)**：使用"预热 + 衰减"的调度策略——前 $warmup\_steps$ 步学习率线性上升，之后按步数的倒数平方根衰减：
  $$
  lr = d_{\text{model}}^{-0.5} \cdot \min(step\_num^{-0.5}, step\_num \cdot warmup\_steps^{-1.5})
  $$
- **Adam 优化器参数**：使用 $\beta_1 = 0.9$，$\beta_2 = 0.98$，$\epsilon = 10^{-9}$。相比默认的 $\beta_2 = 0.999$，更小的 $\beta_2$ 使 SGD 动量衰减更快，更适合 Transformer 的训练。
- **Dropout**：在每一个子层的输出上应用 Dropout（$P_{drop}=0.1$），同时在注意力权重和嵌入层也使用 Dropout。
- **推理时的翻译**：使用束搜索（beam size = 4-12）解码，长度惩罚 (length penalty) 平衡翻译质量和长度。
- **批量翻译 (Batch Translation)**：GPU 推理时，将多个句子组成 batch 同时翻译，但处理变长序列需要填充 (padding)。填充的 token 通过注意力 mask 忽略。

## 数学推导
**Transformer 在翻译任务中的完整信息流**：

源句子 $\mathbf{x} = (x_1, \ldots, x_m)$ 编码为上下文表示 $H$：
$$
H = \text{Encoder}(\text{Embed}(x) + PE(x))
$$

目标句子 $\mathbf{y} = (y_1, \ldots, y_n)$ 的自回归生成：
$$
P(\mathbf{y} | \mathbf{x}) = \prod_{t=1}^{n} P(y_t | y_{<t}, H)
$$

$$
P(y_t | y_{<t}, H) = \text{softmax}(W_o \cdot \text{DecoderLayer}_t + b_o)
$$

**标签平滑**：硬标签分布 $q(k) = \delta_{k, y}$ 替换为：
$$
q'(k) = (1 - \epsilon_{ls}) \delta_{k, y} + \frac{\epsilon_{ls}}{K - 1}(1 - \delta_{k, y})
$$

交叉熵损失：
$$
\mathcal{L} = -\sum_{k=1}^{K} q'(k) \log P(k)
$$

## 直观理解
- **词嵌入缩放**：嵌入向量的方差随维度增大而变化，$\sqrt{d_{model}}$ 缩放使嵌入和位置编码的"信号强度"匹配，就像调整音响的高低音音量一致。
- **标签平滑的哲学**：不给模型 100% 的确信度，而是告诉它"即使正确答案是 'the'，有 5% 的可能性也是 'a' 或 'an'"。这阻止了模型过度自信（overconfident），使输出概率分布更加合理。
- **学习率预热的原因**：模型刚开始训练时参数是随机的，如果直接用大学习率可能导致"原地打转"。预热让模型先用小步探索方向，稳定后再大步前进。

## 代码示例
```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class LabelSmoothing(nn.Module):
    """标签平滑"""
    def __init__(self, smoothing=0.1):
        super().__init__()
        self.smoothing = smoothing

    def forward(self, logits, target):
        # logits: (batch, vocab), target: (batch,)
        vocab_size = logits.size(-1)
        confidence = 1.0 - self.smoothing
        # 构造平滑标签分布
        smooth_label = torch.full_like(logits, self.smoothing / (vocab_size - 1))
        smooth_label.scatter_(1, target.unsqueeze(1), confidence)
        # 交叉熵
        log_probs = F.log_softmax(logits, dim=-1)
        return -(smooth_label * log_probs).sum(dim=1).mean()

class TransformerLearningRateScheduler:
    """Transformer 学习率调度器"""
    def __init__(self, optimizer, d_model=512, warmup_steps=4000):
        self.optimizer = optimizer
        self.d_model = d_model
        self.warmup_steps = warmup_steps
        self.step_num = 0

    def step(self):
        self.step_num += 1
        lr = (self.d_model ** -0.5) * min(
            self.step_num ** -0.5,
            self.step_num * (self.warmup_steps ** -1.5)
        )
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
        return lr

# 模拟调度器
optimizer = torch.optim.AdamW([torch.nn.Parameter(torch.randn(5, 5))])
scheduler = TransformerLearningRateScheduler(optimizer, d_model=512, warmup_steps=4000)

lrs = []
for _ in range(20000):
    lr = scheduler.step()
    lrs.append(lr)

import matplotlib.pyplot as plt
print(f"最大学习率: {max(lrs):.6f}")
print(f"最终学习率: {lrs[-1]:.6f}")
print(f"预热在 {scheduler.warmup_steps} 步后结束")

# WMT 翻译任务中的典型配置
translation_config = {
    "batch_size": 2048,      # token 级 batch size
    "d_model": 512,          # 模型维度
    "n_layers": 6,           # 编码器/解码器层数
    "n_heads": 8,            # 注意力头数
    "d_ff": 2048,            # FFN 隐藏层维度
    "dropout": 0.1,          # Dropout 率
    "label_smoothing": 0.1,  # 标签平滑
    "warmup_steps": 4000,    # 预热步数
    "beam_size": 4,          # 束搜索宽度
    "length_penalty": 0.6,   # 长度惩罚
    "max_tokens": 100,       # 最大翻译长度
}
print(f"\nTransformer 翻译配置: {translation_config}")
```

## 深度学习关联
- **机器翻译的 SOTA 演变**：在 Transformer 出现之前，机器翻译由基于 RNN 的 Seq2Seq+注意力模型主导。Transformer 以显著优势超越 RNN（WMT 英德翻译 BLEU 从 26.8 提升到 28.4），从此成为机器翻译的标准架构。
- **预训练时代的翻译**：mBART、M2M-100 等预训练多语言模型在机器翻译上取得了进一步的突破。M2M-100 使用 Encoder-Decoder Transformer 在 100 种语言上预训练，实现了任意语言对之间的直接翻译。
- **大模型时代的机器翻译**：GPT-4 等大规模语言模型在翻译任务上展现出了强大能力，尤其在低资源语言、翻译变体（风格翻译、术语保留）等方面超越了传统的机器翻译系统。
