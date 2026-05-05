# 65_Repetition Penalty 重复惩罚机制

## 核心概念

- **重复问题 (Repetition Problem)**：语言模型在长文本生成中容易陷入重复循环，如生成"我喜欢猫。我喜欢狗。我喜欢鱼。"或"I love you love you love you..."。
- **重复惩罚 (Repetition Penalty)**：在生成每个 token 时，对之前已经出现过的 token 的概率进行缩减（降低 logits），减少重复生成。参数 $\theta > 1.0$ 控制惩罚强度。
- **N-gram 惩罚**：除了 token 级别的重复惩罚，还有 n-gram 级别的重复避免——例如，禁止生成的 2-gram "love you" 再次出现，可以有效避免短语级别的重复。
- **惩罚实现方式**：在 logits 上施加惩罚：$\text{logits}[t] = \text{logits}[t] - \theta$ 对已出现的 token。或者在 softmax 前的 logits 上除以 $\theta$：$\text{logits}_{\text{penalized}}[t] = \text{logits}[t] / \theta$。
- **频率惩罚 vs 存在惩罚**：存在惩罚 (presence penalty) 只看 token 是否出现过（出现一次就惩罚），频率惩罚 (frequency penalty) 根据出现次数按比例惩罚。两者可以同时使用。
- **HuggingFace 的实现**：HuggingFace Transformers 的 `generate()` 方法支持 `repetition_penalty` 参数。OpenAI API 中对应 `frequency_penalty` 和 `presence_penalty`。
- **重复与模型大小的关系**：小模型更容易产生重复问题，大模型（如 GPT-3 175B）的重复问题相对较轻，但仍然存在。这可能是模型对训练数据中重复模式的"记忆"。
- **重复与上下文窗口**：在超长文本生成中，即使大模型也可能因为失去早期上下文而开始重复。KV cache 的无限推理能力在此处受限。

## 数学推导

**HuggingFace 风格的重复惩罚**：
$$
\text{logits}_{\text{penalized}}[i] = \begin{cases}
\text{logits}[i] / \theta & \text{if } i \in \text{generated\_ids} \\
\text{logits}[i] & \text{otherwise}
\end{cases}
$$

其中 $\theta$ 是惩罚参数，$\theta > 1$。

**OpenAI 风格的频率/存在惩罚**：
$$
\text{logits}_{\text{penalized}}[i] = \text{logits}[i] - \text{frequency\_penalty} \times \text{count}(i) - \text{presence\_penalty} \times \text{has\_appeared}(i)
$$

其中 $\text{count}(i)$ 是 token $i$ 已出现的次数，$\text{has\_appeared}(i) \in \{0, 1\}$ 表示是否出现过。

## 直观理解

- **重复惩罚像"对已经说过的话感到厌烦"**：与人对话时，如果对方反复说同样的词，你会感到不耐烦。重复惩罚赋予模型同样的"厌烦感"：已经出现的词会降低继续选择的概率。
- **存在惩罚 vs 频率惩罚**：存在惩罚像"说过一次就扣分"——无论说多少次惩罚固定。频率惩罚像"说的越多扣得越多"——每多说一次，下一次被选中的概率就进一步降低。频率惩罚更猛烈。
- **θ 的选择**：θ=1.0 是无惩罚。θ=1.2 是通常的有效范围。过高的惩罚（θ > 2.0）可能导致模型避开正常词汇，产生不自然的文本。
- **n-gram 重复避免像"不重复自己"**：除了词级别的重复，n-gram 避免确保模型不会说相同的短语。在生成代码或翻译时特别有用。

## 代码示例

```python
import torch
import torch.nn.functional as F

def apply_repetition_penalty(logits, generated_ids, penalty=1.2):
    """对已生成 token 施加重复惩罚"""
    if penalty <= 1.0:
        return logits
    penalized = logits.clone()
    for token_id in set(generated_ids):
        if penalized[token_id] < 0:
            penalized[token_id] *= penalty  # 负值乘以惩罚（变得更负）
        else:
            penalized[token_id] /= penalty  # 正值除以惩罚（变小）
    return penalized

def apply_frequency_penalty(logits, generated_ids, freq_penalty=0.5, presence_penalty=0.0):
    """OpenAI 风格的频率/存在惩罚"""
    penalized = logits.clone()
    token_counts = {}
    for token_id in generated_ids:
        token_counts[token_id] = token_counts.get(token_id, 0) + 1

    for token_id, count in token_counts.items():
        has_appeared = 1.0
        reduction = freq_penalty * count + presence_penalty * has_appeared
        penalized[token_id] -= reduction
    return penalized

# 演示重复惩罚的效果
vocab_size = 10
logits = torch.tensor([0.5, 1.5, 2.0, 0.8, 1.0, 0.3, 0.1, -0.2, -0.5, -1.0])

# 假设已经生成了 token [2, 2, 2, 7]（token 2 重复了 3 次）
generated_ids = [2, 2, 2, 7]

print("原始 logits:", logits)
print("原始概率:", F.softmax(logits, dim=-1))

# 重复惩罚
penalized_logits = apply_repetition_penalty(logits, generated_ids, penalty=1.2)
print("\n重复惩罚后:")
print(f"  Token 2 (已出现 3 次): {logits[2].item():.2f} -> {penalized_logits[2].item():.2f}")
print(f"  Token 7 (已出现 1 次): {logits[7].item():.2f} -> {penalized_logits[7].item():.2f}")
print(f"  Token 0 (未出现): {logits[0].item():.2f} -> {penalized_logits[0].item():.2f}")

# 频率惩罚
freq_penalized = apply_frequency_penalty(logits, generated_ids, freq_penalty=0.5)
print("\n频率惩罚后:")
print(f"  Token 2 (出现 3 次): {logits[2].item():.2f} -> {freq_penalized[2].item():.2f}")
print(f"  Token 7 (出现 1 次): {logits[7].item():.2f} -> {freq_penalized[7].item():.2f}")

# 使用 HuggingFace 风格的 generate 参数
print("\nHuggingFace generate() 中的重复惩罚参数:")
print("  model.generate(..., repetition_penalty=1.2)")
print("  model.generate(..., no_repeat_ngram_size=3)")
```

## 深度学习关联

- **长文本生成的必要工具**：重复惩罚是长文本生成中不可或缺的技术。在故事写作、长对话、代码生成等场景中，没有重复惩罚的模型几乎必然陷入重复循环。
- **OpenAI API 的标准参数**：ChatGPT/GPT-4 API 中提供了 `frequency_penalty` 和 `presence_penalty` 两个参数供用户调节。它们与 `temperature`、`top_p` 等一起构成了 LLM 生成的核心控制参数集。
- **防止重复的对抗性方法**：除了基于 logits 的惩罚，还有一些对抗性方法——如 Unlikelihood Training (Welleck et al., 2019) 在训练时直接优化减少重复 token 的概率，从根源上减轻重复问题。
