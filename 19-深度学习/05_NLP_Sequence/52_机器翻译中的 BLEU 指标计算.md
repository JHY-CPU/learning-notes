# 52_机器翻译中的 BLEU 指标计算

## 核心概念

- **BLEU (Bilingual Evaluation Understudy)**：由 Papineni et al. (2002) 提出，通过比较机器翻译输出与人工参考翻译的 n-gram 重合度来评估翻译质量。分数范围为 0-100（或 0-1）。
- **n-gram 精确率 (Precision)**：计算机器翻译中出现的 n-gram 在参考翻译中出现的比例。分别计算 unigram（1-gram）到 4-gram 的精确率。
- **裁剪计数 (Clipped Count)**：对于每个 n-gram，其在机器翻译中的计数被裁剪到不超过在任意单个参考翻译中的最大出现次数。防止翻译反复使用同一短语刷分。
- **简短惩罚 (Brevity Penalty, BP)**：当机器翻译长度短于参考翻译时施加的惩罚。防止翻译只输出高精确率的短片段。
- **多个参考翻译**：BLEU 通常使用多个（通常 4 个）人工参考翻译，以覆盖合理的翻译多样性。
- **加权几何平均**：BLEU 对 1-4 的 n-gram 精确率取加权几何平均，通常使用均匀权重。
- **BLEU 的局限**：只衡量词汇重合度，不衡量语义正确性；对同义词和合理意译不敏感；对语序变化不敏感（仅通过 n-gram 隐式考虑）。

## 数学推导

**n-gram 精确率**（带裁剪）：
$$
P_n = \frac{\sum_{C \in \text{candidates}} \sum_{\text{n-gram} \in C} \text{Count}_{\text{clip}}(\text{n-gram})}{\sum_{C \in \text{candidates}} \sum_{\text{n-gram} \in C} \text{Count}(\text{n-gram})}
$$

其中 $\text{Count}_{\text{clip}}(g) = \min(\text{Count}_{\text{machine}}(g), \max_{\text{ref}} \text{Count}_{\text{ref}}(g))$

**简短惩罚**：
$$
BP = \begin{cases}
1 & \text{if } c > r \\
e^{1 - r/c} & \text{if } c \leq r
\end{cases}
$$

其中 $c$ 是机器翻译总长度，$r$ 是参考翻译有效长度（当有多个参考时，选择最接近 $c$ 的参考长度）。

**BLEU 分数**：
$$
\text{BLEU} = BP \cdot \exp\left(\sum_{n=1}^{N} w_n \log P_n\right)
$$

通常 $N = 4$，$w_n = 1/4$（均匀权重）。

## 直观理解

- **BLEU 像"选择题匹配度测试"**：你写的答案（机器翻译）和标准答案（参考翻译）对比，检查使用了多少相同的"词块"（n-gram）。如果用了相同的四个词连在一起的片段（4-gram），说明翻译得非常准确。但 BLEU 只检查"词对不对"，不检查"意思对不对"。
- **裁剪计分的必要性**：如果机器翻译输出"the the the the the"，其中 unigram "the" 精确率高达 100%（因为参考中可能也包含"the"），但这不是一个好翻译。裁剪确保每个词最多被算与参考中出现次数一样多。
- **长度惩罚的意义**：机器翻译可以用"打游击"的策略——只翻译自己有把握的短片段，跳过难的词，这样精确率高但信息不完整。长度惩罚确保翻译的完整性。
- **BLEU 的可靠性争议**：BLEU 高不一定代表翻译质量好（可以用与参考高度重合但语义不通的句子作弊），但 BLEU 低基本说明翻译有问题。

## 代码示例

```python
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
import numpy as np

# 参考翻译（通常多个）
reference = [
    "The cat is sitting on the mat.",
    "There is a cat sitting on the mat.",
    "A cat sits on the mat."
]
reference_tokens = [r.lower().split() for r in reference]

# 候选翻译
candidate = "the cat sat on the mat"
candidate_tokens = candidate.lower().split()

# 计算 BLEU
smoothie = SmoothingFunction().method4  # 平滑处理避免 0 分

bleu_1 = sentence_bleu(reference_tokens, candidate_tokens,
                       weights=(1, 0, 0, 0), smoothing_function=smoothie)
bleu_2 = sentence_bleu(reference_tokens, candidate_tokens,
                       weights=(0.5, 0.5, 0, 0), smoothing_function=smoothie)
bleu_4 = sentence_bleu(reference_tokens, candidate_tokens,
                       weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothie)

print(f"候选: {candidate}")
print(f"参考: {reference[0]}")
print(f"BLEU-1: {bleu_1:.4f}")
print(f"BLEU-2: {bleu_2:.4f}")
print(f"BLEU-4: {bleu_4:.4f}")

# 手动 BLEU 计算（理解原理）
def manual_bleu(references, candidate, n=4):
    ref_tokens = [r.lower().split() for r in references]
    cand_tokens = candidate.lower().split()

    precisions = []
    for n_gram in range(1, n + 1):
        cand_ngrams = {}
        for i in range(len(cand_tokens) - n_gram + 1):
            ng = tuple(cand_tokens[i:i+n_gram])
            cand_ngrams[ng] = cand_ngrams.get(ng, 0) + 1

        # 裁剪计数
        clip_count = 0
        for ng, count in cand_ngrams.items():
            max_ref_count = max(
                sum(1 for j in range(len(r) - n_gram + 1)
                    if tuple(r[j:j+n_gram]) == ng)
                for r in ref_tokens
            )
            clip_count += min(count, max_ref_count)

        total_count = sum(cand_ngrams.values())
        precisions.append(clip_count / max(total_count, 1))

    # 简短惩罚
    c = len(cand_tokens)
    r = min(ref_tokens, key=lambda x: abs(len(x) - c))
    r_len = len(r)
    bp = 1 if c > r_len else np.exp(1 - r_len / max(c, 1))

    # BLEU
    if any(p == 0 for p in precisions):
        return 0.0
    geo_mean = np.exp(np.mean(np.log(precisions)))
    return bp * geo_mean

manual = manual_bleu(reference, candidate, n=4)
print(f"手动 BLEU-4: {manual:.4f}")
```

## 深度学习关联

- **机器翻译评估的基石**：尽管 BLEU 有各种局限，它仍是机器翻译最广泛使用的自动评估指标。在 WMT（机器翻译研讨会）中 BLEU 是强制报告的指标。
- **BLEU 的替代与补充**：BLEU 的局限催生了更多评估指标——METEOR（引入同义词匹配）、TER（编辑距离）、chrF（字符级 F 值）、COMET（基于预训练模型的学习型指标）。
- **大模型时代的翻译评估**：GPT-4 等大模型的翻译质量已经超越了传统 BLEU 的区分能力。对于大模型生成的翻译，人类评估或 LLM-as-judge（使用 GPT-4 给翻译打分）变得更为常见。
