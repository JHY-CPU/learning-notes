# 62_文本生成中的 Beam Search 算法

## 核心概念

- **Beam Search (束搜索)**：在序列生成任务（机器翻译、文本摘要、图像描述）中，每一步保留 $B$ 个（束宽）最高概率的部分序列（hypotheses），而非贪婪地只选一个最佳 token。在搜索质量和计算成本之间取得平衡。
- **束宽 (Beam Size)**：$B$ 是每一步保留的候选序列数。$B=1$ 等价于贪婪解码，$B=10$ 表示保留 10 个最佳候选。束宽越大搜索质量越好，但计算成本线性增长。
- **序列概率**：生成的序列 $\mathbf{y} = (y_1, \ldots, y_T)$ 的概率为各步条件概率的乘积：$P(\mathbf{y}) = \prod_{t=1}^{T} P(y_t | y_{<t}, \text{context})$。
- **束搜索流程**：每步从当前的 $B$ 个候选的扩展中（每个候选中选择 $|V|$ 个可能的下一 token），保留总概率最高的 $B$ 个新候选。
- **长度归一化 (Length Normalization)**：概率乘积随序列长度指数衰减，长序列天然得分更低。使用长度归一化使束搜索更公平：$\frac{1}{L^\alpha} \log P(\mathbf{y})$，或使用 $(\alpha + L)^\beta / (1 + \beta)^\alpha$ 等。
- **停止条件**：当生成到最大长度或所有候选都产生了 [EOS] 标记时停止。通常会在生成长度达到预设最大值时强制结束。
- **与贪婪解码对比**：贪婪解码每一步只选一个最优，可能错过局部次优但全局更好的路径。束搜索通过保留多条路径分摊风险。

## 数学推导

**标准束搜索**：
设第 $t$ 步有 $B$ 个候选序列 $\{\mathbf{y}^{(1)}_t, \ldots, \mathbf{y}^{(B)}_t\}$，每个候选的得分为：
$$
\text{score}(\mathbf{y}^{(b)}_t) = \log P(\mathbf{y}^{(b)}_t) = \sum_{i=1}^{t} \log P(y_i^{(b)} | y_{<i}^{(b)})
$$

在第 $t+1$ 步，每个候选扩展所有可能的 $|V|$ 个 token，共得到 $B \times |V|$ 个新序列：
$$
\text{score}(\mathbf{y}^{(b)}_t, v) = \text{score}(\mathbf{y}^{(b)}_t) + \log P(v | \mathbf{y}^{(b)}_t)
$$

选择其中得分最高的 $B$ 个作为新的候选集。

**长度惩罚**（Wu et al., 2016）：
$$
\text{score}_{\text{norm}}(\mathbf{y}) = \frac{\text{score}(\mathbf{y})}{LP(\mathbf{y})}
$$

$$
LP(\mathbf{y}) = \frac{(5 + |\mathbf{y}|)^\alpha}{(5 + 1)^\alpha}
$$

其中 $\alpha$ 是长度惩罚参数（$\alpha=1$ 完全归一化，$\alpha=0$ 无惩罚）。

## 直观理解

- **Beam Search 像"多条路同时探索"**：从起点出发，不是只走一条最好走的路（贪婪解码），而是同时走 B 条看起来最有希望的路。虽然每条路走的步数相同，但某条路可能前期看起来一般，后期发现是捷径。
- **贪婪 vs 束搜索**：贪婪是"只顾眼前"——选当前概率最高的词。比如 "The cat" 下，$P(\text{sat}) = 0.4$, $P(\text{ran}) = 0.35$, $P(\text{ate}) = 0.25$，贪婪选 "sat"。但实际上"the cat ate the mouse" 整体概率可能更高——束搜索保留 "ate" 作为另一个候选。
- **束宽的选择**：$B=4$ 到 $B=8$ 在翻译中通常就够了。更大的束宽（如 $B=100$）可能带来以下问题：①计算的线性增加；②翻译可能变得过于"安全"（短而常见的翻译）；③长序列可能占优势（所以需要长度惩罚）。
- **长度惩罚的直觉**：概率乘积意味着长句子的概率天然小于短句——就像连乘概率 0.5^10 > 0.5^20。长度惩罚通过除以长度相关因子来纠正这种偏差。

## 代码示例

```python
import torch
import torch.nn.functional as F
import math

def beam_search(model, input_ids, beam_size=4, max_len=50, alpha=0.6):
    """简化的束搜索实现"""
    batch_size = input_ids.size(0)
    device = input_ids.device
    vocab_size = model.config.vocab_size

    # 初始候选
    sequences = [[[input_ids[0].tolist()], 0.0]]  # [(tokens, score), ...]

    for step in range(max_len):
        all_candidates = []
        for seq_tokens, score in sequences:
            if seq_tokens[-1] == model.config.eos_token_id:
                # 已结束的序列保留
                all_candidates.append((seq_tokens, score))
                continue

            # 编码当前序列
            input_tensor = torch.tensor(seq_tokens, device=device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                logits = outputs.logits[0, -1, :]  # 最后一个位置的 logits
                probs = F.log_softmax(logits, dim=-1)

            # 扩展下一个 token
            topk_probs, topk_indices = torch.topk(probs, beam_size)
            for i in range(beam_size):
                new_tokens = seq_tokens + [topk_indices[i].item()]
                new_score = score + topk_probs[i].item()
                all_candidates.append((new_tokens, new_score))

        # 选择 top beam_size 个候选
        # 使用长度归一化
        def length_penalty(tokens, score, alpha=0.6):
            if len(tokens) == 0:
                return score
            return score / ((5 + len(tokens)) ** alpha / (5 + 1) ** alpha)

        all_candidates.sort(
            key=lambda x: length_penalty(x[0], x[1], alpha),
            reverse=True
        )
        sequences = all_candidates[:beam_size]

        # 检查是否所有序列都已结束
        if all(seq[-1] == model.config.eos_token_id for seq, _ in sequences):
            break

    # 返回得分最高的序列
    best_seq = max(sequences, key=lambda x: length_penalty(x[0], x[1], alpha))
    return best_seq[0]

# 演示束搜索（使用模拟 logits）
class MockModel:
    def __init__(self, config):
        self.config = config

    def __call__(self, input_ids):
        class Output:
            pass
        output = Output()
        vocab_size = self.config.vocab_size
        batch, seq = input_ids.shape
        # 模拟 logits
        logits = torch.randn(batch, seq, vocab_size) * 0.5
        output.logits = logits
        return output

class Config:
    def __init__(self):
        self.vocab_size = 100
        self.eos_token_id = 2

print("束搜索示例:")
print(f"  束宽 (beam_size): 4")
print(f"  最大生成长度: 50")
print(f"  长度惩罚 alpha: 0.6")
print(f"\n束搜索 vs 贪婪解码:")
print(f"  贪婪: 每一步只选概率最高的 token -> 可能错过全局最优")
print(f"  束搜索: 保留 {4} 个候选路径 -> 在质量和效率间平衡")
print(f"\n长度惩罚的作用:")
print(f"  不加惩罚: 束搜索倾向于产生更短的序列")
print(f"  加惩罚 (LP): 长度不抑制选择，公平比较长短序列")
```

## 深度学习关联

- **生成任务的标准解码算法**：束搜索是机器翻译、文本摘要、图像描述等几乎所有生成任务的标配解码算法。尽管大语言模型更倾向于使用采样策略，束搜索在需要确定性和高质量输出的场景中仍是首选。
- **KV cache 加速束搜索**：在束搜索中，多个候选序列共享大部分前缀。通过 KV cache 缓存已计算的 Key 和 Value 矩阵，避免重复计算，可以显著加速束搜索。
- **多样化束搜索 (Diverse Beam Search)**：标准束搜索中，不同候选可能非常相似。多样化的束搜索通过引入分组和差异惩罚，鼓励候选之间的多样性，提升输出的整体质量。
