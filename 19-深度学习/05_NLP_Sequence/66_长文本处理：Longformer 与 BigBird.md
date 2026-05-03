# 66_长文本处理：Longformer 与 BigBird

## 核心概念

- **长文本处理的挑战**：标准 Transformer 的自注意力复杂度为 $O(n^2)$，在长序列（如 4096+ tokens）上计算和显存开销过大。长文本处理（文档级理解、书籍摘要、法律文本分析）需要更高效的注意力机制。
- **Longformer (Beltagy et al., 2020)**：使用滑动窗口注意力（Sliding Window Attention）替代全局注意力。每个 token 只关注窗口内的局部邻居，复杂度降至 $O(n \times w)$。
- **BigBird (Zaheer et al., 2020)**：结合三种注意力模式——滑动窗口注意力、全局注意力（特殊 token 关注所有位置）、随机注意力（每个 token 关注少量随机位置）。图论保证信息流通。
- **Longformer 的注意力模式**：
  - 滑动窗口（局部）：每个 token 关注左右 $w$ 个邻居
  - 扩张窗口（可选）：与空洞卷积类似，间隔性地跳过一些位置以扩大感受野
  - 全局注意力（可选）：特殊 token（如 [CLS]）关注所有位置
- **BigBird 的图论保证**：BigBird 的注意力图是"扩展图"(expander graph)，任意两个节点之间的信息通过 $O(\log n)$ 步可达。这保证了长距离信息可以有效传播。
- **位置编码**：Longformer 使用位置编码的扩展版本（在较长序列上外推位置编码），BigBird 使用门控随机位置编码。
- **预训练与微调**：Longformer 和 BigBird 在预训练时就使用长上下文（4096 token），然后在长文本任务上微调。
- **与 Longformer 兼容的预训练**：Longformer 使用 RoBERTa 的权重初始化，逐步从 512 扩展到 4096 的上下文窗口。BigBird 从零开始预训练。

## 数学推导

**Longformer 的滑动窗口注意力**（窗口大小 $w$）：
对于位置 $i$，其可关注的位置集合为：
$$
N(i) = \{j: |i - j| \leq w/2\} \cup \{\text{global tokens}\}
$$

注意力计算：
$$
\text{Attention}_i = \sum_{j \in N(i)} \text{softmax}\left(\frac{q_i \cdot k_j}{\sqrt{d_k}}\right) v_j
$$

复杂度：$O(n \cdot w)$，其中 $w \ll n$。

**BigBird 的注意力组合**：
$$
\text{BigBirdAttn}(x) = \text{softmax}\left(\frac{QK^\top}{\sqrt{d_k}} + M_{\text{bigbird}}\right)V
$$

其中掩码 $M_{\text{bigbird}}$ 包含三种模式：
- 滑动窗口：$M_{\text{swa}}(i, j) = 0$ if $|i-j| \leq w/2$
- 全局：$M_{\text{global}}(i, j) = 0$ 对全局 token $i$ 的所有 $j$
- 随机：$M_{\text{random}}(i, j) = 0$ 对随机选择的 $j$

**理论保证**：BigBird 的注意力图是 $(r, \epsilon)$-扩展图，任意节点的信息可在 $O(\log n)$ 步内传播到图中任意节点。

## 直观理解

- **Longformer 像"戴着望远镜的局部阅读"**：主要关注眼前几行（滑动窗口），但关键位置（全局 token）可以总览全文。就像你阅读合同——大部分时候逐行读，但需要时抬头看整体结构。
- **BigBird 像"三种社交方式"**：你经常和邻居聊天（滑动窗口），偶尔和城市里的名人说话（全局 token），还通过随机朋友认识远方的陌生人（随机注意力）。这确保了你虽然不直接认识所有人，但信息可以通过网络传遍整个城市。
- **长文本 vs 普通文本的效率差异**：对于 4096 长度的序列，标准注意力需要计算 1600 万对；Longformer（$w=128$）只需计算约 50 万对——节省 30 多倍的计算量，而且理论上可以处理 32000+ 的长度。

## 代码示例

```python
from transformers import LongformerModel, LongformerTokenizer, BigBirdModel, BigBirdTokenizer
import torch

# Longformer 加载和基本使用
tokenizer = LongformerTokenizer.from_pretrained('allenai/longformer-base-4096')
model = LongformerModel.from_pretrained('allenai/longformer-base-4096')

# 超长文本（2048 tokens 以上）
long_text = "This is a very long text. " * 200  # 约 2000 tokens
inputs = tokenizer(long_text, return_tensors='pt', max_length=2048, truncation=True)

# Longformer 使用全局注意力关注 [CLS] token
global_attention_mask = torch.zeros(inputs['input_ids'].shape, dtype=torch.long)
global_attention_mask[:, 0] = 1  # [CLS] 使用全局注意力

with torch.no_grad():
    outputs = model(**inputs, global_attention_mask=global_attention_mask)

print(f"Longformer 输入长度: {inputs['input_ids'].shape[1]}")
print(f"Longformer 输出形状: {outputs.last_hidden_state.shape}")

# 标准注意力 vs 滑动窗口注意力复杂度对比
def complexity_comparison(seq_len, window_size=128):
    standard = seq_len ** 2
    longformer = seq_len * window_size
    bigbird = seq_len * (window_size + 2 + 3)  # 滑动窗口 + 2 个全局 + 3 个随机
    print(f"\n序列长度 {seq_len}:")
    print(f"  标准注意力计算量: {standard:,}")
    print(f"  Longformer (w={window_size}): {longformer:,} ({standard / longformer:.1f}x 提升)")
    print(f"  BigBird (w={window_size}+g=2+r=3): {bigbird:,} ({standard / bigbird:.1f}x 提升)")

complexity_comparison(4096)
complexity_comparison(8192)
complexity_comparison(16384)
```

## 深度学习关联

- **长文本 BERT 的实用化**：Longformer 和 BigBird 使得 BERT 可以在文档级、法律文本、科学论文等长文本场景中应用。Longformer 在 PubMed（生物医学文献）和 BigBird 在长文档 QA 任务上取得了显著成果。
- **与 GPT 系列的关系**：GPT-3 等 Decoder-only 模型也面临长文本挑战。GPT-3 的 4K 上下文通过稀疏注意力扩展到 32K (GPT-4)。GPT-4 据推测使用了混合的专家注意力和稀疏化策略。
- **注意力优化的谱系**：Longformer/BigBird 是注意力优化的谱系中重要的一环，串联了稀疏注意力和 FlashAttention 的发展路径。后来 FlashAttention 通过 IO 优化解决了注意力的另一个瓶颈，两者可以互补使用。
