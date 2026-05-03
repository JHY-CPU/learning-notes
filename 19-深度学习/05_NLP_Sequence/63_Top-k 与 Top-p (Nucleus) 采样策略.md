# 63_Top-k 与 Top-p (Nucleus) 采样策略

## 核心概念

- **Top-k 采样**：在生成每个 token 时，只从概率最高的 $k$ 个 token 中采样（重归一化后），去除概率很低的"长尾" token，减少生成不合理内容的概率。
- **Top-p (Nucleus) 采样**：从累计概率达到阈值 $p$（如 0.9）的最小 token 集中采样。与 Top-k 不同，Top-p 动态调整候选集大小——分布尖锐时候选少，分布平坦时候选多。
- **采样 vs 贪婪**：采样引入随机性，使每次生成结果不同，增加创造性和多样性。但采样也增加了生成不连贯或荒谬内容的风险。
- **贪婪解码的局限**：贪婪解码总是选概率最高的词，可能导致"安全但无聊"的文本，容易陷入重复循环或错过有创意的表达。
- **温度参数的影响**：Temperature 与 Top-k/Top-p 结合使用——先做温度缩放，再应用 Top-k/Top-p 采样。高温使分布更均匀，低温使分布更尖锐。
- **平衡创造性与安全性**：Top-k/Top-p 是质量与多样性之间的平衡——过小的 $k$ 或过小的 $p$ 使生成过于保守，过大则让低质量 token 有被选中的风险。

## 数学推导

**Top-k 采样**：
给定概率分布 $P(v | \text{context})$，定义候选集 $V^{(k)}$ 为概率最高的 $k$ 个 token：

$$
V^{(k)} = \arg\max_{V' \subseteq V, |V'| = k} \sum_{v \in V'} P(v | \text{context})
$$

重归一化后采样：
$$
P'(v) = \begin{cases}
\frac{P(v)}{\sum_{u \in V^{(k)}} P(u)} & \text{if } v \in V^{(k)} \\
0 & \text{otherwise}
\end{cases}
$$

**Top-p (Nucleus) 采样**：
选择满足条件的最小候选集 $V^{(p)}$：
$$
V^{(p)} = \text{the smallest set s.t. } \sum_{v \in V^{(p)}} P(v) \geq p
$$

同样重归一化后从 $V^{(p)}$ 中采样。

**两者的关系**：
- Top-k 固定候选数量，适合候选集大小大致稳定时
- Top-p 固定概率质量，适应分布的动态变化——在分布尖锐时自动缩小候选集（减少噪声），分布平坦时自动扩大候选集（增加多样性）

## 直观理解

- **Top-k 像"从最优秀的前 k 人中随机选"**：面试候选人，要求只从前 10 名中随机选一个。这确保了不会选到不合格的，但候选人数固定——如果前 10 名实力相当，合理；如果前 3 名就足够了，后 7 名就是不必要的噪声。
- **Top-p 像"从累计分数达标的人中随机选"**：不固定人数，而是定一个"能力阈值"——前几名的能力总和达到总能力的 90% 就停止。如果只有 3 个人就贡献了 90% 的能力，就只从 3 人中选；如果需要 20 人才累积到 90%，就从 20 人中选。这更加灵活。
- **$p=0.9$ 的含义**：意思是在 90% 的情况下你会选择合理的 token，10% 的情况下（概率质量尾部）被排除。这些尾部 token 通常是不合适的选择。
- **为什么需要采样而不是贪心**：想象写诗——贪心的话每次选最"安全"的词，结果可能平淡无奇。采样（即使是 Top-p 采样）允许偶尔选一个"意料之外"的词，增加文本的创造性和自然性。

## 代码示例

```python
import torch
import torch.nn.functional as F
import numpy as np

def top_k_sampling(logits, k=50):
    """Top-k 采样"""
    top_k_values, top_k_indices = torch.topk(logits, k, dim=-1)
    # 将非 top-k 的值设为 -inf
    probs = F.softmax(logits, dim=-1)
    filtered = torch.zeros_like(probs)
    filtered.scatter_(-1, top_k_indices, probs.gather(-1, top_k_indices))
    # 重归一化
    filtered = filtered / filtered.sum(dim=-1, keepdim=True)
    return filtered

def top_p_sampling(logits, p=0.9):
    """Top-p (Nucleus) 采样"""
    probs = F.softmax(logits, dim=-1)
    sorted_probs, sorted_indices = torch.sort(probs, descending=True, dim=-1)
    cumsum = torch.cumsum(sorted_probs, dim=-1)
    
    # 找到累计概率超过 p 的位置
    mask = cumsum - sorted_probs > p
    sorted_probs[mask] = 0.0
    
    # 重归一化
    sorted_probs = sorted_probs / sorted_probs.sum(dim=-1, keepdim=True)
    return sorted_probs, sorted_indices

# 演示两种采样策略的效果
vocab_size = 1000
batch_size = 1

# 模拟两种不同类型的概率分布
sharp_dist = torch.zeros(1, vocab_size)
sharp_dist[0, 0] = 5.0
sharp_dist[0, 1] = 4.5
sharp_dist[0, 2] = 4.0  # 只有 3 个 token 有显著概率

flat_dist = torch.randn(1, vocab_size) * 0.5  # 分布平坦

print("Sharp 分布 (只有 3 个高概率 token):")
top_k_probs = top_k_sampling(sharp_dist, k=50)
top_p_probs_sorted, _ = top_p_sampling(sharp_dist, p=0.9)
nonzero_k = (top_k_probs > 0).sum().item()
nonzero_p = (top_p_probs_sorted > 0).sum().item()
print(f"  Top-k (k=50): 候选中 {nonzero_k} 个非零概率 token")
print(f"  Top-p (p=0.9): 候选中 {nonzero_p} 个非零概率 token")
print(f"  => Top-p 自动缩小候选集，减少噪声")

print("\nFlat 分布 (所有 token 概率相近):")
top_k_probs_flat = top_k_sampling(flat_dist, k=50)
top_p_flat_sorted, _ = top_p_sampling(flat_dist, p=0.9)
nonzero_k_flat = (top_k_probs_flat > 0).sum().item()
nonzero_p_flat = (top_p_flat_sorted > 0).sum().item()
print(f"  Top-k (k=50): 候选中 {nonzero_k_flat} 个非零概率 token")
print(f"  Top-p (p=0.9): 候选中 {nonzero_p_flat} 个非零概率 token")
print(f"  => Top-p 自动扩大候选集，增加多样性")
```

## 深度学习关联

- **GPT-2 和 GPT-3 的默认参数**：GPT-2 默认使用 Top-k=40 采样，而 GPT-3 和后续模型通常默认使用 Top-p=0.9 或组合使用两者（先 Top-k 再 Top-p）。HuggingFace Transformers 的 generate 方法默认使用 Top-p=0.9。
- **文本生成的质量控制**：Top-k/Top-p 是大语言模型中最重要的质量控制参数之一。在对话场景（ChatGPT）中通常使用采样以追求自然性，在事实性场景（如摘要）中通常使用束搜索或低温度采样以追求准确性。
- **采样的理论分析**：核采样（Top-p）的论文 (Holtzman et al., 2019) 通过信息论分析证明，自然语言中 token 的分布高度动态——有些位置只有几个合理选择（低熵），有些位置有很多合理选择（高熵）。Top-p 自适应地处理了这种动态性。
