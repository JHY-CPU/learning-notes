# 24_ALBERT：跨层参数共享与因式分解嵌入

## 核心概念
- **ALBERT (A Lite BERT)**：Google 于 2019 年提出，旨在降低 BERT 的参数量和内存占用，同时保持或提升性能。通过两种主要技术实现模型轻量化。
- **因式分解嵌入 (Factorized Embedding Parameterization)**：将词嵌入矩阵分解为两个小矩阵的乘积。$V \times E$ 分解为 $V \times H$ 和 $H \times E$（其中 $E \ll H$），大幅减少嵌入层参数量。
- **跨层参数共享 (Cross-layer Parameter Sharing)**：所有 Transformer 层的参数（注意力权重和 FFN 权重）完全共享。只有 12 层共享同一套参数，而非每层独立。
- **层间转换的学习**：参数共享并不意味着所有层的输出相同。每一层前的 LayerNorm 和残差连接提供了不同的上下文，使得不同层仍然可以学习到层次化的特征。
- **SOP 替代 NSP**：ALBERT 用 Sentence Order Prediction (SOP) 替代 NSP——判断两个句子的顺序是否正确，这对句子间关系的理解要求更高。
- **参数量 vs 计算量**：ALBERT 大幅减少了参数量（BERT-large 3.34 亿 vs ALBERT-xxl 2.23 亿在嵌入层），但计算量没有减少（因为层数不变、隐藏维度不变）。
- **性能对比**：ALBERT-xxl 在多项 GLUE 基准上超越了 BERT-large，但计算速度相同甚至更慢（因为参数共享需要更多步数收敛）。

## 数学推导
**因式分解嵌入**：
$$
\text{Embedding}(x) = \text{OneHot}(x) \times W_E \times W_H
$$

其中 $W_E \in \mathbb{R}^{V \times E}$ 是嵌入矩阵，$W_H \in \mathbb{R}^{E \times H}$ 是投影矩阵。参数量从 $V \times H$ 降低到 $V \times E + E \times H$。

当 $V=30000$, $H=1024$, $E=128$ 时：
- BERT：$30000 \times 1024 = 30.72M$ 参数
- ALBERT：$30000 \times 128 + 128 \times 1024 = 3.84M + 0.13M = 3.97M$ 参数，减少约 87%

**跨层参数共享**：第 $l$ 层的输出：
$$
X^l = \text{TransformerLayer}(X^{l-1}; \theta_{\text{shared}})
$$

所有 $L$ 层使用相同的参数 $\theta_{\text{shared}}$，因此总层参数量从 $L \times |\theta_{\text{layer}}|$ 减少到 $|\theta_{\text{layer}}|$。

## 直观理解
- **因式分解嵌入像"嵌入式压缩"**：想象一个大书柜（30000 本书 × 1024 页），直接管理很费空间。现在建一个中转书架（128 页容量），先整理到中转书架再放进大书柜。虽然多了一步，但主书柜容量大幅减小。
- **参数共享像"分身技巧"**：一个人（一套参数）在 12 个不同的房间（12 层）工作。虽然还是同一个人，但在每个房间面对不同的情境（前一层输出的上下文不同），工作的侧重和方式就自然分化了。
- **为什么参数共享有效**：Transformer 各层学习到的特征具有一定的相似性——低层学语法，高层学语义，相邻层之间差异并不大。共享参数相当于"强制复用"，避免了每层从头学习造成的参数冗余。

## 代码示例
```python
from transformers import AlbertTokenizer, AlbertForMaskedLM
import torch

# 加载 ALBERT 模型
tokenizer = AlbertTokenizer.from_pretrained('albert-base-v2')
model = AlbertForMaskedLM.from_pretrained('albert-base-v2')

# ALBERT 的参数量统计
total_params = sum(p.numel() for p in model.parameters())
print(f"ALBERT-base 总参数量: {total_params:,}")

# 对比相同配置 BERT 的理论参数量（仅估计嵌入层差异）
# BERT-base: V=30000, H=768
bert_embed_params = 30000 * 768  # 23,040,000
albert_embed_params = 30000 * 128 + 128 * 768  # 3,936,384 + 98,304 = 4,034,688
print(f"BERT-base 嵌入层: {bert_embed_params:,}")
print(f"ALBERT-base 嵌入层: {albert_embed_params:,}")
print(f"参数节省: {(1 - albert_embed_params / bert_embed_params) * 100:.1f}%")

# 使用 ALBERT 做 MLM 预测
text = "今天天气真[MASK]，适合出去散步。"
inputs = tokenizer(text, return_tensors='pt')

with torch.no_grad():
    outputs = model(**inputs)

mask_token_index = torch.where(inputs['input_ids'] == tokenizer.mask_token_id)[1]
mask_token_logits = outputs.logits[0, mask_token_index, :]
top_3_tokens = torch.topk(mask_token_logits, 3, dim=1).indices[0]

print(f"\n输入: {text}")
for token_id in top_3_tokens:
    print(f"  预测: {tokenizer.decode([token_id])}")
```

## 深度学习关联
- **模型压缩的先驱**：ALBERT 是预训练语言模型压缩的代表工作之一。其参数共享和分解思想影响了后续的 TinyBERT、MobileBERT 等模型压缩研究。
- **大模型的轻量化需求**：ALBERT 表明可以在保持深层架构的同时显著减少参数，这对大模型在资源受限环境中的部署具有重要参考价值。
- **SOP 任务的启发**：ALBERT 对 SOP 的成功验证表明，预训练任务的设计需要更关注"真正理解"而非"简单判别"，推动了 ELECTRA 的判别式预训练和 DeBERTa 的解码注意力等后续工作。
