# 42_P-Tuning v2：连续提示优化

## 核心概念

- **P-Tuning v2**：由 Liu et al. (2021) 提出，是对 P-Tuning v1 和 Prefix Tuning 的改进和统一。通过引入可训练的连续提示嵌入（continuous prompt embeddings），在模型输入的每一层注入可学习参数。
- **连续提示 (Continuous Prompt)**：不同于离散提示（人工编写的文字提示），连续提示是在嵌入空间的连续向量。这些向量由模型自动优化，不需要人工设计提示措辞。
- **深度提示 (Deep Prompting)**：P-Tuning v2 在各种 Transformer 层（不仅是输入嵌入层）都插入可训练的提示向量。相比之下，P-Tuning v1 只在输入层插入提示。
- **多层前缀 (Multi-layer Prefix)**：在每层的 Key 和 Value 之前添加可训练的前缀向量，这些前缀向量是层特定的（每层有自己的前缀）。
- **分类头微调**：与 P-Tuning v1 不同，P-Tuning v2 也微调任务的分类头（如果存在），这提升了序列标注等任务的性能。
- **任务类型泛化**：P-Tuning v2 在序列标注、关系抽取、文本分类等多种任务上超越了 P-Tuning v1 和 Prefix Tuning，并在小模型和大模型上都有效。
- **参数量控制**：可训练参数量通常为模型总参数的 0.1%-3%，与 LoRA 同属参数高效微调范畴。

## 数学推导

对于 Transformer 第 $l$ 层，输入隐藏状态 $H^l \in \mathbb{R}^{n \times d}$，前缀长度为 $m$。

在自注意力层中，拼接前缀到 Key 和 Value：
$$
K^l = [P_k^l; H^l W_K^l], \quad V^l = [P_v^l; H^l W_V^l]
$$

其中 $P_k^l, P_v^l \in \mathbb{R}^{m \times d}$ 是可训练的前缀参数。

注意力计算变为：
$$
\text{Attn}(Q^l, K^l, V^l) = \text{softmax}\left(\frac{Q^l [P_k^l; H^l W_K^l]^\top}{\sqrt{d_k}}\right) [P_v^l; H^l W_V^l]
$$

可训练参数量（仅前缀部分）：
$$
\text{Params} = L \times 2 \times m \times d
$$

其中 $L$ 是层数，$2$ 是 K 和 V 两组前缀。

## 直观理解

- **P-Tuning v2 像给每层都贴了"便利贴"**：模型在处理任务时，每层都可以参考一些额外的"便利贴"（可训练的提示向量）。这些便利贴在训练中不断调整内容，以最好地引导模型完成任务。
- **深层提示 vs 浅层提示**：P-Tuning v1 只在最底层贴"便利贴"，信息在经过多层处理后会衰减。P-Tuning v2 在每层都有便利贴，信息可以在每层被"刷新"和"补充"——对深层网络效果更好。
- **与 Prompt Engineering 的关系**：传统 Prompt Engineering 是写"请翻译以下内容："这样的文字提示。P-Tuning 将提示优化自动化——不是用人写的文字，而是用模型自动搜索到的"连续向量"。
- **软提示的可解释性**：这些训练好的连续提示向量不是人类可读的文字，但可以通过最近邻搜索映射到最近的 token——有时会发现它们学到了类似"请仔细思考并回答"这样的语义概念。

## 代码示例

```python
import torch
import torch.nn as nn

class PTuningV2(nn.Module):
    """P-Tuning v2 实现（简化版）"""
    def __init__(self, num_layers, hidden_size, prefix_length=20, num_heads=12):
        super().__init__()
        # 每层的 Key 和 Value 前缀
        self.prefix_k = nn.Parameter(
            torch.randn(num_layers, prefix_length, hidden_size) * 0.02
        )
        self.prefix_v = nn.Parameter(
            torch.randn(num_layers, prefix_length, hidden_size) * 0.02
        )
        # 可选：前缀的 MLP 重参数化（提升训练稳定性）
        self.prefix_projection = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.Tanh(),
            nn.Linear(hidden_size // 2, hidden_size)
        )

    def forward(self, layer_idx, K, V):
        # 获取对应层的前缀
        pk = self.prefix_k[layer_idx].unsqueeze(0)  # (1, prefix_len, hidden)
        pv = self.prefix_v[layer_idx].unsqueeze(0)

        # MLP 重参数化
        pk = self.prefix_projection(pk)
        pv = self.prefix_projection(pv)

        # 拼接到 K, V 前面
        K_aug = torch.cat([pk.expand(K.size(0), -1, -1), K], dim=1)
        V_aug = torch.cat([pv.expand(V.size(0), -1, -1), V], dim=1)

        return K_aug, V_aug

# 应用 P-Tuning v2 的注意力层
def p_tuning_attention(Q, K, V, prefix_module, layer_idx):
    K_aug, V_aug = prefix_module(layer_idx, K, V)
    # 标准注意力计算（前缀已拼接到 K, V）
    d_k = Q.size(-1)
    scores = torch.matmul(Q, K_aug.transpose(-2, -1)) / (d_k ** 0.5)
    attn_weights = torch.softmax(scores, dim=-1)
    out = torch.matmul(attn_weights, V_aug)
    # 去除前缀部分的输出
    prefix_len = K_aug.size(1) - K.size(1)
    return out[:, :, prefix_len:, :] if len(out.shape) == 4 else out[:, prefix_len:, :]

# 参数量计算
num_layers = 12
hidden_size = 768
prefix_length = 20
total_params = num_layers * 2 * prefix_length * hidden_size  # 12*2*20*768
print(f"P-Tuning v2 参数量: {total_params:,}")
print(f"占 BERT-base 总参数 (110M) 比例: {total_params / 110e6 * 100:.3f}%")
```

## 深度学习关联

- **参数高效微调的重要成员**：P-Tuning v2 与 LoRA、Adapter、Prefix Tuning 共同构成了 PEFT 技术体系。不同方法各有优劣——P-Tuning v2 在生成任务上表现好，LoRA 在理解和分类任务上强。
- **多任务学习的便利性**：由于冻结了主模型，不同任务只需要保存各自的提示前缀（约 1-10MB），适合多任务部署。
- **与开源框架的融合**：P-Tuning v2 已被集成到 HuggingFace PEFT 库，成为 ChatGLM-6B 等模型的默认微调方法之一。
