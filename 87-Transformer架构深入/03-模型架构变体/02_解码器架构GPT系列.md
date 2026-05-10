# 解码器架构 GPT 系列


# 解码器架构 GPT 系列


#### 解码器架构概览


解码器架构（Decoder-only）仅使用 Transformer 的解码器部分，通过因果自注意力（Causal Self-Attention）进行自回归文本生成。这是当前大语言模型（LLM）的主流架构，GPT 系列是其最典型的代表。从 GPT-1 的 117M 参数到 GPT-4 的万亿级参数，解码器架构通过 scaling 展现出了令人惊叹的能力涌现。


## 1. GPT-1（Generative Pre-Training）


### 1.1 核心思想


GPT-1（Radford et al., 2018）首次证明了**无监督预训练 + 有监督微调**范式在 NLP 任务上的有效性：


- **预训练**
   ：在 BooksCorpus 上用语言模型目标（自回归）预训练
- **微调**
   ：在下游任务上用有标签数据微调
- **架构**
   ：12 层 Transformer 解码器，768 维，12 头注意力，117M 参数


$$
预训练目标: L = Σ log P(wi | w1, ..., wi-1; Θ)
$$


### 1.2 GPT-1 的历史意义


#### GPT-1 的贡献


GPT-1 的真正贡献不在于模型架构本身（就是标准 Transformer decoder），而在于它证明了一个重要假设：**在大规模无标注文本上预训练的通用语言表示，可以通过简单的微调迁移到各种下游任务**。这奠定了"预训练-微调"范式的基础，深刻改变了 NLP 的研究方向。


## 2. GPT-2（Language Models are Unsupervised Multitask Learners）


### 2.1 关键改进


| 改进 | 描述 |
| --- | --- |
| 规模扩大 | 1.5B 参数，40GB WebText 数据 |
| Zero-shot 能力 | 不微调，直接用语言模型做下游任务 |
| 任务表述 | 将所有任务统一为文本生成（输入输出都是文本） |
| Layer Norm 位置 | Pre-LN（残差连接前做归一化） |
| 词表扩展 | BPE 词表从 40K 扩展到 50K |


### 2.2 Zero-shot 学习


```
# GPT-2 的 zero-shot 能力示例
# 不需要微调，通过提示词就能完成任务

# 翻译任务:
# 输入: "translate English to French: sea otter => loutre de mer"
# 模型自回归生成: "cheese => fromage"

# 阅读理解:
# 输入: 文章 + 问题
# 模型自回归生成: 答案

# 摘要:
# 输入: 长文章 + "TL;DR:"
# 模型自回归生成: 摘要
```


## 3. GPT-3（Language Models are Few-Shot Learners）


### 3.1 核心发现


GPT-3（Brown et al., 2020）以 175B 参数规模展现了惊人的 in-context learning 能力：


| 规格 | GPT-3 |
| --- | --- |
| 参数量 | 175B |
| 训练数据 | 570GB 文本（Common Crawl 等） |
| 层数 | 96 |
| 隐藏维度 | 12288 |
| 注意力头数 | 96 |
| 上下文长度 | 2048 |
| 训练 FLOPS | 3.14 × 10^23^ |


### 3.2 In-Context Learning


GPT-3 最重要的发现是 **In-Context Learning（ICL）**：不需要梯度更新，仅通过在 prompt 中提供几个示例（few-shot），模型就能"学会"新任务。


```
# In-Context Learning 的三种模式

# 1. Zero-shot: 只有指令，没有示例
# "将以下句子翻译成法语: The cat is on the table."
# 模型直接输出翻译

# 2. One-shot: 提供一个示例
# "sea otter => loutre de mer\nThe cat is on the table =>"
# 模型类比生成

# 3. Few-shot: 提供多个示例
# "sea otter => loutre de mer\ncheese => fromage\nhouse => maison\nThe cat is on the table =>"
# 模型从多个示例中学习模式
```


### 3.3 Scaling Laws（缩放定律）


Kaplan et al., 2020 (OpenAI) 发现模型性能与三个因素之间存在幂律关系：


$$
L(N, D, C) = [(Nc/N)αN + (Dc/D)αD] + 常数
        N: 模型参数量, D: 数据量, C: 计算量
        αN ≈ 0.076, αD ≈ 0.095
$$


> **Warning:** #### Chinchilla 定律的修正
> Hoffmann et al., 2022 (DeepMind) 的 Chinchilla 研究发现，OpenAI 的 scaling laws 低估了数据的重要性。最优策略是**模型参数量和数据量应该等比例增长**：如果计算预算翻倍，模型参数和数据都应该增加约 41%。Chinchilla (70B) 在与 Gopher (280B) 相同的计算量下训练，但使用更多数据（4x），效果更好。这改变了后来的模型训练策略（如 LLaMA）。


## 4. GPT-4（多模态大模型）


### 4.1 已知信息与猜测


| 方面 | 已知/推测 |
| --- | --- |
| 架构 | 传闻为 MoE（混合专家），16 个专家，每个约 220B 参数，总约 1.8T |
| 模态 | 支持文本和图像输入（多模态） |
| 上下文长度 | 32K tokens（GPT-4-32K） |
| 训练 | 据推测使用了大量合成数据 + RLHF |
| 推理成本 | 使用 MoE 后每次推理只激活一部分专家，控制推理成本 |


### 4.2 MoE（Mixture of Experts）架构


MoE 是一种条件计算架构：模型包含多个"专家"网络（Expert），但每次前向传播时只激活其中一部分：


```
class MoELayer(nn.Module):
    def __init__(self, d_model, num_experts, num_active=2):
        super().__init__()
        self.experts = nn.ModuleList([
            FeedForward(d_model) for _ in range(num_experts)
        ])
        self.gate = nn.Linear(d_model, num_experts)  # 路由网络
        self.num_active = num_active

    def forward(self, x):
        # Gate: 决定每个 token 去哪些专家
        gate_scores = self.gate(x)  # (batch, seq_len, num_experts)
        topk_scores, topk_indices = gate_scores.topk(self.num_active, dim=-1)
        # topk_scores: (batch, seq_len, num_active)
        # topk_indices: (batch, seq_len, num_active)

        # 只对选中的专家做计算
        # 实际实现使用负载均衡损失确保各专家利用率均衡
        gate_weights = F.softmax(topk_scores, dim=-1)

        output = torch.zeros_like(x)
        for i in range(self.num_active):
            expert_idx = topk_indices[:, :, i]
            weight = gate_weights[:, :, i].unsqueeze(-1)
            # 对每个 expert_idx，调用对应专家
            # （实际实现更复杂，需要高效的 batch 调度）
            for e in range(len(self.experts)):
                mask = (expert_idx == e)
                if mask.any():
                    output[mask] += weight[mask] * self.experts[e](x[mask])

        return output
```


#### MoE 的优势与挑战


- **优势**
   ：总参数量可以很大（增加模型容量），但每次推理只用一部分（控制推理成本）
- **优势**
   ：不同专家可能学到不同领域的知识，形成隐式分工
- **挑战**
   ：负载均衡——某些专家可能被过度使用而其他闲置
- **挑战**
   ：训练不稳定——路由决策的离散性导致梯度传播困难
- **挑战**
   ：分布式推理——专家可能分布在不同 GPU 上，需要大量通信


## 5. LLaMA 系列


### 5.1 LLaMA-1/2（Meta）


| 特性 | LLaMA-1 (2023) | LLaMA-2 (2023) |
| --- | --- | --- |
| 参数量 | 7B/13B/33B/65B | 7B/13B/70B |
| 训练数据 | 1.4T tokens | 2T tokens |
| 上下文长度 | 2048 | 4096 |
| 位置编码 | RoPE | RoPE |
| 注意力 | Multi-Head | GQA (70B) |
| 归一化 | Pre-RMSNorm | Pre-RMSNorm |
| 激活函数 | SwiGLU | SwiGLU |
| 开源 | 是 | 是（含商用许可） |


### 5.2 LLaMA 的架构细节


```
# LLaMA 架构关键组件

# 1. RMSNorm（替代 LayerNorm）
# 去掉均值归一化，只做方差归一化，更快
class RMSNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(d_model))
        self.eps = eps

    def forward(self, x):
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x / rms * self.weight

# 2. SwiGLU 激活函数
# FFN 层使用 SwiGLU 而非 ReLU
class SwiGLU(nn.Module):
    def forward(self, x, gate):
        return F.silu(gate) * x
# 完整 FFN: FFN(x) = (SwiGLU(x*W1, x*W3)) * W2

# 3. RoPE 位置编码（见位置编码章节）

# 4. 预归一化（Pre-Norm）
# Attention(x + RMSNorm(x)) 而非 RMSNorm(x + Attention(x))
```


### 5.3 LLaMA-3 的改进


| 特性 | LLaMA-3 (2024) |
| --- | --- |
| 参数量 | 8B / 70B / 405B |
| 训练数据 | 15T+ tokens（超大规模数据） |
| 上下文长度 | 8K（8B）/ 128K（405B） |
| 注意力 | 全部使用 GQA |
| 词表 | 128K（tiktoken BPE） |
| 多模态 | 405B 支持图文理解 |


## 6. Mistral / Mixtral


### 6.1 Mistral 7B


Mistral 7B（Jiang et al., 2023）证明了**精心设计的小模型**可以超越更大的模型：


- **滑动窗口注意力（SWA）**
   ：每层只关注前 4096 个 token（而非全局），结合多层堆叠获得长距离信息
- **GQA**
   ：8 KV heads（Q:KV = 4:1），大幅减少 KV Cache
- **Rolling Buffer KV Cache**
   ：配合 SWA，KV Cache 大小固定为窗口大小
- **效果**
   ：在大多数基准上超越 LLaMA-2 13B


### 6.2 Mixtral 8x7B（MoE）


Mixtral 8x7B 是首个大规模开源 MoE 模型：


| 规格 | Mixtral 8x7B |
| --- | --- |
| 总参数 | 46.7B |
| 激活参数 | 12.9B（每次推理） |
| 专家数 | 8 |
| 激活专家数 | 2 |
| FFN 层使用 MoE | 是（注意力层参数共享） |
| 效果 | 接近或超过 LLaMA-2 70B |


## 7. Qwen 系列


### 7.1 Qwen2 / Qwen2.5 架构特点


- **长上下文**
   ：原生支持 128K tokens（Qwen2.5-72B）
- **GQA**
   ：所有规模模型均使用 GQA
- **RoPE + YaRN**
   ：位置编码使用 RoPE，外推使用 YaRN 方案
- **SwiGLU**
   ：激活函数使用 SwiGLU
- **多语言**
   ：中英文双语优势，词表包含大量中文 token
- **Code/Qwen-Coder**
   ：代码能力特别强的变体


## 8. 解码器架构的关键设计选择汇总


| 设计选择 | 选项 | 当前主流 | 代表模型 |
| --- | --- | --- | --- |
| 位置编码 | Learned / RoPE / ALiBi | RoPE | LLaMA, Qwen, Mistral |
| 归一化 | LayerNorm / RMSNorm | Pre-RMSNorm | LLaMA, Qwen |
| 激活函数 | ReLU / GELU / SwiGLU | SwiGLU | 几乎所有新模型 |
| 注意力类型 | MHA / MQA / GQA | GQA | LLaMA-2/3, Qwen2 |
| 归一化位置 | Pre-Norm / Post-Norm | Pre-Norm | 几乎所有模型 |
| FFN 结构 | Standard / SwiGLU / MoE | SwiGLU / MoE | Mixtral, GPT-4 |


## 9. Emergent Abilities（涌现能力）


### 9.1 什么是涌现能力


Wei et al., 2022 定义涌现能力为：在小模型中不存在，但当模型达到某个规模阈值时突然出现的能力。这些能力**无法通过小模型的性能外推来预测**。


### 9.2 已观察到的涌现能力


| 能力 | 出现阈值（近似） | 描述 |
| --- | --- | --- |
| In-Context Learning | ~13B | 从 prompt 中的示例学习新任务 |
| Chain-of-Thought | ~62B | 通过思维链进行多步推理 |
| 指令遵循 | ~13B | 理解和执行复杂指令 |
| 代码生成 | ~13B | 从自然语言描述生成代码 |
| 多步数学推理 | ~62B | 解决需要多步骤的数学问题 |


> **Warning:** #### 涌现能力的争议
>
>
> Schaeffer et al., 2023 认为涌现能力可能只是**评估指标选择的产物**——非线性指标（如精确匹配）导致了"突然出现"的假象，如果使用线性指标（如 token 级准确率），性能提升是平滑连续的。但无论如何，大模型确实展现出了质变级别的新能力，这对实际应用至关重要。


#### 解码器架构成功的核心原因


1. **统一范式**
   ：所有任务都可以转化为文本生成，一个模型解决所有问题
2. **Scaling 友好**
   ：损失函数和架构设计使得增加参数和数据能持续提升性能
3. **In-Context Learning**
   ：无需微调就能适应新任务，极大降低了使用门槛
4. **涌现能力**
   ：在足够大的规模下，模型展现出质变的新能力
5. **生态效应**
   ：RLHF/RLAIF 等对齐技术与解码器架构天然适配

解码器架构GPT系列 - GPT-1/2/3/4、scaling laws、涌现能力、MoE、LLaMA/Mistral/Qwen架构对比完整笔记


<!-- Converted from: 02_解码器架构GPT系列.html -->
