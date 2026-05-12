# 编码器架构 BERT 系列


## 编码器架构 BERT 系列


#### 编码器架构概览


编码器架构（Encoder-only）使用 Transformer 的编码器部分，通过双向自注意力获取每个 token 的上下文表示。这类模型的核心能力是**理解**（understanding），而非生成。通过预训练+微调范式，BERT 家族在 NLU 任务上取得了突破性进展。虽然近年来被 GPT 家族的解码器架构超越，但编码器模型在特定场景（分类、NER、信息检索）中仍有其独特优势。


## 1. BERT（Bidirectional Encoder Representations from Transformers）


### 1.1 模型架构


BERT（Devlin et al., 2019）基于 Transformer 编码器堆叠，核心特点：


- **双向注意力**
   ：每个 token 可以看到序列中所有其他 token（无因果掩码）
- **特殊 token**
   ：[CLS] 放在序列开头，用于分类任务；[SEP] 用于分隔句子对
- **输入表示**
   ：Token Embedding + Segment Embedding + Position Embedding


| 规格 | BERT-Base | BERT-Large |
| --- | --- | --- |
| 层数（L） | 12 | 24 |
| 隐藏维度（d） | 768 | 1024 |
| 注意力头数（h） | 12 | 16 |
| 参数量 | 110M | 340M |
| 词表大小 | 30,522 (WordPiece) | 30,522 |


```
class BERTInput(nn.Module):
    def __init__(self, vocab_size, d_model, max_len, num_segments=2):
        super().__init__()
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.segment_emb = nn.Embedding(num_segments, d_model)
        self.position_emb = nn.Embedding(max_len, d_model)
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(0.1)

    def forward(self, input_ids, segment_ids):
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device)

        embeddings = (self.token_emb(input_ids) +
                      self.segment_emb(segment_ids) +
                      self.position_emb(positions))
        return self.dropout(self.layer_norm(embeddings))
```


### 1.2 预训练任务：MLM（Masked Language Model）


MLM 随机遮盖输入中 15% 的 token，让模型预测被遮盖的原始 token：


- **80% 概率**
   ：替换为 [MASK] token
- **10% 概率**
   ：替换为随机 token
- **10% 概率**
   ：保持原 token 不变


> **Warning:** #### MLM 的训练-推理不一致
>
>
> 预训练时 [MASK] 出现在输入中，但微调/推理时不会出现 [MASK]，导致预训练和下游任务之间的分布偏移（pretrain-finetune discrepancy）。这也是后续模型（如 RoBERTa、ELECTRA）改进的重点之一。


### 1.3 预训练任务：NSP（Next Sentence Prediction）


给定句子对 (A, B)，预测 B 是否是 A 的真实下一句。50% 的样本是真实的连续句子（IsNext），50% 是随机配对（NotNext）。NSP 任务的 [CLS] 输出经过二分类层判断。


> **Warning:** #### NSP 的争议
>
>
> 后续研究（尤其是 RoBERTa）发现 NSP 任务实际上对模型帮助有限，甚至可能有害——因为随机负样本太容易区分，模型主要学到的是主题差异而非连贯性。RoBERTa 移除了 NSP，效果反而更好。


## 2. ALBERT（A Lite BERT）


### 2.1 核心改进


| 改进 | 描述 | 效果 |
| --- | --- | --- |
| 因式分解嵌入参数化 | 将大的词嵌入矩阵分解为两个小矩阵：vocab×E 和 E×d | 大幅减少参数量（E << d） |
| 跨层参数共享 | 所有 Transformer 层共享同一组参数 | 参数量不随层数增长 |
| SOP 替代 NSP | 用句子顺序预测替代下一句预测 | 更有效的预训练任务 |


### 2.2 SOP（Sentence Order Prediction）


SOP 与 NSP 的区别：不用随机句子作为负样本，而是交换两个连续句子的顺序。这让模型学习句子间的连贯关系，而非简单的主题判断。后续的模型（如 BART）也采用了类似的思路。


## 3. RoBERTa（Robustly Optimized BERT Approach）


### 3.1 关键改进


Liu et al., 2019 通过系统性消融实验发现 BERT 实际上**训练不足**，提出了更优的训练方案：


| 改进 | BERT 原版 | RoBERTa |
| --- | --- | --- |
| 训练数据量 | 16GB 文本 | 160GB（+CC-News 等） |
| 训练步数 | 1M 步 | 500K 步 × 8x 数据 |
| Batch size | 256 | 8K（大 batch 更高效） |
| MLM 策略 | 静态 mask（预处理时确定） | 动态 mask（每个 epoch 重新生成） |
| NSP | 使用 | 移除（使用 full-sentence 格式） |
| 词表 | WordPiece 30K | BPE 50K |
| 序列长度 | 512 | 512（偶尔使用更长序列） |


#### RoBERTa 的核心启示


RoBERTa 证明了 BERT 的成功很大程度上来自**架构**和**双向注意力**本身，而非预训练任务的设计。通过更充分的训练（更多数据、更大 batch、更久训练），简单的 MLM 任务就能产生优秀的表示。这对后续大模型训练有深远影响。


## 4. ELECTRA（Efficiently Learning an Encoder that Classifies Token Replacements Accurately）


### 4.1 Replaced Token Detection（RTD）


ELECTRA（Clark et al., 2020）提出了替代 MLM 的预训练方式：不是让模型预测被遮盖的 token，而是让模型判断每个 token 是否被替换过。


```
# ELECTRA 的训练流程
# 1. Generator: 小型 MLM 模型，预测被 [MASK] 位置的 token
#    输入: The [mask] is fast → 输出: The car is fast
#    Generator 用自己的预测结果替换 [MASK]

# 2. Discriminator: 判断每个 token 是否是原始的
#    输入: The car is fast → 输出: [原始, 被替换, 原始, 原始]
#    损失: 二分类交叉熵（每个 token 一个判断）

# 关键: Generator 生成的替换样本更接近真实分布
# 比随机替换更难、更有挑战性
```


### 4.2 与 MLM 的对比


| 对比项 | MLM（BERT） | RTD（ELECTRA） |
| --- | --- | --- |
| 任务类型 | 多分类（预测词表大小个类） | 二分类（是否被替换） |
| 训练效率 | 只在 15% 位置计算损失 | 在所有位置计算损失 |
| 样本利用 | 低效（85% token 不参与训练） | 高效（所有 token 都有信号） |
| 计算量 | 需要大词表的 softmax | 只需 sigmoid |
| 效果 | 基准 | 相同计算量下更优 |


## 5. DeBERTa（Decoding-enhanced BERT with disentangled Attention）


### 5.1 解耦注意力（Disentangled Attention）


DeBERTa（He et al., 2021）的核心创新是将每个 token 的表示分解为**内容**和**相对位置**两个独立的向量：


$$
传统注意力: score(i,j) = qiT · kj
        解耦注意力: score(i,j) = qi,cT · kj,c + qi,cT · kj,r + qi,rT · kj,c
$$


其中 q~i,c~、k~j,c~ 是内容向量，q~i,r~、k~j,r~ 是相对位置向量。这种分解使得位置信息和内容信息在计算注意力时保持独立，模型可以更灵活地学习内容与位置的交互。


### 5.2 Enhanced Mask Decoder


DeBERTa 还在 MLM 解码阶段引入了绝对位置信息：在预测被遮盖的 token 时，不仅使用上下文内容，还显式注入绝对位置信息。这弥补了纯相对位置编码在某些需要绝对位置信息的任务上的不足。


### 5.3 DeBERTa-v3 的改进


DeBERTa-v3 进一步引入了 ELECTRA 风格的 RTD 预训练任务，结合解耦注意力获得了更强的性能。


## 6. 各编码器模型综合对比


| 模型 | 年份 | 核心创新 | 预训练任务 | 参数量 | GLUE 分数 |
| --- | --- | --- | --- | --- | --- |
| BERT | 2019 | 双向注意力 | MLM + NSP | 110M/340M | 80.5 |
| ALBERT | 2019 | 参数共享 + SOP | MLM + SOP | 12M/60M | 82.3 |
| RoBERTa | 2019 | 更充分训练 | MLM（无 NSP） | 125M/355M | 88.0 |
| ELECTRA | 2020 | RTD 判别任务 | MLM + RTD | 14M/110M/335M | 88.1 |
| DeBERTa | 2021 | 解耦注意力 | MLM + EMD | 86M/276M | 90.3 |
| DeBERTa-v3 | 2023 | 解耦 + RTD | MLM + RTD | 86M/304M | 91.5+ |


## 7. 下游任务微调


### 7.1 单句/句子对分类


```
# BERT 用于分类任务
# [CLS] + 句子 + [SEP] → 取 [CLS] 的输出 → 分类头

class BertForSequenceClassification(nn.Module):
    def __init__(self, bert_model, num_classes):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        cls_output = outputs.last_hidden_state[:, 0, :]  # [CLS] 位置
        logits = self.classifier(cls_output)
        return logits
```


### 7.2 Token 分类（NER / 词性标注）


```
# BERT 用于序列标注
# 每个 token 的输出 → 独立分类头

class BertForTokenClassification(nn.Module):
    def __init__(self, bert_model, num_labels):
        super().__init__()
        self.bert = bert_model
        self.classifier = nn.Linear(bert_model.config.hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids, attention_mask=attention_mask)
        # 取所有 token 的输出（非 [CLS] 和 [SEP]）
        token_outputs = outputs.last_hidden_state  # (batch, seq_len, hidden)
        logits = self.classifier(token_outputs)     # (batch, seq_len, num_labels)
        return logits
```


### 7.3 句子对任务（自然语言推理、语义相似度）


```
# BERT 用于句子对任务
# [CLS] + 句子A + [SEP] + 句子B + [SEP]
# segment_ids: [0, 0, ..., 0, 1, 1, ..., 1]
# 取 [CLS] 输出 → 分类头

# NLI: 蕴含/矛盾/中立 (3类)
# STS: 语义相似度 (回归或分类)
```


## 8. 编码器模型的局限性


> **Warning:** #### 为什么编码器架构不再是主流？
>
>
> - **不适合生成任务**
>    ：双向注意力无法做自回归生成，限制了其在文本生成、对话、代码生成等场景的应用
> - **微调成本**
>    ：每个下游任务需要单独微调，缺乏 zero-shot/few-shot 能力
> - **Scaling 不如解码器**
>    ：编码器模型的 scaling 表现不如解码器（GPT 系列），随着模型增大差距更明显
> - **涌现能力**
>    ：大解码器模型涌现出 in-context learning 等能力，编码器模型不具备
> - **当前趋势**
>    ：2023 年后，几乎所有新的基础模型都是解码器架构（GPT-4、LLaMA、Claude、Gemini 等）


#### 编码器模型仍然有价值的场景


1. **嵌入/检索**
   ：BERT 类模型的双向编码非常适合文本嵌入（sentence-transformers）
2. **信息抽取**
   ：NER、关系抽取等任务，编码器模型微调后效率高、效果好
3. **分类任务**
   ：在足够的标注数据下，编码器微调仍是性价比最高的方案
4. **轻量部署**
   ：BERT-Base（110M）可以部署在资源受限的环境中

编码器架构BERT系列 - BERT/ALBERT/RoBERTa/ELECTRA/DeBERTa对比、预训练任务、下游微调完整笔记


<!-- Converted from: 01_编码器架构BERT系列.html -->
