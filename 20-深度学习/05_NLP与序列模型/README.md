# 05_NLP与序列模型

> 从词嵌入到ChatGPT，自然语言处理的技术全景。本目录从经典NLP方法出发，纵贯RNN/LSTM/GRU、注意力机制与Transformer、BERT/GPT系列，延伸到LLM微调（LoRA、P-Tuning）、RAG、模型评估等工程实践。

---

## 基础知识

- **前置知识**：03_神经网络核心; Python（transformers 库）
- **关联目录**：04_计算机视觉（多模态）; 06_生成式AI（生成式模型）
- **笔记数量**：共 70 篇

---

## 内容结构

#### 词表示与嵌入

BoW/TF-IDF、Word2Vec、GloVe、FastText

| 编号 | 笔记 |
|------|------|
| 01 | [词袋模型 (BoW) 与 TF-IDF 局限性](01_词袋模型 (BoW) 与 TF-IDF 局限性.md) |
| 02 | [Word2Vec：Skip-gram 与 CBOW 架构对比](02_Word2Vec：Skip-gram 与 CBOW 架构对比.md) |
| 03 | [GloVe：基于全局词频矩阵的分解](03_GloVe：基于全局词频矩阵的分解.md) |
| 04 | [FastText：子词信息与 OOV 处理](04_FastText：子词信息与 OOV 处理.md) |

#### 循环神经网络

RNN、BPTT、梯度消失、LSTM、GRU、双向RNN

| 编号 | 笔记 |
|------|------|
| 05 | [RNN 循环神经网络与前向传播](05_RNN 循环神经网络与前向传播.md) |
| 06 | [RNN 反向传播与 BPTT 算法推导](06_RNN 反向传播与 BPTT 算法推导.md) |
| 07 | [梯度消失问题与长程依赖挑战](07_梯度消失问题与长程依赖挑战.md) |
| 08 | [LSTM 单元结构：门控机制详解](08_LSTM 单元结构：门控机制详解.md) |
| 09 | [GRU：LSTM 的简化变体](09_GRU：LSTM 的简化变体.md) |
| 10 | [Bi-directional RNN 双向建模](10_Bi-directional RNN 双向建模.md) |

#### 注意力与Transformer

Seq2Seq、Bahdanau/Luong Attention、Self-Attention、Multi-Head、Transformer Encoder/Decoder

| 编号 | 笔记 |
|------|------|
| 11 | [Seq2Seq 框架与编码器-解码器范式](11_Seq2Seq 框架与编码器-解码器范式.md) |
| 12 | [Bahdanau 注意力机制 (Additive Attention)](12_Bahdanau 注意力机制 (Additive Attention).md) |
| 13 | [Luong 注意力机制 (Multiplicative Attention)](13_Luong 注意力机制 (Multiplicative Attention).md) |
| 14 | [Self-Attention 自注意力的并行计算优势](14_Self-Attention 自注意力的并行计算优势.md) |
| 15 | [Scaled Dot-Product Attention 缩放因子推导](15_Scaled Dot-Product Attention 缩放因子推导.md) |
| 16 | [Multi-head Attention 多头机制与子空间表示](16_Multi-head Attention 多头机制与子空间表示.md) |
| 17 | [Positional Encoding 位置编码的频谱分析](17_Positional Encoding 位置编码的频谱分析.md) |
| 18 | [Transformer Encoder 架构细节](18_Transformer Encoder 架构细节.md) |
| 19 | [Transformer Decoder 与 Causal Masking](19_Transformer Decoder 与 Causal Masking.md) |
| 20 | [Encoder-Decoder 交互与 Cross-Attention](20_Encoder-Decoder 交互与 Cross-Attention.md) |

#### 预训练语言模型

BERT系列（RoBERTa/ALBERT/DistilBERT）、GPT系列、T5、BART

| 编号 | 笔记 |
|------|------|
| 21 | [BERT：掩码语言模型 (MLM) 任务设计](21_BERT：掩码语言模型 (MLM) 任务设计.md) |
| 22 | [BERT：下一句预测 (NSP) 任务的争议](22_BERT：下一句预测 (NSP) 任务的争议.md) |
| 23 | [RoBERTa：对 BERT 训练策略的改进](23_RoBERTa：对 BERT 训练策略的改进.md) |
| 24 | [ALBERT：跨层参数共享与因式分解嵌入](24_ALBERT：跨层参数共享与因式分解嵌入.md) |
| 25 | [DistilBERT：知识蒸馏在预训练中的应用](25_DistilBERT：知识蒸馏在预训练中的应用.md) |
| 26 | [GPT 系列：因果语言建模 (CLM)](26_GPT 系列：因果语言建模 (CLM).md) |
| 27 | [GPT-2：零样本学习与大规模数据](27_GPT-2：零样本学习与大规模数据.md) |
| 28 | [GPT-3：In-context Learning 上下文学习](28_GPT-3：In-context Learning 上下文学习.md) |
| 29 | [T5：文本到文本 (Text-to-Text) 统一框架](29_T5：文本到文本 (Text-to-Text) 统一框架.md) |
| 30 | [BART：去噪自编码器与生成任务](30_BART：去噪自编码器与生成任务.md) |

#### 高效Transformer

RoPE、ALiBi、稀疏注意力、FlashAttention、PagedAttention、Tokenizer

| 编号 | 笔记 |
|------|------|
| 31 | [RoPE 旋转位置编码原理与实现](31_RoPE 旋转位置编码原理与实现.md) |
| 32 | [Alibi (ALiBi)：注意力线性偏置](32_Alibi (ALiBi)：注意力线性偏置.md) |
| 33 | [Swarm Attention 与稀疏注意力机制](33_Swarm Attention 与稀疏注意力机制.md) |
| 34 | [FlashAttention：IO 感知的注意力加速](34_FlashAttention：IO 感知的注意力加速.md) |
| 35 | [PagedAttention：vLLM 中的显存优化](35_PagedAttention：vLLM 中的显存优化.md) |
| 36 | [Tokenizer 技术：BPE, WordPiece, Unigram](36_Tokenizer 技术：BPE, WordPiece, Unigram.md) |
| 37 | [词汇表大小对模型性能的影响](37_词汇表大小对模型性能的影响.md) |

#### LLM微调与工程

指令微调、RLHF、PPO、LoRA、P-Tuning v2、RAG、向量数据库、LangChain、ChatGLM、LLaMA

| 编号 | 笔记 |
|------|------|
| 38 | [指令微调 (Instruction Tuning) 原理](38_指令微调 (Instruction Tuning) 原理.md) |
| 39 | [RLHF：基于人类反馈的强化学习对齐](39_RLHF：基于人类反馈的强化学习对齐.md) |
| 40 | [PPO 在 LLM 训练中的稳定性技巧](40_PPO 在 LLM 训练中的稳定性技巧.md) |
| 41 | [LoRA：低秩自适应微调原理](41_LoRA：低秩自适应微调原理.md) |
| 42 | [P-Tuning v2：连续提示优化](42_P-Tuning v2：连续提示优化.md) |
| 43 | [检索增强生成 (RAG) 架构设计](43_检索增强生成 (RAG) 架构设计.md) |
| 44 | [向量数据库与语义检索技术](44_向量数据库与语义检索技术.md) |
| 45 | [LangChain 框架核心组件分析](45_LangChain 框架核心组件分析.md) |
| 46 | [ChatGLM：中英双语大模型实践](46_ChatGLM：中英双语大模型实践.md) |
| 47 | [LLaMA：开源大模型的里程碑](47_LLaMA：开源大模型的里程碑.md) |
| 48 | [Mistral：滑动窗口注意力机制](48_Mistral：滑动窗口注意力机制.md) |
| 49 | [MoE (Mixture of Experts) 混合专家模型](49_MoE (Mixture of Experts) 混合专家模型.md) |

#### NLP任务与解码

NER、依存分析、BLEU、文本摘要、情感分析、QA、对话系统、CRF、CTC

| 编号 | 笔记 |
|------|------|
| 50 | [命名实体识别 (NER) 与 BIO 标注](50_命名实体识别 (NER) 与 BIO 标注.md) |
| 51 | [依存句法分析 (Dependency Parsing)](51_依存句法分析 (Dependency Parsing).md) |
| 52 | [机器翻译中的 BLEU 指标计算](52_机器翻译中的 BLEU 指标计算.md) |
| 53 | [文本摘要：抽取式与生成式对比](53_文本摘要：抽取式与生成式对比.md) |
| 54 | [情感分析与 Aspect-Based Sentiment](54_情感分析与 Aspect-Based Sentiment.md) |
| 55 | [问答系统：Extractive vs Generative QA](55_问答系统：Extractive vs Generative QA.md) |
| 56 | [对话系统与多轮上下文管理](56_对话系统与多轮上下文管理.md) |
| 57 | [文本分类中的 Hierarchical Attention](57_文本分类中的 Hierarchical Attention.md) |
| 58 | [序列标注与 CRF 条件随机场](58_序列标注与 CRF 条件随机场.md) |
| 59 | [语音识别：CTC Loss 原理](59_语音识别：CTC Loss 原理.md) |

#### 前沿与评估

Wav2Vec、Beam Search、采样策略、长文本处理、多语言模型、代码模型、思维链、MMLU

| 编号 | 笔记 |
|------|------|
| 60 | [Wav2Vec 2.0：自监督语音预训练](60_Wav2Vec 2.0：自监督语音预训练.md) |
| 61 | [机器翻译中的 Transformer 实现细节](61_机器翻译中的 Transformer 实现细节.md) |
| 62 | [文本生成中的 Beam Search 算法](62_文本生成中的 Beam Search 算法.md) |
| 63 | [Top-k 与 Top-p (Nucleus) 采样策略](63_Top-k 与 Top-p (Nucleus) 采样策略.md) |
| 64 | [Temperature 参数对生成分布的影响](64_Temperature 参数对生成分布的影响.md) |
| 65 | [Repetition Penalty 重复惩罚机制](65_Repetition Penalty 重复惩罚机制.md) |
| 66 | [长文本处理：Longformer 与 BigBird](66_长文本处理：Longformer 与 BigBird.md) |
| 67 | [多语言模型：mBERT 与 XLM-R](67_多语言模型：mBERT 与 XLM-R.md) |
| 68 | [代码大模型：CodeLlama 与 StarCoder](68_代码大模型：CodeLlama 与 StarCoder.md) |
| 69 | [逻辑推理与大模型的思维链 (CoT)](69_逻辑推理与大模型的思维链 (CoT).md) |
| 70 | [大模型评估基准：MMLU, GSM8K, HumanEval](70_大模型评估基准：MMLU, GSM8K, HumanEval.md) |
---

## 学习建议

1. 按编号顺序阅读每个子主题内的笔记，因为内部存在递进关系
2. 每个子主题完成后，尝试用「深度学习关联」部分串联知识点
3. 代码示例可以直接复制运行（需要 PyTorch 和 transformers 库）
4. 遇到数学推导不熟悉时，回到 01_数学基础 查阅对应基础

---

*本 README 由笔记元数据自动生成。*
