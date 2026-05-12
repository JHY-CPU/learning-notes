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
| 00 | [词袋模型 (BoW) 与 TF-IDF 局限性](0_词袋模型 (BoW) 与 TF-IDF 局限性.md) |
| 01 | [Word2Vec：Skip-gram 与 CBOW 架构对比](1_Word2Vec：Skip-gram 与 CBOW 架构对比.md) |
| 02 | [GloVe：基于全局词频矩阵的分解](2_GloVe：基于全局词频矩阵的分解.md) |
| 03 | [FastText：子词信息与 OOV 处理](3_FastText：子词信息与 OOV 处理.md) |

#### 循环神经网络

RNN、BPTT、梯度消失、LSTM、GRU、双向RNN

| 编号 | 笔记 |
|------|------|
| 04 | [RNN 循环神经网络与前向传播](4_RNN 循环神经网络与前向传播.md) |
| 05 | [RNN 反向传播与 BPTT 算法推导](5_RNN 反向传播与 BPTT 算法推导.md) |
| 06 | [梯度消失问题与长程依赖挑战](6_梯度消失问题与长程依赖挑战.md) |
| 07 | [LSTM 单元结构：门控机制详解](7_LSTM 单元结构：门控机制详解.md) |
| 08 | [GRU：LSTM 的简化变体](8_GRU：LSTM 的简化变体.md) |
| 09 | [Bi-directional RNN 双向建模](9_Bi-directional RNN 双向建模.md) |

#### 注意力与Transformer

Seq2Seq、Bahdanau/Luong Attention、Self-Attention、Multi-Head、Transformer Encoder/Decoder

| 编号 | 笔记 |
|------|------|
| 10 | [Seq2Seq 框架与编码器-解码器范式](10_Seq2Seq 框架与编码器-解码器范式.md) |
| 11 | [Bahdanau 注意力机制 (Additive Attention)](11_Bahdanau 注意力机制 (Additive Attention).md) |
| 12 | [Luong 注意力机制 (Multiplicative Attention)](12_Luong 注意力机制 (Multiplicative Attention).md) |
| 13 | [Self-Attention 自注意力的并行计算优势](13_Self-Attention 自注意力的并行计算优势.md) |
| 14 | [Scaled Dot-Product Attention 缩放因子推导](14_Scaled Dot-Product Attention 缩放因子推导.md) |
| 15 | [Multi-head Attention 多头机制与子空间表示](15_Multi-head Attention 多头机制与子空间表示.md) |
| 16 | [Positional Encoding 位置编码的频谱分析](16_Positional Encoding 位置编码的频谱分析.md) |
| 17 | [Transformer Encoder 架构细节](17_Transformer Encoder 架构细节.md) |
| 18 | [Transformer Decoder 与 Causal Masking](18_Transformer Decoder 与 Causal Masking.md) |
| 19 | [Encoder-Decoder 交互与 Cross-Attention](19_Encoder-Decoder 交互与 Cross-Attention.md) |

#### 预训练语言模型

BERT系列（RoBERTa/ALBERT/DistilBERT）、GPT系列、T5、BART

| 编号 | 笔记 |
|------|------|
| 20 | [BERT：掩码语言模型 (MLM) 任务设计](20_BERT：掩码语言模型 (MLM) 任务设计.md) |
| 21 | [BERT：下一句预测 (NSP) 任务的争议](21_BERT：下一句预测 (NSP) 任务的争议.md) |
| 22 | [RoBERTa：对 BERT 训练策略的改进](22_RoBERTa：对 BERT 训练策略的改进.md) |
| 23 | [ALBERT：跨层参数共享与因式分解嵌入](23_ALBERT：跨层参数共享与因式分解嵌入.md) |
| 24 | [DistilBERT：知识蒸馏在预训练中的应用](24_DistilBERT：知识蒸馏在预训练中的应用.md) |
| 25 | [GPT 系列：因果语言建模 (CLM)](25_GPT 系列：因果语言建模 (CLM).md) |
| 26 | [GPT-2：零样本学习与大规模数据](26_GPT-2：零样本学习与大规模数据.md) |
| 27 | [GPT-3：In-context Learning 上下文学习](27_GPT-3：In-context Learning 上下文学习.md) |
| 28 | [T5：文本到文本 (Text-to-Text) 统一框架](28_T5：文本到文本 (Text-to-Text) 统一框架.md) |
| 29 | [BART：去噪自编码器与生成任务](29_BART：去噪自编码器与生成任务.md) |

#### 高效Transformer

RoPE、ALiBi、稀疏注意力、FlashAttention、PagedAttention、Tokenizer

| 编号 | 笔记 |
|------|------|
| 30 | [RoPE 旋转位置编码原理与实现](30_RoPE 旋转位置编码原理与实现.md) |
| 31 | [Alibi (ALiBi)：注意力线性偏置](31_Alibi (ALiBi)：注意力线性偏置.md) |
| 32 | [Swarm Attention 与稀疏注意力机制](32_Swarm Attention 与稀疏注意力机制.md) |
| 33 | [FlashAttention：IO 感知的注意力加速](33_FlashAttention：IO 感知的注意力加速.md) |
| 34 | [PagedAttention：vLLM 中的显存优化](34_PagedAttention：vLLM 中的显存优化.md) |
| 35 | [Tokenizer 技术：BPE, WordPiece, Unigram](35_Tokenizer 技术：BPE, WordPiece, Unigram.md) |
| 36 | [词汇表大小对模型性能的影响](36_词汇表大小对模型性能的影响.md) |

#### LLM微调与工程

指令微调、RLHF、PPO、LoRA、P-Tuning v2、RAG、向量数据库、LangChain、ChatGLM、LLaMA

| 编号 | 笔记 |
|------|------|
| 37 | [指令微调 (Instruction Tuning) 原理](37_指令微调 (Instruction Tuning) 原理.md) |
| 38 | [RLHF：基于人类反馈的强化学习对齐](38_RLHF：基于人类反馈的强化学习对齐.md) |
| 39 | [PPO 在 LLM 训练中的稳定性技巧](39_PPO 在 LLM 训练中的稳定性技巧.md) |
| 40 | [LoRA：低秩自适应微调原理](40_LoRA：低秩自适应微调原理.md) |
| 41 | [P-Tuning v2：连续提示优化](41_P-Tuning v2：连续提示优化.md) |
| 42 | [检索增强生成 (RAG) 架构设计](42_检索增强生成 (RAG) 架构设计.md) |
| 43 | [向量数据库与语义检索技术](43_向量数据库与语义检索技术.md) |
| 44 | [LangChain 框架核心组件分析](44_LangChain 框架核心组件分析.md) |
| 45 | [ChatGLM：中英双语大模型实践](45_ChatGLM：中英双语大模型实践.md) |
| 46 | [LLaMA：开源大模型的里程碑](46_LLaMA：开源大模型的里程碑.md) |
| 47 | [Mistral：滑动窗口注意力机制](47_Mistral：滑动窗口注意力机制.md) |
| 48 | [MoE (Mixture of Experts) 混合专家模型](48_MoE (Mixture of Experts) 混合专家模型.md) |

#### NLP任务与解码

NER、依存分析、BLEU、文本摘要、情感分析、QA、对话系统、CRF、CTC

| 编号 | 笔记 |
|------|------|
| 49 | [命名实体识别 (NER) 与 BIO 标注](49_命名实体识别 (NER) 与 BIO 标注.md) |
| 50 | [依存句法分析 (Dependency Parsing)](50_依存句法分析 (Dependency Parsing).md) |
| 51 | [机器翻译中的 BLEU 指标计算](51_机器翻译中的 BLEU 指标计算.md) |
| 52 | [文本摘要：抽取式与生成式对比](52_文本摘要：抽取式与生成式对比.md) |
| 53 | [情感分析与 Aspect-Based Sentiment](53_情感分析与 Aspect-Based Sentiment.md) |
| 54 | [问答系统：Extractive vs Generative QA](54_问答系统：Extractive vs Generative QA.md) |
| 55 | [对话系统与多轮上下文管理](55_对话系统与多轮上下文管理.md) |
| 56 | [文本分类中的 Hierarchical Attention](56_文本分类中的 Hierarchical Attention.md) |
| 57 | [序列标注与 CRF 条件随机场](57_序列标注与 CRF 条件随机场.md) |
| 58 | [语音识别：CTC Loss 原理](58_语音识别：CTC Loss 原理.md) |

#### 前沿与评估

Wav2Vec、Beam Search、采样策略、长文本处理、多语言模型、代码模型、思维链、MMLU

| 编号 | 笔记 |
|------|------|
| 59 | [Wav2Vec 2.0：自监督语音预训练](59_Wav2Vec 2.0：自监督语音预训练.md) |
| 60 | [机器翻译中的 Transformer 实现细节](60_机器翻译中的 Transformer 实现细节.md) |
| 61 | [文本生成中的 Beam Search 算法](61_文本生成中的 Beam Search 算法.md) |
| 62 | [Top-k 与 Top-p (Nucleus) 采样策略](62_Top-k 与 Top-p (Nucleus) 采样策略.md) |
| 63 | [Temperature 参数对生成分布的影响](63_Temperature 参数对生成分布的影响.md) |
| 64 | [Repetition Penalty 重复惩罚机制](64_Repetition Penalty 重复惩罚机制.md) |
| 65 | [长文本处理：Longformer 与 BigBird](65_长文本处理：Longformer 与 BigBird.md) |
| 66 | [多语言模型：mBERT 与 XLM-R](66_多语言模型：mBERT 与 XLM-R.md) |
| 67 | [代码大模型：CodeLlama 与 StarCoder](67_代码大模型：CodeLlama 与 StarCoder.md) |
| 68 | [逻辑推理与大模型的思维链 (CoT)](68_逻辑推理与大模型的思维链 (CoT).md) |
| 69 | [大模型评估基准：MMLU, GSM8K, HumanEval](69_大模型评估基准：MMLU, GSM8K, HumanEval.md) |
---

## 学习建议

1. 按编号顺序阅读每个子主题内的笔记，因为内部存在递进关系
2. 每个子主题完成后，尝试用「深度学习关联」部分串联知识点
3. 代码示例可以直接复制运行（需要 PyTorch 和 transformers 库）
4. 遇到数学推导不熟悉时，回到 01_数学基础 查阅对应基础

---

*本 README 由笔记元数据自动生成。*
