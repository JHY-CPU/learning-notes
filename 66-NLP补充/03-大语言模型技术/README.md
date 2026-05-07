# 03-大语言模型技术

大语言模型（Large Language Model, LLM）是当前人工智能领域最核心的研究方向之一。本章从Transformer架构出发，系统介绍预训练语言模型、LLM训练技术、微调方法、对齐技术以及实际应用。

---

## 一、Transformer架构回顾

Transformer是现代所有大语言模型的基础架构，由Vaswani等人在2017年提出。

### 1.1 Self-Attention（自注意力机制）

Self-Attention允许序列中的每个位置关注序列中的所有其他位置：

- **计算过程**：输入向量通过线性变换得到Q、K、V，计算 $Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V$
- **缩放因子** $\sqrt{d_k}$ 防止点积值过大导致softmax梯度消失
- **复杂度**：$O(n^2 \cdot d)$，可并行计算，能捕获长距离依赖

### 1.2 Multi-Head Attention（多头注意力）

将注意力机制扩展为多个"头"，从不同子空间学习不同的注意力模式：

- $MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O$，其中 $head_i = Attention(QW_i^Q, KW_i^K, VW_i^V)$
- 不同头可以关注不同类型的关系（语法关系、语义关系等）

### 1.3 位置编码（Positional Encoding）

Transformer本身不包含位置信息，需要显式添加位置编码：

- **正弦位置编码**：$PE_{(pos, 2i)} = \sin(pos / 10000^{2i/d})$，可泛化到未见过的序列长度
- **可学习位置编码**：GPT、BERT使用可学习的位置嵌入向量
- **旋转位置编码（RoPE）**：LLaMA等现代LLM广泛使用，通过旋转矩阵编码相对位置
- **ALiBi**：在注意力分数上添加线性偏置，不需要额外参数

---

## 二、预训练语言模型

### 2.1 GPT系列

GPT系列是自回归语言模型的代表：

- **GPT-1**（2018）：12层Decoder，1.17亿参数，提出无监督预训练+有监督微调范式
- **GPT-2**（2019）：48层15亿参数，证明零样本学习能力和规模扩展带来的涌现能力
- **GPT-3**（2020）：1750亿参数，提出In-context Learning，引领"大规模预训练+提示"范式
- **GPT-4**（2023）：多模态能力，推理能力大幅提升，传闻为MoE架构

### 2.2 BERT及其变体

BERT是编码器架构的代表，适合理解类任务：

- **BERT**（2018）：双向Transformer编码器，预训练任务为MLM + NSP
- **RoBERTa**（2019）：移除NSP，动态掩码，更大batch和更多数据，更长训练
- **ALBERT**（2019）：跨层参数共享+嵌入矩阵因式分解，大幅减少参数量
- **DeBERTa**（2020）：解耦注意力机制分别编码内容和位置，在NLU基准上超越人类

### 2.3 T5/BART（编码器-解码器）

- **T5**：将所有NLP任务统一为文本到文本格式，预训练使用Span Corruption
- **BART**：编码器双向+解码器自回归，对文本做多种噪声变换后重建

---

## 三、大语言模型的训练

### 3.1 预训练目标

- **Next Token Prediction**：给定前文预测下一个token，GPT系列、LLaMA等使用
- **Masked Language Model**：随机掩码token并预测，BERT等使用，可利用双向上下文

### 3.2 Scaling Laws（缩放定律）

描述模型性能与规模之间的幂律关系：

- **Kaplan定律**（OpenAI, 2020）：损失与参数量、数据量成幂律关系，模型和数据应同步扩展
- **Chinchilla定律**（DeepMind, 2022）：固定计算预算下，参数量和数据量应等比例增加

### 3.3 分布式训练

大模型训练需要多种并行策略组合：

- **数据并行**：数据分片到多GPU，每个GPU持完整模型副本，同步梯度更新
- **张量并行**：将单层矩阵运算拆分到多GPU，需高速互联，代表：Megatron-LM
- **流水线并行**：不同层分配到不同GPU，通过micro-batch减少气泡，代表：GPipe
- **ZeRO优化**（DeepSpeed）：分片优化器状态/梯度/参数，显著降低显存占用

### 3.4 DeepSpeed/Megatron

- **DeepSpeed**：ZeRO系列、混合精度训练、激活检查点，支持100B+模型训练
- **Megatron-LM**：高效张量并行实现，序列并行支持超长上下文

---

## 四、微调技术

全参数微调成本极高，参数高效微调（PEFT）方法应运而生。

### 4.1 全参数微调

更新所有参数，需存储完整优化器状态。对175B模型仅优化器状态就需约1TB显存。

### 4.2 LoRA（Low-Rank Adaptation）

通过低秩分解近似参数更新：$W = W_0 + BA$，其中 $r \ll d$。可训练参数极少（r=8或16），推理时可合并回原始权重，无额外延迟。

### 4.3 QLoRA

在LoRA基础上引入4-bit量化：4-bit NormalFloat量化+双重量化+分页优化器，可在单张24GB GPU上微调65B参数模型。

### 4.4 Prefix Tuning / Prompt Tuning / P-Tuning

- **Prefix Tuning**：在每层KV前添加可学习前缀向量，只训练前缀参数
- **Prompt Tuning**：仅在输入嵌入层添加soft prompt，参数极少
- **P-Tuning v2**：每层都添加前缀，适合理解类任务

### 4.5 Adapter方法

在Transformer层中插入瓶颈结构（down_project → nonlinear → up_project + residual），参数量约为原始模型的0.5%-5%。

---

## 五、指令微调与RLHF

### 5.1 Instruction Tuning

在多种任务的指令-输出对上进行有监督微调，覆盖问答、摘要、翻译、代码生成等，显著提升指令遵循和零样本泛化能力。

### 5.2 RLHF（人类反馈强化学习）

三步流程：(1) SFT有监督微调；(2) 人类偏好排序训练奖励模型；(3) PPO优化。关键技术包括KL散度惩罚和优势函数估计。代表工作：InstructGPT、ChatGPT。

### 5.3 DPO（Direct Preference Optimization）

跳过奖励模型训练，直接从偏好数据优化策略。损失函数基于策略与参考模型的概率比。优势：无需奖励模型，训练更稳定，成本更低。变体：IPO、KTO、ORPO。

---

## 六、提示工程（Prompt Engineering）

### 6.1 基本策略

- **Zero-shot**：直接给任务描述，不提供示例
- **Few-shot**：在提示中提供少量示例引导模型
- **角色设定**：指定模型角色和行为方式

### 6.2 高级策略

- **Chain-of-Thought（CoT）**：引导模型展示推理过程，"让我们一步步思考"，显著提升推理性能
- **Self-Consistency**：多次采样不同推理路径，对答案进行多数投票
- **Tree-of-Thought（ToT）**：推理过程组织为树状搜索，支持回溯和剪枝

---

## 七、RAG（检索增强生成）

将检索系统与生成模型结合，使模型利用外部知识：

- **流程**：(1) 文档切片编码存入向量数据库；(2) 查询编码检索相关片段；(3) 检索结果作为上下文输入LLM生成答案
- **关键组件**：嵌入模型（BGE、E5）、向量数据库（FAISS、Milvus）、重排序模型
- **优化方向**：混合检索、查询改写、迭代检索、图RAG

---

## 八、Agent与工具调用

LLM作为智能体调用外部工具完成复杂任务：

- **架构**：LLM推理决策 + 工具集（搜索、代码执行等）+ 记忆模块
- **范式**：ReAct（推理+行动交替）、Function Calling（结构化函数调用）、Plan-and-Execute
- **框架**：LangChain、AutoGPT、MetaGPT

---

## 九、大模型评估基准

| 基准 | 评估内容 | 说明 |
|------|----------|------|
| MMLU | 多任务语言理解 | 57个学科的多项选择题 |
| HumanEval | 代码生成 | 函数级代码补全 |
| MT-Bench | 多轮对话 | 人工评分评估 |
| GSM8K | 数学推理 | 小学数学应用题 |
| ARC | 科学推理 | 小学到高中科学题目 |
| TruthfulQA | 事实准确性 | 检测虚假信息 |

---

## 十、长上下文处理技术

处理超长文本（100K+ tokens）的技术：

- **稀疏注意力**：Longformer（局部窗口+全局token）、BigBird（随机+局部+全局）
- **位置编码扩展**：位置插值、NTK-aware缩放
- **上下文压缩**：历史摘要、KV-Cache压缩、StreamingLLM

---

## 十一、模型压缩

- **量化**：将FP16量化为INT8/INT4。GPTQ（逐层最小化误差）、AWQ（激活感知）、GGUF（CPU友好）
- **蒸馏**：教师模型输出训练学生模型，代表：DistilBERT
- **剪枝**：非结构化剪枝移除小权重，结构化剪枝移除整个注意力头/层，SparseGPT一次性剪枝

---

## 十二、总结

大语言模型技术涵盖了从架构、预训练、微调到部署的完整技术栈。Transformer奠定了基础，预训练-微调范式提供了通用框架，RLHF/DPO实现了人机对齐，RAG和Agent扩展了能力边界。随着规模增长和效率技术进步，LLM正在深刻改变NLP乃至整个人工智能领域。
