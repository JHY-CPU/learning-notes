import os

titles = [
    "01_词袋模型 (BoW) 与 TF-IDF 局限性",
    "02_Word2Vec：Skip-gram 与 CBOW 架构对比",
    "03_GloVe：基于全局词频矩阵的分解",
    "04_FastText：子词信息与 OOV 处理",
    "05_RNN 循环神经网络与前向传播",
    "06_RNN 反向传播与 BPTT 算法推导",
    "07_梯度消失问题与长程依赖挑战",
    "08_LSTM 单元结构：门控机制详解",
    "09_GRU：LSTM 的简化变体",
    "10_Bi-directional RNN 双向建模",
    "11_Seq2Seq 框架与编码器-解码器范式",
    "12_Bahdanau 注意力机制 (Additive Attention)",
    "13_Luong 注意力机制 (Multiplicative Attention)",
    "14_Self-Attention 自注意力的并行计算优势",
    "15_Scaled Dot-Product Attention 缩放因子推导",
    "16_Multi-head Attention 多头机制与子空间表示",
    "17_Positional Encoding 位置编码的频谱分析",
    "18_Transformer Encoder 架构细节",
    "19_Transformer Decoder 与 Causal Masking",
    "20_Encoder-Decoder 交互与 Cross-Attention",
    "21_BERT：掩码语言模型 (MLM) 任务设计",
    "22_BERT：下一句预测 (NSP) 任务的争议",
    "23_RoBERTa：对 BERT 训练策略的改进",
    "24_ALBERT：跨层参数共享与因式分解嵌入",
    "25_DistilBERT：知识蒸馏在预训练中的应用",
    "26_GPT 系列：因果语言建模 (CLM)",
    "27_GPT-2：零样本学习与大规模数据",
    "28_GPT-3：In-context Learning 上下文学习",
    "29_T5：文本到文本 (Text-to-Text) 统一框架",
    "30_BART：去噪自编码器与生成任务",
    "31_RoPE 旋转位置编码原理与实现",
    "32_Alibi (ALiBi)：注意力线性偏置",
    "33_Swarm Attention 与稀疏注意力机制",
    "34_FlashAttention：IO 感知的注意力加速",
    "35_PagedAttention：vLLM 中的显存优化",
    "36_Tokenizer 技术：BPE, WordPiece, Unigram",
    "37_词汇表大小对模型性能的影响",
    "38_指令微调 (Instruction Tuning) 原理",
    "39_RLHF：基于人类反馈的强化学习对齐",
    "40_PPO 在 LLM 训练中的稳定性技巧"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
