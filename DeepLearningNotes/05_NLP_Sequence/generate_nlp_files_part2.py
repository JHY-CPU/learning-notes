import os

titles = [
    "41_LoRA：低秩自适应微调原理",
    "42_P-Tuning v2：连续提示优化",
    "43_检索增强生成 (RAG) 架构设计",
    "44_向量数据库与语义检索技术",
    "45_LangChain 框架核心组件分析",
    "46_ChatGLM：中英双语大模型实践",
    "47_LLaMA：开源大模型的里程碑",
    "48_Mistral：滑动窗口注意力机制",
    "49_MoE (Mixture of Experts) 混合专家模型",
    "50_命名实体识别 (NER) 与 BIO 标注",
    "51_依存句法分析 (Dependency Parsing)",
    "52_机器翻译中的 BLEU 指标计算",
    "53_文本摘要：抽取式与生成式对比",
    "54_情感分析与 Aspect-Based Sentiment",
    "55_问答系统：Extractive vs Generative QA",
    "56_对话系统与多轮上下文管理",
    "57_文本分类中的 Hierarchical Attention",
    "58_序列标注与 CRF 条件随机场",
    "59_语音识别：CTC Loss 原理",
    "60_Wav2Vec 2.0：自监督语音预训练"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
