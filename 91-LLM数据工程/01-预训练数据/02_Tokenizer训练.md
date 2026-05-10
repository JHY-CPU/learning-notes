# Tokenizer训练 - LLM预训练数据

*分词器是大语言模型的"第一道门"，深入理解BPE、SentencePiece、Unigram等分词算法，以及词表大小、特殊Token、多语言支持等关键设计决策*

BPE算法核心流程

1. **初始化**
   : 将所有单词拆分为字符序列，每个字符作为一个token
2. **统计频率**
   : 统计所有相邻token对的共现频率
3. **合并**
   : 将频率最高的token对合并为新token，加入词表
4. **迭代**
   : 重复步骤2-3，直到词表达到目标大小

Unigram算法流程

1. **初始化**
   : 从一个足够大的候选词表开始（如所有出现的子串）
2. **训练**
   : 在候选词表上训练Unigram语言模型，估计每个token的概率
3. **计算loss**
   : 对每个训练样本，计算去掉每个候选token后loss的增加量
4. **裁剪**
   : 移除loss增加最小的token（约保留80%），缩小词表
5. **迭代**
   : 重复步骤2-4，直到词表达到目标大小


**优势**: 输出带有概率信息，同一文本可产生多种分词结果（适合数据增强）

词表大小的权衡

| 词表大小 | 代表模型 | 优势 | 劣势 |
| --- | --- | --- | --- |
| 32K | LLaMA, T5, Mistral | Embedding参数少，训练快 | 中文等语言序列更长 |
| 50K | BERT | 适中的平衡点 | 对多语言不够友好 |
| 100K+ | Gemini, Qwen | 多语言支持好，中文效率高 | Embedding层参数大 |
| 256 (字节) | GPT-2原始版 | 极小词表，完全无OOV | 序列极长，训练慢 |


**经验值**: 英文为主的模型通常选择32K-50K；中英双语建议64K-150K；多语言模型建议100K-256K。

提升多语言分词质量的方法

- **提高character_coverage**
   : SentencePiece中设置0.9995-0.9999，确保覆盖所有字符
- **增大词表**
   : 多语言模型建议100K-256K词表
- **采样平衡**
   : 对训练语料进行语言平衡采样，避免低资源语言被"吃掉"
- **byte_fallback**
   : 启用字节回退，确保任何罕见字符都能被编码
- **用户自定义符号**
   : 将高频中文词汇作为用户定义符号强制加入词表

分词效率对比（相同文本）

| Tokenizer | 词表大小 | 英文压缩率 | 中文压缩率 | 备注 |
| --- | --- | --- | --- | --- |
| GPT-2 BPE | 50,257 | ~0.25 tok/char | ~1.5 tok/char | 中文效率极低 |
| LLaMA SP | 32,000 | ~0.27 tok/char | ~0.9 tok/char | 中等 |
| Qwen BPE | 151,936 | ~0.23 tok/char | ~0.4 tok/char | 中文效率高 |
| ChatGLM SP | 64,793 | ~0.24 tok/char | ~0.5 tok/char | 中英平衡 |

主流模型Tokenizer对比

| 模型 | 算法 | 词表大小 | 框架 |
| --- | --- | --- | --- |
| GPT-4 / ChatGPT | Byte-level BPE | 100,277 (cl100k) | tiktoken |
| GPT-4o | Byte-level BPE | 200,019 (o200k) | tiktoken |
| LLaMA 2/3 | BPE | 128,256 / 128K | SentencePiece |
| Mistral | BPE | 32,000 | SentencePiece |
| Qwen 2 | Byte-level BPE | 151,936 | tiktoken-based |
| ChatGLM-3 | BPE | 64,793 | SentencePiece |
| Claude | Byte-level BPE | 未公开 | 未公开 |


<!-- Converted from: 02_Tokenizer训练.html -->
