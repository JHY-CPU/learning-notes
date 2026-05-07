# 9_幻觉 (Hallucination) 问题

## 1. 幻觉定义

大语言模型的**幻觉 (Hallucination)** 是指模型生成的内容虽然语法流畅、看起来合理，但**与事实不符、缺乏依据或完全捏造**。这是 LLM 落地应用中最关键的安全问题之一。

### 1.1 幻觉分类

```
幻觉类型
├── 事实幻觉 (Factual Hallucination)
│   ├── 事实错误: 与真实世界事实矛盾
│   ├── 虚构实体: 编造不存在的人物/事件/论文
│   ├── 时间错乱: 事件发生时间错误
│   └── 数字错误: 统计数据、日期、数量错误
├── 忠实幻觉 (Faithfulness Hallucination)
│   ├── 不忠实于输入: 与给定源文档矛盾
│   ├── 添加信息: 超出源文档范围的编造
│   └── 遗漏信息: 关键信息丢失导致误导
└── 指令幻觉 (Instruction Hallucination)
    └── 偏离指令: 未按用户要求回答
```

### 1.2 典型幻觉示例

```
用户: "请介绍一下 2024 年诺贝尔物理学奖得主。"

幻觉回答: "2024 年诺贝尔物理学奖授予了中国科学家张明教授，
以表彰他在量子纠缠理论方面的开创性贡献。张明教授任职于
北京大学物理学院，此前曾获得 2018 年沃尔夫物理学奖。"

问题:
- 张明教授并非真实人物
- 2024 年诺奖得主信息捏造
- 虚构的获奖履历
- 所有细节都看似合理但全部错误
```

## 2. 幻觉来源分析

### 2.1 训练阶段来源

| 来源 | 描述 | 影响 |
|------|------|------|
| 数据噪声 | 训练数据中的错误信息 | 模型学习错误知识 |
| 数据过时 | 知识时效性不足 | 输出过时事实 |
| 数据偏差 | 某些领域/观点过度代表 | 偏向特定立场 |
| 知识冲突 | 不同来源对同一事实有矛盾描述 | 模型产生不确定的输出 |
| 长尾知识 | 罕见实体/事件的信息不足 | 对长尾知识产生幻觉 |

### 2.2 推理阶段来源

| 来源 | 描述 | 影响 |
|------|------|------|
| 解码策略 | 采样温度过高 | 生成多样性增加但准确性下降 |
| 暴露偏差 | 训练时用真实标签、推理时用自身输出 | 误差累积 |
| 过度自信 | 模型对不确定内容给出确定性表述 | 用户难以区分真伪 |
| 位置偏差 | 关注前面的内容、忽略后面的内容 | 信息丢失 |
| 上下文长度限制 | 长文档截断 | 部分信息不可用 |

### 2.3 模型内在原因

```
幻觉的深层原因:

1. 记忆 vs 理解
   - 模型记忆了事实的关联模式，而非真正的因果理解
   - 当关联模式被触发但缺乏约束时，产生看似合理的编造

2. 流畅性与准确性的冲突
   - 模型被训练为生成流畅连贯的文本
   - 流畅性不等于准确性

3. 知识截止问题
   - 模型有知识截止日期
   - 超出训练数据范围时，模型倾向于"合理化"编造

4. 无确定性机制
   - 模型没有内置的事实核查机制
   - 无法区分"知道"和"不知道"
```

## 3. 幻觉的量化评估

### 3.1 评估指标

| 指标 | 定义 | 适用场景 |
|------|------|----------|
| 事实准确率 | 事实性陈述正确的比例 | 开放式问答 |
| 虚构率 | 编造的实体/事件的比例 | 知识密集型任务 |
| 忠实度 | 与源文档一致的程度 | 摘要、翻译 |
| 幻觉率 | 包含幻觉内容的样本比例 | 通用评估 |
| 信息覆盖率 | 正确覆盖关键信息的比例 | 摘要 |

### 3.2 FActScore 评估

```python
def compute_factscore(model_output, knowledge_source):
    """
    FActScore: 将生成文本分解为原子事实并逐一验证
    """
    # 步骤1: 分解为原子事实
    atomic_facts = extract_atomic_facts(model_output)

    # 步骤2: 对每个事实验证
    supported = 0
    for fact in atomic_facts:
        if verify_fact(fact, knowledge_source):
            supported += 1

    # 步骤3: 计算分数
    factscore = supported / len(atomic_facts) if atomic_facts else 0

    return {
        'factscore': factscore,
        'total_facts': len(atomic_facts),
        'supported': supported,
        'hallucinated': len(atomic_facts) - supported
    }

def extract_atomic_facts(text):
    """
    将文本分解为不可再分的原子事实
    """
    # 使用 LLM 进行分解
    prompt = f"""
    请将以下文本分解为独立的原子事实:

    文本: {text}

    每个事实应该是独立的、可验证的陈述。
    输出格式: 每行一个事实
    """

    response = llm.generate(prompt)
    facts = [line.strip() for line in response.split('\n') if line.strip()]
    return facts
```

## 4. 幻觉检测方法

### 4.1 SelfCheckGPT

```python
def selfcheckgpt(original_response, num_samples=5):
    """
    SelfCheckGPT: 通过多次采样检测幻觉
    原理: 真实信息在多次生成中一致性高，幻觉内容一致性低
    """
    # 生成多个采样响应
    samples = []
    for _ in range(num_samples):
        sample = model.generate(prompt, temperature=0.8)
        samples.append(sample)

    # 将原始响应分解为句子
    sentences = split_sentences(original_response)

    # 对每个句子检查一致性
    hallucination_scores = []
    for sentence in sentences:
        # 计算与其他采样的一致性
        consistency = 0
        for sample in samples:
            if sentence_in_text(sentence, sample):
                consistency += 1

        score = 1.0 - (consistency / num_samples)  # 不一致性 = 幻觉概率
        hallucination_scores.append(score)

    return {
        'sentence_scores': list(zip(sentences, hallucination_scores)),
        'avg_hallucination_score': np.mean(hallucination_scores)
    }
```

### 4.2 事实验证方法

```python
def verify_with_knowledge_base(claim, kb):
    """
    使用知识库验证声明
    """
    # 检索相关知识
    relevant_docs = kb.search(claim, top_k=5)

    # 使用 NLI 模型判断是否支持
    for doc in relevant_docs:
        nli_result = nli_model.predict(premise=doc, hypothesis=claim)
        if nli_result == 'entailment':
            return 'supported'
        elif nli_result == 'contradiction':
            return 'contradicted'

    return 'insufficient_evidence'
```

## 5. 缓解策略

### 5.1 训练阶段

| 方法 | 描述 | 效果 |
|------|------|------|
| 数据清洗 | 去除训练数据中的错误信息 | 中 |
| 知识增强训练 | 注入结构化知识 | 高 |
| RLHF 忠实度奖励 | 惩罚幻觉输出 | 高 |
| 对比学习 | 让模型区分真实和虚构内容 | 中 |

### 5.2 推理阶段

| 方法 | 描述 | 效果 |
|------|------|------|
| 降低温度 | 减少随机性 | 中 |
| 检索增强生成 (RAG) | 引入外部知识 | **高** |
| 思维链 (CoT) | 让模型展示推理过程 | 中 |
| 引用要求 | 要求模型标注信息来源 | 中 |
| 不确定性表达 | 允许模型说"我不确定" | 中 |

## 6. 总结

| 要点 | 说明 |
|------|------|
| 定义 | 模型生成看似合理但不真实的内容 |
| 类型 | 事实幻觉、忠实幻觉 |
| 根本原因 | 记忆≠理解、流畅性≠准确性 |
| 检测 | SelfCheckGPT、事实验证 |
| 缓解 | RAG、知识增强训练、RLHF |

---

**参考文献：**

1. Ji, Z. et al. (2023). *Survey of hallucination in natural language generation*. ACM Computing Surveys.
2. Manakul, P. et al. (2023). *SelfCheckGPT: Zero-resource black-box hallucination detection*. EMNLP.
3. Min, S. et al. (2023). *FActScore: Fine-grained atomic evaluation of factual precision*. EMNLP.
