# 70_大模型评估基准：MMLU, GSM8K, HumanEval

## 核心概念
- **大模型评估的挑战**：传统 NLP 评估指标（如 BLEU、ROUGE）对 LLM 的评估过于局限。需要综合考察知识、推理、编码、安全等多维度能力，由此产生了专门的大模型评估基准。
- **MMLU (Massive Multitask Language Understanding)**：由 Hendrycks et al. (2020) 提出，涵盖 57 个学科的知识和理解能力测试，包括人文、社会科学、自然科学、技术等。格式为单选题，每个问题 4 个选项。
- **MMLU 的学科分布**：57 个任务分为人文学科（法律、历史、哲学）、社会科学（政治、经济、心理）、STEM（数学、物理、化学、计算机）、其他（医学、商业等）。
- **GSM8K (Grade School Math 8K)**：由 Cobbe et al. (2021) 提出，包含 8,500 个小学数学文字应用题。评估 LLM 的多步数学推理能力。难度相当于小学 3-8 年级。
- **HumanEval**：由 Chen et al. (2021) 提出，包含 164 个手写的 Python 编程问题。每个问题包含函数签名、文档字符串、测试用例。评估指标为 Pass@k——在 k 次生成中至少一次通过测试的概率。
- **MMLU-Pro**：MMLU 的增强版本，增加问题难度和选项数量（从 4 个增加到 10 个），减少随机猜对概率。
- **Big-Bench (Beyond the Imitation Game Benchmark)**：Google 的更大规模评估套件，包含 204 个任务，涵盖推理、知识、创造力、社会偏见等。
- **HellaSwag**：评估常识推理——从多个选项中选出一个故事最合理的结尾。对模型的反事实推理能力敏感。
- **TruthfulQA**：评估模型产生事实性回答的能力，专门设计"人类常见的误解"类问题，测试模型是否有"说出正确但反直觉事实"的倾向。

## 数学推导
**MMLU 的评分**：
对于 57 个学科的每个学科 $s$，模型在该学科上的准确率为：
$$
\text{Acc}_s = \frac{1}{N_s} \sum_{i=1}^{N_s} \mathbb{I}(\text{prediction}_i = \text{correct}_i)
$$

最终 MMLU 分数是所有学科的平均值：
$$
\text{MMLU} = \frac{1}{57} \sum_{s=1}^{57} \text{Acc}_s
$$

**GSM8K 的评分**：
$$
\text{GSM8K Acc} = \frac{\text{正确解答的问题数}}{\text{总问题数}}
$$

答案必须准确匹配（允许数值等价的不同格式，如"3"和"3.0"）。

**HumanEval Pass@k**：
Pass@k 是无偏估计量：
$$
\text{Pass@k} = \mathbb{E}_{\text{problems}} \left[1 - \frac{\binom{n - c}{k}}{\binom{n}{k}}\right]
$$

其中 $n$ 是每个问题生成的样本数（通常 $n=200$），$c$ 是通过测试的样本数。

## 直观理解
- **MMLU 像美国高考的 57 门科目综合测试**：涵盖历史、法律、物理、数学、医学等人类知识的多个领域。通过 MMLU 相当于让模型参加一个"全科竞赛"。GPT-4 在某些科目上超过了 90% 的人类应试者。
- **GSM8K 像小学数学应用题**："小明有 10 个苹果，给了小红 3 个，又买了 5 个，现在有多少个？"这类问题虽然题目简单，但需要多步推理和数字计算能力。LLM 在 GSM8K 上的表现从 2022 年的不到 20% 提升到 2024 年的 95%+。
- **HumanEval 像编程面试题**：给你一道题的描述和函数签名，让你写出正确的函数实现。Pass@1 是"一次写对的概率"，Pass@10 是"写 10 次至少有一次通过的概率"。GPT-4 的 Pass@1 约 67%，远高于 GPT-3.5 的 48%。
- **为什么需要多个基准**：一个模型可能在 MMLU 上表现很好（知识丰富），但在 GSM8K 上很差（推理能力弱），或者在 HumanEval 上很好（代码能力强）但在 TruthfulQA 上很差（容易产生幻觉）。多个基准全面刻画了模型的能力画像。

## 代码示例
```python
# 模拟大模型评估

# 1. MMLU 风格的多选题
mmlu_example = {
    "subject": "Physics",
    "question": "What is the SI unit of electric current?",
    "choices": ["A) Volt", "B) Ampere", "C) Ohm", "D) Coulomb"],
    "answer": "B"
}

print("MMLU 示例:")
print(f"  学科: {mmlu_example['subject']}")
print(f"  问题: {mmlu_example['question']}")
print(f"  选项: {mmlu_example['choices']}")
print(f"  答案: {mmlu_example['answer']}")

# 2. GSM8K 风格的应用题
gsm8k_example = {
    "question": "Janet's ducks lay 16 eggs per day. She eats three for breakfast every morning "
                "and bakes muffins for her friends every day with four. She sells the rest at the "
                "farmers' market daily for $2 per fresh duck egg. How much money does she make every day?",
    "solution": "Janet's ducks lay 16 eggs per day.\n"
                "She eats 3 per day and bakes with 4 per day.\n"
                "So she uses 3 + 4 = 7 eggs per day.\n"
                "That means she has 16 - 7 = 9 eggs left to sell.\n"
                "At $2 per egg, she makes 9 * 2 = $18 per day.\n"
                "Answer: 18",
    "answer": 18
}

print(f"\nGSM8K 示例:")
print(f"  问题: {gsm8k_example['question'][:60]}...")
print(f"  答案: {gsm8k_example['answer']}")

# 3. HumanEval 风格编程题
humaneval_example = {
    "prompt": "def has_close_elements(numbers: List[float], threshold: float) -> bool:\n"
              "    \"\"\"Check if in given list of numbers, are any two numbers closer to each other than\n"
              "    given threshold.\"\"\"\n",
    "canonical_solution": "    for i in range(len(numbers)):\n"
                          "        for j in range(i + 1, len(numbers)):\n"
                          "            if abs(numbers[i] - numbers[j]) < threshold:\n"
                          "                return True\n"
                          "    return False\n",
    "test": "assert has_close_elements([1.0, 2.0, 3.0], 0.5) == False\n"
            "assert has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3) == True"
}

print(f"\nHumanEval 示例:")
print(f"  函数描述: has_close_elements - 检测列表中是否有两个数的差小于阈值")
print(f"  测试用例: {humaneval_example['test']}")

# 4. 模型排行榜模拟
print(f"\n\n主流模型在三大基准上的表现 (2024 年参考数据):")
print(f"{'模型':<20} {'MMLU':>8} {'GSM8K':>8} {'HumanEval':>12}")
print("-" * 50)
models_scores = [
    ("GPT-4", 86.4, 92.0, 67.0),
    ("GPT-3.5", 70.0, 57.1, 48.1),
    ("Claude 3 Opus", 86.8, 95.0, 84.9),
    ("Gemini Ultra", 83.7, 87.5, 51.1),
    ("LLaMA-2-70B", 68.9, 56.8, 29.9),
    ("Mixtral 8x7B", 70.6, 74.4, 40.2),
]
for name, mmlu, gsm8k, humaneval in models_scores:
    print(f"{name:<20} {mmlu:>7.1f}% {gsm8k:>7.1f}% {humaneval:>10.1f}%")
```

## 深度学习关联
- **基准驱动的进步**：MMLU、GSM8K、HumanEval 是当前 LLM 竞争的核心指标。新模型发布时，这些基准分数被广泛报道——它们成为了模型能力的"代言指标"。但这种关注也可能导致"对基准的过度优化"（benchmark contamination）。
- **数据污染 (Data Contamination)**：一个重大担忧是——这些公开基准的数据可能已经被包含在模型的训练数据中。例如，如果 GSM8K 的题目出现在 CommonCrawl 中，模型可能"记住"了答案而非真正学会了推理。研究者通过创建新版本（MMLU-Pro）和使用动态测试（LiveBench）来应对。
- **从基准到实用性**：这些基准虽然重要，但并不能完全代表模型的"实用性"。一个模型可能在所有基准上排名第一，但在实际对话中的有用性和安全性却不理想。因此 Chatbot Arena 等"人类评估"与实际产品使用反馈是对基准评估的重要补充。
