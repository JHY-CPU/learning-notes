import os

titles = [
    "61_机器翻译中的 Transformer 实现细节",
    "62_文本生成中的 Beam Search 算法",
    "63_Top-k 与 Top-p (Nucleus) 采样策略",
    "64_Temperature 参数对生成分布的影响",
    "65_Repetition Penalty 重复惩罚机制",
    "66_长文本处理：Longformer 与 BigBird",
    "67_多语言模型：mBERT 与 XLM-R",
    "68_代码大模型：CodeLlama 与 StarCoder",
    "69_逻辑推理与大模型的思维链 (CoT)",
    "70_大模型评估基准：MMLU, GSM8K, HumanEval"
]

for t in titles:
    with open(f"{t}.md", "w", encoding="utf-8") as f:
        f.write(f"# {t}\n\n## 核心概念\n- \n\n## 数学推导\n$$\n\n$$\n\n## 深度学习关联\n- \n")
