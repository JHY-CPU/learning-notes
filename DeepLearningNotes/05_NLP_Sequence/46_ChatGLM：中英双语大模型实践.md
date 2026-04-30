# 46_ChatGLM：中英双语大模型实践

## 核心概念
- **ChatGLM**：由清华大学 KEG 团队和智谱 AI 联合开发的中英双语大语言模型系列。从 ChatGLM-6B 到 GLM-130B，再到 ChatGLM-3，在中文 NLP 领域具有重要影响力。
- **GLM 架构 (General Language Model)**：一种统一的预训练框架，结合自编码和自回归两种范式。使用自回归空白填充（Autoregressive Blank Infilling）作为预训练目标。
- **自回归空白填充**：随机遮盖输入中的连续 token 跨度（类似 T5 的 Span Corruption），但使用自回归方式（从左到右）生成被遮盖的内容，而不是并行预测。
- **二维位置编码 (2D Positional Encoding)**：GLM 为每个 token 编码两个位置信息——在原始序列中的位置和在被遮盖跨度中的位置，以支持部分自回归生成。
- **ChatGLM-6B 的"瘦身"实践**：使用 6B 参数量实现了接近 100B 级别模型的对话能力，通过量化（INT4/INT8）可以在消费级 GPU（6GB 显存）上运行。
- **中英双语优化**：训练数据包含大量中文语料（中文书籍、网页、百科、新闻等），使模型在中英文任务上都有良好表现。
- **对话式微调**：ChatGLM 系列在预训练基础上进行了多轮对话微调，支持上下文管理、角色扮演、工具使用等功能。
- **FlashAttention 和 P-Tuning v2 的整合**：ChatGLM-6B 集成了 FlashAttention（加速长序列计算）和 P-Tuning v2（高效微调），使其在实际应用中更有优势。

## 数学推导
GLM 的自回归空白填充目标：
给定输入 $\mathbf{x}$，随机采样若干跨度 $\{s_1, s_2, \ldots, s_k\}$，每个跨度长度为 $l_i$。

输入的噪声版本 $\mathbf{x}_{\text{corrupt}}$ 是原文去掉所有被选中的跨度部分。

预训练目标（自回归方式填充所有空白）：
$$
\max_{\theta} \log P_{\theta}(\mathbf{x}_{\text{span}_1}, \ldots, \mathbf{x}_{\text{span}_k} | \mathbf{x}_{\text{corrupt}})
$$

$$
= \max_{\theta} \sum_{i=1}^{k} \sum_{j=1}^{l_i} \log P_{\theta}(x_{\text{span}_i, j} | \mathbf{x}_{\text{corrupt}}, \mathbf{x}_{\text{span}_{<i}}, x_{\text{span}_i, <j})
$$

这与 T5 的预训练目标类似，但 T5 使用编码器-解码器架构并行输出，GLM 使用唯一的解码器自回归输出。

## 直观理解
- **ChatGLM 像"中英文双语者"**：很多模型是"先学英文再学中文"（英文数据为主），但 ChatGLM 是中英文"双语者"，两种语言都精通。这体现在它对中文 idiom（成语）、诗歌、俗语等的理解力上。
- **GLM 的自回归空白填充像"边想边填"**：不同于 BERT 的 MLM（所有空白同时预测），GLM 的空白填充是自回归的——先填第一个空格，再看结果填第二个，以此类推。这比 MLM 更适合生成任务。
- **2D 位置编码像"双重定位"**：一个 token 既要知道自己在原文中的位置（第几句话），也要知道在被遮盖跨度中的位置（是空格的第几个字）。就像你既要知道自己在哪条街，也要知道自己在这栋楼的第几层。

## 代码示例
```python
# ChatGLM-6B 使用示例
# 需要安装: pip install transformers accelerate

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch

# 注意：需要从 HuggingFace 下载模型权重
# 这里使用代码格式展示使用方式

def chatglm_example():
    """ChatGLM-6B 使用示例（模拟）"""
    # 实际加载（需要下载模型）:
    # tokenizer = AutoTokenizer.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True)
    # model = AutoModel.from_pretrained("THUDM/chatglm-6b", trust_remote_code=True).half().cuda()
    # model = model.eval()

    # ChatGLM 的特色：对话式调用
    # response, history = model.chat(tokenizer, "你好", history=[])
    # print(response)  # "你好！我是 ChatGLM-6B，有什么可以帮助你的吗？"
    #
    # response, history = model.chat(tokenizer, "请介绍一下机器学习", history=history)
    # print(response)

    print("ChatGLM-6B 核心特性:")
    print("  - 6B 参数，INT4 量化后仅需 6GB 显存")
    print("  - 原生支持中英双语")
    print("  - 多轮对话能力")
    print("  - 支持 P-Tuning v2 微调")
    print("  - 基于 GLM 架构（自回归空白填充）")

# ChatGLM 的对话格式
dialogue_example = [
    {"role": "user", "content": "你好"},
    {"role": "assistant", "content": "你好！我是 ChatGLM-6B，很高兴为你服务。"},
    {"role": "user", "content": "请用 Python 写一个冒泡排序"},
    {"role": "assistant", "content": "以下是冒泡排序的 Python 实现：\n\ndef bubble_sort(arr):\n    n = len(arr)\n    for i in range(n-1):\n        for j in range(n-i-1):\n            if arr[j] > arr[j+1]:\n                arr[j], arr[j+1] = arr[j+1], arr[j]\n    return arr"},
]

chatglm_example()
print("\nChatGLM 典型对话流程:")
for msg in dialogue_example:
    print(f"  [{msg['role']}] {msg['content'][:50]}...")
```

## 深度学习关联
- **中文 LLM 的代表**：ChatGLM 与文心一言、通义千问、讯飞星火等共同构成了中文大语言模型的生态。GLM 架构的独特性使其在学术界和开源社区都有重要影响。
- **开源大模型的生态贡献**：ChatGLM-6B 的开源使得研究人员和中小企业可以在消费级硬件上部署和微调大模型，推动了中文 NLP 的民主化。
- **架构创新的探索**：GLM 尝试统一填空和生成两种范式，代表了对"BERT vs GPT"二元对立的一种超越。类似的尝试还有 UniLM (统一 LM) 和 T5。
