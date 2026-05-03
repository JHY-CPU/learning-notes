# 54_情感分析与 Aspect-Based Sentiment

## 核心概念

- **情感分析 (Sentiment Analysis)**：自动识别和提取文本中的主观信息（情绪、态度、意见）。最基础的形式是文本级三分类（积极/消极/中性）。
- **文本级情感分析**：对整个句子或文档判断情感极性（正面/负面/中性）。方法包括词典方法、传统机器学习（SVM、朴素贝叶斯）、深度学习（LSTM、BERT）。
- **方面级情感分析 (Aspect-Based Sentiment Analysis, ABSA)**：更细粒度的情感分析。识别文本中提到的"方面"（aspect，即评价对象）及其对应的情感极性。
- **ABSA 的子任务**：
  - **方面提取 (Aspect Extraction)**：识别文本中评价的对象，如"屏幕"、"电池"、"相机"
  - **观点抽取 (Opinion Extraction)**：提取对应的评价词，如"清晰"、"耐用"
  - **情感极性分类 (Sentiment Polarity)**：判断每个方面的情感是正面、负面还是中性
  - **方面-观点配对**：将方面和对应的观点词正确匹配
- **句子"酒店不错但有点贵"**：方面"酒店" -> 正面；方面"价格" -> 负面。同一句话包含对两个不同方面的矛盾情感。
- **细粒度情感分析**：在方面级的基础上进一步细分——不仅分析"屏幕好"（方面+极性），还要分析"屏幕清楚但太小"（同一方面的不同属性）。
- **端到端 ABSA**：现代模型使用 BERT 等预训练模型联合训练所有子任务，而非分阶段处理。

## 数学推导

**文本级情感分类**（使用 BERT）：
$$
P(y | \text{text}) = \text{softmax}(W [\text{CLS}] + b)
$$

其中 $[\text{CLS}]$ 是 BERT 最后一层的 [CLS] 表示，$y \in \{\text{positive}, \text{negative}, \text{neutral}\}$。

**ABSA 的方面提取 + 极性分类**：
给定输入 $\mathbf{x} = (x_1, \ldots, x_n)$，预测每个 token 的标签 $y_i$（方面位置 + 情感极性）：
$$
\text{Label set} = \{O, B\text{-}POS, I\text{-}POS, B\text{-}NEG, I\text{-}NEG, B\text{-}NEU, I\text{-}NEU\}
$$

损失函数（多任务学习）：
$$
\mathcal{L} = \mathcal{L}_{\text{aspect}} + \lambda \mathcal{L}_{\text{sentiment}}
$$

**方面-观点对的提取**（使用序列标注 + 距离约束）：
$$
P(\text{opinion}_j | \text{aspect}_i) = \text{softmax}(E_{\text{aspect}_i}^\top W E_{\text{opinion}_j})
$$

## 直观理解

- **文本级 vs 方面级**：文本级情感分析就像给整篇作文打分"好"或"差"。方面级分析就像在作文评语中分别评价"论点明确"（正面）、"论据不足"（负面）、"语言流畅"（正面）。同一篇作文可能包含对不同维度的不同评价。
- **ABSA 的实用场景**：餐厅评论"环境很好但服务很差"——文本级分析会困惑（正负混合），ABSA 能给出"环境:+1, 服务:-1"的精确分析。对于商家来说，知道需要改善服务而非环境，是更 actionable 的信息。
- **方面-观点对的匹配**："这款手机屏幕清晰但系统卡顿"——需要把"清晰"匹配给"屏幕"，"卡顿"匹配给"系统"。如果错误匹配成"屏幕卡顿"，就是完全相反的理解。

## 代码示例

```python
from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
import torch

# 1. 文本级情感分析
sentiment_pipeline = pipeline("sentiment-analysis")
result = sentiment_pipeline("I love this product! It's amazing.")
print(f"文本级情感: {result[0]['label']} (score: {result[0]['score']:.4f})")

# 2. 使用 BERT 做情感分类
tokenizer = AutoTokenizer.from_pretrained("nlptown/bert-base-multilingual-uncased-sentiment")
model = AutoModelForSequenceClassification.from_pretrained(
    "nlptown/bert-base-multilingual-uncased-sentiment"
)

def predict_sentiment(text):
    inputs = tokenizer(text, return_tensors='pt')
    with torch.no_grad():
        outputs = model(**inputs)
    scores = torch.softmax(outputs.logits, dim=1)
    rating = torch.argmax(scores, dim=1).item() + 1
    return rating

print(f"\"This movie is great\" 评分: {predict_sentiment('This movie is great')}/5")
print(f"\"This movie is terrible\" 评分: {predict_sentiment('This movie is terrible')}/5")

# 3. 方面级情感分析的模拟流程
def aspect_based_sentiment(text, aspects):
    """简化的 ABSA 流程"""
    results = {}
    for aspect in aspects:
        if aspect in text:
            # 简单策略：根据上下文判断（实际应使用专门的 ABSA 模型）
            if "好" in text or "不错" in text or "喜欢" in text:
                results[aspect] = "positive"
            elif "差" in text or "不好" in text or "贵" in text:
                results[aspect] = "negative"
            else:
                results[aspect] = "neutral"
    return results

review = "这家酒店的早餐很不错，但价格太贵了。"
aspects_to_check = ["早餐", "价格", "服务", "位置"]
abasa_result = aspect_based_sentiment(review, aspects_to_check)
print(f"\nABS 分析结果:")
for aspect, sentiment in abasa_result.items():
    print(f"  方面 '{aspect}': {sentiment}")
```

## 深度学习关联

- **预训练模型提升 ABSA 效果**：BERT 及其变体显著提升了方面级情感分析的性能。基于 BERT 的 ABSA 模型（如 BERT-AD、BERT-PT）在 SemEval ABSA 基准上达到了 90%+ 的 F1 值。
- **多任务联合学习**：ABSA 的各个子任务（方面提取、极性分类、观点抽取）相互关联，多任务联合训练通常优于独立训练。Span-level 的联合模型是当前的主流方法。
- **生成式 ABSA**：大语言模型（ChatGPT、GPT-4）直接将 ABSA 作为结构化文本生成任务——"分析以下评论的情感[积极/消极/中性]，并列出评价对象"。生成式方法避免了对复杂标注体系的工程需求。
