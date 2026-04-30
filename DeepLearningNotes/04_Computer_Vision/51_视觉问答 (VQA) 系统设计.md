# 51_视觉问答 (VQA) 系统设计

## 核心概念

- **视觉问答（Visual Question Answering, VQA）**：给定一张图像和一个关于该图像的自然语言问题，系统需要给出正确的答案。VQA需要同时理解视觉内容和语义问题。
- **VQA v2.0数据集**：标准VQA基准数据集，包含约20万张COCO图像、110万个问题和1000万个答案。训练/验证/测试集分离。问题类型包括"是/否"、"计数"、"物体"、"位置"等。
- **经典VQA框架（两阶段）**：**(1)** 图像编码器（CNN）提取视觉特征，文本编码器（RNN/LSTM/Transformer）提取问题特征；**(2)** 多模态融合模块将两种特征融合后送入分类器预测答案。
- **注意力机制**：在VQA中，注意力机制用于让问题关注图像中的相关区域（视觉注意力），或让图像关注问题中的相关词（文本注意力）。Bottom-Up Top-Down Attention是经典方法。
- **答案空间**：VQA通常被视为分类问题——从预定义的答案词汇表（最频繁的3000-10000个答案）中选择一个答案。开放生成式VQA近年来越来越流行。
- **多模态融合方法**：从简单的拼接、逐元素乘/加，到双线性融合（MCB、MUTAN、BLOCK）、Transformer交叉注意力，融合方法不断改进。

## 数学推导

**Bottom-Up Top-Down Attention：**

自底向上（Bottom-Up）——使用Faster R-CNN检测图像中的显著区域（物体提议），提取每个区域的视觉特征 $\{v_1, \dots, v_K\}, v_i \in \mathbb{R}^{2048}$。

自顶向下（Top-Down）——使用问题特征 $q \in \mathbb{R}^{1024}$ 计算每个视觉区域的注意力权重：
$$
a_i = w^T \cdot f(W_v v_i + W_q q + b)
$$
$$
\alpha_i = \text{softmax}(a_i)
$$
$$
\hat{v} = \sum_{i=1}^K \alpha_i v_i
$$

其中 $W_v, W_q, w$ 是可学习参数，$f$ 是激活函数（如ReLU）。

**多模态融合与答案预测：**
$$
h = \text{LayerNorm}(W_h [\hat{v}; q] + b_h)
$$
$$
p(ans) = \text{softmax}(W_o h + b_o)
$$

**VQA的损失函数（多标签分类）：**
$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^N \sum_{a \in \mathcal{A}} s_{ia} \log(p(a | v_i, q_i))
$$

其中 $s_{ia}$ 是软得分（VQA中每个问题有10个答案，可以通过投票生成软标签）。

## 直观理解

VQA系统的工作方式类似于一个"看图回答问题"的AI。想象你给一个人看一张照片，然后问"图中穿红色衣服的人在做什么？"——这个人需要先理解问题（"红色衣服"和"在做什么"是两个关键信息），再在图像中找到穿红色衣服的人，最后分析他的动作，给出答案。

注意力机制模仿了人类在回答问题时的"关注策略"——当被问到"红色衣服"时，人类的视线自然会关注图像中的红色区域；当被问到"在做什么"时，会关注人的手部和动作。Bottom-Up Attention使用物体检测器预提取图像中的"显著区域"（物体、人、物体部件等），再根据问题计算这些区域的重要性权重。

## 代码示例

```python
import torch
import torch.nn as nn

class BottomUpTopDownAttention(nn.Module):
    """自底向上-自顶向下注意力"""
    def __init__(self, visual_dim=2048, hidden_dim=512, question_dim=1024):
        super().__init__()
        self.W_v = nn.Linear(visual_dim, hidden_dim)
        self.W_q = nn.Linear(question_dim, hidden_dim)
        self.w = nn.Linear(hidden_dim, 1)

    def forward(self, visual_features, question_feature):
        # visual_features: (B, K, 2048) K个区域特征
        # question_feature: (B, 1024)
        q = self.W_q(question_feature).unsqueeze(1)  # (B, 1, hidden)
        v = self.W_v(visual_features)  # (B, K, hidden)
        # 注意力得分
        a = self.w(torch.tanh(v + q))  # (B, K, 1)
        alpha = torch.softmax(a, dim=1)  # (B, K, 1)
        # 注意力加权聚合
        v_attended = (alpha * v).sum(dim=1)  # (B, hidden)
        return v_attended, alpha

class VQAModel(nn.Module):
    """VQA 模型 (简化版)"""
    def __init__(self, vocab_size=10000, answer_size=3000):
        super().__init__()
        # 文本编码器 (简化: LSTM)
        self.word_embed = nn.Embedding(vocab_size, 300)
        self.text_encoder = nn.LSTM(300, 512, bidirectional=True, batch_first=True)
        self.text_proj = nn.Linear(1024, 1024)
        
        # 注意力
        self.attention = BottomUpTopDownAttention(
            visual_dim=2048, hidden_dim=512, question_dim=1024
        )
        
        # 分类器
        self.classifier = nn.Sequential(
            nn.Linear(512 + 1024, 1024),
            nn.Tanh(),
            nn.Dropout(0.5),
            nn.Linear(1024, answer_size),
        )

    def forward(self, images, questions, question_lengths):
        # images: 区域特征 (B, K, 2048)
        # questions: (B, seq_len) 问题词索引
        q_embed = self.word_embed(questions)
        q_packed = nn.utils.rnn.pack_padded_sequence(
            q_embed, question_lengths, batch_first=True, enforce_sorted=False
        )
        q_output, (h_n, _) = self.text_encoder(q_packed)
        # 取最后一层的隐藏状态拼接
        q_feat = torch.cat([h_n[0], h_n[1]], dim=1)  # (B, 1024)
        q_feat = self.text_proj(q_feat)
        
        # 注意力
        v_attended, attn_weights = self.attention(images, q_feat)
        
        # 融合并分类
        combined = torch.cat([v_attended, q_feat], dim=1)
        logits = self.classifier(combined)
        return logits, attn_weights

# 测试
model = VQAModel(vocab_size=5000, answer_size=3000)
img_regions = torch.randn(2, 36, 2048)  # 每张图36个区域
questions = torch.randint(0, 5000, (2, 20))  # 问题序列
lengths = torch.tensor([15, 12])

logits, attn = model(img_regions, questions, lengths)
print(f"答案logits: {logits.shape}")
print(f"注意力权重: {attn.shape}")  # (2, 36, 1)

print(f"\nVQA参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **多模态学习的基准任务**：VQA是视觉-语言多模态学习的基准任务，推动了多模态表示学习、跨模态注意力、细粒度视觉-语言对齐等技术的发展。
- **从分类到生成的演进**：VQA从"分类式VQA"（从固定答案集选择）演进到"生成式VQA"（自由文本生成，如LLaVA、BLIP-2、InstructBLIP），随着大语言模型的发展，现代VQA系统更倾向于使用多模态大模型（MLLM）进行开放式问答。
- **视觉-语言基础模型的评估平台**：VQA已成为评估多模态大模型（如GPT-4V、Gemini、Qwen-VL等）视觉理解能力的重要基准，VQA v2.0、GQA、TextVQA、VizWiz等子任务覆盖了不同维度的视觉理解能力。
