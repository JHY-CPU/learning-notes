# 52_图像描述 (Image Captioning) 生成技术

## 核心概念

- **图像描述（Image Captioning）**：自动生成描述图像内容的自然语言句子。这需要结合计算机视觉（理解图像中的物体和关系）和自然语言生成（产生流畅的句子）的能力。
- **编码器-解码器架构**：经典图像描述框架——编码器（CNN/ViT）提取图像视觉特征，解码器（RNN/LSTM/Transformer）基于视觉特征逐词生成描述文本。
- **Show and Tell (2015)**：Vinyals et al. 的开创性工作，使用Inception V3作为图像编码器，LSTM作为文本解码器，将图像特征作为LSTM的初始状态输入。
- **Show, Attend and Tell (2016)**：引入空间注意力机制，解码器在生成每个词时关注图像的不同空间区域，能够产生更精确的描述。
- **Transformer在Captioning中的应用**：使用Transformer解码器替代RNN/LSTM，通过交叉注意力（cross-attention）让文本Token关注图像特征，实现并行训练和更长的序列建模。
- **评估指标**：BLEU（n-gram精确率）、ROUGE-L（最长公共子序列）、METEOR（基于对齐的评分）、CIDEr（基于TF-IDF的加权n-gram相似度）、SPICE（基于场景图的语义评估）。

## 数学推导

**Show and Tell 的编码器-解码器架构：**

图像编码：$v = \text{CNN}(I) \in \mathbb{R}^{D}$（最后一层池化后的全局特征）

文本解码（LSTM）：
$$
h_t = \text{LSTM}([x_t; v], h_{t-1})
$$
$$
p(y_t | y_{<t}, v) = \text{softmax}(W h_t + b)
$$

其中 $x_t = E[y_{t-1}]$ 是前一个词的词嵌入。

**Show, Attend and Tell 的注意力解码：**

在时间步 $t$，计算空间特征 $\{a_1, \dots, a_L\}, a_i \in \mathbb{R}^{2048}$ 的注意力权重：
$$
e_{ti} = f_{att}(h_{t-1}, a_i)
$$
$$
\alpha_{ti} = \frac{\exp(e_{ti})}{\sum_{k=1}^L \exp(e_{tk})}
$$
$$
\hat{z}_t = \sum_{i=1}^L \alpha_{ti} a_i
$$

加权的图像特征 $\hat{z}_t$ 输入LSTM生成下一个词：
$$
h_t = \text{LSTM}([x_t; \hat{z}_t], h_{t-1})
$$

**自回归解码（推理阶段）：**
生成时，模型在每一步生成一个词的概率分布，选择概率最高的词（贪婪解码）或使用束搜索（Beam Search）：
$$
\hat{y}_t = \arg\max p(y_t | y_{<t}, v)
$$

束搜索在每一步保留 $k$ 个最可能的部分序列，最终选择整体概率最高的完整序列。

**训练损失（交叉熵）：**
$$
\mathcal{L} = -\sum_{t=1}^T \log p(y_t^* | y_{<t}^*, v)
$$

其中 $y_t^*$ 是真实描述的第 $t$ 个词。

## 直观理解

图像描述可以理解为"看图说话"——让AI像人一样，看一眼图片就能说出合理的描述。编码器的作用是"看懂图片"，解码器的作用是"把看懂的内容说出来"。

注意力机制让这个过程更智能——当模型说"狗"这个词时，它会"看"图像中狗的位置；当说"在跑步"时，会"看"狗腿的位置。这种"说哪里看哪里"的能力使描述更加准确和可解释。

束搜索（Beam Search）则可以理解为"写作时的打草稿"——不是每次只选最可能的词（贪婪），而是保留几个候选句子一起写，最后选最好的。这避免了"一着不慎满盘皆输"的问题（早期选错词导致整句不通顺）。

## 代码示例

```python
import torch
import torch.nn as nn

class AttentionCaptioning(nn.Module):
    """带注意力的图像描述模型 (简化)"""
    def __init__(self, vocab_size=10000, embed_dim=256, hidden_dim=512,
                 num_visual_features=49, visual_dim=2048):
        super().__init__()
        # 词嵌入
        self.word_embed = nn.Embedding(vocab_size, embed_dim)
        # 图像特征投影
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        # 注意力
        self.attn = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )
        # LSTM解码器
        self.lstm = nn.LSTMCell(embed_dim + hidden_dim, hidden_dim)
        # 输出分类器
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, vocab_size),
        )

    def forward(self, image_features, captions):
        """
        image_features: (B, num_features, visual_dim) 空间特征
        captions: (B, seq_len) 描述词索引
        """
        B, L, _ = image_features.shape
        seq_len = captions.shape[1]
        
        # 投影视觉特征
        V = self.visual_proj(image_features)  # (B, L, hidden)
        
        # 初始化LSTM状态
        h = torch.zeros(B, self.lstm.hidden_size).to(image_features.device)
        c = torch.zeros(B, self.lstm.hidden_size).to(image_features.device)
        
        # 词嵌入
        captions_embed = self.word_embed(captions)  # (B, seq_len, embed_dim)
        
        outputs = []
        for t in range(seq_len):
            # 注意力
            h_expanded = h.unsqueeze(1).expand(-1, L, -1)  # (B, L, hidden)
            attn_input = torch.cat([V, h_expanded], dim=2)
            attn_scores = self.attn(attn_input).squeeze(2)  # (B, L)
            attn_weights = torch.softmax(attn_scores, dim=1)  # (B, L)
            context = (attn_weights.unsqueeze(2) * V).sum(dim=1)  # (B, hidden)
            
            # LSTM
            lstm_input = torch.cat([captions_embed[:, t, :], context], dim=1)
            h, c = self.lstm(lstm_input, (h, c))
            
            # 输出
            logits = self.classifier(h)
            outputs.append(logits)
        
        return torch.stack(outputs, dim=1)  # (B, seq_len, vocab_size)

# 测试
model = AttentionCaptioning()
img_feat = torch.randn(2, 49, 2048)  # 7x7 grid features
captions = torch.randint(0, 10000, (2, 20))  # 描述词序列
output = model(img_feat, captions)
print(f"输出: {output.shape}")  # (2, 20, 10000)

# 推理: 束搜索 (简化版本)
def greedy_decode(model, image_features, start_token=1, end_token=2, max_len=30):
    model.eval()
    with torch.no_grad():
        B = image_features.shape[0]
        V = model.visual_proj(image_features)
        L = V.shape[1]
        
        h = torch.zeros(B, model.lstm.hidden_size)
        c = torch.zeros(B, model.lstm.hidden_size)
        generated = torch.full((B, 1), start_token, dtype=torch.long)  # <start>
        
        for _ in range(max_len):
            h_exp = h.unsqueeze(1).expand(-1, L, -1)
            attn_scores = model.attn(torch.cat([V, h_exp], dim=2)).squeeze(2)
            attn_weights = torch.softmax(attn_scores, dim=1)
            context = (attn_weights.unsqueeze(2) * V).sum(dim=1)
            
            word_embed = model.word_embed(generated[:, -1])
            h, c = model.lstm(torch.cat([word_embed, context], dim=1), (h, c))
            
            logits = model.classifier(h)
            next_word = logits.argmax(dim=1).unsqueeze(1)
            generated = torch.cat([generated, next_word], dim=1)
            
            if (next_word == end_token).all():
                break
        return generated

print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")
```

## 深度学习关联

- **视觉-语言多模态生成的基础**：图像描述是视觉-语言生成任务的基石，其编码器-解码器+注意力框架被扩展到视频描述（Video Captioning）、图像故事生成（Visual Storytelling）、图文融合生成等任务。
- **从RNN到Transformer的演进**：图像描述的解码器从LSTM演进到Transformer，大大提升了训练效率和长序列建模能力。现代的Captioning模型（如BLIP-2、GIT、Florence-2）使用大规模多模态预训练，在描述质量上大幅超越早期方法。
- **与多模态大模型的融合**：最新的图像描述方法（如LLaVA、GPT-4V）不再将描述视为独立的序列生成任务，而是将其融入多模态对话框架中——"用一句话描述这张图片"只是MLLM的一个简单指令，描述能力是通用多模态理解的副产品。
