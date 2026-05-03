# 25_Text-to-Image 模型中的 CLIP 文本编码器

## 核心概念

- **CLIP (Contrastive Language-Image Pre-training)**：OpenAI 开发的多模态模型，使用对比学习在 4 亿图文对上进行训练，学习将文本和图像映射到同一语义空间。
- **双塔架构**：CLIP 包含两个编码器——文本编码器（Transformer）和图像编码器（ViT 或 ResNet），分别将文本和图像编码为固定维度的向量。
- **对比损失 (InfoNCE)**：训练目标是在 batch 内最大化配对的图文向量相似度，最小化非配对图文向量的相似度。这使编码器学到语义对齐的表示。
- **文本编码器在 T2I 中的作用**：在 Stable Diffusion 中，CLIP 文本编码器将用户输入的自然语言提示转换为条件嵌入 $\tau_\theta(c)$，通过交叉注意力注入到生成过程中。
- **最后隐藏层 vs 倒数第二层**：CLIP 的最后隐藏层主要用于判别任务（图文匹配），而倒数第二层在生成任务中表现更好。SDXL 使用了两个 CLIP 模型（OpenCLIP ViT-bigG 和 CLIP ViT-L）的拼接。
- **开放词汇理解**：CLIP 的训练数据覆盖了几乎所有的视觉概念，使其能够理解自然语言描述而不仅仅是预定义类别——这是开放式文本到图像生成的基石。

## 数学推导

**CLIP 训练目标（InfoNCE 损失）**：

对于 batch 中的 $N$ 个图文对 $(x_i, y_i)$，文本编码器输出 $t_i = \text{TextEncoder}(y_i)$，图像编码器输出 $v_i = \text{ImageEncoder}(x_i)$。

$$
\mathcal{L} = -\frac{1}{2N} \sum_{i=1}^N \left[ \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(v_i, t_j)/\tau)} + \log \frac{\exp(\text{sim}(t_i, v_i)/\tau)}{\sum_{j=1}^N \exp(\text{sim}(t_i, v_j)/\tau)} \right]
$$

其中 $\text{sim}(v, t) = \frac{v \cdot t}{\|v\| \cdot \|t\|}$ 是余弦相似度，$\tau$ 是可学习的温度参数。

**文本编码器结构**（CLIP ViT-L/14 版本）：

- 12 层 Transformer decoder
- 隐藏维度 768（ViT-L）/ 1024（ViT-bigG）
- 多头注意力：12 头
- 最大序列长度：77 tokens
- 特殊 token：[SOS] text [EOS]——输出取 [EOS] 位置的特征作为全局表示

**文本嵌入在交叉注意力中的使用**：

在 U-Net 的交叉注意力层中，文本嵌入作为 Key 和 Value：

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{d}}\right) V
$$

$$
Q = W_Q \cdot \text{U-Net 特征}, \quad K = W_K \cdot \text{CLIP 文本嵌入}, \quad V = W_V \cdot \text{CLIP 文本嵌入}
$$

这使得每个图像 patch 通过注意力权重关注文本中最相关的 token——"一只黑色的猫"中的"猫"和"黑色"分别与图像的不同区域对齐。

## 直观理解

- **CLIP 文本编码器 = 理解你说话的艺术品鉴赏家**：你说"一只戴着红色帽子的柯基犬"，CLIP 能理解"柯基犬"、"红色"、"帽子"这些概念，并知道它们组合在一起应该是怎样的。
- **对比学习的过程**：就像给 AI 看 100 万张图片和对应的描述，每次让它从一堆错误描述中找到正确的那一个。久而久之，AI 学会了"短腿狗"对应的是一张柯基照片，"长鼻子狗"对应的是腊肠犬。
- **为什么需要 77 token 限制**：文本编码器的最大长度是 77 个 token（约 50-60 个英文字符）。这是计算效率和指令完整性之间的折中——太短的信息不全，太长的算不动。
- **[EOS] token 的妙用**：文本末尾的 [EOS] token 的编码被视为"全局文本表示"，它聚合了整个句子的语义。在 Stable Diffusion 中，这个表示被用作无条件生成中的"空文本嵌入"。

## 代码示例

```python
import torch
import torch.nn as nn
import math

class SimpleCLIPTextEncoder(nn.Module):
    """
    简化的 CLIP 文本编码器
    
    实际 CLIP 使用 12 层 Transformer，此处为演示核心概念
    """
    def __init__(self, vocab_size=49408, hidden_size=768, max_length=77, num_layers=12):
        super().__init__()
        self.hidden_size = hidden_size
        self.max_length = max_length
        
        # Token 嵌入 + 位置编码
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.position_embedding = nn.Embedding(max_length, hidden_size)
        
        # Transformer 层（简化）
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=hidden_size, nhead=12, dim_feedforward=3072,
                activation='gelu', batch_first=True
            ) for _ in range(num_layers)
        ])
        
        self.final_ln = nn.LayerNorm(hidden_size)
    
    def forward(self, input_ids, attention_mask=None):
        """
        将文本 token 编码为嵌入
        
        参数:
            input_ids: [B, seq_len] token ID 序列
        返回:
            文本嵌入 [B, seq_len, hidden_size]
            pooled_output [B, hidden_size] ([EOS] 位置的输出)
        """
        seq_len = input_ids.size(1)
        positions = torch.arange(seq_len, device=input_ids.device).unsqueeze(0)
        
        # Token + 位置嵌入
        x = self.token_embedding(input_ids) + self.position_embedding(positions)
        
        # Transformer 编码
        for layer in self.layers:
            x = layer(x)
        
        x = self.final_ln(x)
        
        # 取 [EOS] token 位置（通常在序列末尾）作为全局表示
        pooled = x[torch.arange(x.size(0)), input_ids.argmax(dim=-1)]
        
        return x, pooled

class CrossAttentionWithCLIP(nn.Module):
    """CLIP 文本条件注入的交叉注意力层"""
    def __init__(self, query_dim=320, context_dim=768, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x, context):
        """
        x: U-Net 特征图 [B, C, H, W] -> 展平为 [B, H*W, C]
        context: CLIP 文本嵌入 [B, seq_len, context_dim]
        """
        B, C, H, W = x.shape
        x_flat = x.flatten(2).permute(0, 2, 1)  # [B, H*W, C]
        
        Q = self.to_q(x_flat)
        K = self.to_k(context)
        V = self.to_v(context)
        
        # 多头注意力
        Q = Q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        attn = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.head_dim)
        attn = torch.softmax(attn, dim=-1)
        
        out = torch.matmul(attn, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, C)
        out = self.to_out(out)
        
        # 恢复空间形状
        out = out.permute(0, 2, 1).view(B, C, H, W)
        return out

# 演示 CLIP 文本编码的使用
class T2IPipeline(nn.Module):
    """简化的文生图流程（仅演示文本编码部分）"""
    def __init__(self, text_encoder, unet):
        super().__init__()
        self.text_encoder = text_encoder
        self.unet = unet
    
    def encode_prompt(self, prompt, device='cpu'):
        """编码文本提示"""
        # 实际使用 tokenizer
        # 此处简化：随机生成 token ID 模拟
        dummy_ids = torch.randint(0, 100, (1, 77))
        text_emb, pooled = self.text_encoder(dummy_ids)
        return text_emb  # [1, 77, 768]

print("=== CLIP 文本编码器在 T2I 中的作用 ===")
print()
print("文本编码流程:")
print("  1. Tokenizer: 'a cat sitting on a mat' -> token IDs")
print("  2. CLIP Text Encoder: token IDs -> [1, 77, 768] 嵌入")
print("  3. Cross-Attention: 嵌入注入 U-Net 各层")
print()
print("关键特性:")
print("- 开放词汇: 理解任意自然语言描述")
print("- 对比学习: 文本与图像的语义对齐")
print("- 交叉注意力: 文本 token 对应图像的不同区域")
print(f"- 嵌入维度: 768 (ViT-L) 或 1024 (ViT-bigG)")
print(f"- 最大序列长度: 77 tokens")
```

## 深度学习关联

- **CLIP 的改进**：OpenCLIP（开源复现）、SigLIP（用 Sigmoid 损失替代 Softmax，训练更高效）、DFN（用数据过滤网络提升训练数据质量）。
- **T5 文本编码器**：Imagen (Google) 使用 T5-XXL 编码器替代 CLIP，因为 T5 在理解和遵循复杂指令上更强——但 T5 参数量巨大（11B），需要更多计算资源。
- **双编码器到单编码器的趋势**：最新的模型（如 Stable Diffusion 3 的 MMDeit、DALL-E 3）开始使用多模态 Transformer 同时编码文本和图像，而不是分开的 CLIP 编码器。
- **文本编码的 CFG (Classifier-Free Guidance)**：在推理时，同时计算有条件嵌入和无条件嵌入（空文本）的预测，通过插值放大条件的影响——这是 CLIP 文本编码器在推理时的关键应用。
