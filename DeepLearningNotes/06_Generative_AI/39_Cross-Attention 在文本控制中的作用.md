# 39_Cross-Attention 在文本控制中的作用

## 核心概念
- **交叉注意力 (Cross-Attention)**：扩散模型中将文本条件注入图像生成过程的核心机制——图像特征作为 Query，文本嵌入作为 Key 和 Value，通过注意力机制让图像"看到"文本。
- **Query, Key, Value 的映射**：在 U-Net 中，图像特征 $x \in \mathbb{R}^{H \times W \times C}$ 通过 $W_Q$ 投影为 Query，文本嵌入 $c \in \mathbb{R}^{L \times d}$ 通过 $W_K, W_V$ 投影为 Key 和 Value。
- **注意力图 (Attention Map)**：$\text{Softmax}(QK^T/\sqrt{d})$ 产生的注意力图显示了每个图像区域关注了哪些文本 token——这是理解"模型看到了什么"的窗口。
- **文本到图像的对齐**：通过交叉注意力，图像的不同区域会被对应的文本描述所"激活"——例如，生成"一只黑色的猫"时，"黑色"的 token 注意力集中在猫所在区域的纹理上。
- **层级的语义理解**：U-Net 的低分辨率层（瓶颈）的交叉注意力关注整体语义（如"猫"的形状），高分辨率层关注细节属性（如"黑色"的纹理）。
- **Prompt 工程的基础**：理解交叉注意力的工作原理是提示词工程的核心——通过调整提示词的表达方式（顺序、措辞、分隔）可以改变注意力分布，进而控制生成结果。

## 数学推导

**交叉注意力的前向计算**：

给定图像特征 $x \in \mathbb{R}^{B \times H \times W \times C}$（展平为 $x \in \mathbb{R}^{B \times HW \times C}$）和文本嵌入 $c \in \mathbb{R}^{B \times L \times d}$：

$$
Q = x W_Q, \quad K = c W_K, \quad V = c W_V
$$

其中 $W_Q \in \mathbb{R}^{C \times D}$，$W_K, W_V \in \mathbb{R}^{d \times D}$ 是可学习投影矩阵。

$$
\text{Attention}(Q, K, V) = \text{softmax}\left(\frac{QK^T}{\sqrt{D}}\right) V
$$

输出形状：$[B, HW, D]$，然后投影回原始图像通道维度。

**多头注意力 (Multi-Head Attention)**：

$$
\text{MultiHead}(Q, K, V) = \text{Concat}(\text{head}_1, ..., \text{head}_h) W_O
$$

其中 $\text{head}_i = \text{Attention}(QW_Q^i, KW_K^i, VW_V^i)$。

**注意力图的可视化**：

$A = \text{softmax}(QK^T/\sqrt{D}) \in \mathbb{R}^{B \times HW \times L}$

$A[i, j, k]$ 表示第 $i$ 个样本的第 $j$ 个图像 token 对第 $k$ 个文本 token 的注意力权重。

**交叉注意力层在 U-Net 中的位置**：

U-Net 通常在瓶颈层和低分辨率层（$16 \times 16$ 和 $8 \times 8$）布置交叉注意力——这些分辨率下的特征图有足够的语义抽象度来"理解"文本的语义概念。

**梯度视角**：

交叉注意力层对文本嵌入 $c$ 的梯度 $\frac{\partial L}{\partial c}$ 决定了文本编码器如何更新——这就是为什么 CFG 通过调整文本条件的"强度"来影响生成结果。

## 直观理解
- **交叉注意力 = AI 在"看图识字"**：生成图像时，每个图像区域会"看"文本提示，找到与自己最相关的词。如果提示是"一只黑色的猫坐在红色沙发上"，"猫"的区域注意力集中在"猫"这个词上，"沙发"的区域集中在"沙发"和"红色"上。
- **注意力图 = AI 的思维导图**：可视化注意力图可以看到——当你说"猫"时，U-Net 的某个层的注意力图上猫的轮廓区域是亮的，表明"这里正在画猫"。
- **为什么 prompt 顺序重要**："一只黑色的猫"和"一只猫是黑色的"在 token 层面是不同的排列。注意力机制中，Query 会同时关注所有 Key，但 token 之间的相对位置会影响注意力分布——越靠前的 token 通常获得更多关注。
- **多重主体时的注意力竞争**：当提示包含多个主体时（"一只猫和一只狗"）,"猫"和"狗"的 token 会在注意力图中竞争——空间上相邻的区域会倾向于分配给不同的 token，这可能导致"猫狗融合"或某一方被忽略。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class CrossAttention(nn.Module):
    """
    交叉注意力层
    
    图像特征 <-> 文本嵌入
    """
    def __init__(self, query_dim=320, context_dim=768, n_heads=8):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = query_dim // n_heads
        self.scale = self.head_dim ** -0.5
        
        self.to_q = nn.Linear(query_dim, query_dim, bias=False)
        self.to_k = nn.Linear(context_dim, query_dim, bias=False)
        self.to_v = nn.Linear(context_dim, query_dim, bias=False)
        self.to_out = nn.Linear(query_dim, query_dim)
    
    def forward(self, x, context, return_attention=False):
        """
        x: 图像特征 [B, H, W, C] 或 [B, HW, C]
        context: 文本嵌入 [B, L, d]
        return_attention: 是否返回注意力图（用于可视化）
        """
        # 展平图像特征
        if x.dim() == 4:
            B, C, H, W = x.shape
            x = x.flatten(2).permute(0, 2, 1)  # [B, HW, C]
        else:
            B, HW, C = x.shape
            H = W = int(math.sqrt(HW))
        
        # 投影到 Q, K, V
        Q = self.to_q(x)
        K = self.to_k(context)
        V = self.to_v(context)
        
        # 多头分割
        Q = Q.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        K = K.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        V = V.view(B, -1, self.n_heads, self.head_dim).transpose(1, 2)
        
        # 注意力分数
        attn_scores = torch.matmul(Q, K.transpose(-2, -1)) * self.scale
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, n_heads, HW, L]
        
        # 加权求和
        out = torch.matmul(attn_weights, V)
        out = out.transpose(1, 2).contiguous().view(B, -1, self.n_heads * self.head_dim)
        out = self.to_out(out)
        
        # 恢复空间形状
        out = out.permute(0, 2, 1).reshape(B, -1, H, W)
        
        if return_attention:
            # 返回平均注意力图 [B, HW, L]
            return out, attn_weights.mean(dim=1)
        return out

# 注意力图可视化与分析
def analyze_attention_maps(cross_attn, image_feat, text_emb, text_tokens):
    """分析图像-文本之间的注意力"""
    with torch.no_grad():
        out, attn = cross_attn(image_feat, text_emb, return_attention=True)
    
    # attn: [B, HW, L] 其中 L 是文本 token 数
    B, HW, L = attn.shape
    H = W = int(math.sqrt(HW))
    
    # 计算每个文本 token 的总注意力
    token_attention = attn.mean(dim=1)  # [B, L]
    
    print("文本 token 的注意力分布:")
    for i in range(L):
        if text_tokens is not None:
            token_text = text_tokens[i] if i < len(text_tokens) else f"<{i}>"
        else:
            token_text = f"token_{i}"
        print(f"  {token_text}: {token_attention[0, i].item():.3f}")
    
    # 找到注意力最高的 token
    max_token_idx = token_attention.argmax(dim=-1)[0].item()
    print(f"\n最高注意力 token: {max_token_idx}")
    print(f"  图像区域中对该 token 的平均注意力: {token_attention[0, max_token_idx].item():.3f}")
    
    # 注意力图可以 reshape 为 2D 进行可视化
    # attn_map_for_max = attn[0, :, max_token_idx].reshape(H, W)
    
    return attn

# Prompt 工程中的注意力分析
def prompt_attention_analysis(cross_attn, image_feat):
    """比较不同 prompt 表述对注意力的影响"""
    prompts = [
        "a black cat",
        "cat, black color", 
        "a cat that is black in color",
        "a black cat and a white dog",
    ]
    
    # 模拟文本嵌入
    vocab = {"a": 0, "black": 1, "cat": 2, "color": 3, 
             "that": 4, "is": 5, "in": 6, "and": 7, "white": 8, "dog": 9}
    
    print("不同 Prompt 对 'cat' token 的注意力影响:")
    for prompt in prompts:
        # 模拟文本嵌入
        # 实际应使用 CLIP 编码器
        text_emb_sim = torch.randn(1, len(prompt.split()), 768)
        _, attn = cross_attn(image_feat, text_emb_sim, return_attention=True)
        
        # 计算"cat"对应的注意力（假设每个词一个 token）
        cat_attn = attn.mean(dim=1)[0, 2].item()
        print(f"  '{prompt}': cat attention = {cat_attn:.3f}")

print("=== Cross-Attention 在文本控制中的作用 ===")
print()
ca = CrossAttention(query_dim=320, context_dim=768, n_heads=8)
x = torch.randn(1, 320, 16, 16)  # 图像特征
c = torch.randn(1, 8, 768)  # 文本嵌入（8 个 token）

out, attn = ca(x, c, return_attention=True)
print(f"图像特征形状: {x.shape}")
print(f"文本嵌入形状: {c.shape}")
print(f"输出形状: {out.shape}")
print(f"注意力图形状: {attn.shape}  [batch, HW, L]")
print(f"交叉注意力参数量: {sum(p.numel() for p in ca.parameters()):,}")
```

## 深度学习关联
- **Attention Control (Prompt-to-Prompt)**：通过干预交叉注意力图来实现图像编辑——交换、修改或加权特定 token 的注意力图，可以编辑图像中对应区域的内容而不改变整体结构。
- **Null-Text Inversion**：通过反向传播找到与真实图像匹配的"空文本嵌入"，实现基于 CFG 的真实图像编辑——本质上是利用交叉注意力机制做图像反演。
- **Composable Diffusion / Layout Guidance**：通过分别计算每个 token 的交叉注意力图并施加约束（如"猫"的注意力应该集中在左半图，"狗"在右半图），可以实现细粒度的版面控制。
- **多模态大模型的交叉注意力**：LLaVA、GPT-4V 等多模态模型同样使用交叉注意力让语言模型"看到"图像——只不过 Query 来自语言模型，Key/Value 来自视觉编码器。
