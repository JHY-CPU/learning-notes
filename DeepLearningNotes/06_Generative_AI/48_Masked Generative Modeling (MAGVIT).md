# 48_Masked Generative Modeling (MAGVIT)

## 核心概念
- **掩码生成建模 (Masked Generative Modeling)**：介于自回归和扩散之间的生成范式——通过随机掩码部分 token，训练模型预测被掩码的 token，生成时从全掩码开始逐步去掩码。
- **MAGVIT (Masked Generative Video Transformer)**：Google 提出的视频生成模型，使用掩码建模在离散 token 空间中进行视频生成，支持无条件、条件和视频预测。
- **迭代去掩码 (Iterative Decoding)**：从全部掩码（所有 token 都是 [MASK]）开始，每轮预测并保留高置信度的 token，低置信度的继续掩码——类似扩散模型的"由粗到细"策略。
- **离散 token 空间**：与 VQ-VAE 类似，MAGVIT 使用视频 VAE 将视频帧编码为离散 token 序列（网格），然后在 token 空间进行掩码建模。
- **并行生成**：与 AR 的顺序生成不同，MAGVIT 每轮并行生成所有位置的预测——只保留高置信度的，然后在下轮对剩余的低置信度位置重新预测。
- **灵活的条件控制**：掩码框架天然支持多种条件——类别条件（保留部分 token）、帧预测（保留前 $k$ 帧）、文本条件（通过交叉注意力）。

## 数学推导

**MAGVIT 的训练目标**：

视频 $v$ 被 VQ-VAE 编码为离散 token 网格 $z \in \{1, ..., K\}^{T \times H \times W}$。

在训练时，随机生成掩码 $m \in \{0, 1\}^{T \times H \times W}$，其中 $m_{thw}=1$ 表示被掩码（需要预测），$m_{thw}=0$ 表示可见。

训练损失（预测被掩码的 token）：

$$
\mathcal{L} = \mathbb{E}_{z, m} \left[ \sum_{thw: m_{thw}=1} -\log p_\theta(z_{thw} | z_{\text{visible}}, m) \right]
$$

其中 $p_\theta(z_{thw} | z_{\text{visible}}, m)$ 是模型对掩码位置的条件预测。

**生成时的调度函数 (Schedule Function)**：

定义 $\gamma(t) \in [0, 1]$ 为时间步 $t$ 的掩码率，$\gamma(0)=1$（全部掩码）到 $\gamma(T)=0$（全部可见）。

常见的调度函数：余弦调度 $\gamma(t) = \cos\left(\frac{t\pi}{2T}\right)$

**迭代解码算法**：

1. 设置 $z_T = [MASK]^{T \times H \times W}$（全部掩码）
2. 对于 $t = T-1$ 到 $0$:
   - 模型预测所有掩码位置的 logits
   - 计算置信度分数
   - 保留 $\frac{\gamma(t-1)}{\gamma(t)}$ 比例的最高置信度 token（去掩码）
   - 其他位置保持掩码
3. 返回完全去掩码的 $z_0$

**与 AR 和扩散的数学对比**：

AR：$p(z) = \prod_i p(z_i | z_{<i})$，顺序条件

扩散（连续）：$p(x_{t-1}|x_t)$，逐步去噪

掩码生成：$p(z_{t-1}|z_t)$ 其中 $z_t$ 是部分掩码的离散 token，逐步去掩码。

## 直观理解
- **掩码生成 = AI 版的"填字游戏"**：把一张图的大部分区域都用灰色遮住（掩码），让 AI 猜每个格子下是什么内容，保留最有信心的猜测，然后把其他格子重新遮住再猜——直到所有格子都被填满。
- **为什么比 AR 快**：AR 必须按顺序一个一个填——"先填左上角，然后它的右边，然后...一步也不能跳过"。掩码生成就像是一群人同时填字谜——每个人负责不同的区域，同时下笔。
- **与扩散模型的关系**：掩码生成可以看作是"离散空间中的扩散模型"——扩散在连续空间中加高斯噪声，掩码生成在离散空间中将 token 替换为 [MASK]。两者的迭代精化过程高度相似。
- **置信度驱动的生成**：模型先填充"最有把握"的部分（如大块天空、背景），再处理"不确定"的部分（如人物细节、边缘区域）——这模拟了从宏观到微观的认知过程。

## 代码示例

```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MaskedGenerativeModel(nn.Module):
    """
    掩码生成模型核心
    
    在离散 token 空间中进行迭代去掩码
    """
    def __init__(self, vocab_size=1024, d_model=512, nhead=8, num_layers=6):
        super().__init__()
        self.vocab_size = vocab_size
        
        # Token 嵌入
        self.token_embedding = nn.Embedding(vocab_size + 1, d_model)  # +1 为 [MASK]
        self.pos_embedding = nn.Parameter(torch.randn(1, 100, d_model) * 0.02)
        
        # Transformer
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=2048,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)
        
        self.output_proj = nn.Linear(d_model, vocab_size)
    
    def forward(self, tokens, mask_indices):
        """
        预测被掩码位置的 token
        
        参数:
            tokens: [B, N] token 序列（包含 [MASK]）
            mask_indices: [B, N] 布尔掩码（True=需要预测）
        """
        B, N = tokens.shape
        
        # 嵌入
        x = self.token_embedding(tokens) + self.pos_embedding[:, :N, :]
        h = self.transformer(x)
        
        # 预测被掩码的位置
        logits = self.output_proj(h)  # [B, N, vocab_size]
        
        return logits

def cosine_mask_schedule(t, T):
    """余弦调度函数：返回时间步 t 的掩码率"""
    return math.cos(t * math.pi / (2 * T))

def iterative_decoding(model, seq_len, T=8, temperature=1.0, device='cpu'):
    """
    迭代去掩码解码
    
    参数:
        model: 掩码生成模型
        seq_len: 序列长度
        T: 迭代步数
        temperature: 采样温度
    """
    MASK_TOKEN = model.vocab_size  # [MASK] 的 ID
    B = 1
    
    # 初始全掩码
    tokens = torch.full((B, seq_len), MASK_TOKEN, device=device)
    
    for t in range(T - 1, 0, -1):
        # 当前掩码率
        current_mask_rate = cosine_mask_schedule(t, T)
        next_mask_rate = cosine_mask_schedule(t - 1, T)
        
        # 模型预测
        logits = model(tokens, tokens == MASK_TOKEN)
        
        # 只对掩码位置采样
        mask_positions = (tokens == MASK_TOKEN)
        masked_logits = logits[mask_positions]
        
        # 带温度的采样
        probs = F.softmax(masked_logits / temperature, dim=-1)
        predicted_tokens = torch.multinomial(probs, 1).squeeze(-1)
        
        # 计算置信度
        confidence = probs.max(dim=-1).values
        
        # 计算本轮需要去掩码的数量
        total_mask = mask_positions.sum().item()
        tokens_to_keep = int(total_mask * (1 - next_mask_rate / current_mask_rate))
        tokens_to_keep = max(tokens_to_keep, 1)
        
        # 保留高置信度的预测
        if tokens_to_keep < total_mask:
            # 按置信度排序
            _, indices = torch.topk(confidence, tokens_to_keep)
            
            # 替换高置信度位置
            mask_indices = torch.where(mask_positions.flatten())[0]
            keep_indices = mask_indices[indices]
            
            tokens_flatten = tokens.view(-1)
            tokens_flatten[keep_indices] = predicted_tokens[indices]
            tokens = tokens_flatten.view(B, seq_len)
        else:
            # 最后一步：全部去掩码
            tokens[mask_positions] = predicted_tokens
    
    return tokens

# 演示
print("=== Masked Generative Modeling (MAGVIT) ===")
print()

model = MaskedGenerativeModel(vocab_size=1024, d_model=256, nhead=4, num_layers=4)

# 训练模拟
B, N = 4, 64
vocab_size = 1024
tokens = torch.randint(0, vocab_size, (B, N))
mask = torch.rand(B, N) < 0.7  # 70% 掩码

# 训练损失
train_tokens = tokens.clone()
train_tokens[mask] = vocab_size  # [MASK]

logits = model(train_tokens, mask)
loss = F.cross_entropy(logits[mask], tokens[mask])

print(f"Token 序列长度: {N}")
print(f"词表大小: {vocab_size}")
print(f"掩码率: {mask.float().mean().item():.1%}")
print(f"训练损失: {loss.item():.4f}")
print()

# 迭代解码
print("迭代去掩码过程:")
T = 8
for t in range(T, -1, -1):
    mask_rate = cosine_mask_schedule(t, T)
    remaining = mask_rate * N
    print(f"  步 {T-t:2d}/{T}: 掩码率={mask_rate:.2f}, 已知={N-remaining:.0f}/{N}")
```

## 深度学习关联
- **MaskGIT (Masked Generative Image Transformer)**：MAGVIT 的图像版本，使用掩码生成范式在图像离散 token 上做生成。相比 AR 模型（如 DALL-E），MaskGIT 的采样速度快 30-60 倍。
- **MUSE (由 Google 提出的文本到图像生成模型)**：使用掩码生成范式，结合 VQ-VAE 和 Transformer，在 ImageNet 上取得了当时最好的 FID 指标。MUSE 证明掩码生成可以超越扩散和 AR。
- **MDLM (Masked Diffusion Language Model)**：将掩码生成应用到文本领域——证明了在语言建模中，迭代去掩码也可以达到与 AR 模型竞争的性能，且生成更可控。
- **视频与 3D 的掩码生成**：MAGVIT2 扩展了 MAGVIT 到更高分辨率和更长的视频；CubDiff 将掩码生成用于 3D 表示——证明了掩码范式在多模态生成中的普适性。
