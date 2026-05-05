# 49_多模态大模型 (LMM) 的生成能力

## 核心概念

- **多模态大模型 (Large Multimodal Model, LMM)**：能同时理解和生成多种模态内容（文本、图像、视频、音频）的 AI 模型。代表：GPT-4V, Gemini, LLaVA, Qwen-VL。
- **多模态理解**：LMM 可以"看懂"图像——描述场景、识别物体、阅读 OCR 文字、理解图表。这种能力通过视觉编码器（如 ViT）+ LLM 的连接实现。
- **多模态生成**：LMM 不仅能理解多模态输入，还能生成多模态输出——生成图像（如 Gemini 的图像生成能力）、生成图文混合内容（如 GPT-4 的 DALL-E 集成）。
- **视觉编码器 + LLM 架构**：标准的 LMM 架构——视觉编码器（ViT/CLIP）将图像编码为视觉 token，通过投影层（Projector）对齐到 LLM 的嵌入空间，LLM 做推理和生成。
- **指令微调 (Instruction Tuning)**：对 LMM 进行多模态指令微调，使其能理解和执行复杂的多模态任务——如"解释这张图表并生成类似的数据可视化"。
- **涌现能力 (Emergent Abilities)**：当模型规模足够大和多模态训练数据足够丰富时，LMM 展现出训练数据中未明确包含的能力——如空间推理、情感理解、幽默识别。
- **多模态 in-context learning**：LMM 可以从多模态示例中学习新任务——给模型看几张带标注的图片，模型就能理解任务并执行类似的标注。

## 数学推导

**LMM 的标准架构**：

视觉编码器 $f_{\text{vis}}: \mathbb{R}^{H \times W \times C} \to \mathbb{R}^{N_v \times d_v}$，将图像映射为视觉 token 序列。

投影器 $g_{\text{proj}}: \mathbb{R}^{N_v \times d_v} \to \mathbb{R}^{N_v \times d_{\text{LLM}}}$，将视觉 token 对齐到 LLM 的嵌入空间。

LLM $h_{\text{LLM}}: \mathbb{R}^{(N_v + N_t) \times d_{\text{LLM}}} \to \mathbb{R}^{d_{\text{LLM}}}$，处理视觉 + 文本 token 的序列。

**训练目标（自回归 next-token prediction）**：

给定多模态输入（图像 $I$，文本前缀 $t_{1:L}$），LLM 预测后续文本：

$$
p_\theta(t_{L+1:L+M} | I, t_{1:L}) = \prod_{i=L+1}^{L+M} p_\theta(t_i | I, t_{1:i-1})
$$

训练损失（文本部分的负对数似然）：

$$
\mathcal{L} = -\sum_{i=L+1}^{L+M} \log p_\theta(t_i | I, t_{1:i-1})
$$

**视觉-文本对比对齐**：

在训练初期，使用对比损失对齐视觉和文本表示：

$$
\mathcal{L}_{\text{align}} = -\frac{1}{B} \sum_{i=1}^B \log \frac{\exp(\text{sim}(v_i, t_i)/\tau)}{\sum_{j=1}^B \exp(\text{sim}(v_i, t_j)/\tau)}
$$

**图像生成能力**：

当 LMM 接入图像生成模块（如 DALL-E/扩散模型）时，整体流程为：

$$
\text{Response} = 
\begin{cases}
\text{Text: } h_{\text{LLM}}(\text{prompt}, \text{image}) & \text{如果只输出文本} \\
\text{Image: } G_{\text{gen}}(\text{text from LLM}) & \text{如果输出图像}
\end{cases}
$$

## 直观理解

- **LMM = 一个有眼睛的 GPT**：GPT-4 原本只能"读文字"，现在给它装上了眼睛（视觉编码器）。你给它看一张照片，它不仅能说"这是一只猫"，还能说"这只猫看起来很开心，背景是阳光明媚的下午"。
- **视觉 token = 把图像翻译为 LLM 能理解的语言**：图像通过编码器变成了一串视觉 token——相当于把"图像语言"翻译成了"LLM 语言"。这就像你把一幅画用文字描述出来，然后问 GPT"你觉得这幅画怎么样"。
- **为什么 LMM 能"画"图**：LMM 本身不一定内置图像生成能力——它生成的是"描述图像的文本"（如 Prompt），然后调用外部生成器。但有些 LMM（如 Gemini）直接在模型中融合了扩散解码器。
- **涌现能力的社会意义**：当模型能同时理解文字和图像时，它就解锁了超越简单"看图说话"的能力——比如看 X 光片并生成诊断报告、看工程图纸并解释设计逻辑。

## 代码示例

```python
import torch
import torch.nn as nn

class SimpleLMM(nn.Module):
    """
    简化的多模态大模型架构
    
    视觉编码器 + 投影器 + LLM
    """
    def __init__(self, vision_dim=768, llm_dim=4096, vocab_size=32000, num_layers=4):
        super().__init__()
        self.vision_dim = vision_dim
        self.llm_dim = llm_dim
        
        # 视觉编码器（模拟 ViT）
        self.vision_encoder = nn.Sequential(
            nn.Conv2d(3, 64, 16, 16),  # 简化的 patch embedding
            nn.Flatten(2),  # [B, 64, H*W/256]
            nn.Linear(64, vision_dim),
        )
        
        # 投影器：视觉 token -> LLM token
        self.projector = nn.Sequential(
            nn.Linear(vision_dim, llm_dim),
            nn.GELU(),
            nn.Linear(llm_dim, llm_dim),
        )
        
        # 简化的 LLM（Transformer decoder）
        self.token_embedding = nn.Embedding(vocab_size, llm_dim)
        self.position_embedding = nn.Parameter(torch.randn(1, 2048, llm_dim) * 0.02)
        
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=llm_dim, nhead=8, dim_feedforward=llm_dim * 4,
            batch_first=True, activation='gelu'
        )
        self.transformer = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)
        
        self.lm_head = nn.Linear(llm_dim, vocab_size)
        
        # 图像生成接口（使用扩散模型作为外部模块）
        self.image_gen_module = None
    
    def encode_image(self, images):
        """编码图像为视觉 token"""
        # [B, 3, 224, 224] -> [B, N_v, vision_dim]
        patch_feats = self.vision_encoder(images)
        B, D, N = patch_feats.shape
        patch_feats = patch_feats.permute(0, 2, 1)  # [B, N, D]
        return self.projector(patch_feats)  # [B, N_v, llm_dim]
    
    def forward(self, images, text_tokens, text_mask=None):
        """
        前向：多模态条件文本生成
        
        参数:
            images: 输入图像 [B, C, H, W] 或 None（纯文本）
            text_tokens: 文本 token [B, L]
            text_mask: 文本注意力掩码
        """
        B, L = text_tokens.shape
        
        # 视觉 token
        if images is not None:
            visual_tokens = self.encode_image(images)  # [B, N_v, llm_dim]
        else:
            visual_tokens = None
        
        # 文本 token
        text_emb = self.token_embedding(text_tokens)  # [B, L, llm_dim]
        text_emb = text_emb + self.position_embedding[:, :L, :]
        
        # 拼接视觉和文本 token
        if visual_tokens is not None:
            combined = torch.cat([visual_tokens, text_emb], dim=1)
        else:
            combined = text_emb
        
        # Transformer Decoder
        # 使用 text_tokens 作为记忆（简化）
        memory = text_emb
        output = self.transformer(combined, memory)
        
        # 预测下一个 token
        logits = self.lm_head(output)
        return logits
    
    def set_image_generator(self, gen_module):
        """设置图像生成模块（扩散模型等）"""
        self.image_gen_module = gen_module
    
    @torch.no_grad()
    def generate_image(self, description):
        """根据描述生成图像（简化接口）"""
        if self.image_gen_module is not None:
            return self.image_gen_module.generate(description)
        else:
            print("未安装图像生成模块")
            return None

# 多模态应用的模拟
class MultimodalApp:
    @staticmethod
    def describe_image(lmm, image):
        """图像描述"""
        # 编码图像，生成文本描述
        visual_tokens = lmm.encode_image(image)
        # ... 实际使用 LLM 生成描述文本
        return "A beautiful landscape with mountains and a lake."
    
    @staticmethod
    def visual_qa(lmm, image, question):
        """视觉问答"""
        # 融合图像和问题，生成答案
        return "There are 3 people in the image."

print("=== 多模态大模型 (LMM) 的生成能力 ===")
print()

lmm = SimpleLMM()
image = torch.randn(1, 3, 224, 224)
text = torch.randint(0, 1000, (1, 50))
logits = lmm(image, text)
print(f"图像输入形状: {image.shape}")
print(f"文本输入形状: {text.shape}")
print(f"输出 logits 形状: {logits.shape}")
print(f"LMM 参数量: {sum(p.numel() for p in lmm.parameters()):,}")
print()

print("LMM 的关键能力:")
print("1. 多模态理解: 看图说话、图表理解、OCR")
print("2. 多模态推理: 视觉问答、空间推理")
print("3. 多模态生成: 图文混合内容、可控图像生成")
print("4. In-context learning: 从多模态示例中学习")
```

## 深度学习关联

- **LLaVA (Large Language and Vision Assistant)**：使用 Vicuna/GPT 作为 LLM，CLIP 作为视觉编码器，仅通过一个简单的线性投影层连接，证明了"简单的连接"在大模型下就足够了。
- **GPT-4V / GPT-4o**：OpenAI 的多模态模型，支持图像、音频、视频的输入，并可以生成文本和图像。GPT-4o 进一步将延迟降低到实时水平，支持语音对话。
- **Gemini**：Google 的原生多模态模型，从一开始就是在多模态数据上训练的（而非后期接入视觉编码器），支持图像、视频、音频、代码的联合理解和生成。
- **模型收敛路径**：LMM 的演进方向——从"视觉编码器 + 冻结 LLM"到"全参数多模态训练"再到"原生多模态"（Gemini 路线）。每一步都增加了模态之间的深度融合。
