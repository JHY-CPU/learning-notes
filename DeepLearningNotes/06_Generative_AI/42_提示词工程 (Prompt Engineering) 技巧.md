# 42_提示词工程 (Prompt Engineering) 技巧

## 核心概念
- **提示词工程 (Prompt Engineering)**：通过精心设计输入文本（Prompt）来控制和优化生成式 AI 模型输出结果的技术和策略。
- **描述的结构化**：有效的 prompt 通常包含——主体 (Subject)、动作/状态 (Action/State)、环境 (Environment)、风格 (Style)、光线 (Lighting)、视角 (Viewpoint)、质量 (Quality)。
- **token 顺序的影响**：在 CLIP 文本编码器中，token 的注意力权重受位置影响——靠近开头的词获得更高的注意力。因此将最重要的词放在 prompt 开头。
- **加权语法 (Weighting Syntax)**：使用 `(word:weight)` 或 `(word)` 的语���来调整特定词的影响——如 `(sunset:1.5)` 增强夕阳效果，`(clouds:0.8)` 减弱云的影响。
- **文本交替 (Prompt Alternation)**：用 `[word1:word2:ratio]` 在采样过程中动态切换提示——例如 `[sunset:night:0.5]` 意味着在前 50% 的步数中关注"sunset"，之后切换到"night"。
- **提示词组合 (Prompt Blending)**：将多个提示组合——`landscape, mountains, lake` 会产生综合效果。这种组合不是简单的逻辑与，而是语义空间中的向量插值。
- **负面提示 (Negative Prompt)**：指定不想在生成结果中出现的内容——`(ugly:1.2), (deformed:1.2), blurry`。在 CFG 框架中，从正向提示的预测中减去负面提示的预测。

## 数学推导

**Prompt embedding 的向量表示**：

文本提示 $t$ 通过 CLIP 编码器得到嵌入序列 $c \in \mathbb{R}^{L \times d}$。

对于整个提示的"语义向量"通常取最后一层 [EOS] token 的表示：

$$
c_{\text{prompt}} = \text{CLIP}(t)[\text{EOS}] \in \mathbb{R}^d
$$

**加权 prompt 的实现**：

加权实际上是在编码空间中对 token 嵌入进行缩放：

$$
c_{\text{weighted}}[i] = w_i \cdot c[i]
$$

其中 $w_i$ 是对应第 $i$ 个 token 的权重，默认为 1。

在 CFG 中，加权等效于：

$$
\epsilon_{\text{weighted}} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c_{\text{weighted}}) - \epsilon_\theta(x_t, t, \emptyset))
$$

**负面 prompt 的数学**：

$$
\epsilon_{\text{final}} = \epsilon_\theta(x_t, t, c_{\text{pos}}) - w_{\text{neg}} \cdot (\epsilon_\theta(x_t, t, c_{\text{neg}}) - \epsilon_\theta(x_t, t, \emptyset))
$$

其中 $c_{\text{pos}}$ 是正向提示，$c_{\text{neg}}$ 是负面提示，$w_{\text{neg}}$ 是负面权重。

等价于：

$$
\epsilon_{\text{final}} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c_{\text{pos}}) - \epsilon_\theta(x_t, t, c_{\text{neg}}))
$$

其中 $w$ 是引导尺度。

**Prompt 语义插值**：

在两个提示 $c_1$ 和 $c_2$ 之间做线性插值：

$$
c_{\text{interp}} = (1 - \alpha) \cdot c_1 + \alpha \cdot c_2
$$

$\alpha$ 控制两种语义的混合比例。

## 直观理解
- **Prompt 工程 = 给 AI 下达精准指令**：就像对设计师说"请画一幅画"vs"请画一幅印象派风格的、日落时分的、薰衣草田的油画，温暖色调，柔和的笔触"——越精确，结果越符合预期。
- **Token 位置的影响力**：CLIP 编码器像一个听了前面的话就不太注意后面的人——所以最重要的词说在前面。"猫，黑色，在沙发上"比"沙发上有一只黑色的猫"更好。
- **加权语法 = 加大音量**：给某个词加权重就像说话时重读某个词——"我说的是**红色**的帽子，不是蓝色"。
- **负面提示 = 消除不必要的元素**：就像告诉设计师"不要加人、不要加文字、不要模糊的效果"——在 CFG 的框架中，负面提示提供了一个"反方向"的指引。

## 代码示例

```python
import torch
import torch.nn.functional as F
import math

class PromptOptimizer:
    """提示词优化工具"""
    
    @staticmethod
    def structure_prompt(subject, style, environment, lighting, quality_prompt=True):
        """
        构建结构化 prompt
        
        按照"最重要信息在前"的原则排列
        """
        prompt_parts = [
            f"masterpiece, best quality, highly detailed" if quality_prompt else "",
            f"({style})" if style else "",
            subject,
            environment,
            lighting,
        ]
        
        return ", ".join([p for p in prompt_parts if p])
    
    @staticmethod
    def add_weight(prompt_term, weight):
        """为 prompt 中的词添加权重"""
        return f"({prompt_term}:{weight})"
    
    @classmethod
    def create_weighted_prompt(cls, base_terms, weights):
        """创建加权 prompt"""
        weighted_terms = []
        for term, weight in zip(base_terms, weights):
            if weight != 1.0:
                weighted_terms.append(cls.add_weight(term, weight))
            else:
                weighted_terms.append(term)
        return ", ".join(weighted_terms)
    
    @staticmethod
    def prompt_alternation(term1, term2, switch_ratio):
        """创建交替 prompt"""
        return f"[{term1}:{term2}:{switch_ratio}]"

# 模拟权重对 token 嵌入的影响
class SimulatedCLIPEmbedding:
    """模拟 CLIP 文本编码器中加权语法对嵌入的影响"""
    def __init__(self, dim=768):
        self.dim = dim
    
    def encode_token(self, token, weight=1.0):
        """模拟带权重的 token 编码"""
        # 均值向量
        vec = torch.randn(self.dim)
        vec = vec / vec.norm()  # 归一化
        return vec * weight
    
    def encode_prompt(self, prompt):
        """编码完整 prompt"""
        tokens = prompt.split()
        embeddings = []
        
        for token in tokens:
            if token.startswith("(") and token.endswith(")"):
                # 加权 token: (token:weight) 或 (token)
                inner = token[1:-1]
                if ":" in inner:
                    word, weight_str = inner.split(":")
                    weight = float(weight_str)
                    word = word
                else:
                    word = inner
                    weight = 1.1
            else:
                word = token.strip(",.")
                weight = 1.0
            
            emb = self.encode_token(word, weight)
            embeddings.append(emb)
        
        return torch.stack(embeddings)

# 不同 prompt 风格的注意力分析
def analyze_prompt_effect(encoder, prompts):
    """分析不同 prompt 对 token 表示的影���"""
    print("Prompt 语义分析:")
    for prompt in prompts:
        emb = encoder.encode_prompt(prompt)
        # 语义向量幅度（越高说明该 prompt 整体强度越大）
        magnitude = emb.norm(dim=-1).mean().item()
        
        # token 间的语义分散度（越高说明 prompt 包含更多不同概念）
        cos_sim = F.cosine_similarity(emb[:-1], emb[1:], dim=-1)
        diversity = (1 - cos_sim.mean()).item()
        
        print(f"  '{prompt[:40]}...'")
        print(f"    语义强度: {magnitude:.3f}, 语义分散度: {diversity:.3f}")

print("=== 提示词工程技巧 ===")
print()

encoder = SimulatedCLIPEmbedding()

# 结构化 prompt 示例
prompt_optimized = PromptOptimizer.structure_prompt(
    subject="a majestic wolf",
    style="oil painting, baroque style",
    environment="in a snowy forest at night",
    lighting="moonlight, dramatic shadows"
)
print(f"结构化 Prompt: {prompt_optimized}")
print()

# 加权 Prompt 示例
weighted = PromptOptimizer.create_weighted_prompt(
    ["masterpiece", "cat", "red collar", "sitting on velvet"],
    [1.0, 1.0, 1.5, 1.2]
)
print(f"加权 Prompt: {weighted}")
print()

# 分析不同 prompt 风格
test_prompts = [
    "a cat",
    "masterpiece, best quality, a cute cat, highly detailed",
    "(masterpiece:1.2), (a cute cat:1.5), (highly detailed:1.1)",
    "a cat sitting on a couch",
]
analyze_prompt_effect(encoder, test_prompts)

print()
print("关键工程技巧总结:")
print("1. 重要词放在开头 (CLIP 注意力偏差)")
print("2. 使用 (word:weight) 语法调整强度")
print("3. 结构化: 质量→风格→主体→环境→光线")
print("4. 结合 Negative Prompt 消除不想要的内容")
print("5. 用 [word1:word2:ratio] 做动态交替")
```

## 深度学习关联
- **Prompt 的 tokenizer 差异**：不同模型使用不同的 tokenizer（BPE, SentencePiece, WordPiece），相同的自然语言文本在不同模型中生成不同的 token 序列——因此 prompt 技巧不能跨模型通用。
- **CFG 和 Prompt Weight 的关系**：CFG 的全局引导尺度和 prompt 中的局部权重是正交的控制手段——CFG 控制"整体听话程度"，prompt weight 控制"某个概念的强调程度"。
- **Prompt 自动化优化 (APE)**：通过 LLM 自动生成和优化 prompt（如 OPRO、APE），让 AI 自己调整文本来达到最佳生成结果——这催生了"prompt 优化器"这一新工具类别。
- **Sanctuary 技术**：一些平台使用"prompt 注入检测"来过滤恶意提示（如生成 NSFW 内容、名人仿冒），这是 prompt 工程在安全维度的重要关联应用。
