# 43_Negative Prompt 的实现机制

## 核心概念
- **Negative Prompt (负面提示)**：指定不希望出现在生成图像中的内容，通过 CFG（无分类器引导）框架实现概念"减法"——从正向预测中减去负面提示的预测。
- **数学本质**：Negative Prompt 不是真正"消除"了概念，而是通过向量引导将生成结果推向远离负面概念的方向。在潜空间中，"去"的方向就是负面概念的方向。
- **实现方式**：在推理时同时计算三个预测——无条件预测 $\epsilon_\theta(\emptyset)$、正向提示预测 $\epsilon_\theta(c_{\text{pos}})$、负面提示预测 $\epsilon_\theta(c_{\text{neg}})$。最终预测为 $\epsilon = \epsilon_\theta(\emptyset) + w(\epsilon_\theta(c_{\text{pos}}) - \epsilon_\theta(c_{\text{neg}}))$。
- **与正向权重的关系**：Negative Prompt 权重可以独立于正向 CFG 尺度设置——$w_{\text{pos}}$ 控制正向强度，$w_{\text{neg}}$ 控制负面强度。两者可以有不同的值。
- **多负面提示组合**：可以同时指定多个负面提示（如 `ugly, blurry, distorted, extra limbs`），每个都从正向中减去。
- **过度使用的副作用**：过度强烈的 negative prompt 会导致图像质量下降、过度校正、颜色偏移——因为模型被推向了"不自然"的区域。

## 数学推导

**标准 CFG（无 Negative Prompt）**：

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, c) - \epsilon_\theta(x_t, t, \emptyset))
$$

**带 Negative Prompt 的 CFG**：

$$
\tilde{\epsilon} = \epsilon_\theta(x_t, t, \emptyset) + w_{\text{pos}} \cdot (\epsilon_\theta(x_t, t, c_{\text{pos}}) - \epsilon_\theta(x_t, t, \emptyset)) - w_{\text{neg}} \cdot (\epsilon_\theta(x_t, t, c_{\text{neg}}) - \epsilon_\theta(x_t, t, \emptyset))
$$

其中 $w_{\text{pos}}$ 和 $w_{\text{neg}}$ 分别是正向和负向的引导尺度。

**等价变换**：

$$
\tilde{\epsilon} = \underbrace{\epsilon_\theta(\emptyset)}_{\text{基准}} + \underbrace{\Delta_{\text{pos}}}_{\text{推入正向}} - \underbrace{\Delta_{\text{neg}}}_{\text{推开负向}}
$$

其中 $\Delta_{\text{pos}} = w_{\text{pos}} \cdot (\epsilon_\theta(c_{\text{pos}}) - \epsilon_\theta(\emptyset))$，$\Delta_{\text{neg}} = w_{\text{neg}} \cdot (\epsilon_\theta(c_{\text{neg}}) - \epsilon_\theta(\emptyset))$。

**另一种等效形式**：

合并为一个伪 CFG 表达式：

$$
\tilde{\epsilon} = \epsilon_\theta(\emptyset) + w_{\text{total}} \cdot (\epsilon_\theta(c_{\text{pos}}) - \epsilon_\theta(c_{\text{neg}}^{'}))
$$

其中 $w_{\text{total}} = w_{\text{pos}}$，$c_{\text{neg}}^{'} = c_{\text{pos}} - \frac{w_{\text{neg}}}{w_{\text{pos}}}(c_{\text{neg}} - \emptyset)$ 是一个"修正的负面条件"。

**得分函数形式**：

$$
\tilde{s}(x_t, c_{\text{pos}}, c_{\text{neg}}) = s(x_t, \emptyset) + w_{\text{pos}}(s(x_t, c_{\text{pos}}) - s(x_t, \emptyset)) - w_{\text{neg}}(s(x_t, c_{\text{neg}}) - s(x_t, \emptyset))
$$

## 直观理解
- **Negative Prompt = 告诉模型"不要往这个方向走"**：想象模型在语义空间中朝"正向提示"的方向走——"一只猫"给出一个向量方向。Negative Prompt "狗"意味着"不要往狗的方向偏"——当正向提示本身可能产生狗的模糊语义时，这个修正很有用。
- **为什么 Negative Prompt 不是"反向生成"**：$\epsilon(c_{\text{neg}})$ 本身只是"如果条件为负面提示时的预测"，不是"生成反向内容"。减去它相当于"注意：不要往这个方向生成"。
- **多负面提示的累积效果**：每一个 negative prompt 都提供一个"排斥方向"，多个排斥方向的合力会把生成推向远离所有这些概念的区域。这就是为什么太多负向提示会导致结果偏向"灰色"区域。
- **Negative Prompt 的局限性**：它不能消除"模型已经记住的模式"。如果你说"不要画手"，模型可能还是会画手——因为正向提示中"人"的概念强烈暗示了手的存��。Negative Prompt 只提供纠正，不提供禁止。

## 代码示例

```python
import torch
import torch.nn.functional as F

class NegativePromptCFG:
    """带 Negative Prompt 的 CFG 采样器"""
    def __init__(self, pos_scale=7.5, neg_scale=7.5):
        self.pos_scale = pos_scale
        self.neg_scale = neg_scale
    
    def predict_noise(self, model, x_t, t, uncond_emb, pos_emb, neg_emb=None):
        """
        计算带 Negative Prompt 的 CFG 噪声预测
        
        参数:
            model: U-Net 噪声预测模型
            x_t: 当前噪声图像
            t: 时间步
            uncond_emb: 空文本嵌入（无条件）
            pos_emb: 正向提示嵌入
            neg_emb: 负面提示嵌入（可选）
        """
        # 批量计算三种条件
        batch_inputs = torch.cat([x_t] * 3, dim=0)
        batch_t = torch.cat([t] * 3)
        batch_emb = torch.cat([uncond_emb, pos_emb, neg_emb if neg_emb is not None else uncond_emb], dim=0)
        
        # 模型前向
        noise_preds = model(batch_inputs, batch_t, batch_emb)
        
        # 分割
        noise_uncond, noise_pos, noise_neg = noise_preds.chunk(3, dim=0)
        
        # CFG 合成
        delta_pos = noise_pos - noise_uncond
        delta_neg = noise_neg - noise_uncond
        
        noise_final = noise_uncond + self.pos_scale * delta_pos - self.neg_scale * delta_neg
        
        return noise_final

    def predict_noise_separate(self, model, x_t, t, uncond_emb, pos_emb, neg_emb=None):
        """
        分别计算（可以独立设置正负引导尺度）
        在大多数实现中，正负引导尺度独立设置
        """
        # 分别计算
        noise_uncond = model(x_t, t, uncond_emb)
        noise_pos = model(x_t, t, pos_emb)
        
        if neg_emb is not None:
            noise_neg = model(x_t, t, neg_emb)
            delta_pos = noise_pos - noise_uncond
            delta_neg = noise_neg - noise_uncond
            return noise_uncond + self.pos_scale * delta_pos - self.neg_scale * delta_neg
        else:
            # 无负面提示 → 标准 CFG
            return noise_uncond + self.pos_scale * (noise_pos - noise_uncond)

# Negative Prompt 的效果分析
def analyze_negative_prompt_effect(pos_text="portrait of a person", 
                                    neg_texts=None, 
                                    noise_pred_fn=None):
    """分析不同 negative prompt 的效果"""
    if neg_texts is None:
        neg_texts = ["", "ugly", "blurry", "ugly, blurry, deformed"]
    
    print(f"正向提示: '{pos_text}'")
    print()
    
    for neg in neg_texts:
        if not neg:
            print(f"无负面提示 (标准 CFG):")
        else:
            print(f"负面提示: '{neg}':")
        
        # 模拟噪声预测
        # 实际应用中这里运行真正的模型推理
        print(f"  → 方向: 推离 '{neg}' 的语义方向")
        print()

print("=== Negative Prompt 实现机制 ===")
print()

cfg = NegativePromptCFG(pos_scale=7.5, neg_scale=7.5)

# 模拟
x_t = torch.randn(1, 4, 32, 32)
t = torch.full((1,), 500, dtype=torch.long)
uncond = torch.zeros(1, 77, 768)
pos = torch.randn(1, 77, 768)
neg = torch.randn(1, 77, 768)

# 模拟模型输出
class MockModel(torch.nn.Module):
    def forward(self, x, t, emb):
        return torch.randn_like(x)

model = MockModel()

noise_final = cfg.predict_noise(model, x_t, t, uncond, pos, neg)
noise_standard = cfg.predict_noise(model, x_t, t, uncond, pos, uncond)

print("负面提示效果对比:")
print(f"  标准 CFG (无负面): noise norm = {noise_standard.norm().item():.2f}")
print(f"  带负面提示:       noise norm = {noise_final.norm().item():.2f}")
print(f"  方向偏差: {F.cosine_similarity(noise_final.flatten(), noise_standard.flatten(), dim=0).item():.3f}")

analyze_negative_prompt_effect()

print()
print("常见 Negative Prompt 短语:")
print("  质量负面: ugly, blurry, low quality, distorted, deformed")
print("  结构负面: extra limbs, bad anatomy, disfigured, mutation")
print("  风格负面: 3d, cartoon, painting (当需要照片真实感时)")
print("  元素负面: watermark, text, signature, logo, words")
```

## 深度学习关联
- **Negatives 的自动生成 (AIGC)**：一些工具使用 LLM 自动分析正向提示并推导出对应的 negative prompt——例如正向提示是"portrait"时自动添加"cartoon, 3d, painting, illustration"。
- **Per-Negative Weighting**：高阶工具允许对每个 negative prompt 独立设置权重——`(ugly:1.5), (blurry:0.8)`——实现细粒度的负向控制。
- **Dynamic Negative Prompt**：在采样过程中动态调整 negative prompt 的强度——早期强调结构层面（如`extra limbs`），后期强调纹理层面（如`blurry`）。
- **Negative Prompt 在视频生成中的应用**：在 AnimateDiff 等视频模型中，每帧使用相同的 negative prompt 有助于保持帧间一致性——但如果帧间的 negative prompt 效果不一致，会导致闪烁。
