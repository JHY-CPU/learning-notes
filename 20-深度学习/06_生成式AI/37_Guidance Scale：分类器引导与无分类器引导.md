# 38_Guidance Scale：分类器引导与无分类器引导

## 核心概念

- **引导尺度 (Guidance Scale)**：在条件生成模型中控制"遵循条件"强度的超参数——$w=0$ 为无条件（忽视提示），$w$ 越大条件越强，但 $w$ 过大时生成质量下降。
- **分类器引导 (Classifier Guidance)**：利用一个训练好的分类器 $p(y|x_t)$ 的梯度 $\nabla_{x_t} \log p(y|x_t)$ 来引导去噪过程——在得分函数上加一个"指向分类器认为对的方向"的项。
- **无分类器引导 (Classifier-Free Guidance, CFG)**：不需要额外分类器，同时训练条件模型和无条件模型（通过随机丢弃条件），在推理时插值条件/无条件预测。
- **条件平衡的挑战**：引导太弱（$w$ 太小）→ 生成与条件不对齐；引导太强（$w$ 太大）→ 生成过度饱和、模式单一、伪影增多。
- **截断技巧 (Truncation Trick)**：在 GAN 中通过截断潜空间的采样区域来平衡质量和多样性——$w$ 同样起到类似作用，大 $w$ 质量高但多样性低。
- **动态引导 (Dynamic Guidance)**：在采样过程中动态调整 $w$——早期用大 $w$（快速收敛到条件区域），后期用小 $w$（探索细节）。

## 数学推导

**条件扩散模型的得分函数分解**：

无条件得分：$\nabla_x \log p(x_t)$
条件得分：$\nabla_x \log p(x_t|y) = \nabla_x \log p(x_t) + \nabla_x \log p(y|x_t)$

**分类器引导**：

使用预训练分类器 $p_\phi(y|x_t)$ 的梯度：

$$
\nabla_{x_t} \log p_\theta(x_t|y) \approx \nabla_{x_t} \log p_\theta(x_t) + w \cdot \nabla_{x_t} \log p_\phi(y|x_t)
$$

其中 $w$ 是引导尺度。在 DDPM 中对应噪声预测的改写：

$$
\epsilon_\theta(x_t, t, y) = \epsilon_\theta(x_t, t) - w \cdot \sqrt{1-\bar{\alpha}_t} \nabla_{x_t} \log p_\phi(y|x_t)
$$

**无分类器引导 (CFG)**：

同时训练条件 $\epsilon_\theta(x_t, t, y)$ 和无条件 $\epsilon_\theta(x_t, t, \emptyset)$ 的噪声预测：

$$
\tilde{\epsilon}_\theta(x_t, t, y) = \underbrace{\epsilon_\theta(x_t, t, \emptyset)}_{\text{无条件}} + w \cdot \underbrace{(\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))}_{\text{条件修正}}
$$

当 $w=0$ 时，退化为无条件生成；$w=1$ 时，退化为原始条件生成；$w>1$ 时，增强条件效果。

**无分类器引导的条件丢弃训练**：

训练时以概率 $p_{\text{uncond}}$（通常 10%）将条件替换为空 $\emptyset$，使得同一个模型同时支持条件和无条件预测。

**得分函数的统一公式**：

$$
\tilde{s}_\theta(x_t, t, y) = s_\theta(x_t, t, \emptyset) + w \cdot (s_\theta(x_t, t, y) - s_\theta(x_t, t, \emptyset))
$$

等价于噪声预测的改写：

$$
\tilde{\epsilon}_\theta(x_t, t, y) = \epsilon_\theta(x_t, t, \emptyset) + w \cdot (\epsilon_\theta(x_t, t, y) - \epsilon_\theta(x_t, t, \emptyset))
$$

## 直观理解

- **引导尺度就像"听话程度"**：$w=1$ 是"正常听话"（标准条件生成），$w=7$ 是"非常听话"（Stable Diffusion 默认），$w=15$ 是"过度听话"（生成变得奇怪、过度饱和）。
- **分类器引导 = 老师在一旁指导**：扩散模型在画画，一个图像分类器在边上说"这看起来不像猫"，然后模型根据这个反馈调整。分类器引导需要额外训练一个分类器，而且该分类器必须能处理噪声图像。
- **无分类器引导 = 想象"如果不听条件会怎样"**：模型同时做了一个"如果不看提示会画出什么"的预测，然后比较"看提示"和"不看提示"的差异，把这个差异放大后加入最终预测。这像在说"看看听提示和不听提示的区别，然后我更坚定地朝提示方向走"。
- **为什么默认 $w=7.5$（不是 1）**：条件模型在 $w=1$ 时常常不够"强调"条件——因为训练时条件被丢弃过，模型学会了不完全依赖条件。增加 $w$ 到 7.5 可以补偿这种"条件不充分"的问题。

## 代码示例

```python
import torch
import torch.nn as nn

class ClassifierGuidance:
    """分类器引导（需要额外分类器）"""
    def __init__(self, classifier, guidance_scale=1.0):
        self.classifier = classifier
        self.guidance_scale = guidance_scale
    
    def guided_noise_prediction(self, noise_pred_uncond, x_t, t, y):
        """
        用分类器梯度修正噪声预测
        
        参数:
            noise_pred_uncond: 无条件噪声预测
            x_t: 当前噪声图像
            t: 时间步
            y: 类别标签
        """
        x_t.requires_grad_(True)
        
        # 分类器预测
        logits = self.classifier(x_t, t)
        log_prob = torch.log_softmax(logits, dim=-1)
        log_p_y = log_prob[:, y].sum()
        
        # 分类器梯度
        grad = torch.autograd.grad(log_p_y, x_t)[0]
        
        # 修正噪声预测
        guided_noise = noise_pred_uncond - self.guidance_scale * grad
        
        return guided_noise

class ClassifierFreeGuidance:
    """无分类器引导"""
    def __init__(self, guidance_scale=7.5):
        self.guidance_scale = guidance_scale
    
    def __call__(self, noise_pred_uncond, noise_pred_cond):
        """
        CFG 插值
        
        参数:
            noise_pred_uncond: 无条件预测 [B, C, H, W]
            noise_pred_cond: 条件预测 [B, C, H, W]
        返回:
            修正后的噪声预测
        """
        return noise_pred_uncond + self.guidance_scale * (
            noise_pred_cond - noise_pred_uncond
        )

# 动态 CFG：在采样过程中调整引导尺度
class DynamicCFG:
    """动态引导尺度"""
    def __init__(self, initial_scale=7.5, final_scale=1.0, total_steps=50):
        self.initial_scale = initial_scale
        self.final_scale = final_scale
        self.total_steps = total_steps
    
    def get_scale(self, step):
        """根据当前步数计算引导尺度"""
        progress = step / self.total_steps
        # 线性衰减
        scale = self.initial_scale + (self.final_scale - self.initial_scale) * progress
        return scale
    
    def __call__(self, noise_pred_uncond, noise_pred_cond, step):
        scale = self.get_scale(step)
        return noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)

# CFG 训练时的条件丢弃
class CFGTrainer:
    """带条件丢弃的 CFG 训练"""
    def __init__(self, model, uncond_prob=0.1):
        self.model = model
        self.uncond_prob = uncond_prob
    
    def train_step(self, x_0, text_emb, noise_schedule):
        batch_size = x_0.size(0)
        
        # 随机丢弃条件（10% 概率）
        if torch.rand(1) < self.uncond_prob:
            text_emb = torch.zeros_like(text_emb)  # 空条件
        
        # 加噪
        noise = torch.randn_like(x_0)
        t = torch.randint(0, 1000, (batch_size,))
        x_t = noise_schedule(x_0, t, noise)
        
        # 预测噪声
        noise_pred = self.model(x_t, t, text_emb)
        
        loss = nn.functional.mse_loss(noise_pred, noise)
        return loss

# 演示不同引导尺度的效果
print("=== Guidance Scale 分析 ===")
print()

cfg = ClassifierFreeGuidance(guidance_scale=7.5)
uncond_pred = torch.randn(1, 4, 32, 32)
cond_pred = torch.randn(1, 4, 32, 32)

result = cfg(uncond_pred, cond_pred)
print(f"CFG 应用前后对比:")
print(f"  无条件预测 norm: {uncond_pred.norm().item():.2f}")
print(f"  条件预测 norm:   {cond_pred.norm().item():.2f}")
print(f"  CFG 结果 norm:   {result.norm().item():.2f}")
print()

# 不同 w 值的引导效果分析
print("不同引导尺度的效果（模拟）:")
for w in [0.0, 1.0, 3.0, 7.5, 15.0, 30.0]:
    cfg_w = ClassifierFreeGuidance(guidance_scale=w)
    result_w = cfg_w(uncond_pred, cond_pred)
    # 条件偏差 = CFG 结果与无条件预测的差异
    cond_bias = (result_w - uncond_pred).norm().item()
    print(f"  w={w:5.1f}: 条件偏差={cond_bias:.2f}, "
          f"输出 norm={result_w.norm().item():.2f}")
```

## 深度学习关联

- **CFG 的局限性**：大 $w$ 会降低生成多样性（模式坍塌效应），导致"过度饱和"和"伪影"。$w$ 和 CFG 的精确选择是扩散模型推理中最重要的超参数之一。
- **CFG 的替代方案：FreeU, DDS**：FreeU 通过修改 U-Net 的跳跃连接权重来改善细节；DDS (Detail Division Scale) 通过在不同频率分量上使用不同的引导尺度来平衡细节和过度平滑。
- **自引导 (Self-Guidance)**：不需要条件模型，让模型自己的一些内部表示作为"条件"——如利用模型"知道图像某个区域应该是什么"来引导细节生成。
- **引导在视频生成中的应用**：视频生成中的 CFG 需要更细致的处理——每帧独立的 CFG 会导致闪烁，时序一致的 CFG 需要在帧间平滑 $w$ 值。
