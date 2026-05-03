# 47_Autoregressive vs Diffusion 生成范式对比

## 核心概念

- **自回归生成 (Autoregressive, AR)**：从左到右（或从上到下）逐步生成每个 token/pixel，每一步依赖之前生成的输出。代表：PixelCNN, GPT, DALL-E。
- **扩散生成 (Diffusion)**：从纯噪声开始，通过多步去噪逐步恢复数据。所有位置同时被生成和精化。代表：DDPM, Stable Diffusion, Imagen。
- **生成方向**：AR 是"顺序生成"（每一步看到之前的结果），扩散是"并行精化"（所有位置同时逐步改进）。
- **似然计算**：AR 模型可以直接计算精确的似然 $\log p(x) = \sum \log p(x_i|x_{<i})$；扩散模型需要用 ELBO 近似似然。
- **采样速度**：AR 模型的采样是顺序的（$O(L)$ 步，$L$ 是序列长度），不能并行；扩散模型的采样在原则上是并行的（但需要多步去噪）。
- **质量对比**：在图像生成中，扩散模型在 FID 等指标上超过 AR 模型。但在文本、代码等序列数据中，AR 模型（如 GPT-4）仍然主导。
- **混合架构**：最新的模型（如 DALL-E 3, Stable Diffusion 3）融合了两种范式的优点——AR 用于文本理解/编码，扩散用于图像生成。

## 数学推导

**自回归生成**：

对于序列 $x = (x_1, x_2, ..., x_L)$：

$$
p(x) = \prod_{i=1}^L p(x_i | x_1, ..., x_{i-1})
$$

训练目标（最大似然）：

$$
\mathcal{L}_{\text{AR}} = -\sum_{i=1}^L \log p_\theta(x_i | x_{<i})
$$

采样：$x_i \sim p_\theta(\cdot | x_{<i})$，顺序生成。

**扩散生成**：

对于连续数据 $x \in \mathbb{R}^D$：

前向：$x_t = \sqrt{\bar{\alpha}_t} x_0 + \sqrt{1-\bar{\alpha}_t} \epsilon$

训练：$\mathcal{L}_{\text{diff}} = \mathbb{E}_{t, x_0, \epsilon} [\|\epsilon - \epsilon_\theta(x_t, t)\|^2]$

采样：从 $x_T \sim \mathcal{N}(0, I)$ 开始，逐步去噪到 $x_0$。

**计算复杂度对比**：

AR 图像生成（逐像素）：$O(HW)$ 步，每步 $O(1)$ 计算
AR 图像生成（逐 patch）：$O(HW / p^2)$ 步，每步复杂
扩散生成：$O(T)$ 步（如 50-1000），每步 $O(HW)$ 计算

**并行化能力**：

AR：严格顺序，不可并行（每一步依赖上一步输出）
扩散：所有位置同时生成，但需要多步迭代

## 直观理解

- **AR 生成 = 一个字一个字地写作**：你要写一篇文章，必须先写第一个字，然后写第二个字（基于第一个），依此类推。一旦写错，后续内容都会受影响（误差累积）。
- **扩散生成 = 雕塑家从粗到细雕刻**：你从一块粗糙的石料（噪声）开始，逐渐雕出形状——先确定整体轮廓（前几步），再精化局部细节（后续步）。每一步都在改进所有部分。
- **为什么 AR 在文本上统治**：文本本质上是顺序的——"我今天吃了"之后，自然的下一词是"饭/苹果/面"，这与 AR 的顺序生成完美匹配。
- **为什么扩散在图像上统治**：图像不是顺序的——像素之间是空间关系而非序列关系。所有像素可以同时优化，没有必要从左到右逐像素生成。扩散的并行精化更符合图像的"整体性"性质。
- **误差累积问题**：AR 早期的错误会在后续步骤中被放大——"差之毫厘，谬以千里"。扩散模型的迭代精化允许在每个步骤修正前一步的错误。

## 代码示例

```python
import torch
import torch.nn as nn
import time

# 模拟 AR 生成
def autoregressive_generate(model, prompt, max_length=100):
    """自回归文本生成（简化）"""
    generated = prompt
    start_time = time.time()
    
    for i in range(max_length):
        # 根据已生成的所有 token 预测下一个
        next_token = model.generate_next(generated)
        generated = generated + [next_token]
        
        if next_token == "<EOS>":
            break
    
    elapsed = time.time() - start_time
    return generated, elapsed

# 模拟扩散生成
def diffusion_generate(model, n_steps=50):
    """扩散模型图像生成（简化）"""
    start_time = time.time()
    x_t = torch.randn(1, 3, 256, 256)  # 纯噪声
    
    for t in range(n_steps - 1, -1, -1):
        noise_pred = model.denoise(x_t, t)
        x_t = (x_t - noise_pred) / 1.0  # 简化去噪
    
    elapsed = time.time() - start_time
    return x_t, elapsed

# 计算复杂度对比的分析
print("=== Autoregressive vs Diffusion 生成范式对比 ===")
print()

print("1. 生成顺序:")
print("   AR:      顺序生成 (1D 序列)")
print("   扩散:    并行精化 (任意维度)")
print()

print("2. 计算复杂度:")
print(f"   AR (逐像素):       O(HW) 步, O(HW) 总计算")
print(f"   AR (逐 patch):     O(HW/p²) 步, O(HW) 总计算")
print(f"   扩散 (DDPM 50步):  50 步, O(50·HW) 总计算")
print()

print("3. 条件建模:")
print("   AR: 通过条件概率 $p(x_i|x_{<i}, y)$ 自然支持")
print("   扩散: 通过交叉注意力/条件拼接 $p(x|y)$")
print()

print("4. 似然计算:")
print("   AR: 精确似然 $p(x) = \\prod p(x_i|x_{<i})$")
print("   扩散: 近似似然 (ELBO)")
print()

print("5. 误差行为:")
print("   AR: 误差累积（早期错误被放大）")
print("   扩散: 可逐步修正（每次迭代都可调整）")
print()

# 适用任务分析
print("6. 适用场景:")
tasks = [
    ("文本生成", "AR", "GPT, LLaMA"),
    ("代码生成", "AR", "Codex, StarCoder"),
    ("图像生成", "扩散", "SD, DALL-E 3"),
    ("视频生成", "扩散", "SVD, AnimateDiff"),
    ("音频生成", "AR/扩散", "AudioLM (AR), AudioLDM (扩散)"),
    ("3D 生成", "扩散/AR", "Point-E (扩散), DALL-E (AR)"),
]

print(f"{'任务':10s} | {'常用范式':8s} | {'代表模型'}")
print("-" * 50)
for task, paradigm, model in tasks:
    print(f"{task:10s} | {paradigm:8s} | {model}")
```

## 深度学习关联

- **混合模型 (Hybrid Generation)**：DALL-E 2 使用 AR 生成图像的 CLIP 嵌入（先验模型），然后使用扩散模型生成图像；Parti 使用 AR 生成视频 token 序列。这种混合利用了两种范式的各自优势。
- **MaskGIT / MAGVIT**：提出了一种中间范式——用掩码建模做并行生成。同时预测所有位置的 token，但只保留高置信度的，低置信度位置留作下一轮精化。介于 AR 和扩散之间的"迭代并行"范式。
- **大型扩散语言模型 (Diffusion LM)**：扩散语言模型（如 Diffusion-LM, GENIE）使用连续扩散过程生成文本——在潜空间中做扩散，再投影到 token 空间。虽然目前质量不及 AR，但提供了可控生成的新方向。
- **边缘 AI 部署的考量**：AR 模型需要顺序推理，不适合并行硬件加速；扩散模型可以并行，但需要多步推理。在边缘设备上，需要根据硬件特性选择合适的范式。
