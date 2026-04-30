# 31_Inception Score (IS) 的计算逻辑

## 核心概念
- **Inception Score (IS)**：评估生成图像质量与多样性的指标，使用在 ImageNet 上预训练的 Inception-v3 分类器，不需要真实图像作为参考。
- **两个关键维度**：IS 同时衡量——(1) 每个生成图像是否被分类器以高置信度归入某个类别（质量），(2) 所有生成图像的类别预测分布是否均匀（多样性）。
- **KL 散度的核心作用**：IS 的核心是计算每个图像的预测分布 $p(y|x)$ 和边缘分布 $p(y)$ 之间的 KL 散度——高 KL 意味着每个图像被清晰分类且整体类别多样。
- **分块计算 (Split)**：IS 将生成样本分成 10 个块分别计算再取平均和标准差，以估计指标的统计稳定性。
- **对预训练分类器的依赖**：IS 假设生成图像的类别能被 Inception-v3 正确识别——如果生成图像不属于 ImageNet 的 1000 类（如医学图像），IS 不再有效。
- **已知缺陷**：IS 对模式崩溃不敏感（生成所有类别的一张图 IS 仍可能很高）、对噪声敏感、不能检测过拟合、在非 ImageNet 数据集上失效。

## 数学推导

**IS 的正式定义**：

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g} \left[ KL(p(y|x) \parallel p(y)) \right]\right)
$$

其中 $p(y|x)$ 是 Inception-v3 对生成图像 $x$ 的类别预测分布，$p(y) = \int p(y|x) p_g(x) dx \approx \frac{1}{N} \sum_{i=1}^N p(y|x^{(i)})$ 是边缘类别分布。

**KL 散度的展开**：

$$
KL(p(y|x) \parallel p(y)) = \sum_{y=1}^{1000} p(y|x) \log \frac{p(y|x)}{p(y)}
$$

**分块计算**：

将 $N$ 个生成样本随机分成 10 个大小相等的块 $\{S_1, ..., S_{10}\}$：

$$
\text{IS}_k = \exp\left(\frac{1}{|S_k|} \sum_{x \in S_k} \sum_{y=1}^{1000} p(y|x) \log \frac{p(y|x)}{\frac{1}{|S_k|} \sum_{x' \in S_k} p(y|x')}\right)
$$

最终报告：$\mu = \frac{1}{10} \sum_{k=1}^{10} \text{IS}_k$ 和 $\sigma = \sqrt{\frac{1}{9} \sum_{k=1}^{10} (\text{IS}_k - \mu)^2}$

**IS 的上下界分析**：

- 下界：如果每个图像的预测是完全均匀分布 $p(y|x) = 1/1000$，则 $KL=0$，$IS=1$
- 上界（理论）：如果每个图像被分到唯一类别且置信度为 1，且所有类别被均匀覆盖，$KL = \log 1000$，$IS = 1000$
- 实际范围：GAN 模型的 IS 通常在 2-20 之间

**IS 可分解为两部分**：

$$
\text{IS} = \exp\left(H(p(y)) + \mathbb{E}_x[H(p(y|x))]\right)
$$

其中 $H(p(y))$ 是边缘分布的熵（多样性），$\mathbb{E}_x[H(p(y|x))]$ 是条件熵的期望的相反数（质量，越小越好）。当条件熵小且边缘熵大时，IS 高。

## 直观理解
- **IS 像一次突击测验**：你给 Inception-v3 看生成的图片，它给每张图一个类别判断。如果每张图它都能高自信地说"这肯定是狗"（质量好），而且所有类别的分布比较均匀（不是只认识狗）（多样性好），IS 就高。
- **为什么 IS 有偏差**：如果你的生成器学会了在每张图上画一个 ImageNet 类别的水印，Inception 可能会高自信地识别出"这是金鱼"——IS 会很高，但图像质量很差。这就是"骗过 IS"的经典方法。
- **分块计算的必要性**：一次性计算所有样本的 $p(y)$ 会高估多样性（因为样本越多边缘分布越均匀）。分块计算再平均能更准确地反映模型的真实性能。
- **IS vs FID 的场景选择**：IS 不需要真实数据，适合评估生成图像的"类 ImageNet"质量（如 ImageNet 上的无条件生成）。FID 需要真实数据，适合评估生成分布与特定数据集的匹配度。

## 代码示例

```python
import torch
import torch.nn as nn
import numpy as np
from scipy.stats import entropy

def inception_score_from_probs(preds, num_splits=10, eps=1e-16):
    """
    从 Inception 预测概率计算 IS
    
    参数:
        preds: [N, 1000] Inception-v3 的 softmax 概率输出
        num_splits: 分块数
        eps: 防止 log(0)
    """
    N = preds.shape[0]
    split_size = N // num_splits
    
    scores = []
    for i in range(num_splits):
        # 当前块的预测
        part = preds[i * split_size: (i + 1) * split_size]
        
        # 边缘分布 p(y)
        p_y = np.mean(part, axis=0, keepdims=True)
        
        # 计算 KL 散度: KL(p(y|x) || p(y))
        # 对每个样本计算
        kl_divs = []
        for j in range(len(part)):
            kl = entropy(part[j], p_y[0])
            kl_divs.append(kl)
        
        # 块内平均并取指数
        score = np.exp(np.mean(kl_divs))
        scores.append(score)
    
    return np.mean(scores), np.std(scores)

def analyze_is_components(preds):
    """
    分析 IS 的两个分量：质量和多样性
    质量 = -E_x[H(p(y|x))] （条件熵越小，质量越好）
    多样性 = H(p(y)) （边缘熵越大，多样性越好）
    """
    # 条件熵
    cond_entropy = np.mean([entropy(p) for p in preds])
    
    # 边缘熵
    p_y = np.mean(preds, axis=0)
    marginal_entropy = entropy(p_y)
    
    # IS = exp(marginal_entropy - cond_entropy) 的近似
    # 实际上是 exp(H(p(y)) + E[H(p(y|x))])，条件熵带负号
    
    # 可视化分解（实际 IS 不是简单的 exp(差)）
    is_val, _ = inception_score_from_probs(preds)
    
    return {
        'IS': is_val,
        'conditional_entropy': cond_entropy,
        'marginal_entropy': marginal_entropy,
        'quality_indicator': -cond_entropy,  # 越大质量越好
        'diversity_indicator': marginal_entropy,  # 越大多样性越好
    }

# 模拟不同场景
print("=== Inception Score 分析 ===")
print()

# 场景 1: 高质量、高多样性的生成
np.random.seed(42)
N = 5000
# 置信度高（质量好），类别均匀（多样性好）
preds_good = np.zeros((N, 1000))
for i in range(N):
    preds_good[i, i % 1000] = 0.9
    preds_good[i] += np.random.rand(1000) * 0.1
    preds_good[i] /= preds_good[i].sum()

is_good, is_good_std = inception_score_from_probs(preds_good)
print(f"场景 1 (高质量/高多样): IS = {is_good:.2f} ± {is_good_std:.2f}")

# 场景 2: 高质量、低多样性（模式崩溃）
preds_lowdiv = np.zeros((N, 1000))
for i in range(N):
    preds_lowdiv[i, 0] = 0.9  # 全都预测为同一类（第 0 类）！
    preds_lowdiv[i] += np.random.rand(1000) * 0.1
    preds_lowdiv[i] /= preds_lowdiv[i].sum()

is_lowdiv, is_lowdiv_std = inception_score_from_probs(preds_lowdiv)
print(f"场景 2 (模式崩溃):      IS = {is_lowdiv:.2f} ± {is_lowdiv_std:.2f}")

# 场景 3: 低质量（模糊/噪声图像，预测为均匀分布）
preds_noise = np.random.dirichlet(np.ones(1000), N)
is_noise, is_noise_std = inception_score_from_probs(preds_noise)
print(f"场景 3 (噪声/模糊):     IS = {is_noise:.2f} ± {is_noise_std:.2f}")

# 场景 4: 高质量但只覆盖 10 个类别
preds_10classes = np.zeros((N, 1000))
for i in range(N):
    preds_10classes[i, np.random.randint(0, 10)] = 0.9
    preds_10classes[i] += np.random.rand(1000) * 0.1
    preds_10classes[i] /= preds_10classes[i].sum()

is_10cls, is_10cls_std = inception_score_from_probs(preds_10classes)
print(f"场景 4 (仅 10 类):     IS = {is_10cls:.2f} ± {is_10cls_std:.2f}")

print()
print("=== 组件分析（高质量/高多样场景）===")
analysis = analyze_is_components(preds_good)
for k, v in analysis.items():
    print(f"{k}: {v:.4f}")
```

## 深度学习关联
- **IS 的改进：sFID, FID 的对比**：IS 只衡量"类别级"的质量和多样性，FID 衡量特征空间中的分布距离。两者互补——IS 锚定语义类别，FID 锚定整体分布。
- **IS 在 ImageNet 无条件生成中仍然使用**：尽管有缺陷，IS + FID 的组合仍是 ImageNet 类条件生成（BigGAN, StyleGAN-XL）的标准评估套路。
- **替代指标：Precision & Recall, Density & Coverage**：为弥补 IS 无法区分质量/多样性的问题，出现了解耦指标——Precision（质量）、Recall（多样性）、Density（质量改进）、Coverage（多样性改进）。
- **CLIP Score 的出现**：在文本到图像生成中，IS 逐渐被 CLIP Score 补充或替代——CLIP Score 直接衡量生成图像与文本提示的语义对齐度，不需要固定类别系统。
