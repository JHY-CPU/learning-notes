# 30_生成模型评估指标：FID 与 IS

## 核心概念

- **FID (Fréchet Inception Distance)**：衡量生成图像分布与真实图像分布之间距离的指标。基于 Inception-v3 网络提取的特征层，计算两个高斯分布之间的 Fréchet 距离。
- **IS (Inception Score)**：衡量生成图像的"质量"和"多样性"——用 Inception-v3 分类器的预测置信度（质量）和预测标签的熵（多样性）来评估。
- **FID 的直觉**：FID 越小，生成分布越接近真实分布。FID 同时评估了质量（生成图像是否真实）和多样性（生成图像是否覆盖了整个真实分布）。
- **IS 的直觉**：IS 越大越好——高的 IS 意味着每个生成图像都能被分类器高置信度地归入某一类（质量好），且生成图像的类别分布均匀（多样性好）。
- **共同依赖 Inception-v3**：两者都依赖于在 ImageNet 上预训练的 Inception-v3 网络，因此评估存在对 ImageNet 的偏向性——如果生成的数据分布与 ImageNet 差异大，指标可能不可靠。
- **FID vs IS 的差异**：FID 需要真实图像作为参考（衡量生成分布与真实分布的差异），IS 不需要真实图像（只衡量生成图像的内部特性），但 IS 可能被"欺骗"（生成所有图像均为同一类且置信度高，IS 仍高但多样性差）。

## 数学推导

**FID 的计算**：

设 $\phi(x) \in \mathbb{R}^{2048}$ 是 Inception-v3 倒数第二层的池化特征。

真实分布特征：$\phi_r \sim \mathcal{N}(\mu_r, \Sigma_r)$（假设高斯分布）
生成分布特征：$\phi_g \sim \mathcal{N}(\mu_g, \Sigma_g)$

Fréchet 距离（也称为 Wasserstein-2 距离）定义为：

$$
\text{FID} = \|\mu_r - \mu_g\|_2^2 + \text{Tr}\left(\Sigma_r + \Sigma_g - 2(\Sigma_r \Sigma_g)^{1/2}\right)
$$

其中 $\text{Tr}$ 是矩阵的迹，$(\Sigma_r \Sigma_g)^{1/2}$ 是矩阵平方根。

FID 越低越好。通常 FID < 10 表示非常好的生成质量。

**IS 的计算**：

设 $p(y|x)$ 是 Inception-v3 对生成图像 $x$ 的类别预测分布（1000 类 ImageNet 类别）。

质量：$\mathbb{E}_x[\max_y p(y|x)]$（每个图像被高置信度分类）

多样性：$H(\mathbb{E}_x[p(y|x)])$（经验边缘分布的熵）

IS 综合这两者：

$$
\text{IS} = \exp\left(\mathbb{E}_{x \sim p_g} \left[ KL\left(p(y|x) \parallel \int p(y|x) p_g(x) dx\right) \right] \right)
$$

实际计算时，用蒙特卡洛采样近似：

$$
\text{IS} \approx \exp\left(\frac{1}{N} \sum_{i=1}^N \sum_{y=1}^{1000} p(y|x^{(i)}) \log \frac{p(y|x^{(i)})}{\frac{1}{N} \sum_{j=1}^N p(y|x^{(j)})}\right)
$$

IS 越高越好。通常 IS > 9 表示较好的生成质量。

**FID 与 IS 的对比数学**：

FID：比较生成分布和真实分布的一阶和二阶矩（均值、协方差）
IS：只分析生成分布内部的条件熵和边缘熵

FID 对模式崩溃更敏感（如果模型只生成部分类别，FID 会显著变差），IS 对模式崩溃不敏感（如果缺失的类别在评估集中不存在）。

## 直观理解

- **FID = 两道菜的味道差异程度**：你做了两道鱼香肉丝（真实分布 vs 生成分布），FID 就像用精密仪器分析两道菜的化学成分差异——差异越小说明越接近。
- **IS = 赛事的质量与多样性**：IS 像评估一场选美比赛——每位选手都要漂亮（质量），而且选手之间的风格不能都一个样（多样性）。但 IS 不知道"所有美女"的标准全集，所以如果有某类美女缺席，IS 不会惩罚。
- **为什么 FID 比 IS 更可靠**：因为 IS 可以用"作弊"手段提高（只生成容易分类的图像），而 FID 因为有真实分布作为参考，作弊更难。
- **对 Inception-v3 的依赖是双刃剑**：用 ImageNet 预训练的分类器意味着评估存在"ImageNet 中心化"偏差。如果你在生成医学影像，Inception-v3 可能提取出与医学无关的特征，导致 FID/IS 不合理。

## 代码示例

```python
import torch
import torch.nn as nn
import numpy as np
from scipy import linalg

def calculate_fid(mu1, sigma1, mu2, sigma2):
    """
    计算 FID (Fréchet Inception Distance)
    
    参数:
        mu1, sigma1: 真实分布特征的均值和协方差
        mu2, sigma2: 生成分布特征的均值和协方差
    """
    # 均值差异的 L2 范数
    diff = mu1 - mu2
    mean_diff = np.dot(diff, diff)
    
    # 协方差矩阵的迹项
    covmean, _ = linalg.sqrtm(sigma1 @ sigma2, disp=False)
    
    # 如果 sqrtm 产生复数，取实数部分
    if np.iscomplexobj(covmean):
        covmean = covmean.real
    
    fid = mean_diff + np.trace(sigma1 + sigma2 - 2 * covmean)
    return fid

def calculate_inception_score(preds, num_splits=10):
    """
    计算 Inception Score
    
    参数:
        preds: [N, 1000] Inception 的 softmax 输出
        num_splits: 分块数（用于计算均值和方差）
    """
    N = preds.shape[0]
    split_size = N // num_splits
    
    scores = []
    for i in range(num_splits):
        part = preds[i * split_size: (i + 1) * split_size]
        
        # KL(p(y|x) || p(y))
        p_y = np.mean(part, axis=0, keepdims=True)  # 边缘分布
        kl_div = part * (np.log(part + 1e-16) - np.log(p_y + 1e-16))
        kl_div = np.sum(kl_div, axis=1)
        scores.append(np.exp(np.mean(kl_div)))
    
    return np.mean(scores), np.std(scores)

# 模拟特征提取（模拟 Inception-v3 池化层输出）
def extract_features(images, model=None):
    """
    提取图像特征（模拟 Inception-v3 输出）
    
    实际使用时需加载预训练 Inception-v3
    """
    # 模拟：生成随机特征
    np.random.seed(42)
    features = np.random.randn(len(images), 2048)
    return features

def compute_fid_from_data(real_images, generated_images):
    """从真实和生成的图像计算 FID"""
    # 提取特征
    real_feats = extract_features(real_images)
    gen_feats = extract_features(generated_images)
    
    # 计算统计量
    mu_real = np.mean(real_feats, axis=0)
    sigma_real = np.cov(real_feats, rowvar=False)
    
    mu_gen = np.mean(gen_feats, axis=0)
    sigma_gen = np.cov(gen_feats, rowvar=False)
    
    # 计算 FID
    fid = calculate_fid(mu_real, sigma_real, mu_gen, sigma_gen)
    return fid

# 模拟评估
print("=== 生成模型评估指标 ===")
print()

# 模拟数据
np.random.seed(42)
N_real, N_gen = 5000, 5000
real_images = torch.randn(N_real, 3, 299, 299)
gen_images = torch.randn(N_gen, 3, 299, 299)

# FID
fid_value = compute_fid_from_data(real_images, gen_images)
print(f"FID (随机生成 vs 真实): {fid_value:.2f}")
print(f"  (< 10: 优秀, 10-30: 好, > 50: 差)")
print()

# IS
preds = np.random.dirichlet(np.ones(1000), N_gen)
is_mean, is_std = calculate_inception_score(preds)
print(f"Inception Score: {is_mean:.2f} ± {is_std:.2f}")
print(f"  (IS > 8: 好, IS > 10: 优秀, IS > 15: 极佳)")
print()

# FID 敏感度分析
print("=== FID 对不同缺陷的敏感度 ===")
# 高质量生成
mu_good = np.zeros(2048)
sigma_good = np.eye(2048)
mu_real = np.zeros(2048) + 0.1
sigma_real = np.eye(2048) * 1.1
fid_good = calculate_fid(mu_real, sigma_real, mu_good, sigma_good)
print(f"高质量 (分布接近): FID = {fid_good:.2f}")

# 低质量生成（均值偏差大）
mu_bad = np.zeros(2048) + 1.0
fid_bad = calculate_fid(mu_real, sigma_real, mu_bad, sigma_good)
print(f"低质量 (均值偏移大): FID = {fid_bad:.2f}")

# 低多样性（方差小）
sigma_lowvar = np.eye(2048) * 0.1
fid_lowdiv = calculate_fid(mu_real, sigma_real, mu_good, sigma_lowvar)
print(f"低多样性 (方差小): FID = {fid_lowdiv:.2f}")
```

## 深度学习关联

- **FID 的问题与改进**：FID 假设特征是高斯分布（实际不一定），且对样本量敏感。Clean-FID 统一了不同的预处理和特征提取实现，使 FID 比较更加可靠。
- **FID vs. LPIPS / DISTS**：FID 评估分布级差异，LPIPS/DISTS 评估图像级差异。使用两者互补评估——FID 看全局，LPIPS 看局部。
- **Precision & Recall for GANs**：FID 是一个综合性指标，无法区分"质量差"和"多样性差"。Precision-Recall 指标将两者独立评估——Precision 衡量质量（生成样本属于真实分布的比例），Recall 衡量多样性（真实分布被生成分布覆盖的比例）。
- **人类评估 (Human Evaluation)**：自动化指标无法完全替代人类评估。在文生图领域，常用的指标还包括 CLIP Score（图像与文本的对齐度）、Aesthetic Score（美学评分）等。
